import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2 
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
import shutil

# -------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Dataset Class for Skin Lesions
# -------------------------------------------------------------------------
class SkinLesionDataset(Dataset):
    """
    Dataset class for skin lesion segmentation and classification.
    Expects:
      - A metadata CSV with at least 'image_id' and 'dx' columns.
      - Images are searched for in a list of directories.
      - Segmentation masks are in a given folder. If a mask file is not found, a dummy mask (all ones) is used.
    """
    def __init__(self, image_dirs, mask_dir, metadata_df, phase='train'):
        """
        Args:
            image_dirs (list): List of directories containing images.
            mask_dir (str): Directory with segmentation masks.
            metadata_df (pd.DataFrame): DataFrame with at least 'image_id' and 'dx' columns.
            phase (str): 'train' or 'val'/'test' (affects augmentation).
        """
        self.image_dirs = image_dirs
        self.mask_dir = mask_dir
        self.metadata_df = metadata_df
        self.phase = phase

        # Map lesion types to integer labels.
        self.class_map = {
            'akiec': 0,  # Actinic Keratosis
            'bcc': 1,    # Basal Cell Carcinoma
            'bkl': 2,    # Benign Keratosis
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic Nevus
            'vasc': 6    # Vascular Lesion
        }
        # Reverse mapping for convenience
        self.class_names = {v: k for k, v in self.class_map.items()}
        self.full_class_names = {
            0: 'Actinic Keratosis (akiec)',
            1: 'Basal Cell Carcinoma (bcc)',
            2: 'Benign Keratosis (bkl)',
            3: 'Dermatofibroma (df)',
            4: 'Melanoma (mel)',
            5: 'Melanocytic Nevus (nv)',
            6: 'Vascular Lesion (vasc)'
        }
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.metadata_df)

    def get_transforms(self):
        """Define augmentation transforms (applied jointly on image and mask)."""
        if self.phase == 'train':
            return A.Compose([
                A.Resize(224, 224),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ], p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def find_image_file(self, image_id):
        """Search for the image file in the provided directories."""
        for dir_path in self.image_dirs:
            image_path = os.path.join(dir_path, f"{image_id}.jpg")
            if os.path.exists(image_path):
                return image_path
        logger.error(f"Image file {image_id}.jpg not found in any of the specified directories.")
        raise FileNotFoundError(f"Image file {image_id}.jpg not found in any of the specified directories.")

    def find_mask_file(self, image_id):
        """Search for a mask file using common naming patterns."""
        mask_patterns = [
            f"{image_id}_segmentation.png",
            f"{image_id}_segmented.png",
        ]
        for pattern in mask_patterns:
            mask_path = os.path.join(self.mask_dir, pattern)
            if os.path.exists(mask_path):
                return mask_path
        return None

    def get_class_name(self, idx):
        """Get the class name for an image by index."""
        row = self.metadata_df.iloc[idx]
        return row['dx']

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_id = row['image_id']
        label = self.class_map[row['dx']]

        # Load image from one of the directories
        image_path = self.find_image_file(image_id)
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to the target size first
        image = cv2.resize(image, (224, 224))

        # Load mask (or use a dummy mask if not found)
        mask_path = self.find_mask_file(image_id)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask for {image_id}. Using dummy mask.")
                mask = np.ones((224, 224), dtype=np.uint8)
        else:
            logger.warning(f"No mask found for {image_id}. Using dummy mask.")
            mask = np.ones((224, 224), dtype=np.uint8)

        # Resize mask
        mask = cv2.resize(mask, (224, 224))
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # Apply augmentations simultaneously on image and mask
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].float()  # for BCEWithLogitsLoss

        return image_tensor, mask_tensor, label, image_id

# -------------------------------------------------------------------------
# Self-Attention Module
# -------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """
    Self-Attention block.
    This module computes an attention map over the spatial dimensions
    and re-weights the input feature map accordingly.
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x : input feature maps (B, C, H, W)
        returns:
            out : self-attention value + input feature
            attention: B x N x N (N=H*W)
        """
        B, C, width, height = x.size()
        proj_query = self.query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # (B, N, C//8)
        proj_key = self.key_conv(x).view(B, -1, width * height)                       # (B, C//8, N)
        energy = torch.bmm(proj_query, proj_key)                                      # (B, N, N)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, width * height)                   # (B, C, N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, width, height)

        out = self.gamma * out + x
        return out, attention

# -------------------------------------------------------------------------
# Encoder Creation Function
# -------------------------------------------------------------------------
def create_encoder(base_channels=64):
    return nn.Sequential(
         nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels),
         nn.ReLU(inplace=True),
         nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(2),  # 224 -> 112

         nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels*2),
         nn.ReLU(inplace=True),
         nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels*2),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(2),  # 112 -> 56

         nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels*4),
         nn.ReLU(inplace=True),
         nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels*4),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(2),  # 56 -> 28

         nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels*8),
         nn.ReLU(inplace=True),
         nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
         nn.BatchNorm2d(base_channels*8),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(2)   # 28 -> 14
    )

# -------------------------------------------------------------------------
# UnifiedJNet: A Single Model for Segmentation & Classification with Overlay
# -------------------------------------------------------------------------
class UnifiedJNet(nn.Module):
    """
    A unified model that performs both segmentation and classification in one pass.

    Workflow:
      - A segmentation encoder extracts features from the original image.
      - A self-attention block is applied to the segmentation features.
      - A segmentation decoder (using transposed convolutions) produces raw segmentation logits (224×224).
      - The sigmoid of the logits yields a soft lesion mask.
      - An overlay image is computed by replacing the lesion region in the original image with white.
      - A separate classification encoder processes both the original and the overlay image.
      - Their features are fused (averaged) and fed to a fully-connected layer for classification.
    """
    def __init__(self, num_classes=7, base_channels=64):
        super(UnifiedJNet, self).__init__()
        # Segmentation branch
        self.seg_encoder = create_encoder(base_channels)
        self.attention = SelfAttention(in_dim=base_channels*8)
        # Replace single bilinear upsampling with a multi-stage decoder for sharper masks
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=2, stride=2)
        )

        # Classification branch
        self.cls_encoder = create_encoder(base_channels)
        self.fc = nn.Linear(base_channels*8, num_classes)

        # Precompute "white" in normalized space for overlay
        # (Assuming normalization: (x - mean)/std, white in [0..1] is [1,1,1])
        white_vals = [(1 - m)/s for m, s in zip([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])]
        self.register_buffer('white', torch.tensor(white_vals).view(1, 3, 1, 1))

    def forward(self, x):
        # Segmentation branch
        seg_features = self.seg_encoder(x)
        seg_features, attn_map = self.attention(seg_features)
        raw_seg_logits = self.seg_decoder(seg_features)  # Output size should be 224x224 now
        seg_prob = torch.sigmoid(raw_seg_logits)

        # Create overlay.
        # Option 1 (soft overlay): use seg_prob directly (differentiable)
        overlay = x * (1 - seg_prob) + self.white * seg_prob

        # Option 2 (hard overlay): uncomment the following two lines to use a binary mask.
        # Note: This detaches gradients from the segmentation branch.
        # hard_mask = (seg_prob > 0.5).float()
        # overlay = x * (1 - hard_mask) + self.white * hard_mask

        # Classification branch
        orig_features = self.cls_encoder(x)
        overlay_features = self.cls_encoder(overlay)
        combined_features = (orig_features + overlay_features) / 2.0
        gap = combined_features.mean(dim=(2, 3))
        cls_logits = self.fc(gap)

        return raw_seg_logits, cls_logits, overlay

# -------------------------------------------------------------------------
# Metric Tracker for Training/Evaluation
# -------------------------------------------------------------------------
class MetricTracker:
    """Tracks running losses and classification accuracy."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.seg_loss = 0.0
        self.cls_loss = 0.0
        self.correct = 0
        self.total_samples = 0
        self.batch_count = 0

    def update(self, loss, seg_loss, cls_loss, cls_outputs, labels):
        self.total_loss += loss.item()
        self.seg_loss += seg_loss.item()
        self.cls_loss += cls_loss.item()
        self.batch_count += 1
        self.total_samples += labels.size(0)
        _, preds = cls_outputs.max(1)
        self.correct += preds.eq(labels).sum().item()

    @property
    def avg_loss(self):
        return self.total_loss / self.batch_count if self.batch_count else 0

    @property
    def avg_seg_loss(self):
        return self.seg_loss / self.batch_count if self.batch_count else 0

    @property
    def avg_cls_loss(self):
        return self.cls_loss / self.batch_count if self.batch_count else 0

    @property
    def accuracy(self):
        return 100. * self.correct / self.total_samples if self.total_samples else 0

# -------------------------------------------------------------------------
# Helper: Plot and save training history
# -------------------------------------------------------------------------
def plot_training_history(history, save_path):
    """Plot and save the training and validation metrics history."""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_total_loss'], label='Train Total Loss')
    plt.plot(history['val_total_loss'], label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_seg_loss'], label='Train Seg Loss')
    plt.plot(history['val_seg_loss'], label='Val Seg Loss')
    plt.title('Segmentation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_cls_loss'], label='Train Cls Loss')
    plt.plot(history['val_cls_loss'], label='Val Cls Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Classification Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------------------------------------------------------
# Helper: Save overlay comparison images (GT vs. Pred) + classification info
# -------------------------------------------------------------------------
def save_overlay_comparison(
    image_tensor, mask_tensor, pred_mask_tensor,
    gt_label, pred_label, class_names, image_id,
    epoch, output_folder
):
    """
    Save a figure with:
      1) Original image
      2) Ground truth mask
      3) Predicted mask
      4) Overlay (predicted) on the original image
    Also show GT class and predicted class in the title.

    Args:
        image_tensor: (C,H,W) tensor
        mask_tensor: (H,W) ground truth
        pred_mask_tensor: (H,W) predicted
        gt_label: int
        pred_label: int
        class_names: dict {class_index: class_str}
        image_id: str
        epoch: int
        output_folder: where to save the result
    """
    os.makedirs(output_folder, exist_ok=True)

    # Move to CPU numpy
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mask_np = mask_tensor.cpu().numpy()
    pred_mask_np = pred_mask_tensor.cpu().numpy()

    # Denormalize if needed (because we used ImageNet means/std)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    # Create overlay
    overlay_np = image_np.copy()
    # Where predicted mask > 0.5, we highlight
    overlay_mask = (pred_mask_np > 0.5)
    # You can color the overlay region in red, for example:
    overlay_np[overlay_mask, 0] = 1.0  # Red channel
    overlay_np[overlay_mask, 1] = 0.0
    overlay_np[overlay_mask, 2] = 0.0

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(image_np)
    axs[0].set_title(f"Original\nID: {image_id}")
    axs[0].axis('off')

    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(pred_mask_np, cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    axs[3].imshow(overlay_np)
    axs[3].set_title("Overlay (Predicted)")
    axs[3].axis('off')

    # Super-title with classification
    fig.suptitle(
        f"Epoch {epoch} | GT Class: {class_names[gt_label]} | Pred Class: {class_names[pred_label]}",
        fontsize=14, y=1.08
    )

    save_path = os.path.join(output_folder, f"epoch_{epoch}_img_{image_id}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# Training Function
# -------------------------------------------------------------------------
def train_model(
    train_loader, 
    val_loader, 
    model, 
    num_epochs=50, 
    device='cuda',
    lambda_seg=1.0, 
    lambda_cls=1.0, 
    output_dir="results"
):
    """
    Train the UnifiedJNet model for both segmentation and classification.
    Save outputs (checkpoints, visualizations, overlays) in an organized manner.
    """
    # Create main result directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(output_dir, timestamp)

    # Create subdirectories for different outputs
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    segmentation_dir = os.path.join(result_dir, "segmentation")
    overlay_dir = os.path.join(result_dir, "overlay_images")  # as per user request
    visualization_dir = os.path.join(result_dir, "visualizations")

    # Create all directories
    for directory in [checkpoint_dir, segmentation_dir, overlay_dir, visualization_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"\nAll outputs will be saved in: {result_dir}")

    # Losses and optimizer
    seg_criterion = nn.BCEWithLogitsLoss()
    cls_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    history = {
        'train_total_loss': [], 'train_seg_loss': [], 'train_cls_loss': [], 'train_acc': [],
        'val_total_loss': [], 'val_seg_loss': [], 'val_cls_loss': [], 'val_acc': [],
        'learning_rates': []
    }

    print("\nTraining Configuration:")
    print("=" * 80)
    print("UnifiedJNet: Single model for segmentation & classification with overlay")
    print(f"Optimizer: AdamW (lr={optimizer.param_groups[0]['lr']})")
    print(f"Scheduler: CosineAnnealingLR, Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Checkpoint Saving: Every 10 epochs")
    print("=" * 80)

    model.to(device)

    # We'll keep class_names from the dataset for easy reference
    class_names = getattr(train_loader.dataset, 'full_class_names', None)
    if class_names is None:
        class_names = {
            0: 'akiec',
            1: 'bcc',
            2: 'bkl',
            3: 'df',
            4: 'mel',
            5: 'nv',
            6: 'vasc'
        }

    for epoch in range(num_epochs):
        model.train()
        train_metrics = MetricTracker()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, masks, labels, _ in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            seg_logits, cls_logits, _ = model(images)
            seg_loss = seg_criterion(seg_logits, masks.unsqueeze(1))
            cls_loss = cls_criterion(cls_logits, labels)
            loss = lambda_seg * seg_loss + lambda_cls * cls_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_metrics.update(loss, seg_loss, cls_loss, cls_logits, labels)
            train_pbar.set_postfix({
                'Total Loss': f'{train_metrics.avg_loss:.4f}',
                'Cls Acc': f'{train_metrics.accuracy:.2f}%'
            })

        # Validation phase
        model.eval()
        val_metrics = MetricTracker()
        with torch.no_grad():
            for images, masks, labels, image_ids in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                seg_logits, cls_logits, _ = model(images)

                seg_loss = seg_criterion(seg_logits, masks.unsqueeze(1))
                cls_loss = cls_criterion(cls_logits, labels)
                loss = lambda_seg * seg_loss + lambda_cls * cls_loss
                val_metrics.update(loss, seg_loss, cls_loss, cls_logits, labels)

                preds = cls_logits.argmax(dim=1)
                for i in range(images.size(0)):
                    pred_mask = seg_logits[i, 0]  # shape: (224,224)
                    save_overlay_comparison(
                        image_tensor=images[i],
                        mask_tensor=masks[i],
                        pred_mask_tensor=pred_mask,
                        gt_label=labels[i].item(),
                        pred_label=preds[i].item(),
                        class_names=class_names,
                        image_id=image_ids[i],
                        epoch=epoch+1,
                        output_folder=overlay_dir
                    )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Record training/val stats
        history['train_total_loss'].append(train_metrics.avg_loss)
        history['train_seg_loss'].append(train_metrics.avg_seg_loss)
        history['train_cls_loss'].append(train_metrics.avg_cls_loss)
        history['train_acc'].append(train_metrics.accuracy)
        history['val_total_loss'].append(val_metrics.avg_loss)
        history['val_seg_loss'].append(val_metrics.avg_seg_loss)
        history['val_cls_loss'].append(val_metrics.avg_cls_loss)
        history['val_acc'].append(val_metrics.accuracy)
        history['learning_rates'].append(current_lr)

        print(f"Epoch {epoch+1:2d}: "
              f"Train Loss={train_metrics.avg_loss:.4f} "
              f"(Seg: {train_metrics.avg_seg_loss:.4f}, Cls: {train_metrics.avg_cls_loss:.4f}), "
              f"Train Acc={train_metrics.accuracy:.2f}%, "
              f"Val Loss={val_metrics.avg_loss:.4f} "
              f"(Seg: {val_metrics.avg_seg_loss:.4f}, Cls: {val_metrics.avg_cls_loss:.4f}), "
              f"Val Acc={val_metrics.accuracy:.2f}%, LR={current_lr:.2e}")

        # Save best model checkpoint
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': val_metrics.avg_loss,
                'val_acc': val_metrics.accuracy,
                'history': history
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  --> New best model saved at epoch {epoch+1} (Val Loss: {val_metrics.avg_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"  --> Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

    # Save training history plot
    plot_training_history(history, os.path.join(visualization_dir, 'training_history.png'))
    
    print("\nTraining Complete!")
    print("=" * 80)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Model checkpoints saved in: {checkpoint_dir}")
    print(f"All outputs saved in: {result_dir}")

    # Load the best model for return
    best_checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['state_dict'])
    
    return model, history, result_dir

# -------------------------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------------------------
def evaluate_model(model, test_loader, device='cuda', visualization_dir=None):
    """
    Evaluate the unified model on the test set.
    Computes segmentation loss and classification accuracy.
    Also saves confusion matrix if visualization_dir is provided.
    """
    model.eval()
    model.to(device)

    seg_criterion = nn.BCEWithLogitsLoss()
    cls_criterion = nn.CrossEntropyLoss()

    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_loss = 0.0
    correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []
    class_correct = [0] * 7
    class_total = [0] * 7

    test_overlay_dir = None
    if visualization_dir is not None:
        test_overlay_dir = os.path.join(visualization_dir, "test_overlays")
        os.makedirs(test_overlay_dir, exist_ok=True)

    class_names = getattr(test_loader.dataset, 'full_class_names', {
        0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
    })

    with torch.no_grad():
        for images, masks, labels, image_ids in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            seg_logits, cls_logits, _ = model(images)
            seg_loss = seg_criterion(seg_logits, masks.unsqueeze(1))
            cls_loss = cls_criterion(cls_logits, labels)
            loss = seg_loss + cls_loss

            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()

            total_samples += labels.size(0)
            _, preds = cls_logits.max(1)
            correct += preds.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label_i = labels[i].item()
                class_total[label_i] += 1
                if preds[i].item() == label_i:
                    class_correct[label_i] += 1

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if test_overlay_dir is not None:
                for i in range(images.size(0)):
                    pred_mask = seg_logits[i, 0]
                    save_overlay_comparison(
                        image_tensor=images[i],
                        mask_tensor=masks[i],
                        pred_mask_tensor=pred_mask,
                        gt_label=labels[i].item(),
                        pred_label=preds[i].item(),
                        class_names=class_names,
                        image_id=image_ids[i],
                        epoch=0,  # or "test"
                        output_folder=test_overlay_dir
                    )

    avg_seg_loss = total_seg_loss / len(test_loader)
    avg_cls_loss = total_cls_loss / len(test_loader)
    avg_total_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total_samples

    print("\nTest Evaluation Results:")
    print("=" * 80)
    print(f"Avg Segmentation Loss: {avg_seg_loss:.4f}")
    print(f"Avg Classification Loss: {avg_cls_loss:.4f}")
    print(f"Avg Total Loss: {avg_total_loss:.4f}")
    print(f"Test Classification Accuracy: {accuracy:.2f}%")

    print("\nPer-Class Accuracy:")
    for i in range(7):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    print("=" * 80)

    if visualization_dir is not None:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[class_names[i].split()[0] for i in range(7)],
                    yticklabels=[class_names[i].split()[0] for i in range(7)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(visualization_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
    return {
        'avg_seg_loss': avg_seg_loss,
        'avg_cls_loss': avg_cls_loss,
        'avg_total_loss': avg_total_loss,
        'accuracy': accuracy
    }

def save_binary_segmentations(model, dataset, output_dir, device='cuda'):
    """
    Process all images in 'dataset' with 'model' and save binary segmentation masks.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(len(dataset)):
        image, mask, label, image_id = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            seg_logits, _, _ = model(image_tensor)
            seg_prob = torch.sigmoid(seg_logits)
            seg_mask = (seg_prob > 0.5).float().squeeze().cpu().numpy() * 255
            seg_mask = seg_mask.astype(np.uint8)
        output_path = os.path.join(output_dir, f"{image_id}_binary.png")
        cv2.imwrite(output_path, seg_mask)
    print("Binary segmentation masks saved.")

def save_color_overlay_segmentation(model, dataset, output_dir, device='cuda'):
    """
    Process all images in 'dataset' with 'model' and save color overlay images.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(len(dataset)):
        image, mask, label, image_id = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, overlay = model(image_tensor)
            overlay_np = overlay.squeeze().cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            overlay_np = overlay_np * std + mean
            overlay_np = np.clip(overlay_np, 0, 1)
            overlay_np = (overlay_np * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, f"{image_id}_overlay.png")
        cv2.imwrite(output_path, cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR))
    print("Color overlay images saved.")

def visualize_segmentation_results(model, dataset, device='cuda', num_examples=5, save_folder='visualisation'):
    """
    Randomly select a few images from 'dataset' and save plots of:
      - The original image
      - Ground truth mask
      - Predicted segmentation mask
    The plots are saved in the specified folder instead of being displayed.
    """
    model.eval()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    for idx in indices:
        image, mask, label, image_id = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            seg_logits, _, _ = model(image_tensor)
            seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np)
        axs[0].set_title(f"Original: {image_id}")
        axs[0].axis('off')
        
        axs[1].imshow(mask.cpu().numpy(), cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis('off')
        
        axs[2].imshow(seg_prob, cmap='gray')
        axs[2].set_title("Predicted Mask")
        axs[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_folder, f"segmentation_result_{image_id}.png")
        plt.savefig(save_path)
        plt.close()

def main():
    try:
        torch.manual_seed(42)
        np.random.seed(42)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        metadata_path = r"/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_metadata.csv"
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded dataset with {len(metadata_df)} images")

        image_dirs = [
            r"/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_1",
            r"/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000/HAM10000_images_part_2"
        ]
        mask_dir = r"/dist_home/suryansh/dl/Skin-Lesion-Classification/Datasets/HAM10000_segmentations_lesion_tschandl"
        full_dataset = SkinLesionDataset(
            image_dirs=image_dirs,
            mask_dir=mask_dir,
            metadata_df=metadata_df,
            phase='val'
        )

        train_df, temp_df = train_test_split(
            metadata_df,
            test_size=0.3,
            stratify=metadata_df['dx'],
            random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df['dx'],
            random_state=42
        )
        logger.info(f"Train set: {len(train_df)} images")
        logger.info(f"Validation set: {len(val_df)} images")
        logger.info(f"Test set: {len(test_df)} images")

        train_dataset = SkinLesionDataset(
            image_dirs=image_dirs,
            mask_dir=mask_dir,
            metadata_df=train_df,
            phase='train'
        )
        val_dataset = SkinLesionDataset(
            image_dirs=image_dirs,
            mask_dir=mask_dir,
            metadata_df=val_df,
            phase='val'
        )
        test_dataset = SkinLesionDataset(
            image_dirs=image_dirs,
            mask_dir=mask_dir,
            metadata_df=test_df,
            phase='val'
        )

        num_workers = 2
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        model = UnifiedJNet(num_classes=7).to(device)

        model, history, result_dir = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            num_epochs=100,
            device=device
        )

        logger.info("Starting final model evaluation...")
        test_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device
        )

        binary_seg_output_dir = os.path.join("segmented_images", datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(binary_seg_output_dir, exist_ok=True)
        save_binary_segmentations(model, full_dataset, binary_seg_output_dir, device=device)
        logger.info(f"Binary segmentation masks saved in: {binary_seg_output_dir}")

        overlay_output_dir = os.path.join("overlay_images", datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(overlay_output_dir, exist_ok=True)
        save_color_overlay_segmentation(model, full_dataset, overlay_output_dir, device=device)
        logger.info(f"Color overlay images saved in: {overlay_output_dir}")

        visualize_segmentation_results(model, full_dataset, device=device)

        logger.info("\nFinal Results Summary:")
        logger.info("=" * 80)
        logger.info(f"Training Accuracy: {history['train_acc'][-1]:.2f}%")
        logger.info(f"Validation Accuracy: {history['val_acc'][-1]:.2f}%")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
