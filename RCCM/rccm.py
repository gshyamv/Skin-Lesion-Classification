import os
import random
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Define Paths and Constants
# ------------------------------
image_dirs = [
    r'/dist_home/suryansh/dl/Skin-Lesion-Classification/SMOTE/smote_maximum_clarity'
]
mask_dir = r'/dist_home/suryansh/dl/Skin-Lesion-Classification/SMOTE/smote_maximum_clarity_masks'
metadata_path = r'/dist_home/suryansh/dl/Skin-Lesion-Classification/SMOTE/smote_maximum_clarity/smote_metadata.csv'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
# For saving training metrics plot
TRAIN_METRICS_OUTPUT = r"/dist_home/suryansh/dl/Skin-Lesion-Classification/RCCM/rcmm_output/training_metrics.png"
# For saving segmentation overlays
OVERLAY_OUTPUT_DIR = r"/dist_home/suryansh/dl/Skin-Lesion-Classification/RCCM/rcmm_output/overlays"

# ------------------------------
# Synchronized Transformations
# ------------------------------
class SynchronizedTransform:
    """
    Applies the same random flips and rotations to both image and mask.
    """
    def __init__(self, flip_prob=0.5, rotation_degrees=(0, 90, 180, 270)):
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if random.random() < self.flip_prob:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        angle = random.choice(self.rotation_degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return image, mask

# ------------------------------
# Custom Dataset Definition
# ------------------------------
class HAM10000Dataset(Dataset):
    """
    Custom Dataset for HAM10000 data that loads:
      - An image (resized to IMAGE_SIZE)
      - A corresponding segmentation mask (resized to IMAGE_SIZE)
      - A classification label from the dx_type column in metadata
    """
    def __init__(self, df, image_dirs, mask_dir, label_dict=None, transform=None, sync_transform=None, image_size=(256,256)):
        """
        Args:
            df: pandas DataFrame with columns 'image_id' and 'dx_type'.
            image_dirs: List of directories containing image files (with .jpg extension).
            mask_dir: Directory containing the segmentation masks (with _mask_mask.png suffix).
            label_dict: Mapping from dx_type (string) to integer label.
            transform: Transformations to be applied to both image and mask (e.g. ToTensor).
            sync_transform: Synchronized transformations (e.g., flips and rotations) for both image and mask.
            image_size: Tuple (width, height) for resizing images and masks.
        """
        self.df = df
        self.image_dirs = image_dirs
        self.mask_dir = mask_dir
        self.label_dict = label_dict if label_dict is not None else {"histo": 0}
        self.transform = transform
        self.sync_transform = sync_transform
        self.image_size = image_size

        # Build list of indices that have both image and mask files.
        self.valid_indices = []
        for idx in range(len(self.df)):
            image_id = self.df.iloc[idx]["image_id"]
            if self.image_exists(image_id) and self.mask_exists(image_id):
                self.valid_indices.append(idx)

    
    def apply_clahe(self, image):
        # Convert PIL image to NumPy array
        img_np = np.array(image)

        # Convert RGB to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        # Merge channels and convert back to RGB
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Convert back to PIL
        return Image.fromarray(final)
    
    def image_exists(self, image_id):
        for d in self.image_dirs:
            if os.path.exists(os.path.join(d, f"{image_id}.jpg")):
                return True
        return False

    def mask_exists(self, image_id):
        return os.path.exists(os.path.join(self.mask_dir, f"{image_id}_mask_mask.png"))

    def load_image(self, image_id):
        for d in self.image_dirs:
            path = os.path.join(d, f"{image_id}.jpg")
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        return None

    def load_mask(self, image_id):
        path = os.path.join(self.mask_dir, f"{image_id}_mask_mask.png")
        if os.path.exists(path):
            return Image.open(path).convert("L")
        return None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Use the valid index from our filtered list
        row_idx = self.valid_indices[idx]
        row = self.df.iloc[row_idx]
        image_id = row["image_id"]
        dx_type_str = row["dx_type"]
        label = self.label_dict[dx_type_str] if dx_type_str in self.label_dict else 0

        # Load image and mask
        image = self.load_image(image_id)
        mask = self.load_mask(image_id)

        # Resize both image and mask
        if self.image_size:
            image = image.resize(self.image_size)
            mask = mask.resize(self.image_size)
            
        # Apply CLAHE
        image = self.apply_clahe(image)

        # Apply synchronized transformations if provided
        if self.sync_transform:
            image, mask = self.sync_transform(image, mask)

        # Apply basic transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()  # Binarize mask
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()

        return image, mask, label

# ------------------------------
# Load Metadata and Split Data
# ------------------------------
metadata = pd.read_csv(metadata_path)
label_dict = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42)

# ------------------------------
# Define Transformations
# ------------------------------
basic_transform = transforms.ToTensor()
sync_transform = SynchronizedTransform(flip_prob=0.5)

# ------------------------------
# Create Datasets and DataLoaders
# ------------------------------
train_dataset = HAM10000Dataset(
    df=train_df,
    image_dirs=image_dirs,
    mask_dir=mask_dir,
    label_dict=label_dict,
    transform=basic_transform,
    sync_transform=sync_transform,
    image_size=IMAGE_SIZE
)

val_dataset = HAM10000Dataset(
    df=val_df,
    image_dirs=image_dirs,
    mask_dir=mask_dir,
    label_dict=label_dict,
    transform=basic_transform,
    sync_transform=None,  # No augmentation for validation
    image_size=IMAGE_SIZE
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

# ------------------------------
# Model Definition (Rccm)
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Basic convolutional block used in the encoder and decoder
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class RCCMNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, num_classes=7):
        super(RCCMNet, self).__init__()
        
        # Encoder
        self.conv1 = ConvBlock(img_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = ConvBlock(512, 1024)
        
        # Decoder for Segmentation (Multi-scale outputs)
        self.att4 = AttentionGate(512, 512, 256)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.seg_out4 = nn.Conv2d(512, output_ch, kernel_size=1)
        
        self.att3 = AttentionGate(256, 256, 128)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.seg_out3 = nn.Conv2d(256, output_ch, kernel_size=1)
        
        self.att2 = AttentionGate(128, 128, 64)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.seg_out2 = nn.Conv2d(128, output_ch, kernel_size=1)
        
        self.att1 = AttentionGate(64, 64, 32)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.seg_out1 = nn.Conv2d(64, output_ch, kernel_size=1)
        
        # Region Confidence Module (RCM)
        common_ch = 256
        self.proj_e1 = nn.Conv2d(64, common_ch, kernel_size=1)
        self.proj_e2 = nn.Conv2d(128, common_ch, kernel_size=1)
        self.proj_e3 = nn.Conv2d(256, common_ch, kernel_size=1)
        self.proj_e4 = nn.Conv2d(512, common_ch, kernel_size=1)
        
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0
        self.alpha4 = 1.0
        
        # Classification Branch
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(common_ch, num_classes)

    def forward(self, x):
        e1 = self.conv1(x)
        p1 = self.pool1(e1)
        
        e2 = self.conv2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.conv3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.conv4(p3)
        p4 = self.pool4(e4)
        
        e5 = self.conv5(p4)

        # Decoder with attention gates
        d4 = self.up4(e5)
        x4 = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)
        s4 = self.seg_out4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        s3 = self.seg_out3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        s2 = self.seg_out2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        s1 = self.seg_out1(d1)

        s2_up = F.interpolate(s2, size=s1.shape[2:], mode='bilinear', align_corners=True)
        s3_up = F.interpolate(s3, size=s1.shape[2:], mode='bilinear', align_corners=True)
        s4_up = F.interpolate(s4, size=s1.shape[2:], mode='bilinear', align_corners=True)
        seg_out = (s1 + s2_up + s3_up + s4_up) / 4.0
        seg_out = torch.sigmoid(seg_out)

        target_size = e4.shape[2:]
        p1 = torch.sigmoid(F.interpolate(s1, size=target_size, mode='bilinear', align_corners=True))
        p2 = torch.sigmoid(F.interpolate(s2, size=target_size, mode='bilinear', align_corners=True))
        p3 = torch.sigmoid(F.interpolate(s3, size=target_size, mode='bilinear', align_corners=True))
        p4 = torch.sigmoid(s4)

        f1 = F.interpolate(self.proj_e1(e1), size=target_size, mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.proj_e2(e2), size=target_size, mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.proj_e3(e3), size=target_size, mode='bilinear', align_corners=True)
        f4 = self.proj_e4(e4)

        weighted_f1 = self.alpha1 * f1 * p1
        weighted_f2 = self.alpha2 * f2 * p2
        weighted_f3 = self.alpha3 * f3 * p3
        weighted_f4 = self.alpha4 * f4 * p4

        fused_feature = weighted_f1 + weighted_f2 + weighted_f3 + weighted_f4

        pooled = self.global_pool(fused_feature)
        pooled = pooled.view(pooled.size(0), -1)
        cls_out = self.classifier(pooled)

        return seg_out, cls_out

    def compute_ccm_weight(self, cls_out, target_cls):
        log_probs = F.log_softmax(cls_out, dim=1)
        kl_div = F.kl_div(log_probs, target_cls, reduction='none').sum(dim=1)
        weight = 1.0 / (1.0 + kl_div + 1e-8)
        return weight

    def compute_loss(self, seg_out, cls_out, target_seg, target_cls, lambda_cls=1.0, lambda_entropy_seg=0.1, lambda_entropy_cls=0.1):
        eps = 1e-8
        
        seg_loss = F.binary_cross_entropy(seg_out, target_seg)
        seg_entropy = - (seg_out * torch.log(seg_out + eps) + (1 - seg_out) * torch.log(1 - seg_out + eps))
        seg_entropy_loss = torch.mean(seg_entropy)
        
        cls_loss = F.cross_entropy(cls_out, torch.argmax(target_cls, dim=1))
        cls_probs = F.softmax(cls_out, dim=1)
        cls_entropy = - torch.sum(cls_probs * torch.log(cls_probs + eps), dim=1)
        cls_entropy_loss = torch.mean(cls_entropy)
        
        ccm_weight = self.compute_ccm_weight(cls_out, target_cls)
        weighted_seg_loss = seg_loss * ccm_weight.mean()
        
        total_seg_loss = weighted_seg_loss + lambda_entropy_seg * seg_entropy_loss
        total_cls_loss = cls_loss + lambda_entropy_cls * cls_entropy_loss
        total_loss = total_seg_loss + lambda_cls * total_cls_loss
        
        return total_loss, total_seg_loss, total_cls_loss, ccm_weight

# ------------------------------
# Model Loading
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Alias the old class name to your current class (safe if trusted)
AttentionUNet = RCCMNet
torch.serialization.add_safe_globals([AttentionUNet])
pretrained_path = r'/dist_home/suryansh/dl/Skin-Lesion-Classification/attention unet/attention_unet_model_final.pth'
pretrained_weights = torch.load(pretrained_path, map_location=device, weights_only=False)

model = RCCMNet(img_ch=3, output_ch=1, num_classes=len(label_dict)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------
# Losses and Metrics for Training
# ------------------------------
def dice_coefficient(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, pred, target):
        return 1.0 - dice_coefficient(pred, target)

def iou_score(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

cls_criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    dice_scores = []
    iou_scores = []
    pixel_accuracies = []

    for images, masks, labels in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        #print("labels.shape:", labels.shape)
        #print("labels[0]:", labels[0])
        #print("labels.dtype:", labels.dtype)

        optimizer.zero_grad()
        seg_out, cls_out = model(images)

        seg_loss = DiceLoss()(seg_out, masks)
        cls_loss = cls_criterion(cls_out, labels)
        loss = seg_loss + cls_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_seg_loss += seg_loss.item()
        running_cls_loss += cls_loss.item()

        seg_preds = (seg_out > 0.5).float()
        dice_scores.append(dice_coefficient(seg_preds, masks).item())
        iou_scores.append(iou_score(seg_preds, masks).item())
        pixel_accuracies.append(pixel_accuracy(seg_preds, masks))

    epoch_loss = running_loss / len(dataloader)
    epoch_seg_loss = running_seg_loss / len(dataloader)
    epoch_cls_loss = running_cls_loss / len(dataloader)
    epoch_dice = sum(dice_scores) / len(dice_scores)
    epoch_iou = sum(iou_scores) / len(iou_scores)
    epoch_pixel_acc = sum(pixel_accuracies) / len(pixel_accuracies)

    return epoch_loss, epoch_seg_loss, epoch_cls_loss, epoch_dice, epoch_iou, epoch_pixel_acc

# ------------------------------
# Training Loop with Metric Plotting
# ------------------------------
num_epochs = 20

train_loss_history      = []
train_seg_loss_history  = []
train_cls_loss_history  = []
train_dice_history      = []
train_iou_history       = []
train_pixel_acc_history = []

import os
import torch.optim as optim

# Define a fixed checkpoint path
checkpoint_path = r"/dist_home/suryansh/dl/Skin-Lesion-Classification/RCCM/rcmm_output/overlays/checkpoints/checkpoint.pth"

for epoch in range(num_epochs):
    epoch_loss, epoch_seg_loss, epoch_cls_loss, epoch_dice, epoch_iou, epoch_pixel_acc = train_one_epoch(model, train_loader, optimizer, device)
    
    train_loss_history.append(epoch_loss)
    train_seg_loss_history.append(epoch_seg_loss)
    train_cls_loss_history.append(epoch_cls_loss)
    train_dice_history.append(epoch_dice)
    train_iou_history.append(epoch_iou)
    train_pixel_acc_history.append(epoch_pixel_acc)
    
    print(f"[Epoch {epoch+1}/{num_epochs}] Total Loss: {epoch_loss:.4f} | Seg Loss: {epoch_seg_loss:.4f} | Cls Loss: {epoch_cls_loss:.4f} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f} | Pixel Acc: {epoch_pixel_acc:.4f}")
    
    # Save a checkpoint every 10 epochs and delete the previous one
    if (epoch + 1) % 10 == 0:
        # If a checkpoint already exists, delete it
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        # Save the current checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")


epochs_range = range(1, num_epochs+1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs_range, train_loss_history, label='Total Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss over Epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs_range, train_seg_loss_history, label='Seg Loss', marker='o')
plt.plot(epochs_range, train_cls_loss_history, label='Cls Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Segmentation and Classification Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs_range, train_dice_history, label='Dice Coefficient', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.title('Dice Coefficient over Epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs_range, train_iou_history, label='IoU', marker='o')
plt.plot(epochs_range, train_pixel_acc_history, label='Pixel Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('IoU and Pixel Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(TRAIN_METRICS_OUTPUT)

# ------------------------------
# Generate and Save Segmentation Overlays Automatically
# ------------------------------
os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)

def visualize_segmentation_overlay(image_tensor, gt_mask_tensor, pred_mask_tensor, alpha=0.4):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)
    gt_mask_np = gt_mask_tensor.squeeze().cpu().numpy()
    pred_mask_np = pred_mask_tensor.squeeze().cpu().numpy()
    
    color_gt = np.array([0, 1, 0])      # Green for ground truth
    color_pred = np.array([1, 0, 0])    # Red for prediction
    color_overlap = np.array([1, 0, 1]) # Purple for overlap
    
    overlay = image_np.copy()
    
    overlap_mask = (gt_mask_np == 1) & (pred_mask_np == 1)
    just_gt_mask = (gt_mask_np == 1) & (pred_mask_np == 0)
    just_pred_mask = (gt_mask_np == 0) & (pred_mask_np == 1)
    
    overlay[just_gt_mask] = (1 - alpha) * overlay[just_gt_mask] + alpha * color_gt
    overlay[just_pred_mask] = (1 - alpha) * overlay[just_pred_mask] + alpha * color_pred
    overlay[overlap_mask] = (1 - alpha) * overlay[overlap_mask] + alpha * color_overlap
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(overlay)
    ax.set_title("Overlay: GT (Green), Pred (Red), Overlap (Purple)")
    ax.axis('off')
    legend_patches = [
        Patch(color=(0,1,0), label='Ground Truth'),
        Patch(color=(1,0,0), label='Prediction'),
        Patch(color=(1,0,1), label='Overlap')
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    return fig

def generate_and_save_overlays(model, dataloader, device, output_dir, max_images=10):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= max_images:
                break
            images, masks, labels = batch  # Using batch size of 1 is recommended for overlay generation
            images = images.to(device)
            masks = masks.to(device)
            
            seg_out, _ = model(images)
            seg_pred = (seg_out > 0.5).float()
            
            fig = visualize_segmentation_overlay(images[0].cpu(), masks[0].cpu(), seg_pred[0].cpu(), alpha=0.4)
            output_file = os.path.join(output_dir, f"overlay_{count}.png")
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved overlay to {output_file}")
            count += 1

# Create a DataLoader with batch_size=1 for overlay generation
overlay_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
generate_and_save_overlays(model, overlay_loader, device, OVERLAY_OUTPUT_DIR, max_images=10)

confusion_output_path = r"/dist_home/suryansh/dl/Skin-Lesion-Classification/RCCM/rcmm_output/confusion.png"

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_classification(model, dataloader, device):
    """
    Runs inference on the dataloader, collects ground truth and predicted labels.
    Returns:
        all_labels: list of true labels.
        all_preds: list of predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, masks, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # We only need the classification output here
            _, cls_out = model(images)
            preds = torch.argmax(cls_out, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds

def plot_confusion_matrix(true_labels, pred_labels, class_names, output_path=None):
    """
    Plots the confusion matrix for the classification task.
    
    Args:
        true_labels: List or array of ground truth labels.
        pred_labels: List or array of predicted labels.
        class_names: List of class names (in order of label indices).
        output_path: Optional path to save the figure.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Example usage:
# Assume 'model', 'val_loader', and 'device' are already defined.
true_labels, pred_labels = evaluate_classification(model, val_loader, device)
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# Optionally, define an output path to save the confusion matrix plot
plot_confusion_matrix(true_labels, pred_labels, class_names, output_path=confusion_output_path)