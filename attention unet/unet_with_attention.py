import torch
print(torch.version.cuda)          # Should show 12.6
print(torch.cuda.is_available())   # Should return True

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

# Paths
image_dirs = [r'C:\profolders\Collage stuff\Sem 4 project\DL\Datasets\HAM10000\HAM10000_images_part_1',
              r'C:\profolders\Collage stuff\Sem 4 project\DL\Datasets\HAM10000\HAM10000_images_part_2']
mask_dir = r'C:\profolders\Collage stuff\Sem 4 project\DL\Datasets\HAM10000_segmentations_lesion_tschandl'
metadata_path = r'C:\profolders\Collage stuff\Sem 4 project\DL\Datasets\HAM10000\HAM10000_metadata.csv'

# Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16

# Load metadata
metadata = pd.read_csv(metadata_path)
image_ids = metadata['image_id'].values

# Synchronized Transformations
class SynchronizedTransform:
    def __init__(self, flip_prob=0.5, rotation_degrees=(0, 90, 180, 270)):
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees

    def __call__(self, image, mask):
        # Random Horizontal Flip
        if random.random() < self.flip_prob:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random Vertical Flip
        if random.random() < self.flip_prob:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # Random Rotation
        angle = random.choice(self.rotation_degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)

        return image, mask

# Custom Dataset
class HAM10000Dataset(Dataset):
    def __init__(self, image_ids, image_dirs, mask_dir, transform=None, sync_transform=None):
        self.image_dirs = image_dirs
        self.mask_dir = mask_dir
        self.transform = transform
        self.sync_transform = sync_transform

        # Filter valid image-mask pairs at initialization
        self.valid_ids = [img_id for img_id in image_ids if self.image_exists(img_id) and self.mask_exists(img_id)]

    def image_exists(self, image_id):
        return any(os.path.exists(os.path.join(dir, f"{image_id}.jpg")) for dir in self.image_dirs)

    def mask_exists(self, image_id):
        return os.path.exists(os.path.join(self.mask_dir, f"{image_id}_segmentation.png"))

    def load_image(self, image_id):
        for dir in self.image_dirs:
            path = os.path.join(dir, f"{image_id}.jpg")
            if os.path.exists(path):
                return Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        return None

    def load_mask(self, image_id):
        mask_path = os.path.join(self.mask_dir, f"{image_id}_segmentation.png")
        if os.path.exists(mask_path):
            return Image.open(mask_path).convert("L").resize(IMAGE_SIZE)
        return None

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        image_id = self.valid_ids[idx]
        image = self.load_image(image_id)
        mask = self.load_mask(image_id)

        # Apply synchronized transformations
        if self.sync_transform:
            image, mask = self.sync_transform(image, mask)

        # Apply basic transformations (ToTensor)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()  # Binarize mask

        return image, mask

# Transformations
basic_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Synchronized Transform
sync_transform = SynchronizedTransform(flip_prob=0.5)

# Create datasets
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

train_dataset = HAM10000Dataset(train_ids, image_dirs, mask_dir, transform=basic_transform, sync_transform=sync_transform)
val_dataset = HAM10000Dataset(val_ids, image_dirs, mask_dir, transform=basic_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# Define Attention Gate
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

# Define the Attention U-Net
class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()

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

        # Decoder with Attention
        self.att4 = AttentionGate(512, 512, 256)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.att3 = AttentionGate(256, 256, 128)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.att2 = AttentionGate(128, 128, 64)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.att1 = AttentionGate(64, 64, 32)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        p1 = self.pool1(e1)

        e2 = self.conv2(p1)
        p2 = self.pool2(e2)

        e3 = self.conv3(p2)
        p3 = self.pool3(e3)

        e4 = self.conv4(p3)
        p4 = self.pool4(e4)

        e5 = self.conv5(p4)

        # Decoder with Attention
        d4 = self.up4(e5)
        x4 = self.att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)

        output = torch.sigmoid(self.final(d1))
        return output

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionUNet().to(device)
print(model)
print(f"Model loaded on {device}.")

def dice_coefficient(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

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
    total = target.size(0)
    return correct / total

import torch.optim as optim
import os

# Dice Coefficient and Loss
def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - dice_coefficient(pred, target)

# Loss and Optimizer
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

average_loss_list = []
average_dice_list = []
average_iou_list = []
average_pixel_acc_list = []

# Training loop
num_epochs = 2
checkpoint_path = 'attention_unet_checkpoint.pth'

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    dice_scores = []
    iou_scores = []
    pixel_accuracies = []

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5  # Binarize predictions

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        # Calculate metrics
        dice_scores.append(dice_coefficient(preds, masks).item())
        iou_scores.append(iou_score(preds, masks).item())
        pixel_accuracies.append(pixel_accuracy(preds, masks))  # Removed .item()

    avg_loss = train_loss / len(train_loader)
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_pixel_acc = sum(pixel_accuracies) / len(pixel_accuracies)
    average_loss_list.append(avg_loss)
    average_dice_list.append(avg_dice)
    average_iou_list.append(avg_iou)
    average_pixel_acc_list.append(avg_pixel_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, Pixel Acc: {avg_pixel_acc:.4f}")

    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)  # Delete previous checkpoint
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

import matplotlib.pyplot as plt

# Assuming you stored metrics in lists
epochs_range = len(average_loss_list)
plt.figure(figsize=(12, 4))

# Dice Coefficient
plt.subplot(1, 3, 1)
plt.plot(epochs_range, average_dice_list, label='Dice Coefficient')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.title('Dice Coefficient over Epochs')
plt.legend()

# IoU Score
plt.subplot(1, 3, 2)
plt.plot(epochs_range, average_iou_list, label='IoU Score', color='orange')
plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.title('IoU Score over Epochs')
plt.legend()

# Pixel Accuracy
plt.subplot(1, 3, 3)
plt.plot(epochs_range, average_pixel_acc_list, label='Pixel Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Pixel Accuracy')
plt.title('Pixel Accuracy over Epochs')
plt.legend()

# Loss Curve
plt.figure(figsize=(6, 4))
plt.plot(epochs_range, average_loss_list, label='Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()


plt.tight_layout()
plt.savefig('attention_unet_metrics.png')
plt.close()

# Save the entire model (architecture + weights)
torch.save(model, 'attention_unet_model_final.pth')   
ho = None