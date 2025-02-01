import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 450
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_PATH = "/content/drive/MyDrive/UNet2/Data/trainimages.npy"
TRAIN_MASK_PATH = "/content/drive/MyDrive/UNet2/Data/trainmasks.npy"
VAL_IMG_PATH = "/content/drive/MyDrive/UNet2/Data/valimages.npy"
VAL_MASK_PATH = "/content/drive/MyDrive/UNet2/Data/valmasks.npy"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, dtype=torch.float32)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.amp.autocast(device_type="cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose([
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_PATH, TRAIN_MASK_PATH, VAL_IMG_PATH, VAL_MASK_PATH,
        BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY
    )

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()