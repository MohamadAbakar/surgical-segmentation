import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from dataset import CholecSegDataset, GOOD_VIDEOS
from model import UNet

# ── Config ────────────────────────────────────────────────
ROOT       = "/kaggle/input/datasets/newslab/cholecseg8k"
EPOCHS     = 10
BATCH_SIZE = 8
LR         = 1e-3
SAVE_PATH  = "unet_cholecseg.pth"

# ── Detect num classes ────────────────────────────────────
first_mask = None
for video in GOOD_VIDEOS:
    video_path = os.path.join(ROOT, video)
    for clip in os.listdir(video_path):
        clip_path = os.path.join(video_path, clip)
        masks = [f for f in os.listdir(clip_path) if f.endswith("_endo_mask.png")]
        if masks:
            first_mask = os.path.join(clip_path, masks[0])
            break
    if first_mask:
        break

NUM_CLASSES = len(np.unique(np.array(Image.open(first_mask).convert("L"))))
print(f"Num classes: {NUM_CLASSES}")

# ── Data ──────────────────────────────────────────────────
dataset    = CholecSegDataset(ROOT, num_classes=NUM_CLASSES)
train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

# ── Model ─────────────────────────────────────────────────
model     = UNet(num_classes=NUM_CLASSES).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Train ─────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for imgs, masks in loop:
        imgs, masks = imgs.cuda(), masks.cuda()
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=f"{loss.item():.4f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# ── Visualize ─────────────────────────────────────────────
model.eval()
imgs, masks = next(iter(test_loader))

with torch.no_grad():
    preds = model(imgs.cuda()).argmax(dim=1).cpu()

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i in range(3):
    img = imgs[i].permute(1, 2, 0).numpy()
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

    axes[i][0].imshow(img)
    axes[i][0].set_title("Raw Frame")
    axes[i][0].axis('off')

    axes[i][1].imshow(masks[i], cmap='tab20', vmin=0, vmax=NUM_CLASSES)
    axes[i][1].set_title("Actual Mask")
    axes[i][1].axis('off')

    axes[i][2].imshow(preds[i], cmap='tab20', vmin=0, vmax=NUM_CLASSES)
    axes[i][2].set_title("Predicted Mask")
    axes[i][2].axis('off')

plt.tight_layout()
plt.savefig("results/segmentation_results.png")
plt.show()
