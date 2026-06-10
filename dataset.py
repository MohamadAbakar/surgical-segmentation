import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

GOOD_VIDEOS = [
    "video01", "video09", "video25", "video35",
    "video37", "video48", "video52", "video55"
]


class CholecSegDataset(Dataset):
    def __init__(self, root_dir, good_videos=GOOD_VIDEOS, num_classes=13):
        self.samples = []
        self.num_classes = num_classes

        for video in good_videos:
            video_path = os.path.join(root_dir, video)
            for clip in os.listdir(video_path):
                clip_path = os.path.join(video_path, clip)
                files = os.listdir(clip_path)
                frames = [f for f in files if f.endswith("_endo.png")]
                for frame in frames:
                    raw  = os.path.join(clip_path, frame)
                    mask = os.path.join(clip_path, frame.replace("_endo.png", "_endo_mask.png"))
                    if os.path.exists(mask):
                        self.samples.append((raw, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, mask_path = self.samples[idx]

        img  = Image.open(raw_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)

        img  = transforms.ToTensor()(img)
        img  = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])(img)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = mask.clamp(0, self.num_classes - 1)

        return img, mask
