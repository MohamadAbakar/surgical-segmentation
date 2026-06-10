# Surgical Scene Segmentation

Pixel-level semantic segmentation of laparoscopic surgery footage using a U-Net trained on CholecSeg8k.

Before a surgical robot can make a single decision inside a body, it has to know what it's looking at — tissue vs tool vs organ, at the pixel level. This is the perception layer that surgical autonomy gets built on top of.

![results](results/segmentation_results.png)

## What it does

Takes raw laparoscopic camera footage as input and outputs a segmentation mask labeling every pixel as a specific anatomical structure or surgical instrument.

## Dataset

[CholecSeg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k) — 8080 frames extracted from Cholec80 laparoscopic surgery videos, annotated with 13 tissue/instrument classes.

## Model

A lightweight U-Net architecture with:
- 3-level encoder/decoder
- Skip connections to preserve spatial detail
- CrossEntropyLoss for multi-class pixel classification
- Adam optimizer, lr=1e-3

## Results

Trained on 8 videos with clean segmentation masks, 80/20 train/test split, 10 epochs.
The model correctly identifies dominant tissue regions and surgical instruments from raw footage.

## Setup

```bash
pip install -r requirements.txt
python train.py
```

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- Pillow
- numpy
- matplotlib
- tqdm

## Stack

- PyTorch
- CholecSeg8k dataset
- U-Net architecture
- Kaggle (GPU training)

## Next

This perception layer feeds into a broader surgical robotics stack. Next step is integrating with ROS 2 to bridge segmentation output to robot action.
