"""Augmentation pipelines + dataset class for State Farm.

Heavy aug story: break driver-identity shortcuts (face, shirt, seat) so
the subject-wise val acc actually improves. NO horizontal flip — c1/c3
and c2/c4 are left/right-specific classes, so HFlip injects label noise.
See docs/adr/0002-no-hflip.md.

CutMix is implemented as a *batch-level* op applied inside the training
loop (see train.py), not in the per-image transform, because it mixes
two samples and rewrites the label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = [f"c{i}" for i in range(10)]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_stats(stats_path: Path) -> tuple[list[float], list[float]]:
    data = json.loads(Path(stats_path).read_text())
    return data["mean"], data["std"]


def build_train_transform(mean: list[float], std: list[float], size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
    ])


def build_trivialaugment_transform(mean: list[float], std: list[float], size: int = 224) -> transforms.Compose:
    """TrivialAugmentWide stack. No HFlip (op not in TrivialAugmentWide).

    Replaces ColorJitter+Grayscale+Blur+RandomErasing in build_train_transform
    with single learned-by-search policy. Stronger and simpler.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
    ])


def build_eval_transform(mean: list[float], std: list[float], size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class StateFarmDataset(Dataset):
    def __init__(self, csv_path: Path, img_root: Path, transform: transforms.Compose):
        import pandas as pd
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = self.img_root / row["classname"] / row["img"]
        with Image.open(path) as im:
            img = self.transform(im.convert("RGB"))
        return img, CLASS_TO_IDX[row["classname"]]


class StateFarmCropDataset(StateFarmDataset):
    """StateFarmDataset that keeps only the left `crop_left_frac` of each frame.

    Run 10 recipe: camera is dashboard-mounted; the right portion of the
    image is the empty passenger seat / area behind the driver and adds
    no signal. Cropping to left 80% pushes the driver to fill the frame.
    """

    def __init__(self, csv_path: Path, img_root: Path,
                 transform: transforms.Compose, crop_left_frac: float = 0.8):
        super().__init__(csv_path, img_root, transform)
        if not 0.0 < crop_left_frac <= 1.0:
            raise ValueError(f"crop_left_frac must be in (0, 1], got {crop_left_frac}")
        self.crop_left_frac = crop_left_frac

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = self.img_root / row["classname"] / row["img"]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.crop_left_frac < 1.0:
                w, h = im.size
                im = im.crop((0, 0, int(round(w * self.crop_left_frac)), h))
            img = self.transform(im)
        return img, CLASS_TO_IDX[row["classname"]]


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 1.0,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mix random patches between samples in a batch.

    Returns (mixed_images, labels_a, labels_b, lam) so the training loop
    can compute `lam * loss(a) + (1 - lam) * loss(b)`. When CutMix is not
    applied (random skip), `labels_a == labels_b` and `lam == 1.0`.
    """
    if np.random.rand() > p:
        return images, labels, labels, 1.0

    batch_size, _, h, w = images.shape
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(batch_size, device=images.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))
    return images, labels, labels[perm], lam


def mixup_loss(criterion, logits, labels_a, labels_b, lam: float) -> torch.Tensor:
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
