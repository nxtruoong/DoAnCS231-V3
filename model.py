"""ResNet-18 + CBAM, from-scratch (Kaiming init, no pretrained weights).

CBAM (Woo et al. 2018, https://arxiv.org/abs/1807.06521):
- Channel attention: MLP over avg+max channel pool, ratio=16.
- Spatial attention: 7x7 conv over avg+max spatial pool.
- Applied sequentially: `x = SAM(CAM(x) * x) * (CAM(x) * x)`.

Inserted after each of the 4 ResNet stages (layer1..layer4). The SAM map
from layer4 is exposed via a forward hook (see `last_sam` attribute) so
`demo/app.py` can produce demo heatmaps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return attn  # (B, 1, H, W)


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.cam = ChannelAttention(channels, ratio)
        self.sam = SpatialAttention(spatial_kernel)
        self.last_sam: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.cam(x)
        sam = self.sam(x)
        self.last_sam = sam.detach()
        return x * sam


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResNet18CBAM(nn.Module):
    """ResNet-18 with optional CBAM after each stage.

    Args:
        num_classes: output dim (10 for State Farm).
        use_cbam: insert CBAM blocks after each stage. Set False for the
            ablation baseline.
    """

    def __init__(self, num_classes: int = 10, use_cbam: bool = True):
        super().__init__()
        self.use_cbam = use_cbam
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        if use_cbam:
            self.cbam1 = CBAM(64)
            self.cbam2 = CBAM(128)
            self.cbam3 = CBAM(256)
            self.cbam4 = CBAM(512)
        else:
            self.cbam1 = self.cbam2 = self.cbam3 = self.cbam4 = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward to pre-fc 512-d feature vector. Used by two-stream wrapper."""
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))

    def last_spatial_attention(self) -> torch.Tensor | None:
        """SAM map from the last CBAM block (layer4), shape (B, 1, 7, 7)
        at 224x224 input. Returns None if `use_cbam=False`.
        """
        if not self.use_cbam:
            return None
        return self.cbam4.last_sam


def build_model(num_classes: int = 10, use_cbam: bool = True) -> ResNet18CBAM:
    return ResNet18CBAM(num_classes=num_classes, use_cbam=use_cbam)
