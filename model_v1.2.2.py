"""
train_doa.py — Direction of Arrival (DoA) Estimator
=====================================================
Input  : (N, 180) MVDR spatial power spectra
Target : (N, 180) one-hot vectors  →  class indices (N,) for CrossEntropyLoss
Model  : 1D CNN with residual blocks
Dataset: 2,000,000 synthetic samples, 90/10 train/val split

Run:
    python train_doa.py                        # full run
    python train_doa.py --epochs 5 --batch 512 # quick smoke-test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary


# ─────────────────────────────────────────────
# 0.  Device selection
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] Using: {device}")
    return device
                         

# ─────────────────────────────────────────────
# 3.  Model
# ─────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    Residual block: two Conv1d layers with BN + ReLU,
    plus a 1×1 projection shortcut when channels change.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class DoACNN(nn.Module):
    """
    1-D Residual CNN for DoA classification.

    Architecture
    ------------
    Input  : (B, 1, 180)
    Stem   : Conv1d 1→32, BN, ReLU
    Stage 1: 2× ResBlock  32→ 64, dilation 1
    Stage 2: 2× ResBlock  64→128, dilation 2
    Stage 3: 2× ResBlock 128→256, dilation 4
    GAP    : (B, 256)
    Head   : FC 256→180 logits
    """

    def __init__(self, n_classes: int = 180, dropout: float = 0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Residual stages with increasing dilation
        self.stage1 = nn.Sequential(ResBlock1D(32,  64,  dilation=1),
                                    ResBlock1D(64,  64,  dilation=1))
        self.stage2 = nn.Sequential(ResBlock1D(64,  128, dilation=2),
                                    ResBlock1D(128, 128, dilation=2))
        self.stage3 = nn.Sequential(ResBlock1D(128, 256, dilation=4),
                                    ResBlock1D(256, 256, dilation=4))

        # Global average pool → classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # (B, 256, 1)
            nn.Flatten(),                      # (B, 256)
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)                   # (B, 180) logits

model = DoACNN()
summary(model)
