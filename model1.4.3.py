import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from torchinfo import summary

# ==============================
# Device Selection (Auto)
# ==============================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")



# ==============================
# Model: 1D CNN for DoA
# ==============================
class DOA_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),  # 180 → 90

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(2),  # 90 → 45

            nn.Flatten(),

            nn.Linear(128 * 45, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 180)  # 180 classes
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 180)
        return self.net(x)



    # --------------------------
    # Model
    # --------------------------
model = DOA_CNN()
summary(model)

    
