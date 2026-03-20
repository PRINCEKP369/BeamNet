import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchinfo
import matplotlib.pyplot as plt
import h5py
import numpy as np
from random import sample
import os


# ===============================
# Lazy HDF5 Dataset
# Reads slices from disk on-the-fly — no full RAM load for 500k samples
# ===============================
class BeamDataset(Dataset):
    """
    Lazy loader for large HDF5 files.
    Reads individual samples on demand instead of loading everything into RAM.
    Assumes:
        input_path  : HDF5 with dataset key 'DS1'  shape [N, 180]  (MVDR power)
        target_path : HDF5 with dataset key 'DS2'  shape [N, 180]  (sparse target)
    """
    def __init__(self, input_path, target_path,
                 input_key='DS1', target_key='DS2',
                 mean=None, std=None):

        self.input_path  = input_path
        self.target_path = target_path
        self.input_key   = input_key
        self.target_key  = target_key

        # Open once just to get length and compute normalization stats
        with h5py.File(input_path, 'r') as f:
            self.N = f[input_key].shape[0]
            if mean is None or std is None:
                # Compute stats over a 10k subsample to avoid full RAM load
                idx = np.random.choice(self.N, min(10000, self.N), replace=False)
                idx.sort()
                data = np.float32(f[input_key][idx])
                self.mean = float(data.mean())
                self.std  = float(data.std()) + 1e-8
            else:
                self.mean = mean
                self.std  = std

        # File handles opened per-worker in __getitem__ (HDF5 not fork-safe)
        self._input_f  = None
        self._target_f = None

    def __len__(self):
        return self.N

    def _open(self):
        if self._input_f is None:
            self._input_f  = h5py.File(self.input_path,  'r')
            self._target_f = h5py.File(self.target_path, 'r')

    def __getitem__(self, idx):
        self._open()
        x = np.float32(self._input_f [self.input_key ][idx])   # (180,)
        y = np.float32(self._target_f[self.target_key][idx])   # (180,)

        # Normalize input only
        x = (x - self.mean) / self.std

        x = torch.from_numpy(x).unsqueeze(0)   # (1, 180)
        y = torch.from_numpy(y).unsqueeze(0)   # (1, 180)
        return x, y

    def __del__(self):
        if self._input_f  is not None: self._input_f.close()
        if self._target_f is not None: self._target_f.close()






# ===============================
# Combined Loss: MSE + Peak-Aware BCE
# ===============================
class BeamLoss(nn.Module):
    """
    Combines:
      - MSE  : penalizes squared deviation across the full spectrum
      - Focal-BCE : heavily up-weights the rare positive (peak) pixels
                    so the network does not just predict all zeros

    alpha  : weight of MSE vs BCE  (0 = pure BCE, 1 = pure MSE)
    gamma  : focal exponent — higher values focus more on hard positives
    pos_weight : BCE class weight for positive class (target=1 pixels)
                 Set ~= (negatives / positives) ≈ 179 for single-target case
    """
    def __init__(self, alpha=0.5, gamma=2.0, pos_weight=100.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse   = nn.MSELoss()
        self.bce   = nn.BCELoss(reduction='none')
        self.pw    = pos_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)

        # Focal-weighted BCE
        bce_raw  = self.bce(pred, target)
        # Up-weight positive (peak) locations
        weight   = torch.where(target > 0.5,
                               torch.full_like(target, self.pw),
                               torch.ones_like(target))
        # Focal factor: down-weight easy negatives
        pt       = torch.where(target > 0.5, pred, 1.0 - pred)
        focal    = (1.0 - pt) ** self.gamma
        bce_loss = (focal * weight * bce_raw).mean()

        return self.alpha * mse_loss + (1.0 - self.alpha) * bce_loss
