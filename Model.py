
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchinfo
import matplotlib.pyplot as plt
import h5py
import numpy as np
from random import sample
import os
from Essentials import BeamDataset,BeamLoss
# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===============================
# Building Blocks
# ===============================

class DilatedResBlock(nn.Module):
    """
    Two dilated Conv1d branches (different dilation rates) fused with a residual.
    Captures both local and wider angular context in a single block.
    """
    def __init__(self, channels, dilation):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.branch(x))


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Lets the network re-weight feature maps globally (across all 180 angles).
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),       # Global average over 180 angles
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        w = self.gate(x).unsqueeze(-1)    # (B, C, 1)
        return x * w


class MultiScaleFusion(nn.Module):
    """
    Parallel convolutions with kernel sizes 3, 7, 15 to capture
    fine peak detail, mid-range sidelobes, and broad angular structure.
    Outputs are concatenated then projected back.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 3
        self.k3  = nn.Conv1d(in_ch, mid,      kernel_size=3,  padding=1)
        self.k7  = nn.Conv1d(in_ch, mid,      kernel_size=7,  padding=3)
        self.k15 = nn.Conv1d(in_ch, out_ch - 2*mid, kernel_size=15, padding=7)
        self.proj = nn.Sequential(
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.proj(torch.cat([self.k3(x), self.k7(x), self.k15(x)], dim=1))


# ===============================
# Main Model: BeamNet
# ===============================
class BeamNet(nn.Module):
    """
    Multi-scale dilated CNN with channel attention for beamformer
    power spectrum refinement.

    Input  : (B, 1, 180)  — normalized MVDR power spectrum
    Output : (B, 1, 180)  — refined sparse target (peaks at source angles)

    Design rationale:
      1. Multi-scale entry captures peaks, sidelobes, and background in parallel.
      2. Dilated residual blocks (dilation 1,2,4,8,16) give a receptive field
         of ~120 angles without downsampling — important because a sidelobe
         at +30° influences the mainlobe estimate.
      3. Channel attention re-weights feature channels after every two blocks,
         allowing the network to focus on the most discriminative patterns.
      4. Sigmoid output constrains predictions to [0,1], matching the
         binary-like sparse target.
    """
    def __init__(self, base_ch=64):
        super().__init__()

        # ---- Entry: multi-scale feature extraction ----
        self.entry = MultiScaleFusion(1, base_ch)

        # ---- Encoder: progressively wider receptive field ----
        self.enc1 = DilatedResBlock(base_ch, dilation=1)
        self.enc2 = DilatedResBlock(base_ch, dilation=2)
        self.attn1 = ChannelAttention(base_ch)

        self.enc3 = DilatedResBlock(base_ch, dilation=4)
        self.enc4 = DilatedResBlock(base_ch, dilation=8)
        self.attn2 = ChannelAttention(base_ch)

        self.enc5 = DilatedResBlock(base_ch, dilation=16)
        self.enc6 = DilatedResBlock(base_ch, dilation=1)   # close-range refinement
        self.attn3 = ChannelAttention(base_ch)

        # ---- Output head ----
        self.head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_ch // 2, 1, kernel_size=1),
            nn.Sigmoid()          # target is in [0,1]
        )

        self._init_weights()

    def forward(self, x):
        z = self.entry(x)

        z = self.enc1(z)
        z = self.enc2(z)
        z = self.attn1(z)

        z = self.enc3(z)
        z = self.enc4(z)
        z = self.attn2(z)

        z = self.enc5(z)
        z = self.enc6(z)
        z = self.attn3(z)

        return self.head(z)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ===============================
# Load Data (lazy, no full RAM load)
# ===============================
INPUT_PATH  = 'Noisy.h5'
TARGET_PATH = 'Clear.h5'

full_dataset = BeamDataset(INPUT_PATH, TARGET_PATH,
                           input_key='DS1', target_key='DS2')

print(f"Total samples : {len(full_dataset)}")
print(f"Input mean    : {full_dataset.mean:.4f}")
print(f"Input std     : {full_dataset.std:.4f}")

# ---- Train / Val split (90/10) ----
N          = len(full_dataset)
val_size   = int(0.1 * N)
train_size = N - val_size
train_set, val_set = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size  = 512    # Large batch — fine for 500k dataset, faster convergence
num_workers = 4      # Parallel HDF5 readers; set to 0 if issues on Windows

train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers,
                          pin_memory=True, persistent_workers=(num_workers > 0))
val_loader   = DataLoader(val_set,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True, persistent_workers=(num_workers > 0))

# ---- Quick sanity plot ----
idx_list = sample(range(train_size), 4)
fig, axs = plt.subplots(4, 2, figsize=(12, 10))
for i, idx in enumerate(idx_list):
    xv, yv = full_dataset[idx]
    axs[i, 0].plot(xv.squeeze().numpy(), color='steelblue')
    axs[i, 0].set_title(f'Normalized MVDR Input [sample {idx}]')
    axs[i, 1].plot(yv.squeeze().numpy(), color='darkorange')
    axs[i, 1].set_title(f'Sparse Target [sample {idx}]')
plt.tight_layout()
plt.savefig('sample_preview.png', dpi=100)
plt.show()


# ===============================
# Model, Loss, Optimizer, Scheduler
# ===============================
model     = BeamNet(base_ch=64).to(device)
criterion = BeamLoss(alpha=0.5, gamma=2.0, pos_weight=100.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine annealing: smoothly decays LR to lr_min over all epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-5
)

torchinfo.summary(model, input_size=(batch_size, 1, 180), device=device)

# ---- Resume checkpoint (uncomment to use) ----
# checkpoint = torch.load("beamnet_checkpoint.pth", map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# start_epoch = checkpoint['epoch'] + 1
# print(f"Resuming from epoch {start_epoch}")
start_epoch = 0

epochs      = 50
best_val    = float('inf')
train_losses, val_losses = [], []

torch.manual_seed(42)

# ===============================
# Training Loop
# ===============================
for epoch in range(start_epoch, epochs):

    # ---- Train ----
    model.train()
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            pred     = model(batch_x)
            val_loss += criterion(pred, batch_y).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch [{epoch+1:>3}/{epochs}]  "
          f"Train: {epoch_loss:.6f}  Val: {val_loss:.6f}  "
          f"LR: {current_lr:.2e}")

    # ---- Save best model ----
    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            'epoch'               : epoch,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss'            : best_val,
            'input_mean'          : full_dataset.mean,
            'input_std'           : full_dataset.std,
        }, "beamnet_best.pth")
        print(f"  ✓ Best model saved  (val_loss={best_val:.6f})")

    # ---- Periodic checkpoint every 10 epochs ----
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch'               : epoch,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss'            : val_loss,
            'input_mean'          : full_dataset.mean,
            'input_std'           : full_dataset.std,
        }, f"beamnet_epoch{epoch+1}.pth")


# ===============================
# Loss Curve
# ===============================
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('BeamNet Training Curve')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=100)
plt.show()


# ===============================
# Inference Example
# ===============================
model.eval()
checkpoint = torch.load("beamnet_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")

fig, axs = plt.subplots(4, 3, figsize=(15, 10))
with torch.no_grad():
    for i, idx in enumerate(idx_list):
        xv, yv = full_dataset[idx]
        pred = model(xv.unsqueeze(0).to(device)).cpu().squeeze().numpy()
        axs[i, 0].plot(xv.squeeze().numpy(), color='steelblue')
        axs[i, 0].set_title(f'MVDR Input [{idx}]')
        axs[i, 1].plot(yv.squeeze().numpy(), color='darkorange')
        axs[i, 1].set_title('Ground Truth')
        axs[i, 2].plot(pred, color='green')
        axs[i, 2].set_title('BeamNet Prediction')
plt.tight_layout()
plt.savefig('inference_preview.png', dpi=100)
plt.show()

print("Done.")
