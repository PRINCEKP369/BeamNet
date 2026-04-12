"""
Beamforming Classification MLP
================================
Input  : (N, 180)  MVDR power spectrum (log-scaled, one value per degree)
Target : (N,)      class index in [0, 179]  — the dominant bearing
         (derived from a (N, 180) one-hot by argmax at dataset build time)

Hardware : auto-selects CUDA > MPS > CPU
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary

# ─────────────────────────────────────────────────────────────
# 1.  CONFIG
# ─────────────────────────────────────────────────────────────
C = dict(
    num_samples   = 2_000_000,
    num_bearings  = 180,          # degrees: 0 … 179
    # MLP architecture
    hidden_dims   = [512, 512, 256, 128],
    dropout_p     = 0.3,
    # Training
    batch_size    = 4096,
    num_epochs    = 20,
    lr            = 1e-3,
    weight_decay  = 1e-4,
    val_split     = 0.10,
    num_workers   = 4,
    seed          = 42,
)



# ─────────────────────────────────────────────────────────────
# 2.  DEVICE SELECTION
# ─────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[device] using → {dev}")
    return dev


DEVICE = get_device()
  


# ─────────────────────────────────────────────────────────────
# 4.  MODEL
# ─────────────────────────────────────────────────────────────
class BeamformMLP(nn.Module):
    """
    Multi-layer perceptron with:
        BatchNorm1d  — stabilises training, accelerates convergence
        Dropout      — regularises to prevent overfitting
        GELU         — smooth activation, works well with BN
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list[int],
        output_dim:  int,
        dropout_p:   float = 0.3,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_p),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))  # logits; no softmax (CE handles it)

        self.net = nn.Sequential(*layers)

        # Kaiming init for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = BeamformMLP(
    input_dim   = C["num_bearings"],
    hidden_dims = C["hidden_dims"],
    output_dim  = C["num_bearings"],
    dropout_p   = C["dropout_p"],
).to(DEVICE)
summary(model)


