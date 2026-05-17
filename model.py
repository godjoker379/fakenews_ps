"""
model.py — FakeNewsANN architecture for TruthLens

Input  : 384-dim L2-normalised sentence-transformer embedding
Output : scalar in [0, 1]  →  probability text is FAKE

Architecture: 384 → 256 → 128 → 64 → 32 → 1
BatchNorm + Dropout for stable generalisation.
"""

import torch
import torch.nn as nn


class FakeNewsANN(nn.Module):
    def __init__(self, input_dim: int = 384):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> float:
        """Convenience: return scalar fake-probability for a single sample."""
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return float(self.forward(x).item())