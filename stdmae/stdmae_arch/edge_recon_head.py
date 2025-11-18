import torch
import torch.nn as nn


class EdgeReconHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, H, edges):
        # H: [B, N, D], edges: [E,2] (long)
        u = H[:, edges[:, 0], :]
        v = H[:, edges[:, 1], :]
        pair = torch.cat([u, v], dim=-1)  # [B, E, 2D]
        logits = self.scorer(pair).squeeze(-1)  # [B, E]
        return logits
