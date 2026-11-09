from __future__ import annotations
import math
import torch
from torch import nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=x.device, dtype=emb.dtype)], dim=-1)
        return emb


class FlowMatchingTimestepPosEmb(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        min_period: float = 4.0e-3,
        max_period: float = 4.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dim ({dim}) must be divisible by 2")
        self.dim = int(dim)
        fraction = torch.linspace(0.0, 1.0, self.dim // 2, dtype=torch.float32)
        period = float(min_period) * (float(max_period) / float(min_period)) ** fraction
        scaling_factor = 1.0 / period * 2.0 * math.pi
        self.register_buffer("scaling_factor", scaling_factor)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.view(-1).to(dtype=torch.float32)
        sin_input = self.scaling_factor.view(1, -1) * time.unsqueeze(-1)
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)

