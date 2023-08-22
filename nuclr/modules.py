import torch
from torch import nn


class PeriodicEmbedding(nn.Embedding):
    def __init__(self, d_model):
        super().__init__(1, d_model)

    def forward(self, x):
        freq = self.weight.sigmoid()
        sin = torch.sin(x.unsqueeze(-1) * freq[:, ::2])
        cos = torch.cos(x.unsqueeze(-1) * freq[:, 1::2])
        return torch.cat([sin, cos], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, beta=False):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        if beta:
            self.beta = nn.Parameter(torch.ones(1))
        else:
            self.beta = 1

    def forward(self, x, dim=-1):
        x_gate, x_out = x.chunk(2, dim=dim)
        x_gate = self.sigmoid(self.beta * x_gate) * x_gate
        return x_gate * x_out


class RNNCell(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 2 * d_model)
        self.activation = SwiGLU()
        self.norm = nn.LayerNorm(d_model)
        self.numerical_emb = PeriodicEmbedding(d_model)
        self.linear.weight.data = (
            torch.randn_like(self.linear.weight.data) / d_model**0.5
        )

    def forward(self, x, n):
        x = x + self.numerical_emb(n)
        return self.norm(self.activation(self.linear(x)) + x)