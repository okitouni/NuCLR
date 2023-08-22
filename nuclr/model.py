import torch
from torch import nn
from .modules import RNNCell, SwiGLU
from .utils.tensor_dict import TensorDict
from .utils.tensor_dict import Fields


class NuCLRWrapper(nn.Module):
    def __init__(self, model: nn.Module, pred_fields: Fields):
        """Wrapper to make NuCLR models compatible with TensorDicts.

        Args:
            model (nn.Module): The NuCLR model to wrap. Expects tensors as input.
            pred_fields (Fields): The name and order of the fields to predict.
        """
        super().__init__()
        self.model = model
        self.pred_fields = pred_fields

    def forward(self, batch: TensorDict):
        preds = TensorDict(fields=self.pred_fields)
        tensor_preds = self.model(batch.to_tensor())
        for i, field in enumerate(self.pred_fields.all_fields):
            preds[field] = tensor_preds[:, [i]]
        return preds


class RetNet(nn.Module):
    def __init__(self, d_model, n_heads=1, num_layers=1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RetBlock(d_model, n_heads=n_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RetBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=1):
        # TODO n_heads
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 2
        self.msr = MSR(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.msr(x) + x
        return self.norm(self.ffn(y) + y)


class MSR(nn.Module):
    # TODO actually need multiscale and multihead
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.ret = Retention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), SwiGLU(dim=-1), nn.Linear(d_model, d_model)
        )
        self.norm = nn.GroupNorm(n_heads, d_model)

    def forward(self, x):
        x = self.ffn(self.ret(x))
        # transpose for group norm
        x = x.transpose(1, 2)
        return self.norm(x).transpose(1, 2)


class Retention(nn.Module):
    def __init__(self, d_model, n_heads=1):
        # TODO implement multihead, missing gamma
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.kqv = nn.Linear(d_model, d_model * 3, bias=False)

    def forward(self, x):
        K, Q, V = self.kqv(x).chunk(3, dim=-1)
        QKT = (
            torch.einsum("bij, bkj -> bik", Q, K) / self.d_model**0.5
        )  # [batch, seq_len, seq_len]
        norm = 1 + torch.arange(QKT.shape[1], device=QKT.device).sqrt().view(1, -1, 1)
        R = torch.tril(QKT) / norm  # [batch, seq_len, seq_len]
        R /= R.sum(-1, keepdim=True).abs().clamp(min=1)
        return R.bmm(V)

    def recursive_forward(self, xn, hidden_prev):
        assert tuple(xn.shape) == (1, self.d_model)
        # TODO missing gamma
        Kn, Qn, Vn = self.kqv(xn).chunk(3, dim=-1)
        Sn = Kn.T @ Vn + hidden_prev
        return Qn @ Sn, Sn

    def recursive_forward_all(self, x):
        # TODO missing gamma
        K, Q, V = self.kqv(x).chunk(3, dim=-1)
        # Sn = ( gamma * ) S_{n-1} + K^T_n V_n
        KV = torch.einsum("bsd, bse -> bsde", K, V)
        S = torch.cumsum(KV, 1)
        rets = torch.einsum("bsd, bs -> bsd", Q, S)
        return rets, S


class RNN(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(2 + 2, d_model) / d_model**0.5)
        self.protonet = RNNCell(d_model)
        self.neutronet = RNNCell(d_model)
        self.readout = nn.Linear(2 * d_model, output_dim)

        self.min_proton = 8
        self.min_neutron = 8
        self.max_proton = 150
        self.max_neutron = 200

    def _protons(self, n_p, n_n):
        emb = self.embeddings[0]
        for _ in range(n_p - self.min_proton + 1):
            emb = self.protonet(emb, n_n / self.max_neutron)
        return emb

    def _neutrons(self, n_n, n_p):
        emb = self.embeddings[1]
        for _ in range(n_n - self.min_neutron + 1):
            emb = self.neutronet(emb, n_p / self.max_proton)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        protons = torch.vstack([self._protons(n_p, n_n) for n_p, n_n in x[:, :2]])
        neutrons = torch.vstack([self._neutrons(n_n, n_p) for n_p, n_n in x[:, :2]])
        out = torch.hstack([protons, neutrons])
        return self.readout(out)
