import torch
from torch import nn
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


class SwiGLU(nn.Module):
    def __init__(self, dim=-1, beta=False):
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


class RetBlock:
    def __init__(self, d_model, d_ff=None, n_heads=1):
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
        QKT = torch.einsum("bij, bkj -> bik", Q, K)  # [batch, seq_len, seq_len]
        # V is [batch, seq_len, d_model]
        return (torch.tril(QKT)).bmm(V) / self.d_model**0.5

    def recursive_forward(self, xn, hidden_prev):
        assert tuple(xn.shape) == (1, self.d_model)
        # TODO missing gamma
        Kn, Qn, Vn = self.kqv(xn).chunk(3, dim=-1)
        Sn = Kn.T @ Vn + hidden_prev
        return Qn @ Sn / self.d_model**0.5, Sn

    def recursive_forward_all(self, x):
        # TODO missing gamma
        K, Q, V = self.kqv(x).chunk(3, dim=-1)
        # Sn = ( gamma * ) S_{n-1} + K^T_n V_n
        KV = torch.einsum("bsd, bse -> bsde", K, V)
        S = torch.cumsum(KV, 1)
        rets = torch.einsum("bsd, bs -> bsd", Q, S)
        return rets / self.d_model**0.5, S


def toy_data(n, d_model=2):
    # p times proton emb, n times neutron emb
    n_protons_max = 15
    n_neutrons_max = 20
    n_p = torch.randint(1, n_protons_max + 1, (n, 1))
    n_n = torch.randint(1, n_neutrons_max + 1, (n, 1))
    emb_p = torch.randn(d_model)
    emb_n = torch.randn(d_model)
    seqs = torch.zeros((n, n_protons_max + n_neutrons_max, d_model))
    seqs[:, :n_protons_max] = emb_p
    seqs[:, :n_protons_max] *= torch.arange(n_protons_max) < n_p
    seqs[:, n_protons_max:] = emb_n
    seqs[:, n_protons_max:] *= torch.arange(n_neutrons_max) < n_n

    def out_fn(ps, ns):
        return ps.sum(dim=-1) ** 0.8 - ns.sum(dim=-1) ** 0.5

    ys = out_fn(n_p, n_n)
    return seqs, ys


def test_retention():
    d_model = 4

    X_train, Y_train = toy_data(1000)
    Y_test, Y_test = toy_data(100)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ret = Retention(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                SwiGLU(dim=-1),
                nn.Linear(d_model, d_model),
            )
        
        def forward(self, x):
            return self.ffn(self.ret(x).sum(1))
    
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(1000):
        opt.zero_grad()
        y_pred = model(X_train)
        loss = ((y_pred - Y_train) ** 2).mean()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(loss.item())


def test_msr():
    d_model = 4
    msr = MSR(d_model)
    x = torch.randn(2, 3, d_model)
    print(x)
    print(msr(x))


if __name__ == "__main__":
    test_retention()
    test_msr()
