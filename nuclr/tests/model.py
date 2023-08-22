import torch
from torch import nn

from ..modules import SwiGLU
from ..model import Retention, MSR


def toy_data(n, d_model=2):
    # p times proton emb, n times neutron emb
    n_protons_max = 15
    n_neutrons_max = 20
    n_p = torch.randint(n_protons_max + 1, (n, 1))
    n_n = torch.randint(n_neutrons_max + 1, (n, 1))
    emb_p = torch.randn(d_model)
    emb_n = torch.randn(d_model)
    seqs = torch.zeros((n, n_protons_max + n_neutrons_max, d_model))
    for b, seq in enumerate(seqs):
        for i in range(n_protons_max):
            if i < n_p[b]:
                seq[i] = emb_p
        for i in range(n_neutrons_max):
            if i < n_n[b]:
                seq[n_protons_max + i] = emb_n
    
    ys = n_p.sum(dim=-1, keepdim=True) - n_n.sum(dim=-1, keepdim=True)
    return seqs, ys


def test_retention():
    d_model = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, Y_train = toy_data(1000, d_model)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ret = Retention(d_model)
            # self.ret = RetNet(d_model, num_layers=2)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                SwiGLU(dim=-1),
                nn.Linear(d_model, 1),
            )

        def forward(self, x):
            return self.ffn(self.ret(x).sum(1))

    model = Model().to(device)
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
