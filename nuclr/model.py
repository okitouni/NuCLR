import torch
from torch import nn
from typing import Callable
import mup
from .modules import RNNCell, SwiGLU
from .utils.tensor_dict import TensorDict
from .utils.tensor_dict import Fields

def _append_readout(model_fn: Callable) -> Callable:
    """Append a muP readout to a model. If the model is a sequential model,
    the readout replaces the last element in the sequence. Otherwise,
    the readout layer is expected to be an attribute.

    Args:
        model_fn (callable): Function which returns a model.
    """

    def model_fn_with_readout(*args, **kwargs):
        model = model_fn(*args, **kwargs)
        # check if model already has a readout, FIXME: this is a hack
        if any([isinstance(x, mup.MuReadout) for x in model.modules()]):
            return model
        if isinstance(model, nn.Sequential):
            assert isinstance(
                model[-1], nn.Linear
            ), "Last layer of sequential model must be linear (readout)"
            old_readout = model.pop(len(model) - 1)
            model.append(mup.MuReadout(*old_readout.weight.T.shape))
        else:
            assert hasattr(
                model, "readout"
            ), "Model must be sequential or have a readout attribute"
            old_readout = model.readout
            model.readout = mup.MuReadout(*old_readout.weight.T.shape)
        return model

    return model_fn_with_readout


def make_mup(model_fn, **scale_kwargs) -> nn.Module:
    """Reinitialize model with mup scaling of relevant dimensions. Takes a function which returns a model and returns a model with mup scaling.
    Assumes the model has a readout linear layer which is either the last layer in a sequential model or an attribute of the model.

    Args:
        model_fn (Callable): Function which returns a nn.Module model.
        init_fn (Callable, optional): Function which initializes the model parameters in-place. Defaults to Kaiming uniform with a = sqrt(5).

    Raises:
        ValueError: If depth is in scale_kwargs. Depth is not a scaling parameter.

    Returns:
        nn.Module: Model with mup scaling.
    """

    model_fn = _append_readout(model_fn)
    base_kwargs = {k: 32 for k in scale_kwargs}
    delta_kwargs = {k: 64 for k in scale_kwargs}
    base = model_fn(**base_kwargs)
    delta = model_fn(**delta_kwargs)
    model = model_fn(**scale_kwargs)
    mup.set_base_shapes(model, base, delta=delta)
    del base, delta
    for name, param in model.named_parameters():
        if "weight" in name.lower():  # FIXME or not
            mup.init.uniform_(param, -.3, .3)
            #mup.init.kaiming_uniform_(param, a=5**0.5, nonlinearity="leaky_relu")
    return model


class NuCLRWrapper(nn.Module):
    def __init__(self, model_fn, d_model, pred_fields: Fields):
        """Wrapper to make NuCLR models compatible with TensorDicts.

        Args:
            model (nn.Module): The NuCLR model to wrap. Expects tensors as input.
            pred_fields (Fields): The name and order of the fields to predict.
        """
        super().__init__()
        self.model = make_mup(model_fn, d_model=d_model)
        self.pred_fields = pred_fields
        self.readout = self.model.readout

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
            nn.Linear(d_model, d_model * 2), SwiGLU(), nn.Linear(d_model, d_model)
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

class Model(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        # embeddings via RNN

        # proton number = z -> z times application of RNN
        # neutron number = n -> n times application of RNN
        # one for proton, one for neutron
        self.embeddings = nn.Parameter(torch.randn(2, d_model) / d_model**0.5)

        # RNN output (the entire sequence) is concatenated and fed into a Transformer
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=d_model,
        #         nhead=2,
        #         dim_feedforward=d_model * 4,
        #         dropout=0.0,
        #         activation="gelu",
        #         batch_first=True,
        #     ),
        #     num_layers=2,
        #     norm=nn.LayerNorm(d_model),
        # )
        self.interaction_model = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        self.proton_rnn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.neutron_rnn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.proton_rnn[-1].weight.data = torch.randn(d_model, d_model) / d_model**0.5
        self.neutron_rnn[-1].weight.data = (
            torch.randn(d_model, d_model) / d_model**0.5
        )

        self.readout = nn.Linear(d_model, output_dim)

    def _protons(self, z):
        p = self.embeddings[0]
        return torch.vstack([(p := self.proton_rnn(p)) for _ in range(z)])

    def _neutrons(self, n):
        p = self.embeddings[1]
        return torch.vstack([(p := self.neutron_rnn(p)) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neutrons = x[:, 1]
        protons = x[:, 0]
        nmax = neutrons.amax().item()
        pmax = protons.amax().item()
        protons = self._protons(pmax)[protons-1]
        neutrons = self._neutrons(nmax)[neutrons-1]
        out = torch.cat([protons, neutrons], dim=1)
        out = self.interaction_model(out)
        return torch.sigmoid(self.readout(out))
        # dev = x.device
        # neutrons = x[:, 1]
        # protons = x[:, 0]
        # nmax = neutrons.amax().item()
        # pmax = protons.amax().item()
        # neutrons = self._neutrons(nmax)  # [nmax, d_model]
        # protons = self._protons(pmax)  # [pmax, d_model]
        # make sequences: (batch, seq_len, d_model)
        # seq_len == pmax + nmax
        # for shorter elements in the batch, we can mask things
        # pn_embeddings = torch.cat([protons, neutrons], dim=0)  # [pmax + nmax, d_model]
        # make one sequence for each batch element
        # # sequence = sequence[None].repeat(
        # #     x.shape[0], 1, 1
        # # )  # [ batch, pmax + nmax, d_model]
        # mask elements
        # proton_mask = torch.arange(pmax, device=dev) >= x[:, [0]]
        # neutron_mask = torch.arange(nmax, device=dev) >= x[:, [1]]
        # sequence_mask = torch.cat([proton_mask, neutron_mask], dim=1)
        # sequence[sequence_mask] = 0
        # sequence = self.transformer(sequence)  # batch, seq, d_model
        # sequence = sequence.amax(dim=1)  # batch, d_model
        # return self.readout(sequence)  # batch, output_dim
