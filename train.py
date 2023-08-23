import torch
from torch import nn
from nuclr.data import prepare_nuclear_data
from nuclr.config import config, datadir
from nuclr.loss import loss_by_task, metric_by_task
from nuclr.model import NuCLRWrapper, RetNet
from nuclr.model import RNN
from nuclr.utils import tprint, Fields
import tqdm


EPOCHS = 10000
BATCH_SIZE = 512
LR = 1e-3
WD = 1e-5
d_model = 32
n_head = 2
torch.manual_seed(42)


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
        return self.readout(out)
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


input_fields = ["z", "n"]
output_fields = ["binding", "radius"]

data = prepare_nuclear_data(config)

output_dim = sum(data.all_fields[field] for field in output_fields)

tensor_dict_train = data.tensor_dict.iloc[data.train_mask]
tensor_dict_valid = data.tensor_dict.iloc[data.valid_mask]

fields = Fields({"numerical": list(output_fields), "categorical": []})

model = NuCLRWrapper(RNN(d_model, output_dim), fields).to(config.DEV)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)


for epoch in (pbar := tqdm.trange(EPOCHS)):
    perm = torch.randperm(tensor_dict_train.size())
    for batch_idx in range(0, tensor_dict_train.size(), BATCH_SIZE):
        batch = tensor_dict_train.iloc[perm[batch_idx : batch_idx + BATCH_SIZE]]
        pred = model(batch[input_fields])
        loss_dict = loss_by_task(pred, batch)
        metric_train = metric_by_task(pred, batch, data.regression_transformers)
        loss_train = sum(loss_dict.values())
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        scheduler.step()
    if epoch % 10 == 0:
        pred = model(tensor_dict_valid[input_fields])
        loss_valid = sum(loss_by_task(pred, tensor_dict_valid).values())
        metric_valid = metric_by_task(
            pred, tensor_dict_valid, data.regression_transformers
        )
        pbar.set_description(
            f"loss: {loss_train.item():.3e}|{loss_valid.item():.3e}, {tprint(metric_train, metric_valid)}"
        )
