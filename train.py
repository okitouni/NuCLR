import torch
from torch import nn
from nuclr.data import prepare_nuclear_data
from nuclr.config import config, datadir
from nuclr.loss import loss_by_task, metric_by_task
from nuclr.model import NuCLRWrapper
from nuclr.utils import tprint, Fields
import tqdm
import wandb
import mup
from argparse import ArgumentParser

WANDB=True

# EPOCHS = 1000000
# BATCH_SIZE = 512
# LR = 1e-3
# WD = 1e-4
# d_model = 64

parser = ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000000)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--dropout", type=float, default=.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exp_name", type=str, default="exp")

args = parser.parse_args()

torch.manual_seed(args.seed)
config.seed = args.seed

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WD = args.wd
d_model = args.d_model
dropout_prob = args.dropout

if WANDB:
  wandb.init(project="nuclr", config=args, name=args.exp_name)

class MLPMixerLayer(nn.Module):
    def __init__(self, num_tokens, d_model):
        super(MLPMixerLayer, self).__init__()

        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens*4),
            nn.GELU(),
            nn.Linear(num_tokens*4, num_tokens),
        )

        self.sequence_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        # Apply token mixing along the token dimension
        x = x + self.token_mlp(x.transpose(1, 2)).transpose(1, 2)

        # Apply channel mixing along the channel dimension
        x = x + self.sequence_mlp(x)

        return x

class Model(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.proton_embedding_model = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.neutron_embedding_model = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        self.interaction_model = nn.Sequential(
            MLPMixerLayer(2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            MLPMixerLayer(2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.readout = nn.Linear(d_model, output_dim)

        self.z_max = 120
        self.n_max = 180

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neutrons = x[:, [1]] / self.n_max
        protons = x[:, [0]] / self.z_max
        protons = self.proton_embedding_model(protons)
        neutrons = self.neutron_embedding_model(neutrons)
        out = torch.stack([protons, neutrons], dim=1)
        out = self.interaction_model(out)
        out = out.amax(dim=1)
        return self.readout(out)


input_fields = ["z", "n"]
output_fields = ["binding", "radius"]

data = prepare_nuclear_data(config)

output_dim = sum(data.all_fields[field] for field in output_fields)

tensor_dict_train = data.tensor_dict.iloc[data.train_mask]
tensor_dict_valid = data.tensor_dict.iloc[data.valid_mask]

fields = Fields({"numerical": list(output_fields), "categorical": []})

model = NuCLRWrapper(lambda d_model: Model(d_model, output_dim), d_model, fields).to(config.DEV)

optimizer = mup.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

if WANDB:
    pbar = range(EPOCHS)
else:
    pbar = tqdm.trange(EPOCHS)

for epoch in pbar:
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
    if epoch % 50 == 0:
        pred = model(tensor_dict_valid[input_fields])
        loss_valid = sum(loss_by_task(pred, tensor_dict_valid).values())
        metric_valid = metric_by_task(
            pred, tensor_dict_valid, data.regression_transformers
        )
        if WANDB:
          log_dict = {
            "loss_train": loss_train.item(),
            "loss_valid": loss_valid.item(),
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
          }
          log_dict.update({f"train/{k}": v for k, v in metric_train.items()})
          log_dict.update({f"valid/{k}": v for k, v in metric_valid.items()})
          wandb.log(log_dict)
        else:
          pbar.set_description(
              f"loss: {loss_train.item():.3e}|{loss_valid.item():.3e}, {tprint(metric_train, metric_valid)}"
          )

