import torch
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
