import torch
from nuclr.data import prepare_nuclear_data
from nuclr.config import config, datadir
from nuclr.loss import loss_by_task, metric_by_task
from nuclr.model import NuCLRWrapper
from nuclr.tensor_dict import Fields
import tqdm

EPOCHS = 1000
BATCH_SIZE = 1024
LR = 1e-3
WD = 1e-3
d_model = 6

input_fields = ["n", "z"]
output_fields = ["binding", "radius"]

data = prepare_nuclear_data(config)

output_dim = sum(data.all_fields[field] for field in output_fields)

tensor_dict_train = data.tensor_dict.iloc[data.train_mask]
tensor_dict_valid = data.tensor_dict.iloc[data.valid_mask]

fields = Fields({"numerical": list(output_fields), "categorical": []})


torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Embedding(sum(data.vocab_size), d_model),
    torch.nn.Flatten(),
    torch.nn.Linear(d_model * 2, d_model),
    torch.nn.ReLU(),
    torch.nn.Linear(d_model,output_dim),
).to(config.DEV)

model = NuCLRWrapper(model, fields)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

for epoch in (pbar:=tqdm.trange(EPOCHS)):
  perm = torch.randperm(len(tensor_dict_train))
  for batch_idx in range(0, len(tensor_dict_train), BATCH_SIZE):
    batch = tensor_dict_train.iloc[perm[batch_idx:batch_idx+BATCH_SIZE]]
    pred = model(batch[input_fields])
    loss_dict = loss_by_task(pred, batch)
    metric_dict = metric_by_task(pred, batch, data.regression_transformers)
    loss = sum(loss_dict.values())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_description(f"loss: {loss.item():.3f}, metric: {metric_dict}")
  if epoch % 10 == 0:
    pred = model(tensor_dict_valid[input_fields])
    loss_valid = sum(loss_by_task(pred, tensor_dict_valid).values())
    metric_valid = metric_by_task(pred, tensor_dict_valid, data.regression_transformers)
    print(f"epoch {epoch}")
    print(f"loss: {loss_valid.item():.3f}")
    print(f"metric: {metric_valid}")
    print("")
    
    