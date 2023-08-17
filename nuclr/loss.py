import torch
import typing as T
from torch.nn import functional as F
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from .tensor_dict import TensorDict


def accuracy(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    masked_target = targets[mask]
    masked_output = output[mask]
    return (masked_output.argmax(dim=-1) == masked_target).float().mean(dim=0)


def rmse(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    masked_target = targets[mask]
    masked_output = output[mask]
    return torch.sqrt(F.mse_loss(masked_output, masked_target, reduction="mean"))


def loss_by_task(
    prediction: TensorDict,
    target: TensorDict,
) -> TensorDict:
    assert all(p in target.keys() for p in prediction.keys())

    loss = prediction.copy()
    for key in loss.keys():
        mask = ~torch.isnan(target[key])
        pred = prediction[key][mask]
        tgt = target[key][mask]
        if key in loss.categorical:
            loss[key] = F.cross_entropy(pred, tgt.long())
        elif key in loss.numerical:
            loss[key] = F.mse_loss(pred, tgt)
        else:
           raise ValueError(f"Key {key} not classified as categorical or numerical.")

    return loss 


def get_eval_fn_for(task_name):
  if task_name == "binding":
    def eval_fn(output, nprotons, nneutrons):
      return output * (nprotons + nneutrons)
    return eval_fn
  else:
    return lambda x, *_: x

@torch.inference_mode()
def metric_by_task(
    prediction: TensorDict,
    target: TensorDict,
    feature_transformers: dict,
) -> TensorDict:
    assert all(p in target.keys() for p in prediction.keys())

    metrics = prediction.copy()
    for key in metrics.keys():
        mask = ~torch.isnan(target[key]).flatten()
        pred = prediction[key][mask]
        tgt = target[key][mask]
        eval_fn = get_eval_fn_for(key)
        nprotons = target["z"][mask]
        nneutrons = target["n"][mask]
        pred = eval_fn(pred, nprotons, nneutrons)
        tgt = eval_fn(tgt, nprotons, nneutrons)
        if key in metrics.categorical:
            metrics[key] = accuracy(pred, tgt)
        elif key in metrics.numerical:
            pred = torch.tensor(feature_transformers[key].inverse_transform(pred.cpu()))
            tgt = torch.tensor(feature_transformers[key].inverse_transform(tgt.cpu()))
            metrics[key] = rmse(pred, tgt)
        else:
           raise ValueError(f"Key {key} not classified as categorical or numerical.")

    return metrics 

