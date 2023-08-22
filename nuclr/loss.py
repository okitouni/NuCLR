import torch
from .utils import TensorDict, Fields


mse = torch.nn.functional.mse_loss
cce = torch.nn.functional.cross_entropy


def loss_by_task(preds: TensorDict, targets: TensorDict):
    """Compute the loss for each task.

    Args:
        preds (TensorDict): The predictions.
        targets (TensorDict): The targets.

    Returns:
        TensorDict: The loss for each task.
    """
    losses = TensorDict(fields=preds.fields)
    for field in preds.fields.all_fields:
        mask = ~targets[field].isna()
        pred = preds[field][mask]
        target = targets[field][mask]
        if field in preds.fields.numerical:
            losses[field] = mse(pred, target)
        elif field in preds.fields.categorical:
            losses[field] = cce(pred, target)
    return losses

def metric_by_task(preds: TensorDict, targets: TensorDict, transforms: dict):    
    metrics = TensorDict(fields=preds.fields)
    for field in preds.fields.all_fields:
        mask = ~targets[field].isna()
        pred = preds[field][mask]
        target = targets[field][mask]
        if field in preds.fields.numerical:
            pred = transforms[field].inverse_transform(pred)
            target = transforms[field].inverse_transform(target)
            eval_fn = get_eval_fn_for(field)
            pred = eval_fn(pred, targets["z"], targets["n"])
            target = eval_fn(target, targets["z"], targets["n"])
            metrics[field] = mse(pred, target).sqrt()
        elif field in preds.fields.categorical:
            metrics[field] = (pred.argmax(dim=-1) == target).float().mean()
    return metrics

def get_eval_fn_for(task_name):
  if task_name == "binding":
    def eval_fn(output, n, z):
      return output * (n + z)
    return eval_fn
  else:
    return lambda x, _: x