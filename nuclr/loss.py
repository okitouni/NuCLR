import torch
from .utils import TensorDict

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
        mask = ~targets[field].isnan().view(-1)
        pred = preds[field][mask]
        target = targets[field][mask]
        if field in preds.fields["numerical"]:
            losses[field] = mse(pred, target)
        else:
            losses[field] = cce(pred, target)
    return losses

def metric_by_task(preds: TensorDict, targets: TensorDict, transforms: dict):    
    metrics = TensorDict(fields=preds.fields)
    for field in preds.fields.all_fields:
        mask = ~targets[field].isnan()
        pred = preds[field][mask].view(-1, 1)
        target = targets[field][mask].view(-1, 1)
        if field in preds.fields["numerical"]:
            pred = transforms[field].inverse_transform(pred.cpu().detach().numpy())
            target = transforms[field].inverse_transform(target.cpu().detach().numpy())
            eval_fn = get_eval_fn_for(field)
            z = targets["z"][mask].view(-1, 1).cpu().detach().numpy()
            n = targets["n"][mask].view(-1, 1).cpu().detach().numpy()
            
            pred = eval_fn(pred, z, n)
            target = eval_fn(target, z, n)
            
            rmse = ((pred-target)**2).mean()**0.5
            metrics[field] = torch.tensor([rmse])
        else:
            metrics[field] = (pred.argmax(dim=-1, keepdim=True) == target).float().mean()
    return metrics

def get_eval_fn_for(task_name):
  if task_name == "binding":
    def eval_fn(output, n, z):
      return output * (n + z)
    return eval_fn
  else:
    return lambda x, *_: x