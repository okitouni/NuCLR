import torch
from torch import nn
from .tensor_dict import TensorDict

class NuCLRWrapper(nn.Module):
    def __init__(self, model, pred_fields):
        super().__init__()
        self.model = model
        self.pred_fields = pred_fields

    def forward(self, batch: TensorDict):
        preds = TensorDict(fields=self.pred_fields)
        tensor_preds = self.model(batch.to_tensor())
        for i, field in enumerate(self.pred_fields.all_fields):
            preds[field] = tensor_preds[:, [i]]
        return preds