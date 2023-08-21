import torch


def tprint(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    dict1 = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in dict1.items()
    }
    dict2 = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in dict2.items()
    }
    return ", ".join(f"{k}: {dict1[k]:.3f}|{dict2[k]:.3f}" for k in dict1)