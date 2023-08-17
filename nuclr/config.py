import os
from argparse import Namespace
import torch

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(rootdir, "data")

defaults_dict = {
    "DEV": "cuda:0" if torch.cuda.is_available() else "cpu",
    "TARGETS_CLASSIFICATION": [],
    "TARGETS_REGRESSION": ["binding", "radius"],
    "TRAIN_FRAC": 0.8,
    "SEED": 0,
}

config = Namespace(**defaults_dict)

__all__ = ["config", "rootdir"]