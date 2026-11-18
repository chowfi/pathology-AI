"""One place to seed every random number generator the project uses."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch, and make cuDNN deterministic.

    Call this at the start of every entry-point script. The seed itself lives in conf/*.yaml.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
