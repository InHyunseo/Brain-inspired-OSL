import random

import numpy as np
import torch


def set_global_seed(seed, deterministic_torch=False):
    """Seed Python/numpy/torch. Set deterministic_torch=True for full
    reproducibility (slower; disables some CUDA kernels)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
