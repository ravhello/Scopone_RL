import random
import torch

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def set_global_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # GPU-only: always seed CUDA
    torch.cuda.manual_seed_all(seed)


