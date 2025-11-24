import random
import torch
import os
import time
from contextlib import contextmanager
try:
    import numpy as _np
    _NP_AVAILABLE = True
except Exception:
    _NP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def set_global_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if _NP_AVAILABLE:
        _np.random.seed(seed)


def resolve_seed(seed: int) -> int:
    """
    Return a concrete non-negative seed. If seed < 0, generate a random seed,
    log it to stdout, and return it. Otherwise, return the given seed.
    """
    s = int(seed)
    if s >= 0:
        return s
    # Generate a 31-bit random seed from os.urandom with time fallback
    rb = os.urandom(8)
    s = int.from_bytes(rb, byteorder='little', signed=False) & 0x7fffffff
    print(f"[seed] Using randomly generated seed: {s}")
    return s


@contextmanager
def temporary_seed(seed: int):
    """
    Temporarily set RNG seeds for Python, NumPy and Torch (CPU only), restoring
    previous RNG states on exit. Useful to make eval/MCTS deterministic without
    affecting training randomness.
    """
    # Snapshots
    py_state = random.getstate()
    if _NP_AVAILABLE:
        np_state = _np.random.get_state()
    else:
        np_state = None
    torch_state = torch.get_rng_state()
    cuda_states = None

    # Apply temporary seed
    set_global_seeds(int(seed))
    try:
        yield
    finally:
        # Restore
        if py_state is not None:
            random.setstate(py_state)
        if _NP_AVAILABLE and np_state is not None:
            _np.random.set_state(np_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        # No CUDA state restore in CPU-only mode


