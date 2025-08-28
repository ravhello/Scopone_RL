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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _NP_AVAILABLE:
        try:
            _np.random.seed(seed)
        except Exception:
            pass


def resolve_seed(seed: int) -> int:
    """
    Return a concrete non-negative seed. If seed < 0, generate a random seed,
    log it to stdout, and return it. Otherwise, return the given seed.
    """
    try:
        s = int(seed)
    except Exception:
        s = 0
    if s >= 0:
        return s
    # Generate a 31-bit random seed from os.urandom with time fallback
    try:
        rb = os.urandom(8)
        s = int.from_bytes(rb, byteorder='little', signed=False) & 0x7fffffff
    except Exception:
        try:
            base = int(time.time() * 1e6) ^ int(os.getpid())
        except Exception:
            from utils.fallback import notify_fallback
            notify_fallback('seed.resolve_seed.time_pid_fallback')
        s = (int(time.time() * 1000) ^ os.getpid()) & 0x7fffffff
    try:
        print(f"[seed] Using randomly generated seed: {s}")
    except Exception:
        pass
    return s


@contextmanager
def temporary_seed(seed: int):
    """
    Temporarily set RNG seeds for Python, NumPy and Torch (CPU/GPU), restoring
    previous RNG states on exit. Useful to make eval/MCTS deterministic without
    affecting training randomness.
    """
    # Snapshots
    try:
        py_state = random.getstate()
    except Exception:
        py_state = None
    if _NP_AVAILABLE:
        try:
            np_state = _np.random.get_state()
        except Exception:
            np_state = None
    else:
        np_state = None
    try:
        torch_state = torch.get_rng_state()
    except Exception:
        torch_state = None
    try:
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    except Exception:
        cuda_states = None

    # Apply temporary seed
    set_global_seeds(int(seed))
    try:
        yield
    finally:
        # Restore
        try:
            if py_state is not None:
                random.setstate(py_state)
        except Exception:
            pass
        if _NP_AVAILABLE and np_state is not None:
            try:
                _np.random.set_state(np_state)
            except Exception:
                pass
        try:
            if torch_state is not None:
                torch.set_rng_state(torch_state)
        except Exception:
            pass
        try:
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass


