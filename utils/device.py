import os
import torch


def _safe_make_device(device_str: str) -> torch.device:
    """
    Create a torch.device from a string, falling back to CPU if the requested
    CUDA device is unavailable. Accepts values like 'cuda', 'cuda:0', 'cpu'.
    """
    try:
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(device_str)
    except Exception:
        return torch.device('cpu')


def get_compute_device() -> torch.device:
    """
    Select the compute device for models and training.
    Priority:
    1) SCOPONE_DEVICE env var, if set
    2) 'cuda' if available and TESTS_FORCE_CPU != '1'
    3) 'cpu'
    """
    forced_cpu = os.environ.get('TESTS_FORCE_CPU') == '1'
    env_dev = os.environ.get('SCOPONE_DEVICE')
    if env_dev:
        return _safe_make_device(env_dev)
    if (not forced_cpu) and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_env_device() -> torch.device:
    """
    Select the device for the environment internals.
    Best-performance default: match compute device (CUDA when available).
    Override with ENV_DEVICE if explicitly set.
    """
    env_dev = os.environ.get('ENV_DEVICE')
    if env_dev:
        return _safe_make_device(env_dev)
    # Default to compute device for end-to-end GPU pipeline
    return get_compute_device()


def get_amp_dtype() -> torch.dtype:
    """
    Return autocast dtype based on AMP_DTYPE env var: 'bf16' or 'fp16' (default).
    """
    amp = os.environ.get('AMP_DTYPE', 'fp16').lower()
    return torch.bfloat16 if amp == 'bf16' else torch.float16


