import inspect
from typing import Any, Optional

import torch

try:
    _SUPPORTS_WEIGHTS_ONLY = 'weights_only' in inspect.signature(torch.load).parameters
except (TypeError, ValueError):
    _SUPPORTS_WEIGHTS_ONLY = False


def safe_torch_load(path: str, map_location: Optional[Any] = None, **kwargs: Any) -> Any:
    """
    Wrapper around torch.load that enables weights_only when supported by the running PyTorch.
    Falls back to the legacy behaviour on older versions that do not accept the flag.
    """
    params = dict(kwargs)
    if map_location is not None:
        params['map_location'] = map_location
    use_weights_only = _SUPPORTS_WEIGHTS_ONLY and 'weights_only' not in params
    if use_weights_only:
        params['weights_only'] = True
    try:
        return torch.load(path, **params)
    except TypeError:
        if use_weights_only:
            params.pop('weights_only', None)
            return torch.load(path, **params)
        raise
