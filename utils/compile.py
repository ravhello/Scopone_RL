import os
from typing import Any, Callable, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _env_enabled() -> bool:
    """Return True if global torch.compile is requested via unified or legacy env vars.
    Defaults to enabled.
    """
    # Unified flag (default OFF; opt-in to avoid backend failures by default)
    if os.environ.get('SCOPONE_TORCH_COMPILE', '0') == '1':
        return True
    # Back-compat: legacy flags
    if os.environ.get('TORCH_COMPILE', '0') == '1':
        return True
    if os.environ.get('ENABLE_TORCH_COMPILE', '0') == '1':
        return True
    return False


def _env_mode() -> str:
    """Return compile mode from unified or legacy env vars, with a sensible default (default)."""
    return os.environ.get(
        'SCOPONE_TORCH_COMPILE_MODE',
        os.environ.get('TORCH_COMPILE_MODE', 'default')
    )


def _env_backend() -> str:
    """Return backend for torch.compile. Default to 'inductor' (best on CPU and CUDA)."""
    return os.environ.get('SCOPONE_TORCH_COMPILE_BACKEND', os.environ.get('TORCH_COMPILE_BACKEND', 'inductor'))


def _maybe_setup_dynamo() -> None:
    """Best-effort configuration for torch._dynamo to avoid hard failures when compiling."""
    if torch is None:
        return
    import logging as _logging  # type: ignore
    import torch._dynamo as _dynamo  # type: ignore
    # Avoid throwing on graphs that the compiler cannot handle; fallback to eager.
    _dynamo.config.suppress_errors = True
    # Prefer dynamic shapes to reduce shape constraint eval failures
    _dynamo.config.dynamic_shapes = True
    # Avoid graph breaks from scalar .item() by capturing scalar outputs in graphs
    _dynamo.config.capture_scalar_outputs = True
    # Increase cache size limit to reduce recompilation churn
    limit = int(os.environ.get('SCOPONE_DYNAMO_CACHE_SIZE_LIMIT', os.environ.get('TORCHDYNAMO_CACHE_SIZE_LIMIT', '32')))
    _dynamo.config.cache_size_limit = limit
    # Silence noisy FX/shape logs
    for _name in (
        'torch.fx.experimental.symbolic_shapes',
        'torch.fx.experimental.recording',
    ):
        _lg = _logging.getLogger(_name)
        _lg.setLevel(_logging.CRITICAL)
        _lg.propagate = False
        _lg.disabled = True
    # Use new torch._logging to lower internal categories if present
    import torch._logging as _tlog  # type: ignore
    _tlog.set_logs(dynamo="error", inductor="error", aot="error")
    # Inductor autotune knobs: respect env if provided; otherwise don't override defaults
    from torch._inductor import config as _inductor_cfg  # type: ignore
    _auto_env = os.environ.get('SCOPONE_INDUCTOR_AUTOTUNE', None)
    if _auto_env is not None:
        _want_auto = (_auto_env == '1')
        if hasattr(_inductor_cfg, 'max_autotune'):
            _inductor_cfg.max_autotune = _want_auto
        if hasattr(_inductor_cfg, 'max_autotune_pointwise'):
            _inductor_cfg.max_autotune_pointwise = _want_auto
        if hasattr(_inductor_cfg, 'max_autotune_gemm'):
            _inductor_cfg.max_autotune_gemm = _want_auto
    # Disable cudagraphs if present (helps reduce noisy skips on some CUDA setups)
    if hasattr(_inductor_cfg, 'triton') and hasattr(_inductor_cfg.triton, 'cudagraphs'):
        _inductor_cfg.triton.cudagraphs = False
    # Also lower log level for inductor utils
    _logging.getLogger('torch._inductor.utils').setLevel(_logging.ERROR)


def is_enabled() -> bool:
    """Public checker for whether torch.compile should be enabled globally."""
    return bool(_env_enabled() and (torch is not None) and hasattr(torch, 'compile'))


def get_mode() -> str:
    """Public accessor for the selected compile mode. If cudagraphs are disabled,
    prefer a no-cudagraphs mode variant to avoid noisy skips.
    """
    mode = _env_mode()
    from torch._inductor import config as _inductor_cfg  # type: ignore
    cg_enabled = True
    if hasattr(_inductor_cfg, 'triton') and hasattr(_inductor_cfg.triton, 'cudagraphs'):
        cg_enabled = bool(_inductor_cfg.triton.cudagraphs)
    if not cg_enabled and mode == 'max-autotune':
        return 'max-autotune-no-cudagraphs'
    return mode


def maybe_compile_module(module: Any, name: Optional[str] = None) -> Any:
    """
    Compile a nn.Module (or any callable) with torch.compile if enabled and available.
    Returns the compiled object or the original on failure/disabled.
    """
    if not is_enabled():
        return module
    _maybe_setup_dynamo()
    mode = get_mode()
    backend = _env_backend()
    compiled = torch.compile(module, mode=mode, backend=backend)  # type: ignore[attr-defined]
    if os.environ.get('SCOPONE_COMPILE_VERBOSE', '0') == '1':
        nm = name or getattr(module, '__class__', type(module)).__name__
        print(f"[compile] Compiled module: {nm} (mode={mode}, backend={backend})")
    return compiled


def maybe_compile_function(fn: Callable, name: Optional[str] = None) -> Callable:
    """
    Compile a plain function or bound method with torch.compile if enabled and available.
    Returns the compiled callable or the original on failure/disabled.
    """
    if not is_enabled():
        return fn
    _maybe_setup_dynamo()
    mode = get_mode()
    backend = _env_backend()
    compiled = torch.compile(fn, mode=mode, backend=backend)  # type: ignore[attr-defined]
    if os.environ.get('SCOPONE_COMPILE_VERBOSE', '0') == '1':
        nm = name or getattr(fn, '__name__', str(fn))
        print(f"[compile] Compiled function: {nm} (mode={mode}, backend={backend})")
    return compiled


