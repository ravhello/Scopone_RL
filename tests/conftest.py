import pytest

try:
    import torch
    _has_cuda = torch.cuda.is_available()
except Exception:
    _has_cuda = False

if not _has_cuda:
    pytest.skip("CUDA required for GPU-only pipeline; skipping tests on systems without NVIDIA driver", allow_module_level=True)



