import torch

# Add numpy-like method for dtype casting
if not hasattr(torch.Tensor, 'astype'):
    def _astype(self, dtype):
        return self.to(dtype)
    torch.Tensor.astype = _astype  # type: ignore

# Provide numpy-like copy() on tensors
if not hasattr(torch.Tensor, 'copy'):
    def _copy(self):
        return self.clone()
    torch.Tensor.copy = _copy  # type: ignore

class _Random:
    @staticmethod
    def seed(seed: int):
        torch.manual_seed(seed)

    @staticmethod
    def rand(*shape):
        return torch.rand(shape, device=torch.device('cuda'))

class _NPShim:
    float32 = torch.float32
    float64 = torch.float64
    random = _Random

    @staticmethod
    def array(x, dtype=None):
        # If x is already a tensor, just cast/return on CUDA
        if torch.is_tensor(x):
            return x.to(dtype=dtype or x.dtype, device=torch.device('cuda'))
        try:
            # Handle sequences of equal-size tensors by stacking
            if isinstance(x, (list, tuple)) and len(x) > 0 and all(torch.is_tensor(e) for e in x):
                return torch.stack([e.to(device=torch.device('cuda')) for e in x], dim=0).to(dtype=dtype or torch.float32)
        except Exception:
            pass
        return torch.tensor(x, dtype=dtype or torch.float32, device=torch.device('cuda'))

    @staticmethod
    def zeros(shape, dtype=None):
        return torch.zeros(shape, dtype=dtype or torch.float32, device=torch.device('cuda'))

    @staticmethod
    def ones(shape, dtype=None):
        return torch.ones(shape, dtype=dtype or torch.float32, device=torch.device('cuda'))

    @staticmethod
    def zeros_like(x, dtype=None):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'))
        return torch.zeros_like(xt, dtype=dtype or xt.dtype, device=torch.device('cuda'))

    @staticmethod
    def ones_like(x, dtype=None):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'))
        return torch.ones_like(xt, dtype=dtype or xt.dtype, device=torch.device('cuda'))

    @staticmethod
    def stack(seq, axis=0):
        return torch.stack(list(seq), dim=axis)

    @staticmethod
    def any(x):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'))
        return bool(torch.any(xt).item())

    @staticmethod
    def all(x):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'))
        return bool(torch.all(xt).item())

    @staticmethod
    def array_equal(a, b):
        at = a if torch.is_tensor(a) else torch.tensor(a, device=torch.device('cuda'))
        bt = b if torch.is_tensor(b) else torch.tensor(b, device=torch.device('cuda'))
        return bool(torch.equal(at, bt))

    @staticmethod
    def isclose(a, b, rtol=1e-5, atol=1e-8):
        at = a if torch.is_tensor(a) else torch.tensor(a, device=torch.device('cuda'), dtype=torch.float32)
        bt = b if torch.is_tensor(b) else torch.tensor(b, device=torch.device('cuda'), dtype=torch.float32)
        return bool(torch.isclose(at, bt, rtol=rtol, atol=atol).item())

    @staticmethod
    def allclose(a, b, rtol=1e-5, atol=1e-8):
        at = a if torch.is_tensor(a) else torch.tensor(a, device=torch.device('cuda'), dtype=torch.float32)
        bt = b if torch.is_tensor(b) else torch.tensor(b, device=torch.device('cuda'), dtype=torch.float32)
        res = torch.allclose(at, bt, rtol=rtol, atol=atol)
        return bool(res) if isinstance(res, bool) else bool(res.item())

    @staticmethod
    def count_nonzero(x):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'))
        return int(torch.count_nonzero(xt).item())

    @staticmethod
    def sum(x):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'), dtype=torch.float32)
        return torch.sum(xt)

    @staticmethod
    def mean(x):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'), dtype=torch.float32)
        return torch.mean(xt)

    @staticmethod
    def log(x):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'), dtype=torch.float32)
        return torch.log(xt)

    @staticmethod
    def argmax(x):
        xt = x if torch.is_tensor(x) else torch.tensor(x, device=torch.device('cuda'))
        return int(torch.argmax(xt).item())

    @staticmethod
    def unravel_index(index, shape):
        # Returns tuple of indices into array with given shape
        idx = int(index)
        dims = list(shape)
        coords = []
        for size in reversed(dims):
            coords.append(idx % size)
            idx //= size
        return tuple(reversed(coords))

    @staticmethod
    def to_tensor(x, dtype=torch.float32):
        return x if torch.is_tensor(x) else torch.tensor(x, dtype=dtype, device=torch.device('cuda'))

np = _NPShim()


