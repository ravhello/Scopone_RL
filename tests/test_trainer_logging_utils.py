import os

import torch
import pytest

from trainers.train_ppo import (
    _flatten_cpu,
    _tensor_basic_stats,
    _format_stats,
    _maybe_log_ppo_batch,
    _to_cpu_float32,
)


def test_flatten_cpu_preserves_empty_and_casts_dtype():
    assert _flatten_cpu(None) is None
    empty = torch.tensor([], dtype=torch.float64)
    assert _flatten_cpu(empty).numel() == 0
    t = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    flat = _flatten_cpu(t, dtype=torch.float32)
    assert flat.shape == (2,)
    assert flat.dtype == torch.float32
    assert torch.allclose(flat, torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_tensor_basic_stats_and_formatting():
    t = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    stats = _tensor_basic_stats(t)
    assert stats['count'] == 3
    assert stats['mean'] == pytest.approx(3.0)
    assert stats['min'] == pytest.approx(1.0)
    assert stats['max'] == pytest.approx(5.0)
    msg = _format_stats('sample', stats)
    assert 'sample=c3' in msg and 'min1.0000' in msg and 'max5.0000' in msg
    msg_empty = _format_stats('empty', {'count': 0})
    assert msg_empty == 'empty=c0'


def test_to_cpu_float32_handles_lists_and_dtypes():
    out = _to_cpu_float32([1, 2, 3])
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (3,)
    t = torch.tensor([1.0], dtype=torch.float64)
    out2 = _to_cpu_float32(t)
    assert out2.dtype == torch.float32
    assert torch.allclose(out2, torch.tensor([1.0], dtype=torch.float32))


def test_maybe_log_ppo_batch_emits_output(monkeypatch, capsys):
    import trainers.train_ppo as train_mod
    monkeypatch.setenv('SCOPONE_PPO_DEBUG', '1')
    monkeypatch.setattr(train_mod, '_PPO_DEBUG', True, raising=False)
    rew = torch.tensor([1.0, -1.0])
    ret = torch.tensor([1.0, -1.0])
    adv = torch.tensor([0.5, -0.5])
    val = torch.tensor([0.2, -0.2])
    done = torch.tensor([0.0, 1.0])
    seat = torch.zeros((2, 6), dtype=torch.float32)
    seat[0, 0] = 1.0
    seat[1, 1] = 1.0
    _maybe_log_ppo_batch(
        'test',
        rew,
        ret,
        adv,
        val,
        val,
        done_mask=done,
        logp=torch.tensor([0.1, 0.2]),
        seat_tensor=seat,
        episode_lengths=[1, 1],
        extra={'note': 'ok'},
    )
    captured = capsys.readouterr()
    merged = (captured.out or '') + (captured.err or '')
    assert '[ppo-debug test]' in merged
