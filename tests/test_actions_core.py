import torch
from actions import encode_action, decode_action_ids, encode_action_from_ids_tensor, find_sum_subsets_ids


def test_encode_decode_roundtrip_ids():
    # play 7 of spade (rank=7, suit=2) -> id= (7-1)*4+2 = 26
    action = encode_action((7, 'spade'), [(1, 'denari'), (3, 'coppe')])
    pid, caps = decode_action_ids(action)
    assert 0 <= pid < 40
    assert all(0 <= c < 40 for c in caps)
    # Ensure played bit set
    played_mat = action[:40].reshape(10, 4)
    assert torch.isclose(played_mat.sum(), torch.tensor(1.0))


def test_encode_from_ids_gpu_matches_cpu_encoding():
    pid = torch.tensor(9, dtype=torch.long)
    captured = torch.tensor([0, 15, 39], dtype=torch.long)
    a_gpu = encode_action_from_ids_tensor(pid, captured)
    a_cpu = encode_action(int(pid.item()), [int(x.item()) for x in captured])
    assert torch.allclose(a_gpu, a_cpu)


def test_find_sum_subsets_ids_basic():
    # table ids ranks: 1,2,3 (ids 0,4,8)
    table = [0, 4, 8]
    subs = find_sum_subsets_ids(table, target_rank=3)
    # possible: [0,4] (1+2) or [8] (3)
    flat = [sorted(s) for s in subs]
    assert [0,4] in flat or [8] in flat
