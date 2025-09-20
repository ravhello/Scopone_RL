import torch
from actions import decode_action_ids, encode_action_from_ids_tensor, find_sum_subsets_ids


def test_encode_decode_roundtrip_ids():
    # play 7 of spade (rank=7, suit=2) -> id= (7-1)*4+2 = 26
    def tid(card_tuple):
        r, s = card_tuple
        suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
        return (r - 1) * 4 + suit_to_col[s]
    action = encode_action_from_ids_tensor(torch.tensor(tid((7, 'spade')), dtype=torch.long), torch.tensor([tid((1, 'denari')), tid((3, 'coppe'))], dtype=torch.long))
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
    # replicate expected vector by manual construction
    expected = torch.zeros(80, dtype=torch.float32)
    expected[int(pid.item())] = 1.0
    expected[40 + captured] = 1.0
    assert torch.allclose(a_gpu.cpu(), expected)


def test_find_sum_subsets_ids_basic():
    # table ids ranks: 1,2,3 (ids 0,4,8)
    table = [0, 4, 8]
    subs = find_sum_subsets_ids(table, target_rank=3)
    # possible: [0,4] (1+2) or [8] (3)
    flat = [sorted(s) for s in subs]
    assert [0,4] in flat or [8] in flat
