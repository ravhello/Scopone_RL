# actions.py
import os
import torch

# Device for action encoding/decoding — controlled exclusively via ACTIONS_DEVICE (default CPU)
_ACTIONS_DEVICE_STR = os.environ.get("ACTIONS_DEVICE", "cpu")
device = torch.device(_ACTIONS_DEVICE_STR)
def encode_action_from_ids_tensor(played_id_t: torch.Tensor, captured_ids_t: torch.Tensor) -> torch.Tensor:
    """
    Variant tensor-native: costruisce il vettore azione 80-d direttamente da tensori int
    (CPU o CUDA) senza conversioni in Python int.
    Accetta uno scalare int64 per la carta giocata e un tensore 1-D int64 per le catture (anche vuoto).
    """
    # Ensure tensors are on target device (CPU by default) and correct dtypes/shapes
    pid = played_id_t.to(device=device, dtype=torch.long).reshape(())
    cap = captured_ids_t.to(device=device, dtype=torch.long).reshape(-1)

    # Fast flat 80-d vector: 0..39 played one-hot, 40..79 captured multi-hot
    vec = torch.zeros((80,), dtype=torch.float32, device=device)
    # Played bit
    pid_clamped = torch.clamp(pid, 0, 39)
    vec[pid_clamped] = 1.0
    # Captured bits
    if cap.numel() > 0:
        cap_clamped = torch.clamp(cap, 0, 39)
        idx = cap_clamped + 40
        # Avoid out-of-bounds and duplicate writes are fine (idempotent to 1.0)
        vec.index_fill_(0, idx, 1.0)
    return vec


def _decode_ids(vec_t: torch.Tensor):
    if vec_t.dim() != 1 or vec_t.numel() != 80:
        raise ValueError("decode_action_ids richiede un vettore 80-dim")
    if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
        if not torch.isfinite(vec_t).all():
            raise RuntimeError("decode_action_ids: vettore azione contiene valori non-finiti")
    # Valori ammessi in [0,1] con tolleranza numerica
    vmin = float(vec_t.min().item())
    vmax = float(vec_t.max().item())
    if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
        if vmin < -1e-6 or vmax > 1.0 + 1e-6:
            raise RuntimeError(f"decode_action_ids: valori fuori range [0,1] (min={vmin}, max={vmax})")
    # Esattamente 1 bit per la carta giocata
    played_sum = vec_t[:40].sum()
    s_val = float(played_sum.item())
    if s_val <= 0:
        raise ValueError("decode_action_ids: nessuna carta giocata indicata (somma sezione 0..39 == 0)")
    # Esattamente un bit a 1 nella sezione played (tolleranza numerica minima)
    if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
        if abs(s_val - 1.0) > 1e-6:
            raise RuntimeError(f"decode_action_ids: sezione played deve avere somma 1.0, trovata {s_val}")
    # Sezione captured: consentiti solo 0/1 (entro tolleranza)
    cap = vec_t[40:]
    cap_bad = ((cap > 0.0 + 1e-6) & (cap < 1.0 - 1e-6)) | (cap < -1e-6) | (cap > 1.0 + 1e-6)
    if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
        if bool(cap_bad.any().item() if torch.is_tensor(cap_bad) else cap_bad.any()):
            raise RuntimeError("decode_action_ids: sezione captured deve essere binaria (0/1)")
    # Avoid per-element .item(); move once to CPU for indexing scalars
    played_idx = torch.argmax(vec_t[:40])
    captured_ids = torch.nonzero(vec_t[40:] > 0, as_tuple=False).flatten()
    played_idx_cpu = int(played_idx.item())
    captured_cpu = captured_ids.tolist()
    return played_idx_cpu, captured_cpu

def decode_action_ids(action_vec: torch.Tensor):
    """
    Decodifica un vettore azione 80-dim in (played_id, [captured_ids]) con card IDs 0..39.
    Accetta solo torch.Tensor per API coerente nel core.
    """
    if not torch.is_tensor(action_vec):
        raise TypeError("decode_action_ids expects a torch.Tensor of shape (80,)")
    vec_t = action_vec.detach().to('cpu', dtype=torch.float32).reshape(-1)
    return _decode_ids(vec_t)

# ----- FAST PATH: subset-sum su ID con numba -----
def find_sum_subsets_ids(table_ids, target_rank: int):
    """
    Restituisce liste di ID dal tavolo la cui somma dei rank è target_rank.
    Usa numba se disponibile per enumerare i sottoinsiemi via bitmask sulle rank.
    """
    if not table_ids:
        return []
    ids = torch.as_tensor([int(x) for x in table_ids], dtype=torch.long, device=device)
    ranks = (ids // 4) + 1
    n = int(ids.numel())
    if n <= 0:
        return []
    pos = torch.arange(n, device=device, dtype=torch.long)
    masks = torch.arange(1, 1 << n, device=device, dtype=torch.long)
    sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.long)
    sums = (sel * ranks.unsqueeze(0)).sum(dim=1)
    good = (sums == int(target_rank)).nonzero(as_tuple=False).flatten()
    results = []
    for gi in good:
        subset = ids[((masks[gi].unsqueeze(0) >> pos) & 1).bool()].tolist()
        results.append([int(x) for x in subset])
    return results
