import threading

_lock = threading.Lock()
_stateproj = {
    't_state_enc': 0.0,
    't_belief_logits': 0.0,
    't_belief_probs': 0.0,
    't_belief_gates': 0.0,
    't_merge': 0.0,
    't_proj': 0.0,
    'count': 0,
}

def accum_actor_stateproj(t_state_enc: float,
                          t_belief_logits: float,
                          t_belief_probs: float,
                          t_belief_gates: float,
                          t_merge: float,
                          t_proj: float) -> None:
    with _lock:
        _stateproj['t_state_enc'] += float(t_state_enc)
        _stateproj['t_belief_logits'] += float(t_belief_logits)
        _stateproj['t_belief_probs'] += float(t_belief_probs)
        _stateproj['t_belief_gates'] += float(t_belief_gates)
        _stateproj['t_merge'] += float(t_merge)
        _stateproj['t_proj'] += float(t_proj)
        _stateproj['count'] += 1

def snapshot_actor_stateproj() -> dict:
    with _lock:
        return dict(_stateproj)

def reset_actor_stateproj() -> None:
    with _lock:
        for k in _stateproj.keys():
            _stateproj[k] = 0.0 if k != 'count' else 0

