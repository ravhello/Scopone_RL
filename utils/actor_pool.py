import io
import os
import warnings
from typing import List, Tuple

import torch
import torch.multiprocessing as mp

from models.action_conditioned import ActionConditionedActor, StateEncoderCompact


def _actor_worker_main(state_blob: bytes,
                       obs_dim: int,
                       action_dim: int,
                       k_history_hint,
                       out_dim: int,
                       worker_idx: int,
                       total_workers: int,
                       task_q: mp.Queue,
                       result_q: mp.Queue) -> None:
    torch.set_grad_enabled(False)
    try:
        # Conservative thread allocation per replica
        max_threads = max(1, torch.get_num_threads() // max(1, total_workers))
        torch.set_num_threads(max_threads)
    except RuntimeError:
        pass
    actor = ActionConditionedActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_encoder=StateEncoderCompact(k_history=k_history_hint),
    )
    actor.detach_inference_pool()
    buffer = io.BytesIO(state_blob)
    state_dict = torch.load(buffer, map_location='cpu', weights_only=True)
    actor.load_state_dict(state_dict)
    actor.eval()
    while True:
        msg = task_q.get()
        if msg is None:
            break
        tag = msg[0]
        if tag == 'stop':
            break
        if tag == 'sync':
            blob = msg[1]
            buf = io.BytesIO(blob)
            new_state = torch.load(buf, map_location='cpu', weights_only=True)
            actor.load_state_dict(new_state)
            result_q.put(('sync', None, None))
            continue
        if tag != 'run':
            raise ValueError(f"Actor worker received unknown message tag '{tag}'")
        _, task_id, obs_chunk, seat_chunk = msg
        obs_t = torch.as_tensor(obs_chunk, dtype=torch.float32)
        seat_t = torch.as_tensor(seat_chunk, dtype=torch.float32)
        with torch.no_grad():
            out = actor._compute_state_proj_direct(obs_t, seat_t)
        result_q.put(('run', task_id, out.cpu()))


class ActorReplicaPool:
    def __init__(self, actor: ActionConditionedActor, replicas: int = 2) -> None:
        if replicas < 2:
            raise ValueError("ActorReplicaPool richiede almeno 2 repliche")
        self._replicas = replicas
        self.replicas = replicas
        self._ctx = mp.get_context('spawn')
        self._workers: List[Tuple[mp.Process, mp.Queue, mp.Queue]] = []
        self._task_counter = 0
        self._obs_dim = actor.obs_dim
        self._action_dim = actor.action_dim
        self._k_history_hint = getattr(actor.state_enc, 'k_history_hint', None)
        self._out_dim = actor.state_to_action.out_features
        self._dtype = next(actor.parameters()).dtype
        state_blob = io.BytesIO()
        torch.save(actor.state_dict(), state_blob)
        self._state_blob = state_blob.getvalue()
        # Start workers
        for idx in range(replicas):
            task_q: mp.Queue = self._ctx.Queue(maxsize=2)
            result_q: mp.Queue = self._ctx.Queue(maxsize=2)
            proc = self._ctx.Process(
                target=_actor_worker_main,
                args=(
                    self._state_blob,
                    self._obs_dim,
                    self._action_dim,
                    self._k_history_hint,
                    self._out_dim,
                    idx,
                    replicas,
                    task_q,
                    result_q,
                ),
            )
            proc.daemon = True
            proc.start()
            self._workers.append((proc, task_q, result_q))

    def _split_batch(self, total: int) -> List[Tuple[int, int]]:
        base = total // self._replicas
        remainder = total % self._replicas
        splits: List[Tuple[int, int]] = []
        start = 0
        for idx in range(self._replicas):
            end = start + base + (1 if idx < remainder else 0)
            splits.append((start, end))
            start = end
        return splits

    def compute_state_proj(self, obs: torch.Tensor, seat: torch.Tensor) -> torch.Tensor:
        if obs.device.type != 'cpu' or seat.device.type != 'cpu':
            raise RuntimeError("ActorReplicaPool supporta solo tensori su CPU")
        B = obs.size(0)
        if B == 0:
            return torch.empty((0, self._out_dim), dtype=self._dtype)
        obs_cpu = obs.detach().contiguous()
        seat_cpu = seat.detach().contiguous()
        splits = self._split_batch(B)
        handles = []
        for (start, end), worker in zip(splits, self._workers):
            if start >= end:
                continue
            proc, task_q, _ = worker
            if not proc.is_alive():
                warnings.warn("Una replica dell'attore si è terminata inaspettatamente; si ricade in eager.", RuntimeWarning)
                return self._fallback(obs_cpu, seat_cpu)
            task_id = self._task_counter
            self._task_counter += 1
            task_q.put(('run', task_id, obs_cpu[start:end].clone(), seat_cpu[start:end].clone()))
            handles.append((task_id, start, end, worker[2]))
        result = torch.empty((B, self._out_dim), dtype=self._dtype)
        for task_id, start, end, result_q in handles:
            tag, rid, chunk = result_q.get()
            if tag != 'run' or rid != task_id:
                raise RuntimeError("ActorReplicaPool ha ricevuto una risposta incoerente dai worker")
            result[start:end] = chunk.to(dtype=self._dtype)
        return result

    def _fallback(self, obs: torch.Tensor, seat: torch.Tensor) -> torch.Tensor:
        # Lazy import per evitare ciclo
        actor = ActionConditionedActor(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            state_encoder=StateEncoderCompact(k_history=self._k_history_hint),
        )
        actor.load_state_dict(torch.load(io.BytesIO(self._state_blob), map_location='cpu', weights_only=True))
        actor.eval()
        with torch.no_grad():
            return actor._compute_state_proj_direct(obs, seat)

    def sync_weights(self, actor: ActionConditionedActor) -> None:
        state_blob = io.BytesIO()
        torch.save(actor.state_dict(), state_blob)
        blob = state_blob.getvalue()
        self._state_blob = blob
        for _, task_q, _ in self._workers:
            task_q.put(('sync', blob))
        for _, _, result_q in self._workers:
            result_q.get()

    def close(self) -> None:
        for proc, task_q, _ in self._workers:
            try:
                task_q.put(('stop', None))
            except Exception:
                pass
        for proc, _, _ in self._workers:
            if proc.is_alive():
                proc.join(timeout=1.0)
        self._workers.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
