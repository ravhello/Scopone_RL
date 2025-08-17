import os
import random
from typing import List, Tuple, Dict
import math
import json


class League:
    """
    Mantiene un pool di checkpoint (main + storici) e fornisce sampling per partner/avversari.
    Tiene un Elo semplice e campiona con softmax su Elo.
    """
    def __init__(self, base_dir: str = 'checkpoints/league', seed: int = 123):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.history: List[str] = []  # percorsi ai checkpoint storici
        self.elo: Dict[str, float] = {}
        self.rng = random.Random(seed)
        self._state_path = os.path.join(self.base_dir, 'league.json')
        self._load()

    def register(self, ckpt_path: str, init_elo: float = 1000.0):
        if os.path.isfile(ckpt_path) and ckpt_path not in self.history:
            self.history.append(ckpt_path)
            self.elo[ckpt_path] = init_elo
            self._save()

    def update_elo(self, ckpt_a: str, ckpt_b: str, result_a: float, k: float = 16.0):
        # result_a: 1 win, 0.5 draw, 0 loss
        ra = self.elo.get(ckpt_a, 1000.0)
        rb = self.elo.get(ckpt_b, 1000.0)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 / (1.0 + 10 ** ((ra - rb) / 400.0))
        self.elo[ckpt_a] = ra + k * (result_a - ea)
        self.elo[ckpt_b] = rb + k * ((1.0 - result_a) - eb)
        self._save()

    def _softmax_sample(self, items: List[str], temp: float = 1.0) -> str:
        if not items:
            return None
        elos = [self.elo.get(x, 1000.0) for x in items]
        if temp <= 0:
            return max(zip(items, elos), key=lambda t: t[1])[0]
        exps = [math.exp(e / (400.0 * temp)) for e in elos]
        s = sum(exps)
        probs = [x / s for x in exps]
        r = self.rng.random()
        acc = 0.0
        for item, p in zip(items, probs):
            acc += p
            if r <= acc:
                return item
        return items[-1]

    def sample_pair(self, temp: float = 1.0) -> Tuple[str, str]:
        """Ritorna (partner_ckpt, opponent_ckpt) dal pool via softmax su Elo. Fallback a None se vuoto."""
        if not self.history:
            return None, None
        partner = self._softmax_sample(self.history, temp)
        opponent = self._softmax_sample(self.history, temp)
        return partner, opponent

    def _save(self):
        try:
            data = {"history": self.history, "elo": self.elo}
            with open(self._state_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.isfile(self._state_path):
                with open(self._state_path, 'r') as f:
                    data = json.load(f)
                self.history = data.get('history', [])
                self.elo = {k: float(v) for k, v in data.get('elo', {}).items()}
        except Exception:
            pass



