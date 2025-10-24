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

    def update_elo(self, ckpt_a: str, ckpt_b: str, result_a: float, k: float = 128.0):
        # result_a: 1 win, 0.5 draw, 0 loss
        ra = self.elo.get(ckpt_a, 1000.0)
        rb = self.elo.get(ckpt_b, 1000.0)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 / (1.0 + 10 ** ((ra - rb) / 400.0))
        self.elo[ckpt_a] = ra + k * (result_a - ea)
        self.elo[ckpt_b] = rb + k * ((1.0 - result_a) - eb)
        self._save()

    def update_elo_from_diff(self, ckpt_a: str, ckpt_b: str, avg_point_diff_a: float, k: float = 128.0, tau: float = None):
        """
        Aggiorna l'Elo usando la differenza media di punti (reward) con mappatura LINEARE a score [0,1].
        score = clamp(0.5 + diff / (2*scale), 0, 1)
        Dove scale è controllabile via env SCOPONE_ELO_DIFF_SCALE e default=6.0.
        """
        # Leggi la scala da env; consenti override via arg 'tau' (alias di scale)
        try:
            _scale_env = float(os.environ.get('SCOPONE_ELO_DIFF_SCALE', '6.0'))
        except Exception:
            _scale_env = 6.0
        scale = float(tau) if (tau is not None) else float(_scale_env)
        scale = max(1e-6, scale)
        try:
            d = float(avg_point_diff_a)
        except Exception:
            d = 0.0
        # Linear mapping with hard clamp
        score_a = 0.5 + (d / (2.0 * scale))
        if score_a < 0.0:
            score_a = 0.0
        elif score_a > 1.0:
            score_a = 1.0
        return self.update_elo(ckpt_a, ckpt_b, score_a, k=k)

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
        """Ritorna (partner_ckpt, opponent_ckpt) dal pool via softmax su Elo. Nessun fallback consentito."""
        if not self.history:
            from utils.fallback import notify_fallback
            notify_fallback('selfplay.league.empty_history')
        partner = self._softmax_sample(self.history, temp)
        opponent = self._softmax_sample(self.history, temp)
        return partner, opponent

    def _save(self):
        data = {"history": self.history, "elo": self.elo}
        with open(self._state_path, 'w') as f:
            json.dump(data, f)

    def _load(self):
        if os.path.isfile(self._state_path):
            with open(self._state_path, 'r') as f:
                data = json.load(f)
            self.history = data.get('history', [])
            self.elo = {k: float(v) for k, v in data.get('elo', {}).items()}



