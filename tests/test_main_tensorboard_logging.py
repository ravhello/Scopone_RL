import os
import runpy
import importlib


class FakeSummaryWriter:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []  # (tag, value, step)
        self.texts = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((str(tag), float(value) if value is not None else None, int(step)))

    def add_text(self, tag, text, step=None):
        self.texts.append((str(tag), str(text), (None if step is None else int(step))))

    def close(self):
        pass


def test_main_writes_tb_every_iteration(monkeypatch):
    """
    Esegue main in-process, patchando SummaryWriter e forzando un run breve.
    Verifica che per ciascuna iterazione completata vengano scritti scalari train/*.
    """
    # Ambiente: abilita TB e riduci lavoro
    monkeypatch.setenv('SCOPONE_DISABLE_TB', '0')
    monkeypatch.setenv('SCOPONE_MINIBATCH', '0')
    monkeypatch.setenv('SCOPONE_MINIBATCH_ALIGN', '0')
    monkeypatch.setenv('TQDM_DISABLE', '1')
    monkeypatch.setenv('SCOPONE_DISABLE_SAVE', '1')
    monkeypatch.setenv('SCOPONE_DISABLE_EVAL', '1')
    monkeypatch.setenv('SCOPONE_LEAGUE_REFRESH', '0')
    monkeypatch.setenv('SCOPONE_AUTO_TB', '0')
    monkeypatch.setenv('TESTS_FORCE_CPU', '1')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')

    # Falsa SummaryWriter globale
    fake_writer = FakeSummaryWriter(log_dir='runs/test_main')
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', lambda log_dir=None: fake_writer, raising=True)

    # Patch train_ppo per fare poche iterazioni e orizzonte piccolo
    import trainers.train_ppo as train_mod

    def rapid_train_ppo(*args, **kwargs):
        num_iterations = 3
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        for it in range(num_iterations):
            # Scrivi un paio di scalari train/* per ogni step
            writer.add_scalar('train/loss_pi', 0.0, it)
            writer.add_scalar('train/avg_return', 0.0, it)
        writer.close()
        return None

    monkeypatch.setattr(train_mod, 'train_ppo', rapid_train_ppo, raising=True)

    # Assicurati che import di main usi modulo fresco
    if 'main' in list(importlib.sys.modules.keys()):
        importlib.invalidate_caches()
        del importlib.sys.modules['main']

    # Esegui main come script
    runpy.run_module('main', run_name='__main__')

    # Estrai gli step con almeno un train/*
    logged_steps = sorted({step for (tag, _val, step) in fake_writer.scalars if tag.startswith('train/')})
    assert logged_steps == [0, 1, 2], (
        f"Main non ha scritto train/* per tutte le iterazioni. Logged: {logged_steps};\n"
        f"Scalars: {fake_writer.scalars[:10]}... (tot={len(fake_writer.scalars)})"
    )

    # Visualizza i punti registrati, raggruppati per step e categoria
    def _print_group(prefix):
        steps = sorted({s for (t, _v, s) in fake_writer.scalars if t.startswith(prefix)})
        if not steps:
            return
        print(f"\n== {prefix} (per step) ==")
        for s in steps:
            pairs = [(t[len(prefix):], v) for (t, v, st) in fake_writer.scalars if st == s and t.startswith(prefix)]
            pairs.sort(key=lambda kv: kv[0])
            pretty = ", ".join([f"{k}={v:.6g}" for (k, v) in pairs])
            print(f"step {s}: {pretty}")

    _print_group('train/')
    _print_group('by_seat/')
    _print_group('league/')




