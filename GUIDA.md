## Guida all'uso della repository Scopone_RL

Questa guida spiega come installare, allenare, fare benchmark e profilare l'agente di Scopone. Elenca tutte le opzioni disponibili a riga di comando per gli script principali e chiarisce i parametri di default scelti nel codice.

### Installazione
- Requisiti: Python 3.10+, CUDA opzionale per GPU.
- Installazione pacchetti:
```bash
pip install -r requirements.txt
```

In alternativa, con i target del `Makefile` (richiede l'ambiente conda `gym_tf` già creato):
```bash
make install
```

### Modi di esecuzione principali
- Allenamento PPO: `python trainers/train_ppo.py` (consigliato)
- Benchmark policy/IS-MCTS: `python tools/benchmark_ac.py`
- Confronto checkpoint Team 0: `python benchmark.py`
- Profiling breve dell'allenamento: `python tools/profile_ppo.py`
- Target di comodo del `Makefile`:
  - `make run` → `python main.py` (codice sperimentale separato, senza CLI)
  - `make benchmark` → `python benchmark.py`
  - `make test` → esegue test con pytest

Nota: la GUI (`scopone_gui.py`) è WIP e non espone una CLI stabile.

---

## Allenamento PPO (Action-Conditioned)
Script: `trainers/train_ppo.py`

Esempio rapido (consigliato):
```bash
python trainers/train_ppo.py \
  --iters 2000 --horizon 256 \
  --compact --k-history 12 \
  --seed 0 \
  --ckpt checkpoints/ppo_ac.pth
```

Opzioni CLI e significato:
- `--iters` (int, default 2000): numero di iterazioni PPO da eseguire.
- `--horizon` (int, default 256): numero di step per iterazione (rollout). Con solo reward finale, ~horizon//40 episodi per iter.
- `--save-every` (int, default 200): salva un checkpoint ogni N iterazioni.
- `--ckpt` (str, default `checkpoints/ppo_ac.pth`): percorso file checkpoint.
- `--compact` (flag): usa osservazione compatta (raccomandato in produzione). Se assente, usa la modalità di default non compatta.
- `--k-history` (int, default 39): numero di mosse recenti nella storia compatta. Valori più bassi riducono costo computazionale (es. 12).
- `--seed` (int, default 0): seed di riproducibilità.
- `--entropy-schedule` (str, default `linear`, scelte: `linear`, `cosine`): schedule del coefficiente di entropia durante il training.
- `--eval-every` (int, default 0): ogni quante iterazioni eseguire una mini-valutazione (0 = disabilitata).
- `--eval-games` (int, default 10): numero di partite nella mini-valutazione.
- `--belief-particles` (int, default 512): numero di particelle del filtro di credenze (belief) usato dal trainer.
- `--belief-ess-frac` (float, default 0.5): soglia ESS frazionaria per il resampling del belief.
- `--mcts-eval` (flag): abilita l'uso dell'IS-MCTS nella mini-valutazione.
- `--mcts-train` (flag, default True): presente nella CLI ma attualmente non utilizzato nel trainer. L'allenamento usa sempre IS-MCTS con warmup (disattivo per ~500 iterazioni) e poi attivo; non è disattivabile da CLI.
- `--train-both-teams` (flag): presente nella CLI ma attualmente non utilizzato; il trainer alterna i seat principali tra 0/2 e 1/3 e chiama `collect_trajectory(..., train_both_teams=False)`.
- `--mcts-sims` (int, default 128): numero di simulazioni per mossa nell'IS-MCTS (per train/eval se attivo).
- `--mcts-dets` (int, default 4): numero di determinizzazioni del belief per ricerca MCTS (train/eval se attivo).
- `--mcts-c-puct` (float, default 1.0): costante di esplorazione PUCT.
- `--mcts-root-temp` (float, default 0.0): temperatura alla radice per campionamento basato su visite (0 = argmax visite).
- `--mcts-prior-smooth-eps` (float, default 0.0): smoothing dei prior (1−eps)·p + eps/|A| (default NEUTRO).
- `--mcts-dirichlet-alpha` (float, default 0.25): parametro alpha per rumore Dirichlet alla radice (se `--mcts-dirichlet-eps` > 0).
- `--mcts-dirichlet-eps` (float, default 0.0): mixing con rumore Dirichlet alla radice (default NEUTRO).

Output e logging:
- Checkpoint salvato su `--ckpt` ogni `--save-every` iterazioni (contiene pesi actor/critic e ottimizzatori).
- Log per TensorBoard in `runs/` se disponibile: eseguire `tensorboard --logdir runs` per visualizzare.

Note utili:
- È fortemente consigliato `--compact` con un `--k-history` moderato (es. 12) per tenere bassa la dimensione dell'osservazione.

Parametri PPO interni (scelte di default nel codice `algorithms/ppo_ac.py`):
- `lr=3e-4`, `clip_ratio=0.2`, `value_coef=0.5`, `entropy_coef=0.01`, `value_clip=0.2`, `target_kl=0.02`.
- Early stop se KL supera il target (con pazienza interna); riduzione LR automatica se persiste.
- Mixed precision CUDA (AMP) e clip del gradiente a 0.5.

Regole di gioco (default effettive nel trainer):
- Salvo diversa indicazione, tutte le varianti sono disattivate (`asso_piglia_tutto`, `scopa_on_asso_piglia_tutto`, `re_bello`, `napola`, ecc.).
- `last_cards_to_dealer`: True (a fine mano le carte rimaste vanno all'ultimo catturante quando la regola è attiva).
- `shape_scopa`: False (nessun reward shaping intermedio durante l'episodio).
- Altre chiavi supportate ma non impostate dal trainer restano ai default hard-coded del codice (es. `napola_scoring='fixed3'`, `max_consecutive_scope=None`).

---

## Benchmark policy / IS-MCTS
Script: `tools/benchmark_ac.py`

Esempi:
```bash
# Solo policy AC
python tools/benchmark_ac.py --games 100 --compact --k-history 12 --ckpt checkpoints/ppo_ac.pth --out-json summary.json

# Policy + IS-MCTS (booster)
python tools/benchmark_ac.py --mcts --sims 256 --dets 16 --games 50 \
  --compact --k-history 12 --ckpt checkpoints/ppo_ac.pth --out-json summary_mcts.json
```

Opzioni CLI e significato:
- `--mcts` (flag): abilita IS-MCTS come booster decisionale.
- `--sims` (int, default 128): simulazioni MCTS per mossa.
- `--dets` (int, default 16): determinizzazioni del belief per ricerca.
- `--compact` (flag): usa osservazione compatta.
- `--k-history` (int, default 12): mosse recenti nella storia compatta.
- `--ckpt` (str, default vuoto): percorso checkpoint per actor/critic (opzionale). Se vuoto, usa pesi iniziali.
- `--games` (int, default 50): numero di partite.
- `--seed` (int, default 0): seed.
- `--c-puct` (float, default 1.0): costante PUCT.
- `--root-temp` (float, default 0.0): temperatura alla radice (0 = scelta deterministica per visite).
- `--prior-smooth-eps` (float, default 0.0): smoothing dei prior.
- `--belief-particles` (int, default 256): particelle del belief.
- `--belief-ess-frac` (float, default 0.5): soglia ESS per resampling del belief.
- `--robust-child` (flag): seleziona il figlio con più visite (robust child). Se non impostato, usa max-Q.
- `--root-dirichlet-alpha` (float, default 0.0): alpha del Dirichlet alla radice.
- `--root-dirichlet-eps` (float, default 0.0): mixing con il rumore Dirichlet alla radice (default NEUTRO).
- `--out-csv` (str): salva report per-partita in CSV.
- `--out-json` (str): salva sommario aggregato in JSON.

Output:
- Stampa un sommario con win rate e statistiche di punteggio.
- Opzionalmente, salva CSV per partita e JSON riassuntivo.

---

## Confronto checkpoint Team 0 (pairwise)
Script: `benchmark.py` (entrypoint principale del file)

Esempio:
```bash
python benchmark.py --checkpoint_dir checkpoints/ --checkpoint_pattern "*team0*ep*.pth" --games 5000 --excel confronto.xlsx
```

Opzioni CLI e significato:
- `--checkpoints` (lista di path): file o directory di checkpoint da includere (Team 0). Accetta multipli.
- `--checkpoint_dir` (str): directory contenente checkpoint (filtrati con `--checkpoint_pattern`).
- `--checkpoint_pattern` (str, default `*team0*ep*.pth`): pattern per il match dei file.
- `--games` (int, default 10000): partite per ogni scontro testa-a-testa.
- `--output` (str): file di testo per i risultati (default auto-generato con timestamp).
- `--excel` (str): file Excel per la matrice comparativa (default auto-generato con timestamp).
- `--limit` (int): limita il numero di checkpoint valutati (se > numero trovato, riduce con campionamento equispaziato).

Output:
- Report testuale e file Excel con matrici di win rate, differenza punteggio, vantaggio del primo giocatore e lunghezza media partite.

Note:
- Lo script filtra automaticamente solo i checkpoint di Team 0 (nel nome deve comparire `team0`).

---

## Profiling dell'allenamento
Script: `tools/profile_ppo.py`

Esempi:
```bash
# Profiling con torch.profiler
python tools/profile_ppo.py --iters 50 --horizon 256

# Profiling line-by-line (richiede profiler interno)
python tools/profile_ppo.py --iters 50 --horizon 256 --line --wrap-update --report
```

Opzioni CLI e significato:
- `--iters` (int, default 50): numero di iterazioni da profilare (run breve).
- `--horizon` (int, default 256): horizon per iterazione durante il profiling.
- `--line` (flag): abilita il profiler per-linea con tempi per riga.
- `--wrap-update` (flag): profila anche `ActionConditionedPPO.update` (più lento).
- `--report` (flag): stampa un report esteso del line-profiler.

---

## Consigli pratici e note progettuali
- Osservazione compatta: usare `--compact` e regolare `--k-history` per bilanciare informatività e costo computazionale. Valori tipici: 12–39.
- Entropia: il coefficiente di entropia segue uno schedule (`linear` o `cosine`) per facilitare esplorazione iniziale e stabilizzazione successiva.
- Belief + IS-MCTS: aumentare `--belief-particles`, `--dets` e `--sims` migliora la qualità della ricerca ma scala i tempi.
- Riproducibilità: impostare sempre `--seed`. Se si passa un seed negativo, il codice genera un seed casuale non-negativo e lo stampa. I checkpoint includono un `run_config` minimale.
- Dipendenze: la repo usa `gymnasium` (non `gym`).
- Dispositivi/AMP: su CUDA, il trainer usa AMP con GradScaler (API unificata); su CPU la mixed precision è disabilitata.
- TensorBoard: avviare `tensorboard --logdir runs` per monitorare loss, KL, entropia, grad norm, ecc.
- Dispositivo ambiente: per impostazione predefinita l'ambiente gira su CPU. È possibile forzare il device impostando `ENV_DEVICE` (es. `ENV_DEVICE=cuda`), ma in generale è sconsigliato per via di micro-kernel poco efficienti.

---

## Riepilogo cartelle output
- `checkpoints/`: salvataggi periodici (`--save-every`) del trainer PPO.
- `runs/`: log per TensorBoard.
- Output benchmark: file `.csv`, `.json`, `.txt`, `.xlsx` a seconda delle opzioni.


Comandi comuni:
python /home/rikyravi/Scopone_RL/tools/profile_ppo.py --cprofile --iters 1
python /home/rikyravi/Scopone_RL/tools/profile_ppo.py --line --profile-all  --line-full --iters 1
python /home/rikyravi/Scopone_RL/tools/profile_ppo.py --scalene --scalene-out html --iters 1
python /home/rikyravi/Scopone_RL/tools/profile_ppo.py --scalene --iters 1 --no-scalene-cpu-only --scalene-gpu-modes --scalene-out html
python /home/rikyravi/Scopone_RL/tools/profile_ppo.py --torch-profiler --iters 1 --horizon 2048

Analizza i profiling del mio codice e dimmi dove viene speso più tempo. Poi ottimizza il mio codice in modo da risolvere il problema partendo dalle cose che perdono più tempo. Non usare fallback per gli errori (piuttosto causano raise, ma meglio che fallback).

Attivazione server:
gcloud compute config-ssh

tmux new -s scopone 'source /home/rikyr/Scopone_RL/.venv/bin/activate && exec bash'
python3 main.py | tee -a run.log
Ctrl+B seguito da D
tmux attach -t scopone
Per verificare che il processo resti vivo dopo il distacco: ps -ef | grep main.py o tmux ls
tmux kill-session -t scopone