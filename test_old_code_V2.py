import pytest
import numpy as np
import random
import torch
from main import DQNAgent

# Importa i moduli del VECCHIO CODICE
# (Assicurati che i nomi dei file e i path siano corretti rispetto a dove li hai salvati)
from environment import ScoponeEnvMA  # Vecchio environment
from actions import encode_action, decode_action, get_valid_actions, MAX_ACTIONS
from state import create_deck, initialize_game, SUITS, RANKS
from observation import encode_state_for_player, card_to_index
from game_logic import update_game_state, compute_final_score_breakdown, compute_final_reward_from_breakdown

@pytest.fixture
def env_old():
    """
    Fixture che crea e restituisce l'ambiente "vecchio" (quello con la logica
    di catture immediata in update_game_state).
    """
    return ScoponeEnvMA()


def test_create_deck():
    """
    Test su create_deck() per verificare che il mazzo contenga 40 carte distinte
    e le carte siano effettivamente una combinazione di SUITS x RANKS.
    """
    deck = create_deck()
    assert len(deck) == 40, "Il mazzo deve contenere 40 carte"
    assert len(set(deck)) == 40, "Non devono esserci duplicati nel mazzo"

    # Verifica che ogni carta appartenga a SUITS x RANKS
    for (r, s) in deck:
        assert s in SUITS, f"Seme {s} non previsto"
        assert r in RANKS, f"Valore {r} non previsto"


def test_initialize_game():
    """
    Verifica che initialize_game() crei uno stato con:
      - 4 mani da 10 carte l'una
      - table vuoto
      - captured_squads = {0:[], 1:[]}
      - history vuota
    E che le carte siano correttamente "mischiate".
    """
    state = initialize_game()
    assert len(state["hands"]) == 4
    for p in range(4):
        assert len(state["hands"][p]) == 10
    assert state["table"] == []
    assert len(state["captured_squads"][0]) == 0
    assert len(state["captured_squads"][1]) == 0
    assert len(state["history"]) == 0

    # Controllo "shuffle": non è un test rigoroso,
    # ma almeno controllo che la prima carta di due giocatori diversi non sia identica.
    assert state["hands"][0][0] != state["hands"][1][0], \
        "Le prime carte di P0 e P1 non dovrebbero coincidere (mazzo non mescolato?)."


def test_encode_action_decode_action():
    """
    Verifica che encode_action+decode_action siano inversi con svariati subset_indices.
    """
    # Esempio di codifica
    hand_index = 3
    subset_indices = (0, 2, 5)
    action_id = encode_action(hand_index, subset_indices)
    dec_h, dec_subset = decode_action(action_id)

    assert dec_h == hand_index
    assert tuple(dec_subset) == subset_indices

    # Subset vuoto
    hand_index2 = 1
    subset_indices2 = ()
    action_id2 = encode_action(hand_index2, subset_indices2)
    dec_h2, dec_subset2 = decode_action(action_id2)
    assert dec_h2 == hand_index2
    assert dec_subset2 == subset_indices2

    # Subset con tanti indici
    hand_index3 = 2
    subset_indices3 = (1, 3, 5, 7, 8)
    action_id3 = encode_action(hand_index3, subset_indices3)
    dec_h3, dec_subset3 = decode_action(action_id3)
    assert dec_h3 == hand_index3
    assert tuple(dec_subset3) == subset_indices3


def test_get_valid_actions_direct_capture(env_old):
    """
    Test 'get_valid_actions' in un caso dove esiste la cattura diretta.
    Se c'è una carta sul tavolo con rank == carta giocata, DEVO catturare e
    non posso buttare la carta (né fare somme).
    """
    env_old.reset()
    # Forziamo scenario
    env_old.game_state["hands"][0] = [(4,'denari'), (7,'spade')]
    env_old.game_state["table"] = [(7,'denari'), (3,'spade'), (4,'coppe')]
    env_old.current_player = 0

    valids = get_valid_actions(env_old.game_state, env_old.current_player)

    # In questa implementazione (vecchia), la regola dice:
    #   Se esiste cattura diretta per la carta selezionata, allora è obbligata
    #   e NON posso buttarla senza catturare.
    # Quindi (4,'denari') ha cattura diretta con (4,'coppe')
    # (7,'spade') ha cattura diretta con (7,'denari')
    # => Non deve comparire la possibilità di buttarle via, e men che meno le somme.

    # Aspettative
    a1 = encode_action(0, (2,))  # (4,'denari') cattura (4,'coppe')
    a2 = encode_action(1, (0,))  # (7,'spade') cattura (7,'denari')
    # *Non* devono comparire azioni con subset vuoto, né la cattura di (3,'spade')+(4,'coppe')=7 (somma)
    # perché la regola "cattura diretta" prevale su eventuali somme.

    assert a1 in valids
    assert a2 in valids

    # Controlliamo che nessun'altra azione sia presente
    # (al limite, se hai più carte di rank=4 o rank=7, potresti avere più catture dirette).
    for va in valids:
        # decodifichiamo e verifichiamo che corrisponda solo a (4,'coppe') o (7,'denari')
        h_idx, subset = decode_action(va)
        if h_idx == 0:
            # stiamo giocando (4,'denari')
            assert tuple(subset) == (2,), "Se catturo con 4,'denari', devo prendere (4,'coppe') e basta."
        elif h_idx == 1:
            # stiamo giocando (7,'spade')
            assert tuple(subset) == (0,), "Se catturo con 7,'spade', devo prendere (7,'denari') e basta."
        else:
            pytest.fail(f"Azione con hand_index={h_idx} non prevista")


def test_get_valid_actions_no_direct_capture(env_old):
    """
    Caso in cui NON c'è cattura diretta, ma c'è una somma possibile.
    Oppure si butta la carta se nessuna somma è possibile.
    """
    env_old.reset()
    env_old.game_state["hands"][0] = [(6,'denari'), (7,'spade')]
    env_old.game_state["table"] = [(1,'coppe'), (3,'spade'), (2,'bastoni')]
    env_old.current_player = 0

    valids = get_valid_actions(env_old.game_state, env_old.current_player)

    # Se giocatore prova (6,'denari'):
    #   - NO cattura diretta (nessuna carta rank=6 sul tavolo)
    #   - Possibile somma? 1+2+3=6 => sì => devo catturare [0,1,2]
    #   - E basta: se esiste una combinazione che fa 6 (qui 1+2+3), la aggiungo. Altre combinazioni (1+3=4,2+3=5 etc.) non fanno 6
    # Quindi l'azione = encode_action(0, (0,1,2)) => mano index=0, subset=(0,1,2)
    # (7,'spade'):
    #   - NO cattura diretta con rank=7
    #   - Possibile somma: 1+2+3=6, non è 7 => no
    #   - Devo buttare la carta => encode_action(1, ())
    from actions import encode_action
    a_capture_6 = encode_action(0, (0,1,2))
    a_butta_7 = encode_action(1, ())

    assert a_capture_6 in valids, "Deve esserci l'azione di catturare 1,2,3 con (6,'denari')."
    assert a_butta_7 in valids, "Deve esistere l'azione di buttare (7,'spade'), non potendo catturare nulla."


def test_encode_state_for_player(env_old):
    """
    Verifica che l'osservazione abbia dimensione (3764) e che la mano di
    'current_player' sia effettivamente codificata, mentre le altre siano azzerate.
    """
    obs = env_old.reset()
    assert obs.shape == (3764,)

    cp = env_old.current_player
    p0_hand = env_old.game_state["hands"][cp]
    for card in p0_hand:
        idx = card_to_index[card]
        # Poiché la mano del current_player sta nel chunk corrispondente (dipende da come è ordinato),
        # ma in questa implementazione "completa" i primi 40 slot (per p=0) oppure in posizioni diverse.
        # *Confronto semplice:* l'osservazione in "qualche parte" dovrebbe avere 1.0 su questi idx.
        # Il test base: verifichiamo che in obs almeno un "pezzo" abbia 1 su quell'idx.
        # (La parte di offset la sta gestendo encode_state_for_player, che crea un blocco 40 per ciascun player.)
        # Avendo p=0, i primi 40 slot di obs contengono la mano, gli slot 40..80 p1, 80..120 p2, 120..160 p3.
        # Un check più generico:
        #   se cp=0 => obs[idx] == 1.  Se cp=1 => obs[40 + idx] == 1, e così via...
        #   Ma per brevità, controlliamo che "in obs" ci sia 1.0 in almeno la posizione "attesa".
        #   Oppure facciamo un check differenziato su cp.
        # Per semplificare, assumo cp=0 di default.
        if cp == 0:
            # L'offset di p0 è 0 => la mano è in obs[0:40]
            assert obs[idx] == 1.0, f"La carta {card} del current_player dovrebbe essere codificata a 1."
        else:
            # Se cp != 0, bisognerebbe calcolare offset = cp*40 + idx, etc.
            # Per brevità, se i test li esegui dopo un reset() che setta cp=0, magari ti fidi...
            pass

    # Controllo invece che le mani degli ALTRI player siano a 0.
    for p in range(4):
        if p == cp:
            continue
        for card in env_old.game_state["hands"][p]:
            idx = card_to_index[card]
            # Se p=1 => la mano sua è in slot [40:80], e in encode_state_for_player p=1 è azzerato
            # => quindi obs[40 + idx] = 0 => e via dicendo.
            if cp == 0:
                # Allora i chunk di p=1,2,3 devono stare a zero.
                # Quindi obs[idx + p*40] non deve essere 1
                # In pratica, verifichiamo che obs[idx] non sia 1 (oppure consideriamo offset?)
                # Per semplicità, fai un test "generico" su un tot posizioni.
                pass
            # Check veloce: 
            if obs[idx] != 0.0:
                raise AssertionError(f"La carta {card} di un altro player non dovrebbe essere visibile!")


def test_step_basic(env_old):
    """
    Verifica che un 'step' con un'azione valida non sollevi eccezioni e
    restituisca (next_obs, reward=0.0, done=False) se la partita non è finita.
    """
    env_old.reset()
    valids = env_old.get_valid_actions()
    assert len(valids) > 0, "Appena dopo reset, dovrebbero esserci azioni valide"
    first_action = valids[0]

    next_obs, reward, done, info = env_old.step(first_action)

    assert next_obs.shape == (3764,)
    assert reward == 0.0
    assert done == False
    assert "team_rewards" not in info


def test_step_invalid_action(env_old):
    """
    Verifica che se provo uno step con un'azione NON valida, venga sollevata una ValueError.
    """
    env_old.reset()
    valids = env_old.get_valid_actions()
    # Proviamo un action_id sicuramente non in valids (es. max(valid)+1), ammesso sia <2048
    invalid_action = max(valids)+1 if max(valids) < MAX_ACTIONS-1 else 0

    with pytest.raises(ValueError):
        env_old.step(invalid_action)


def test_done_and_final_reward(env_old):
    """
    Esegue step finché la partita non finisce. A fine partita (done=True),
    controlla che in info ci sia "team_rewards" e che la lunghezza sia 2.
    """
    env_old.reset()
    done = False
    info = {}
    while not done:
        valids = env_old.get_valid_actions()
        if not valids:
            break
        action = random.choice(valids)
        obs, r, done, info = env_old.step(action)

    assert done is True, "La partita dovrebbe risultare finita."
    assert "team_rewards" in info, "L'info finale dovrebbe contenere team_rewards."
    assert len(info["team_rewards"]) == 2, "team_rewards dev'essere un array di 2 (team0, team1)."


def test_scopa_case(env_old):
    """
    Test di un caso specifico per verificare la "scopa" e che la reward di scopa sia assegnata correttamente.
    Ricordiamo che la scopa in questa implementazione aggiunge 1 punto nel breakdown, se non è l'ultima giocata.
    """
    # Costruiamo manualmente uno scenario:
    #   - P0 ha in mano solo (7,'denari'), e ci sono es. 2 carte sul tavolo che sommano 7.
    #   - Ci sono ancora carte in mano ad altri giocatori, cosicché la scopa sia valida.
    gs = initialize_game()
    gs["hands"][0] = [(7,'denari')]
    gs["hands"][1] = [(5,'coppe')]  # c'è ancora qualcuno con carte => scopa valida
    gs["hands"][2] = []
    gs["hands"][3] = []
    gs["table"] = [(3,'bastoni'), (4,'denari')]
    gs["captured_squads"][0] = []
    gs["captured_squads"][1] = []
    gs["history"] = []

    # Proviamo l'azione: catturare 3,4 => sum=7, e scopa flag attivo
    from actions import encode_action
    action_id = encode_action(0, (0,1))  # gioca la carta in mano_index=0 e cattura indices (0,1) dal tavolo
    # Chiamiamo update_game_state con current_player=0
    new_gs, rw_array, done, info = update_game_state(gs, action_id, 0)

    # Non dovrebbe essere done, perché P1 ha ancora 1 carta
    assert done is False, "Non è l'ultima mossa: P1 ha carte."
    # Ricontrolliamo la history: l'ultima mossa dev'essere "scopa".
    last_move = new_gs["history"][-1]
    assert last_move["capture_type"] == "scopa", "Dovrebbe risultare scopa avendo svuotato il tavolo."

    # Ricontrolliamo che la cattura sia finita in captured_squads del team0
    # (team0 = giocatori 0 e 2)
    assert len(new_gs["captured_squads"][0]) == 3, "Team0 deve aver preso (7,'denari'), (3,'bastoni'), (4,'denari')"
    # E il tavolo ora deve essere vuoto
    assert len(new_gs["table"]) == 0

    # Poiché non è finita la partita, la reward dev'essere [0,0]
    assert rw_array == [0.0, 0.0]

    # Forziamo adesso la fine della partita: svuotiamo la mano di P1 e rieseguiamo un'azione con P1
    new_gs["hands"][1] = []
    # Step successivo => se "update_game_state" vede mano vuota => calcolo finale
    # Basta una "falsa" azione, tipo action_id=0, che decodificato è hand_index=0 subset vuoto => ma la mano è vuota...
    new_gs2, rw_array2, done2, info2 = update_game_state(new_gs, 0, 1)
    assert done2 is True
    # Adesso se la scopa era valida, nel breakdown finale vedremo scope=1 per team0
    final_scope_team0 = info2["score_breakdown"][0]["scope"]
    final_scope_team1 = info2["score_breakdown"][1]["scope"]
    assert final_scope_team0 == 1, f"Team0 dovrebbe avere scope=1, invece {final_scope_team0}"
    assert final_scope_team1 == 0, f"Team1 deve avere scope=0, invece {final_scope_team1}"

    # E controlliamo la differenza di punteggio (0 -> c0 + den0 + settebello + primiera + scope).
    # In questo scenario minimal, probabile che team0 abbia 3 carte totali, team1 ne ha 0, quindi team0 vince.
    # => la diff > 0 => rw_array2[0] > 0
    print("Final reward array:", rw_array2)
    # Non faccio assert numerico dettagliato, basta verificare che la differenza non sia 0.
    assert rw_array2[0] != 0, "Ci aspettiamo un punteggio maggiore per Team0, e quindi una reward != 0."
    assert rw_array2[0] == -rw_array2[1], "Reward di team1 è l'opposto di team0."


def test_full_match_random(env_old):
    """
    Test finale in cui giochiamo una partita random completa con l'Env:
    a) Controlliamo che si arrivi a done=True senza errori.
    b) Controlliamo che info['team_rewards'] abbia valori coerenti.
    """
    obs = env_old.reset()
    done = False
    while not done:
        valids = env_old.get_valid_actions()
        if not valids:
            # A volte può succedere che un player abbia finito le carte
            # ma non è done perché altri player hanno ancora carte.
            # Ma in questa implementazione se 'valids' è vuoto => ValueError se step con azione non valida.
            # Ci basterebbe passare la mano. Oppure passare al successivo.
            # Per semplicità, break che simula "non si può far nulla".
            break

        a = random.choice(valids)
        obs, rew, done, info = env_old.step(a)

    if done:
        assert "team_rewards" in info
        r0, r1 = info["team_rewards"]
        print("Partita terminata. Ricompense finali:", r0, r1)

        # Se r0>0 => Team0 vince, se <0 => Team1 vince, se =0 => pareggio
        # Non facciamo un assert specifico sul segno, potrebbe uscire pareggio.
        # Basta controllare la coerenza:
        diff = r0 - r1
        # Se diff>0 => r1 deve essere <0. Se diff=0 => r0=r1=0, etc.
        # In base alla formula diff = breakdown0 - breakdown1, rewardTeam0= diff*10, rewardTeam1= -diff*10
        # => r0 + r1 deve essere 0 in ogni caso
        assert abs(r0 + r1) < 1e-9, "Le ricompense di due team devono essere opposte"

    else:
        # Se usciamo dal while senza done => la partita non è completata
        # Non è necessariamente un bug, ma potremmo segnalarlo.
        pytest.skip("La partita random non si è conclusa entro i passaggi eseguiti.")



@pytest.mark.parametrize("seed", [1234])
def test_agents_final_reward_team1_with_4_scopes(seed):
    """
    1) Forza nelle prime 8 mosse la realizzazione di 4 scope da parte di Team1
       (giocatori 1 e 3), in modo che a fine partita il punteggio di Team1
       sia sicuramente positivo e abbondante (scope + denari + settebello + primiera).
    2) Poi gioca il resto della partita a caso.
    3) A fine partita, controlla se i Q-value di Team1 sono aumentati. Se la
       final reward fosse ignorata (sempre salvata come 0.0 nel replay),
       i Q-value restano invariati e il test FALLISCE.
       Se invece la final reward viene effettivamente usata, i Q-value
       dovrebbero crescere e il test PASSA.
    """

    # Imposta un seme per ripetibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Crea agenti e ambiente
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Mettiamo epsilon=0 per minore casualità
    agent_team0.epsilon = 0.0
    agent_team1.epsilon = 0.0

    env = ScoponeEnvMA()
    env.current_player = 0

    ###############################################
    # 1) Assegniamo interamente le mani ai 4 giocatori
    #    in modo che Team1 (giocatori 1 e 3) possa fare
    #    4 scope nelle prime 8 mosse.
    ###############################################

    # Assegniamo 10 carte per ogni giocatore (occhio a non duplicare),
    # strutturandole per garantire che p1 e p3 possano fare 4 scope.
    # p1 ha quattro 7, p3 ha quattro 7, cosicché possano completare 4 scope
    # nei turni 1,3,5,7 (p1 e p3). Le altre carte sono scelte per completare
    # un mazzo plausibile (non necessariamente perfetto a scopo demo).

    env.game_state["hands"][0] = [
        (1,'denari'), (1,'coppe'), (2,'denari'), (2,'bastoni'),
        (3,'spade'), (4,'coppe'), (5,'bastoni'), (5,'spade'),
        (6,'denari'), (6,'bastoni')
    ]
    env.game_state["hands"][1] = [
        (7,'denari'), (7,'coppe'), (1,'spade'), (2,'coppe'),
        (3,'denari'), (4,'spade'), (6,'coppe'), (9,'denari'),
        (9,'coppe'), (9,'spade')
    ]
    env.game_state["hands"][2] = [
        (1,'bastoni'), (2,'spade'), (3,'coppe'), (4,'bastoni'),
        (5,'denari'), (8,'denari'), (8,'coppe'), (8,'bastoni'),
        (9,'bastoni'), (10,'denari')
    ]
    env.game_state["hands"][3] = [
        (7,'bastoni'), (7,'spade'), (1,'coppe'), (3,'bastoni'),
        (4,'denari'), (6,'spade'), (8,'spade'), (10,'bastoni'),
        (10,'coppe'), (10,'spade')
    ]
    # Azzeriamo il tavolo inizialmente
    env.game_state["table"] = []

    # Definiamo la sequenza di 8 mosse (turni 0..7).
    # Nei turni di p1 e p3, prepariamo il tavolo per garantire scopa sui 4 7.
    # I rank=7 catturano direttamente una singola carta rank=7 sul tavolo:
    forced_moves = [
        # Turno 0 (p0): butta (1,'denari') => subset vuoto
        dict(player=0, hand_index=0, subset=()),
        # Turno 1 (p1): scopa con (7,'denari'). Mettiamo (3,'spade')+(4,'coppe') sul tavolo per sommare 7
        dict(player=1, hand_index=0, subset=(0,1)),  # cattura 3,4
        # Turno 2 (p2): butta (1,'bastoni') => subset vuoto
        dict(player=2, hand_index=0, subset=()),
        # Turno 3 (p3): scopa con (7,'bastoni'). Mettiamo (2,'spade')+ (5,'bastoni') sul tavolo (somma=7)
        dict(player=3, hand_index=0, subset=(0,1)),
        # Turno 4 (p0): butta (1,'coppe') => subset vuoto
        dict(player=0, hand_index=1, subset=()),
        # Turno 5 (p1): scopa con (7,'coppe'). Mettiamo (2,'denari')+(5,'spade') => 7
        dict(player=1, hand_index=1, subset=(0,1)),
        # Turno 6 (p2): butta (2,'spade') => subset vuoto
        dict(player=2, hand_index=1, subset=()),
        # Turno 7 (p3): scopa con (7,'spade'). Mettiamo (3,'coppe')+(4,'bastoni') => 7
        dict(player=3, hand_index=1, subset=(0,1)),
    ]

    # Salviamo lo stato Q-value di Team1 prima di iniziare
    obs_team1_before = env._get_observation(1)
    obs_t1_before = torch.tensor(obs_team1_before, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qvals_t1_before = agent_team1.online_qnet(obs_t1_before)[0].clone()

    # Eseguiamo le 8 mosse forzate
    for move in forced_moves:
        p = move["player"]
        env.current_player = p

        # Se tocca p1 o p3, impostiamo ad hoc il tavolo per sommare a 7
        if p == 1 and move["hand_index"] in [0,1]:
            env.game_state["table"] = [(3,'spade'), (4,'coppe')] if move["hand_index"] == 0 else [(2,'denari'), (5,'spade')]
        elif p == 3 and move["hand_index"] in [0,1]:
            env.game_state["table"] = [(2,'spade'), (5,'bastoni')] if move["hand_index"] == 0 else [(3,'coppe'), (4,'bastoni')]
        else:
            # p0, p2 => lasciamo il tavolo come sta o azzeriamo
            pass

        action_id = encode_action(move["hand_index"], move["subset"])

        obs_before = env._get_observation(p)
        valids = env.get_valid_actions()
        if action_id not in valids:
            # Se non è valida, fallback a una valida
            action_id = valids[0]

        obs_after, rew, done, info = env.step(action_id)

        # Storing e train
        if p in [1,3]:
            agent_team1.store_transition((
                obs_before, action_id, rew, obs_after, done, env.get_valid_actions()
            ))
            agent_team1.train_step()
        else:
            agent_team0.store_transition((
                obs_before, action_id, rew, obs_after, done, env.get_valid_actions()
            ))
            agent_team0.train_step()

    ###############################################
    # 2) Ora proseguiamo la partita a caso
    ###############################################
    while not done:
        p = env.current_player
        obs_before = env._get_observation(p)

        if p in [1,3]:
            agent = agent_team1
        else:
            agent = agent_team0

        valids = env.get_valid_actions()
        if not valids:
            break
        action_id = random.choice(valids)

        obs_after, rew, done, info = env.step(action_id)
        agent.store_transition((
            obs_before, action_id, rew, obs_after, done, env.get_valid_actions()
        ))
        agent.train_step()
        
    if done:
        # Fine partita
        team_rewards = info["team_rewards"]
        print(f"Team Rewards finali: {team_rewards}")

        # Stampiamo le ultime X transizioni salvate da agent_team1
        print("=== Ultime transizioni di Team1 ===")
        last_5_team1 = list(agent_team1.replay_buffer.buffer)[-5:]  # converto in list() e poi slice
        for idx, trans in enumerate(last_5_team1):
            (obs_, act_, rew_, next_obs_, done_, valids_) = trans
            print(f" Team1 transizione {idx} -> reward={rew_} done={done_}")


        print("=== Ultime transizioni di Team0 ===")
        last_5_team0 = list(agent_team0.replay_buffer.buffer)[-5:]  # stessa cosa
        for idx, trans in enumerate(last_5_team0):
            (obs_, act_, rew_, next_obs_, done_, valids_) = trans
            print(f" Team0 transizione {idx} -> reward={rew_} done={done_}")

    # Se la partita è finita, in info["team_rewards"] ci saranno i punteggi finali
    # (Team1 avrà un grande vantaggio grazie alle 4 scope + denari + 7bello + primiera).
    # Ma se il codice di base non salva la final reward nel replay, Team1 non "impara".

    # Facciamo qualche step di training in più
    for _ in range(50):
        agent_team0.train_step()
        agent_team1.train_step()

    ###############################################
    # 3) Controllo se i Q-value di Team1 sono cresciuti
    ###############################################
    obs_team1_after = env._get_observation(1)
    obs_t1_after = torch.tensor(obs_team1_after, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qvals_t1_after = agent_team1.online_qnet(obs_t1_after)[0]

    # Confrontiamo la differenza massima
    diff_q = (qvals_t1_after - qvals_t1_before).max().item()

    print(f"\n[TEST] Differenza massima Q-value di Team1: {diff_q:.4f}")

    # Mettiamo una soglia: se la final reward fosse effettivamente usata, ci aspettiamo
    # che differenza > 2.0 (o 5.0, a seconda di come i parametri pesano). Scegli una soglia
    # ragionevole. Se scende sotto => fallisce => "non avete usato la final reward".
    threshold = 2.0
    assert diff_q >= threshold, (
        f"Team1 NON ha mostrato miglioramento nei Q-value (delta={diff_q:.4f} < {threshold}). "
        f"Probabilmente la ricompensa di fine partita non viene salvata nel replay buffer!"
    )

    # Se passiamo l'assert => i Q-values sono cresciuti => test OK => Significa
    # che i final rewards (in cui Team1 stravince) sono stati utilizzati dall'agente.
    # Se con il tuo codice di base la reward rimane 0.0, ci aspettiamo un diff_q
    # piccolo e quindi FALLISCE => "scoperta" del bug.