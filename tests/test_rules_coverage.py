import pytest
import numpy as np

from environment import ScoponeEnvMA
from actions import decode_action_ids, encode_action_from_ids_tensor
from rewards import compute_final_score_breakdown


# Helper per conversione tuple->ID
SUIT_TO_COL = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
def tid(card_tuple):
    r, s = card_tuple
    return (r - 1) * 4 + SUIT_TO_COL[s]


def test_direct_capture_blocks_sum_actions():
    env = ScoponeEnvMA()
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((6, 'denari'))]
    # Sul tavolo c'è una carta di pari-rank e una somma equivalente (1+5)
    six_coppe = tid((6, 'coppe'))
    one_spade = tid((1, 'spade'))
    five_bastoni = tid((5, 'bastoni'))
    env.game_state["table"] = [six_coppe, one_spade, five_bastoni]
    env._rebuild_id_caches()

    valids = env.get_valid_actions()
    pid_target = tid((6, 'denari'))
    sum_caps = {one_spade, five_bastoni}
    has_direct = False
    has_sum = False
    has_throw = False
    for v in valids:
        pid, caps = decode_action_ids(v)
        if pid != pid_target:
            continue
        if len(caps) == 1 and caps[0] == six_coppe:
            has_direct = True
        if set(caps) == sum_caps:
            has_sum = True
        if len(caps) == 0:
            has_throw = True
    assert has_direct, "Deve esistere almeno una presa diretta quando presente carta pari-rank"
    assert not has_sum, "Le somme non devono essere permesse quando esiste la presa diretta"
    assert not has_throw, "Non si può scartare quando esiste la presa diretta"


def test_step_raises_when_capturing_card_not_on_table():
    env = ScoponeEnvMA()
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((7, 'denari'))]
    env.game_state["table"] = [tid((3, 'bastoni'))]
    env._rebuild_id_caches()

    # Prova a catturare una carta non presente sul tavolo
    import torch
    bad_action = encode_action_from_ids_tensor(torch.tensor(tid((7, 'denari')), dtype=torch.long), torch.tensor([tid((2, 'coppe'))], dtype=torch.long))
    with pytest.raises(ValueError):
        env.step(bad_action)


def test_compute_final_score_breakdown_ties_and_scope_only():
    # Scegli carte che pareggiano primiera e non coinvolgono denari né settebello
    # Team 0: 7 coppe, 6 spade  → primiera 21 + 18, denari=0
    # Team 1: 7 bastoni, 6 coppe → primiera 21 + 18, denari=0
    game_state = {
        "captured_squads": {
            0: [tid((7, 'coppe')), tid((6, 'spade'))],
            1: [tid((7, 'bastoni')), tid((6, 'coppe'))]
        },
        # Una scopa del team1 (player 1 o 3)
        "history": [
            {"player": 1, "played_card": tid((7, 'coppe')), "capture_type": "scopa", "captured_cards": []}
        ]
    }
    bd = compute_final_score_breakdown(game_state, rules={})
    assert bd[0]["carte"] == 0 and bd[1]["carte"] == 0
    assert bd[0]["denari"] == 0 and bd[1]["denari"] == 0
    assert bd[0]["primiera"] == 0 and bd[1]["primiera"] == 0
    assert bd[0]["settebello"] == 0 and bd[1]["settebello"] == 0
    assert bd[0]["scope"] == 0 and bd[1]["scope"] == 1
    assert bd[0]["total"] + 1 == bd[1]["total"]


def test_redeal_when_three_kings_on_table_in_non_scientifico(monkeypatch):
    import random
    orig_shuffle = random.shuffle
    calls = {"n": 0}

    def fake_shuffle(lst):
        calls["n"] += 1
        # Prima chiamata: non importa (pre-shuffle)
        if calls["n"] == 1:
            lst[:] = list(range(39, -1, -1))
        elif calls["n"] == 2:
            # Forza 3 Re (IDs 36,37,38) tra le prime 4 carte
            rest = [i for i in range(40) if i not in [36, 37, 38, 5]]
            lst[:] = [36, 37, 38, 5] + rest
        else:
            # Successivo shuffle: meno di 3 Re tra le prime 4
            rest = [i for i in range(40) if i not in [36, 5, 6, 7]]
            lst[:] = [36, 5, 6, 7] + rest

    monkeypatch.setattr(random, 'shuffle', fake_shuffle)
    try:
        from state import initialize_game
        gs = initialize_game(rules={"variant": "scopone_non_scientifico"})
    finally:
        monkeypatch.setattr(random, 'shuffle', orig_shuffle)

    tbl = gs["table"]
    kings_on_table = sum(1 for cid in tbl if (cid // 4 + 1) == 10)
    assert kings_on_table < 3, f"Dovrebbe ridistribuire se 3 o 4 Re in apertura, trovato {kings_on_table}"
    # Verifica schema della variante
    for p in range(4):
        assert len(gs["hands"][p]) == 9
    assert len(gs["table"]) == 4


def test_ap_single_ace_capture_counts_scopa_when_only_one_card_on_table():
    env = ScoponeEnvMA(rules={"asso_piglia_tutto": True, "scopa_on_asso_piglia_tutto": False})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((1, 'denari'))]
    env.game_state["hands"][1] = [tid((2, 'coppe'))]
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = [tid((1, 'spade'))]
    env._rebuild_id_caches()

    # Trova l'azione che cattura l'unica carta (asso) sul tavolo con l'asso
    valids = env.get_valid_actions()
    target = None
    for v in valids:
        pid, caps = decode_action_ids(v)
        if pid == tid((1, 'denari')) and caps == [tid((1, 'spade'))]:
            target = v
            break
    assert target is not None, "Azione AP attesa non trovata"

    _, _, done, info = env.step(target)
    assert not done
    assert info["last_move"]["capture_type"] == "scopa", "La presa dell'unico asso sul tavolo deve contare scopa"


def test_forced_ace_capture_on_empty_table_counts_scopa():
    # Con scopa_on_asso_piglia_tutto=True conta scopa anche su tavolo vuoto
    env = ScoponeEnvMA(rules={"asso_piglia_tutto": True, "scopa_on_asso_piglia_tutto": True})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((1, 'denari'))]
    env.game_state["hands"][1] = [tid((2, 'coppe'))]
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = []
    env._rebuild_id_caches()

    act = encode_action_from_ids_tensor(torch.tensor(tid((1, 'denari')), dtype=torch.long), torch.tensor([], dtype=torch.long))
    _, _, done, info = env.step(act)
    assert not done
    assert info["last_move"]["capture_type"] == "scopa", "Su tavolo vuoto e AP non posabile, l'asso conta come scopa"


def test_table_empty_throw_actions_count_equals_hand_size():
    env = ScoponeEnvMA()
    env.reset()
    env.current_player = 0
    env.game_state["table"] = []
    env.game_state["hands"][0] = [tid((2, 'coppe')), tid((4, 'bastoni')), tid((6, 'denari'))]
    env._rebuild_id_caches()

    valids = env.get_valid_actions()
    throws = [(pid, tuple(caps)) for pid, caps in (decode_action_ids(v) for v in valids) if len(caps) == 0]
    assert len(throws) == len(env.game_state["hands"][0])



def test_cannot_capture_two_same_rank_cards():
    env = ScoponeEnvMA()
    env.reset()
    env.current_player = 0
    # Mano con un 7; sul tavolo due 7
    env.game_state["hands"][0] = [tid((7, 'denari'))]
    env.game_state["table"] = [tid((7, 'coppe')), tid((7, 'spade'))]
    env._rebuild_id_caches()

    # Tentare di catturare due carte pari-rank deve sollevare errore
    import torch
    bad = encode_action_from_ids_tensor(torch.tensor(tid((7, 'denari')), dtype=torch.long), torch.tensor([tid((7, 'coppe')), tid((7, 'spade'))], dtype=torch.long))
    with pytest.raises(ValueError):
        env.step(bad)


def test_scopa_shaped_reward_applies():
    env = ScoponeEnvMA(rules={"shape_scopa": True, "scopa_reward": 0.33})
    env.reset()
    env.current_player = 0
    # P0 fa scopa (3+4) con 7; restano carte a P1
    env.game_state["hands"][0] = [tid((7, 'denari'))]
    env.game_state["hands"][1] = [tid((2, 'coppe'))]
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = [tid((3, 'bastoni')), tid((4, 'denari'))]
    env.game_state["captured_squads"] = {0: [], 1: []}
    env.game_state["history"] = []
    env._rebuild_id_caches()

    act = encode_action_from_ids_tensor(torch.tensor(tid((7, 'denari')), dtype=torch.long), torch.tensor([tid((3, 'bastoni')), tid((4, 'denari'))], dtype=torch.long))
    _, r, done, info = env.step(act)
    assert done is False
    assert info["last_move"]["capture_type"] == "scopa"
    import numpy as _np
    assert _np.isclose(r, 0.33)


def test_scoring_awards_each_category():
    # Team0 vince carte, denari, settebello e primiera
    gs = {
        "captured_squads": {
            0: [
                tid((7, 'denari')), tid((7, 'coppe')), tid((7, 'spade')), tid((7, 'bastoni')),
                tid((6, 'denari')),  # extra denari per majority
                tid((3, 'spade'))    # carta extra per majority carte
            ],
            1: [
                tid((1, 'denari')),  # denari ma in minoranza
                tid((4, 'bastoni'))
            ]
        },
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={})
    assert bd[0]["carte"] == 1 and bd[1]["carte"] == 0
    assert bd[0]["denari"] == 1 and bd[1]["denari"] == 0
    assert bd[0]["settebello"] == 1 and bd[1]["settebello"] == 0
    assert bd[0]["primiera"] == 1 and bd[1]["primiera"] == 0
    assert bd[0]["scope"] == 0 and bd[1]["scope"] == 0
    assert bd[0]["total"] == 4 and bd[1]["total"] == 0


def test_ap_take_all_demotes_scopa_when_disabled():
    # scopa_on_asso_piglia_tutto=False -> presa totale con asso non conta scopa
    env = ScoponeEnvMA(rules={"asso_piglia_tutto": True, "scopa_on_asso_piglia_tutto": False})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((1, 'denari'))]
    env.game_state["hands"][1] = [tid((2, 'coppe'))]
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = [tid((2, 'spade')), tid((3, 'bastoni'))]
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((1, 'denari')), dtype=torch.long), torch.tensor([], dtype=torch.long))
    _, _, done, info = env.step(act)
    assert not done
    assert info["last_move"]["capture_type"] == "capture"


def test_force_ace_self_capture_on_empty_once_flag():
    # Se force_ace_self_capture_on_empty_once=True e tavolo vuoto, posa asso -> scopa e flag si consuma
    env = ScoponeEnvMA(rules={"asso_piglia_tutto": True, "scopa_on_asso_piglia_tutto": True, "force_ace_self_capture_on_empty_once": True})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((1, 'denari'))]
    env.game_state["hands"][1] = [tid((2, 'coppe'))]
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = []
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((1, 'denari')), dtype=torch.long), torch.tensor([], dtype=torch.long))
    _, _, done, info = env.step(act)
    assert not done
    assert info["last_move"]["capture_type"] == "scopa"
    # Flag dovrebbe essere consumato
    assert not env.rules.get("force_ace_self_capture_on_empty_once", False)


def test_scopa_on_last_capture_toggle_with_forced_empty():
    # Ultima presa che svuota il tavolo: dipende da scopa_on_last_capture
    env = ScoponeEnvMA(rules={"scopa_on_last_capture": False})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((3, 'denari'))]
    env.game_state["hands"][1] = []
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = [tid((1, 'spade')), tid((2, 'coppe'))]
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((3, 'denari')), dtype=torch.long), torch.tensor([tid((1, 'spade')), tid((2, 'coppe'))], dtype=torch.long))
    _, _, done, info = env.step(act)
    assert done
    assert env.game_state["history"][-1]["capture_type"] == "capture"

    env2 = ScoponeEnvMA(rules={"scopa_on_last_capture": True})
    env2.reset()
    env2.current_player = 0
    env2.game_state["hands"][0] = [tid((3, 'denari'))]
    env2.game_state["hands"][1] = []
    env2.game_state["hands"][2] = []
    env2.game_state["hands"][3] = []
    env2.game_state["table"] = [tid((1, 'spade')), tid((2, 'coppe'))]
    env2._rebuild_id_caches()
    act2 = encode_action_from_ids_tensor(torch.tensor(tid((3, 'denari')), dtype=torch.long), torch.tensor([tid((1, 'spade')), tid((2, 'coppe'))], dtype=torch.long))
    _, _, done2, info2 = env2.step(act2)
    assert done2
    assert env2.game_state["history"][-1]["capture_type"] == "scopa"


def test_scopa_limit_resets_on_team_change():
    # max_consecutive_scope=1 -> due scope di seguito dallo stesso team: seconda demota. Team cambia -> si azzera il contatore
    env = ScoponeEnvMA(rules={"max_consecutive_scope": 1})
    env.reset()
    # pre-carica una scopa del team0
    env.game_state["history"] = [{"player": 2, "played_card": tid((7, 'spade')), "capture_type": "scopa", "captured_cards": [tid((3, 'denari')), tid((4, 'coppe'))]}]
    # ora team1 fa scopa -> deve restare scopa
    env.current_player = 1
    env.game_state["hands"][1] = [tid((7, 'denari'))]
    env.game_state["hands"][0] = []
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = [tid((2, 'denari'))]  # evita ultima presa
    env.game_state["table"] = [tid((3, 'spade')), tid((4, 'bastoni'))]
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((7, 'denari')), dtype=torch.long), torch.tensor([tid((3, 'spade')), tid((4, 'bastoni'))], dtype=torch.long))
    _, _, done, info = env.step(act)
    assert env.game_state["history"][-1]["capture_type"] == "scopa"


def test_scopa_limit_ignores_opponent_moves():
    # Le mosse dell'avversario non interrompono la serie di scope della squadra
    env = ScoponeEnvMA(rules={"max_consecutive_scope": 1})
    env.reset()
    # Team0 fa scopa, poi una mossa dell'avversario (no_capture)
    env.game_state["history"] = [
        {"player": 0, "played_card": tid((7, 'denari')), "capture_type": "scopa", "captured_cards": [tid((3, 'denari')), tid((4, 'coppe'))]},
        {"player": 1, "played_card": tid((5, 'spade')), "capture_type": "no_capture", "captured_cards": []}
    ]
    env.current_player = 0
    env.game_state["hands"][0] = [tid((7, 'coppe'))]
    env.game_state["hands"][1] = []
    env.game_state["hands"][2] = [tid((2, 'denari'))]  # evita ultima presa
    env.game_state["hands"][3] = []
    env.game_state["table"] = [tid((3, 'spade')), tid((4, 'bastoni'))]
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((7, 'coppe')), dtype=torch.long), torch.tensor([tid((3, 'spade')), tid((4, 'bastoni'))], dtype=torch.long))
    _, _, done, info = env.step(act)
    # Dato max=1 e la precedente scopa del team0, anche se c'è stata una mossa avversaria in mezzo,
    # questa seconda scopa della stessa squadra deve essere demota a capture
    assert env.game_state["history"][-1]["capture_type"] == "capture"


def test_scopa_limit_resets_on_same_team_non_scopa():
    # Una giocata senza scopa della stessa squadra interrompe la serie
    env = ScoponeEnvMA(rules={"max_consecutive_scope": 1})
    env.reset()
    # Team0 scopa, poi team0 fa una giocata senza scopa
    env.game_state["history"] = [
        {"player": 0, "played_card": tid((7, 'denari')), "capture_type": "scopa", "captured_cards": [tid((3, 'denari')), tid((4, 'coppe'))]},
        {"player": 2, "played_card": tid((5, 'spade')), "capture_type": "no_capture", "captured_cards": []}
    ]
    env.current_player = 0
    env.game_state["hands"][0] = [tid((7, 'coppe'))]
    env.game_state["hands"][1] = []
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = [tid((2, 'denari'))]  # evita ultima presa
    env.game_state["table"] = [tid((3, 'spade')), tid((4, 'bastoni'))]
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((7, 'coppe')), dtype=torch.long), torch.tensor([tid((3, 'spade')), tid((4, 'bastoni'))], dtype=torch.long))
    _, _, done, info = env.step(act)
    # La serie è stata interrotta da una giocata della stessa squadra senza scopa,
    # quindi questa scopa deve rimanere scopa
    assert env.game_state["history"][-1]["capture_type"] == "scopa"


def test_raise_if_hand_ends_with_no_captures():
    # La mano non può terminare senza alcuna presa: deve sollevare ValueError
    env = ScoponeEnvMA(rules={"last_cards_to_dealer": True})
    env.reset()
    env.current_player = 0
    env.game_state["hands"] = {0: [tid((5, 'denari'))], 1: [], 2: [], 3: []}
    env.game_state["table"] = [tid((9, 'coppe'))]
    env.game_state["captured_squads"] = {0: [], 1: []}
    env.game_state["history"] = []
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((5, 'denari')), dtype=torch.long), torch.tensor([], dtype=torch.long))
    with pytest.raises(ValueError):
        env.step(act)


def test_ace_place_action_when_posabile_non_empty():
    # Se posabile=True e only_empty=False, su tavolo non vuoto deve esistere l'azione di posa asso []
    rules = {
        "asso_piglia_tutto": True,
        "asso_piglia_tutto_posabile": True,
        "asso_piglia_tutto_posabile_only_empty": False,
    }
    env = ScoponeEnvMA(rules=rules)
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((1, 'denari'))]
    env.game_state["table"] = [tid((4, 'coppe'))]
    env._rebuild_id_caches()
    valids = env.get_valid_actions()
    assert any(decode_action_ids(v)[0] == tid((1, 'denari')) and decode_action_ids(v)[1] == [] for v in valids)


def test_napola_length_counts_run_length_4():
    # A-2-3-4 di denari -> napola length=4
    gs = {
        "captured_squads": {
            0: [tid((1,'denari')), tid((2,'denari')), tid((3,'denari')), tid((4,'denari'))],
            1: []
        },
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={"napola": True, "napola_scoring": "length"})
    assert bd[0]["napola"] == 4 and bd[1]["napola"] == 0


def test_napola_requires_123():
    # Mancando il 2 di denari, napola deve essere 0
    gs = {
        "captured_squads": {
            0: [tid((1,'denari')), tid((3,'denari')), tid((4,'denari'))],
            1: []
        },
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={"napola": True, "napola_scoring": "length"})
    assert bd[0]["napola"] == 0 and bd[1]["napola"] == 0


def test_denari_tie_gives_no_points():
    # Stesso numero di denari -> 0 punti denari per entrambi
    gs = {
        "captured_squads": {
            0: [tid((1,'denari')), tid((3,'denari'))],
            1: [tid((5,'denari')), tid((7,'denari'))]
        },
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={})
    assert bd[0]["denari"] == 0 and bd[1]["denari"] == 0


def test_primiera_tie_gives_no_points():
    # Primiera pari -> 0 punti
    gs = {
        "captured_squads": {
            0: [tid((7,'coppe')), tid((6,'spade')), tid((5,'bastoni')), tid((4,'denari'))],
            1: [tid((7,'bastoni')), tid((6,'coppe')), tid((5,'spade')), tid((4,'denari'))]
        },
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={})
    assert bd[0]["primiera"] == 0 and bd[1]["primiera"] == 0


def test_max_consecutive_scopa_limit_two():
    # Limite 2: due scope della stessa squadra consentite; la terza viene demotata
    env = ScoponeEnvMA(rules={"max_consecutive_scope": 2})
    env.reset()
    # Due scope precedenti del team0 (players 0 e 2)
    env.game_state["history"] = [
        {"player": 0, "played_card": tid((7,'denari')), "capture_type": "scopa", "captured_cards": [tid((3,'denari')), tid((4,'coppe'))]},
        {"player": 2, "played_card": tid((6,'spade')), "capture_type": "scopa", "captured_cards": [tid((2,'coppe')), tid((4,'spade'))]},
    ]
    env.current_player = 0
    env.game_state["hands"][0] = [tid((7,'coppe'))]
    env.game_state["hands"][1] = [tid((2,'denari'))]  # per evitare fine mano
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = [tid((3,'spade')), tid((4,'bastoni'))]
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((7,'coppe')), dtype=torch.long), torch.tensor([tid((3,'spade')), tid((4,'bastoni'))], dtype=torch.long))
    _, _, done, info = env.step(act)
    assert env.game_state["history"][-1]["capture_type"] == "capture"


def test_cards_tie_gives_no_points():
    # Stesso numero di carte catturate -> 0 punti carte
    gs = {
        "captured_squads": {
            0: [tid((2,'denari'))] * 10,
            1: [tid((3,'coppe'))] * 10,
        },
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={})
    assert bd[0]["carte"] == 0 and bd[1]["carte"] == 0


def test_ap_direct_capture_blocks_sum_even_with_ap_enabled():
    # Con AP attivo, se esiste carta pari-rank, deve catturare quella (non combinazione),
    # a meno che non prenda tutto il tavolo (AP take-all)
    env = ScoponeEnvMA(rules={"asso_piglia_tutto": True})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((6,'denari'))]
    six_coppe = tid((6,'coppe'))
    env.game_state["table"] = [six_coppe, tid((1,'spade')), tid((5,'bastoni'))]
    env._rebuild_id_caches()
    valids = env.get_valid_actions()
    has_sum = False
    has_direct = False
    for v in valids:
        pid, caps = decode_action_ids(v)
        if pid != tid((6,'denari')):
            continue
        if len(caps) == 1 and caps[0] == six_coppe:
            has_direct = True
        if set(caps) == {tid((1,'spade')), tid((5,'bastoni'))}:
            has_sum = True
    assert has_direct and not has_sum


def test_re_bello_disabled_by_default():
    # Senza flag re_bello=True, non deve assegnare punto al 10 di denari
    gs = {
        "captured_squads": {
            0: [tid((10,'denari'))],
            1: []
        },
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={})
    assert bd[0]["re_bello"] == 0 and bd[1]["re_bello"] == 0


def test_ace_place_only_on_empty_when_only_empty_true():
    rules = {
        "asso_piglia_tutto": True,
        "asso_piglia_tutto_posabile": True,
        "asso_piglia_tutto_posabile_only_empty": True,
    }
    env = ScoponeEnvMA(rules=rules)
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((1,'coppe'))]
    # tavolo non vuoto: non deve esserci l'azione []
    env.game_state["table"] = [tid((4,'denari'))]
    env._rebuild_id_caches()
    valids = env.get_valid_actions()
    assert not any(decode_action_ids(v)[0] == tid((1,'coppe')) and decode_action_ids(v)[1] == [] for v in valids)
    # tavolo vuoto: l'azione [] ora è consentita
    env.game_state["table"] = []
    env._rebuild_id_caches()
    valids2 = env.get_valid_actions()
    assert any(decode_action_ids(v)[0] == tid((1,'coppe')) and decode_action_ids(v)[1] == [] for v in valids2)


def test_ap_take_all_allowed_even_if_same_rank_exists():
    # Se sul tavolo c'è un asso, con AP attivo l'asso in mano può prendere tutto il tavolo
    env = ScoponeEnvMA(rules={"asso_piglia_tutto": True})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((1,'denari'))]
    env.game_state["table"] = [tid((1,'spade')), tid((3,'coppe'))]
    env._rebuild_id_caches()
    valids = env.get_valid_actions()
    take_all = set(env.game_state["table"])
    assert any(decode_action_ids(v)[0] == tid((1,'denari')) and set(decode_action_ids(v)[1]) == take_all for v in valids)


def test_ap_forced_empty_last_capture_toggle():
    # Ultima mossa: tavolo vuoto, AP attivo
    # Regole desiderate:
    # - Se scopa_on_asso_piglia_tutto=False: mai scopa a tavolo vuoto (anche all'ultima)
    # - Se scopa_on_asso_piglia_tutto=True e scopa_on_last_capture=False: all'ultima presa fa scopa
    # Caso 1: scopa_on_asso_piglia_tutto=False -> capture
    env = ScoponeEnvMA(rules={"asso_piglia_tutto": True, "scopa_on_asso_piglia_tutto": False, "scopa_on_last_capture": True})
    env.reset()
    env.current_player = 0
    env.game_state["hands"] = {0: [tid((1,'denari'))], 1: [], 2: [], 3: []}
    env.game_state["table"] = []
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((1,'denari')), dtype=torch.long), torch.tensor([], dtype=torch.long))
    _, _, done, info = env.step(act)
    assert done
    assert env.game_state["history"][-1]["capture_type"] == "capture"
    
    # Caso 2: AP scopa ON ma scopa_on_last_capture OFF: ultima presa deve essere scopa
    env2 = ScoponeEnvMA(rules={"asso_piglia_tutto": True, "scopa_on_asso_piglia_tutto": True, "scopa_on_last_capture": False})
    env2.reset()
    env2.current_player = 0
    env2.game_state["hands"] = {0: [tid((1,'denari'))], 1: [], 2: [], 3: []}
    env2.game_state["table"] = []
    env2._rebuild_id_caches()
    act2 = encode_action_from_ids_tensor(torch.tensor(tid((1,'denari')), dtype=torch.long), torch.tensor([], dtype=torch.long))
    _, _, done2, info2 = env2.step(act2)
    assert done2
    assert env2.game_state["history"][-1]["capture_type"] == "scopa"


def test_shape_scopa_default_reward_is_0_1():
    # Senza scopa_reward impostato, shape_scopa usa default 0.1
    env = ScoponeEnvMA(rules={"shape_scopa": True})
    env.reset()
    env.current_player = 0
    env.game_state["hands"][0] = [tid((7,'denari'))]
    env.game_state["hands"][1] = [tid((2,'coppe'))]  # evitare fine mano
    env.game_state["hands"][2] = []
    env.game_state["hands"][3] = []
    env.game_state["table"] = [tid((3,'bastoni')), tid((4,'denari'))]
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((7,'denari')), dtype=torch.long), torch.tensor([tid((3,'bastoni')), tid((4,'denari'))], dtype=torch.long))
    _, r, done, info = env.step(act)
    assert not done
    import numpy as _np
    assert _np.isclose(r, 0.1)


def test_env_raises_no_capture_end():
    # La mano non può terminare senza alcuna presa: conferma via env.step
    import torch
    env = ScoponeEnvMA()
    env.reset()
    env.game_state["hands"] = {0: [tid((5,'denari'))], 1: [], 2: [], 3: []}
    env.game_state["table"] = [tid((9,'coppe'))]
    env.game_state["captured_squads"] = {0: [], 1: []}
    env.game_state["history"] = []
    env._rebuild_id_caches()
    act = encode_action_from_ids_tensor(torch.tensor(tid((5,'denari')), dtype=torch.long), torch.tensor([], dtype=torch.long))
    import pytest as _pytest
    with _pytest.raises(ValueError):
        env.step(act)


def test_settebello_on_table_does_not_score():
    # Settebello sul tavolo (non catturato) non assegna punti
    gs = {
        "captured_squads": {0: [], 1: []},
        "table": [tid((7,'denari'))],
        "history": []
    }
    bd = compute_final_score_breakdown(gs, rules={})
    assert bd[0]["settebello"] == 0 and bd[1]["settebello"] == 0

