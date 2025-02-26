import pytest
import numpy as np
import random
import torch

# Import dal tuo codice
from main import DQNAgent
from environment import ScoponeEnvMA  # Nuovo environment (senza done)
from actions import encode_action, decode_action, get_valid_actions, MAX_ACTIONS
from state import create_deck, initialize_game, SUITS, RANKS
from observation import encode_state_for_player, card_to_index
from game_logic import update_game_state
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

@pytest.fixture
def env_simplified():
    """
    Crea e restituisce l'ambiente semplificato.
    """
    return ScoponeEnvMA()

def test_create_deck():
    deck = create_deck()
    assert len(deck) == 40
    assert len(set(deck)) == 40
    for (r, s) in deck:
        assert s in SUITS, f"Seme {s} non previsto"
        assert r in RANKS, f"Valore {r} non previsto"

def test_initialize_game():
    state = initialize_game()
    assert len(state["hands"]) == 4
    for p in range(4):
        assert len(state["hands"][p]) == 10
    assert state["table"] == []
    assert len(state["captured_squads"][0]) == 0
    assert len(state["captured_squads"][1]) == 0
    assert len(state["history"]) == 0
    # Check "shuffle"
    assert state["hands"][0][0] != state["hands"][1][0]

def test_encode_action_decode_action():
    # Esempio con subset non vuoto
    hand_index = 3
    subset_indices = (0, 2, 5)
    action_id = encode_action(hand_index, subset_indices)
    dec_h, dec_subset = decode_action(action_id)
    assert dec_h == hand_index
    assert tuple(dec_subset) == subset_indices

    # Esempio con subset vuoto
    hand_index2 = 1
    subset_indices2 = ()
    action_id2 = encode_action(hand_index2, subset_indices2)
    dec_h2, dec_subset2 = decode_action(action_id2)
    assert dec_h2 == hand_index2
    assert dec_subset2 == subset_indices2

def test_get_valid_actions_direct_capture(env_simplified):
    env_simplified.reset()
    env_simplified.game_state["hands"][0] = [(4,'denari'), (7,'spade')]
    env_simplified.game_state["table"] = [(7,'denari'), (3,'spade'), (4,'coppe')]
    env_simplified.current_player = 0

    valids = get_valid_actions(env_simplified.game_state, 0)
    a1 = encode_action(0, (2,))  # (4,'denari') deve catturare (4,'coppe')
    a2 = encode_action(1, (0,))  # (7,'spade') deve catturare (7,'denari')

    assert a1 in valids
    assert a2 in valids

def test_get_valid_actions_no_direct_capture(env_simplified):
    env_simplified.reset()
    env_simplified.game_state["hands"][0] = [(6,'denari'), (7,'spade')]
    env_simplified.game_state["table"] = [(1,'coppe'), (3,'spade'), (2,'bastoni')]
    env_simplified.current_player = 0

    valids = get_valid_actions(env_simplified.game_state, 0)
    a_capture_6 = encode_action(0, (0,1,2))  # Deve catturare 1,2,3
    a_butta_7 = encode_action(1, ())         # Deve buttare (7,'spade')

    assert a_capture_6 in valids
    assert a_butta_7 in valids

def test_encode_state_for_player(env_simplified):
    obs = env_simplified.reset()
    assert obs.shape == (3764,)

    cp = env_simplified.current_player
    hand_cp = env_simplified.game_state["hands"][cp]
    for card in hand_cp:
        idx = card_to_index[card]
        if cp == 0:
            assert obs[idx] == 1.0
    # (Il test si limita a controllare il current player)

def test_step_basic(env_simplified):
    env_simplified.reset()
    valids = env_simplified.get_valid_actions()
    first_action = valids[0]
    next_obs, reward, info = env_simplified.step(first_action)
    assert next_obs.shape == (3764,)
    assert reward == 0.0
    # Non viene gestito done

def test_step_invalid_action(env_simplified):
    env_simplified.reset()
    valids = env_simplified.get_valid_actions()
    invalid = max(valids) + 1 if max(valids) < (MAX_ACTIONS - 1) else 0
    with pytest.raises(ValueError):
        env_simplified.step(invalid)

def test_scopa_case():
    """
    Testa la logica in update_game_state per la scopa e il calcolo finale.
    Poiché update_game_state ora restituisce sempre reward [0.0, 0.0],
    verifichiamo solamente che la mossa venga registrata correttamente.
    """
    gs = initialize_game()
    gs["hands"][0] = [(7,'denari')]
    gs["hands"][1] = [(5,'coppe')]
    gs["hands"][2] = []
    gs["hands"][3] = []
    gs["table"] = [(3,'bastoni'), (4,'denari')]
    gs["captured_squads"] = {0:[], 1:[]}
    gs["history"] = []

    from actions import encode_action
    action_id = encode_action(0, (0,1))  # Deve catturare 3,4
    new_gs, rw_array, info = update_game_state(gs, action_id, 0)
    # In questa funzione, reward è sempre [0.0,0.0]
    assert rw_array == [0.0, 0.0]
    last_move = new_gs["history"][-1]
    assert last_move["capture_type"] == "scopa"
    assert len(new_gs["captured_squads"][0]) == 3
    assert len(new_gs["table"]) == 0
    # Per simulare il calcolo finale, possiamo usare compute_final_reward_from_breakdown
    final_reward = compute_final_reward_from_breakdown(compute_final_score_breakdown(new_gs))
    assert final_reward[0] != 0
    assert final_reward[0] == -final_reward[1]

def test_40_mosse_e_calcolo_finale():
    """
    Simula la nuova logica: esegue 40 mosse e poi calcola manualmente il punteggio finale.
    """
    env = ScoponeEnvMA()
    for i in range(40):
        cp = env.current_player
        valids = env.get_valid_actions()
        if not valids:
            break
        action = random.choice(valids)
        obs, rew, info = env.step(action)
        # rew è sempre 0.0

    breakdown = compute_final_score_breakdown(env.game_state)
    final_reward = compute_final_reward_from_breakdown(breakdown)
    print("Final breakdown:", breakdown)
    print("Final reward:", final_reward)
    r0, r1 = final_reward[0], final_reward[1]
    assert abs(r0 + r1) < 1e-9, "La somma delle reward deve essere 0 (opposte)"

@pytest.mark.parametrize("seed", [1234])
def test_agents_final_reward_team1_with_4_scopes(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)
    agent_team0.epsilon = 0.0
    agent_team1.epsilon = 0.0
    env = ScoponeEnvMA()
    env.current_player = 0
    # Assegna manualmente le mani
    env.game_state["hands"][0] = [
        (1, 'denari'), (1, 'coppe'), (2, 'denari'), (2, 'bastoni'),
        (3, 'spade'), (4, 'coppe'), (5, 'bastoni'), (5, 'spade'),
        (6, 'denari'), (6, 'bastoni')
    ]
    env.game_state["hands"][1] = [
        (7, 'denari'), (7, 'coppe'), (1, 'spade'), (2, 'coppe'),
        (3, 'denari'), (4, 'spade'), (6, 'coppe'), (9, 'denari'),
        (9, 'coppe'), (9, 'spade')
    ]
    env.game_state["hands"][2] = [
        (1, 'bastoni'), (2, 'spade'), (3, 'coppe'), (4, 'bastoni'),
        (5, 'denari'), (8, 'denari'), (8, 'coppe'), (8, 'bastoni'),
        (9, 'bastoni'), (10, 'denari')
    ]
    env.game_state["hands"][3] = [
        (7, 'bastoni'), (7, 'spade'), (1, 'coppe'), (3, 'bastoni'),
        (4, 'denari'), (6, 'spade'), (8, 'spade'), (10, 'bastoni'),
        (10, 'coppe'), (10, 'spade')
    ]
    env.game_state["table"] = []
    # Q-value di Team1 prima
    obs_t1_before = env._get_observation(1)
    tensor_before = torch.tensor(obs_t1_before, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qvals_before = agent_team1.online_qnet(tensor_before)[0].clone()

    # Forced moves: 8 mosse predefinite
    forced_moves = [
        dict(player=0, hand_index=0, subset=()),
        dict(player=1, hand_index=0, subset=(0, 1)),
        dict(player=2, hand_index=0, subset=()),
        dict(player=3, hand_index=0, subset=(0, 1)),
        dict(player=0, hand_index=1, subset=()),
        dict(player=1, hand_index=1, subset=(0, 1)),
        dict(player=2, hand_index=1, subset=()),
        dict(player=3, hand_index=1, subset=(0, 1)),
    ]
    for move in forced_moves:
        p = move["player"]
        env.current_player = p
        if p == 1 and move["hand_index"] in [0, 1]:
            env.game_state["table"] = [(3, 'spade'), (4, 'coppe')] if move["hand_index"] == 0 else [(2, 'denari'), (5, 'spade')]
        elif p == 3 and move["hand_index"] in [0, 1]:
            env.game_state["table"] = [(2, 'spade'), (5, 'bastoni')] if move["hand_index"] == 0 else [(3, 'coppe'), (4, 'bastoni')]
        action = encode_action(move["hand_index"], move["subset"])
        obs_before = env._get_observation(p)
        valid_actions = env.get_valid_actions()
        if action not in valid_actions:
            action = valid_actions[0]
        next_obs, rew, info = env.step(action)
        if p in [1, 3]:
            agent_team1.store_transition((obs_before, action, 0.0, next_obs, env.get_valid_actions()))
            agent_team1.train_step()
        else:
            agent_team0.store_transition((obs_before, action, 0.0, next_obs, env.get_valid_actions()))
            agent_team0.train_step()
    # Random moves fino a completare 40 mosse
    step_count = 8
    while step_count < 40:
        cp = env.current_player
        obs_before = env._get_observation(cp)
        team_id = 0 if cp in [0, 2] else 1
        agent = agent_team0 if team_id == 0 else agent_team1
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        action = random.choice(valid_actions)
        next_obs, rew, info = env.step(action)
        agent.store_transition((obs_before, action, 0.0, next_obs, env.get_valid_actions()))
        agent.train_step()
        step_count += 1

    # Al termine delle 40 mosse, il tuo codice (train_agents) dovrebbe aver inserito le dummy transitions.
    # Qui controlliamo che, ad esempio, il replay buffer del team 1 contenga 21 transizioni
    breakdown = compute_final_score_breakdown(env.game_state)
    final_reward = compute_final_reward_from_breakdown(breakdown)
    r0, r1 = final_reward[0], final_reward[1]
    replay_team0 = agent_team0.replay_buffer.buffer
    replay_team1 = agent_team1.replay_buffer.buffer
    print("Transitions for Team 0:")
    for transition in replay_team0:
        print(f"State: {transition[0]}, Action: {transition[1]}, Reward: {transition[2]}, Next State: {transition[3]}, Valid Actions: {transition[4]}")

    print("Transitions for Team 1:")
    for transition in replay_team1:
        print(f"State: {transition[0]}, Action: {transition[1]}, Reward: {transition[2]}, Next State: {transition[3]}, Valid Actions: {transition[4]}")
    assert len(replay_team0) >= 21, "Il replay buffer di Team0 deve contenere almeno 21 transizioni"
    assert len(replay_team1) >= 21, "Il replay buffer di Team1 deve contenere almeno 21 transizioni"
    last_transition_team0 = replay_team0[-1]
    last_transition_team1 = replay_team1[-1]
    dummy_reward_team0 = last_transition_team0[2]
    dummy_reward_team1 = last_transition_team1[2]
    assert dummy_reward_team0 == r0, "La dummy transition di Team0 deve contenere la final reward"
    assert dummy_reward_team1 == r1, "La dummy transition di Team1 deve contenere la final reward"

    for _ in range(50):
        agent_team0.train_step()
        agent_team1.train_step()

    obs_t1_after = env._get_observation(1)
    tensor_after = torch.tensor(obs_t1_after, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qvals_after = agent_team1.online_qnet(tensor_after)[0]
    diff_q = (qvals_after - qvals_before).max().item()
    print(f"Diff Q team1 = {diff_q}")
    threshold = 2.0
    assert diff_q >= threshold, "Team1 Q-value non è cresciuto a sufficienza!"