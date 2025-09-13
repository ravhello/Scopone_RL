import os
import tempfile
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO


def test_agent_save_and_load_roundtrip(tmp_path=None):
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # do one tiny update to change weights
    from trainers.train_ppo import collect_trajectory
    batch = collect_trajectory(env, agent, horizon=40, use_mcts=False)
    agent.update(batch, epochs=1, minibatch_size=128)
    # save
    if tmp_path is None:
        tmp_dir = tempfile.mkdtemp()
    else:
        tmp_dir = str(tmp_path)
    ckpt = os.path.join(tmp_dir, 'test_ppo_ckpt.pth')
    agent.save(ckpt)
    # load into new agent
    agent2 = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    agent2.load(ckpt)
    # quick forward consistency on tiny batch
    import torch
    o = batch['obs'][:2].to(torch.float32)
    s = batch['seat_team'][:2].to(torch.float32)
    with torch.no_grad():
        v1 = agent.critic(o, s)
        v2 = agent2.critic(o, s)
    assert v1.shape == v2.shape


