import torch
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO


def test_actor_state_proj_under_autocast():
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    device = torch.device('cpu')
    obs = env._get_observation(env.current_player).unsqueeze(0).to(device=device, dtype=torch.float32)
    seat = torch.zeros((1,6), dtype=torch.float32, device=device)
    seat[0, 4] = 1.0
    if device.type == 'cuda':
        ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
    with ctx:
        out = agent.actor.compute_state_proj(obs, seat)
    assert out.dtype in (torch.float16, torch.float32)
    assert out.device.type == device.type


