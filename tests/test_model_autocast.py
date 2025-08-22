import torch
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO


def test_actor_state_proj_under_autocast():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    obs = env._get_observation(env.current_player).unsqueeze(0).to('cuda', dtype=torch.float32)
    seat = torch.zeros((1,6), dtype=torch.float32, device='cuda')
    seat[0, 4] = 1.0
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = agent.actor.compute_state_proj(obs, seat)
    assert out.dtype in (torch.float16, torch.float32)
    assert out.device.type == 'cuda'


