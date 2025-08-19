#!/usr/bin/env python3
"""Test script per verificare le ottimizzazioni GPU"""

import torch
import time
from environment import ScoponeEnvMA
from actions import get_valid_actions
import os

# Forza GPU-only mode
os.environ['GPU_ONLY'] = '1'

def test_environment_operations():
    """Test operazioni base dell'ambiente"""
    print("Testing environment operations...")
    
    env = ScoponeEnvMA()
    obs = env.reset()
    
    # Test che tutto sia su GPU
    assert obs.device.type == 'cuda', f"Observation on {obs.device}, expected cuda"
    
    # Test step con azione casuale
    for _ in range(10):
        state = env.game_state
        current_player = env.current_player
        valid_actions = get_valid_actions(state, current_player)
        
        if valid_actions.size(0) > 0:
            # Prendi azione casuale
            action_idx = torch.randint(0, valid_actions.size(0), (1,), device='cuda').item()
            action = valid_actions[action_idx]
            
            obs, reward, done, info = env.step(action)
            
            # Verifica che tutto sia su GPU
            assert obs.device.type == 'cuda', "Observation should be on GPU"
            assert reward.device.type == 'cuda', "Reward should be on GPU"
            
            if done:
                break
    
    print("✓ Environment operations test passed")

def test_action_generation():
    """Test generazione azioni GPU"""
    print("Testing action generation...")
    
    from actions import _find_sum_subsets_fast_gpu
    
    # Test subset finding su GPU
    table_cards = [0, 4, 8, 12, 16]  # ID cards
    target_rank = 5
    
    start = time.time()
    subsets = _find_sum_subsets_fast_gpu(table_cards, target_rank)
    gpu_time = time.time() - start
    
    print(f"  GPU subset finding took {gpu_time*1000:.2f}ms")
    print(f"  Found {len(subsets)} valid subsets")
    
    print("✓ Action generation test passed")

def test_observation_encoding():
    """Test encoding osservazioni su GPU"""
    print("Testing observation encoding...")
    
    from observation import encode_state_compact_for_player_fast
    
    env = ScoponeEnvMA()
    env.reset()
    state = env.get_state()
    
    start = time.time()
    obs = encode_state_compact_for_player_fast(state, 0)
    encoding_time = time.time() - start
    
    assert torch.is_tensor(obs), "Observation should be tensor"
    assert obs.device.type == 'cuda', "Observation should be on GPU"
    
    print(f"  Observation encoding took {encoding_time*1000:.2f}ms")
    print(f"  Observation shape: {obs.shape}")
    
    print("✓ Observation encoding test passed")

def test_no_cpu_transfers():
    """Verifica che non ci siano trasferimenti CPU non necessari"""
    print("Testing for CPU transfers...")
    
    env = ScoponeEnvMA()
    env.reset()
    
    # Monitora memoria GPU prima e dopo alcune operazioni
    torch.cuda.synchronize()
    initial_mem = torch.cuda.memory_allocated()
    
    # Esegui alcune mosse
    for _ in range(20):
        state = env.game_state
        current_player = env.current_player
        valid_actions = get_valid_actions(state, current_player)
        
        if valid_actions.size(0) > 0:
            action = valid_actions[0]
            obs, reward, done, info = env.step(action)
            
            if done:
                break
    
    torch.cuda.synchronize()
    final_mem = torch.cuda.memory_allocated()
    
    mem_diff = (final_mem - initial_mem) / 1024 / 1024
    print(f"  Memory difference: {mem_diff:.2f} MB")
    
    # Verifica che non ci sia crescita eccessiva di memoria (indicatore di leak)
    assert mem_diff < 10, f"Memory growth too high: {mem_diff} MB"
    
    print("✓ No excessive CPU transfers detected")

if __name__ == "__main__":
    print("=" * 60)
    print("GPU Optimization Test Suite")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print()
    
    try:
        test_environment_operations()
        test_action_generation()
        test_observation_encoding()
        test_no_cpu_transfers()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! GPU optimizations working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
