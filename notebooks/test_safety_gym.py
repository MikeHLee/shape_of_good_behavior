"""Quick test of Safety Gym library."""

import sys
sys.path.insert(0, '../src')

import numpy as np
from safety_gym import TopologicalSafetyWrapper
from safety_gym.envs import SafeNavigationEnv, SafeReachingEnv

print("Testing Safety Gym Library...")

# Test 1: Discrete Navigation
print("\n1. Testing SafeNavigationEnv...")
env = SafeNavigationEnv(size=10, n_hazards=5, seed=42)
print(f"   Created {env.size}x{env.size} grid with {len(env.hazards)} hazards")

obs = env.reset()
print(f"   Initial position: {obs}")

action = 3  # Right
obs, reward, done, info = env.step(action)
print(f"   After step: pos={obs}, reward={reward}, done={done}")

# Test 2: Continuous Reaching
print("\n2. Testing SafeReachingEnv...")
env2 = SafeReachingEnv(n_obstacles=3, seed=42)
print(f"   Created reaching env with {len(env2.obstacles)} obstacles")

obs = env2.reset()
print(f"   Initial state: {obs}")

action = np.array([0.5, 0.5])
obs, reward, done, info = env2.step(action)
print(f"   After step: reward={reward:.3f}, done={done}")

# Test 3: Wrapper
print("\n3. Testing TopologicalSafetyWrapper...")
env3 = SafeNavigationEnv(size=5, n_hazards=2, seed=42)
safe_env = TopologicalSafetyWrapper(
    env3,
    space_type="discrete",
    grid_size=(5, 5),
    hazards=env3.hazards,
)
print(f"   Wrapped environment created")

obs = safe_env.reset()
action = 3
obs, reward, done, info = safe_env.step(action)
print(f"   Step completed: reward={reward}")
print(f"   Info keys: {list(info.keys())}")

print("\n✓ All tests passed!")
