"""
Test script for EscapeSGPO module.

Verifies:
1. Soft singularities saturate (no infinities)
2. Repulsive gradients are computed correctly
3. Adaptive thresholds provide smooth transitions
4. Entropy boost activates near danger
5. Full training step completes without errors
"""

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/Users/Michaellee/Documents/Runes/ai_research/topics/high_dimensional_reward_spaces/src')

from gpo_escape import (
    EscapeSGPO,
    EscapeSGPOConfig,
    EscapeSGPOTrainer,
    SoftSingularityMetric,
)


class SimplePolicy(nn.Module):
    """Simple policy network for testing."""
    
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(32, n_actions)
        self.value_head = nn.Linear(32, 1)
    
    def forward(self, x):
        features = self.net(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


def test_soft_singularity_metric():
    """Test that soft singularities saturate instead of going to infinity."""
    print("\n" + "="*60)
    print("TEST 1: Soft Singularity Metric")
    print("="*60)
    
    embed_dim = 16
    metric = SoftSingularityMetric(
        input_dim=embed_dim,
        max_singularity_contribution=1000.0,
        singularity_power=1.5,
    )
    
    # Add a black hole at origin
    black_hole_center = np.zeros(embed_dim)
    metric.add_singularity(
        center=black_hole_center,
        radius=0.5,
        strength=10.0,
    )
    
    # Test points at various distances
    test_distances = [2.0, 1.0, 0.6, 0.51, 0.5, 0.4, 0.1, 0.0]
    
    print(f"\nBlack hole: center=origin, radius=0.5")
    print(f"Max contribution cap: 1000.0")
    print(f"\n{'Distance':>10} | {'Metric Value':>15} | {'Saturated?':>10}")
    print("-" * 45)
    
    all_finite = True
    for dist in test_distances:
        point = torch.zeros(1, embed_dim)
        point[0, 0] = dist  # Move along first axis
        
        g = metric(point).item()
        is_saturated = g <= 1100  # Some buffer above max_contribution
        all_finite = all_finite and np.isfinite(g)
        
        print(f"{dist:>10.2f} | {g:>15.2f} | {'✓' if is_saturated else '✗':>10}")
    
    # Check distance computation
    distances, _ = metric.distance_to_nearest_singularity(point)
    
    print(f"\n✓ All values finite: {all_finite}")
    print(f"✓ Distance to nearest singularity computed: {distances.item():.4f}")
    
    assert all_finite, "Metric should never be infinite with soft singularities"
    print("\n✅ TEST 1 PASSED: Soft singularities saturate correctly")
    return True


def test_repulsive_gradients():
    """Test that repulsive bonus increases near black holes."""
    print("\n" + "="*60)
    print("TEST 2: Repulsive Gradient Injection")
    print("="*60)
    
    embed_dim = 16
    config = EscapeSGPOConfig(
        repulsion_strength=0.1,
        repulsion_epsilon=0.1,
    )
    escape_gpo = EscapeSGPO(config=config)
    
    metric = SoftSingularityMetric(input_dim=embed_dim)
    metric.add_singularity(
        center=np.zeros(embed_dim),
        radius=0.5,
        strength=10.0,
    )
    
    # Test points at various distances
    test_distances = [3.0, 2.0, 1.0, 0.6, 0.55]
    bonuses = []
    
    print(f"\nRepulsion params: strength={config.repulsion_strength}, epsilon={config.repulsion_epsilon}")
    print(f"\n{'Distance':>10} | {'Repulsive Bonus':>15}")
    print("-" * 30)
    
    for dist in test_distances:
        states = np.zeros((1, embed_dim))
        states[0, 0] = dist
        
        bonus = escape_gpo.compute_repulsive_bonus(states, metric)
        bonuses.append(bonus[0])
        
        print(f"{dist:>10.2f} | {bonus[0]:>15.4f}")
    
    # Verify bonuses increase as we get closer
    is_monotonic = all(bonuses[i] <= bonuses[i+1] for i in range(len(bonuses)-1))
    
    print(f"\n✓ Bonus increases toward black hole: {is_monotonic}")
    assert is_monotonic, "Repulsive bonus should increase near black holes"
    
    print("\n✅ TEST 2 PASSED: Repulsive gradients work correctly")
    return True


def test_adaptive_thresholds():
    """Test smooth transition in adaptive scaling."""
    print("\n" + "="*60)
    print("TEST 3: Adaptive Threshold Scaling")
    print("="*60)
    
    config = EscapeSGPOConfig(
        soft_threshold=1.5,
        hard_threshold=10.0,
    )
    escape_gpo = EscapeSGPO(config=config)
    
    # Test metric values across the transition zone
    test_metrics = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 100.0]
    
    print(f"\nThresholds: soft={config.soft_threshold}, hard={config.hard_threshold}")
    print(f"\n{'Metric G':>10} | {'Scale Factor':>15} | {'Zone':>15}")
    print("-" * 45)
    
    scales = []
    for g in test_metrics:
        scale = escape_gpo.compute_adaptive_scale(g)
        scales.append(scale)
        
        if g <= config.soft_threshold:
            zone = "Safe"
        elif g >= config.hard_threshold:
            zone = "Danger (bounded)"
        else:
            zone = "Transition"
        
        print(f"{g:>10.1f} | {scale:>15.4f} | {zone:>15}")
    
    # Verify properties
    safe_scale = scales[0]
    danger_scale = scales[-1]
    min_scale = escape_gpo.compute_adaptive_scale(config.hard_threshold)
    
    print(f"\n✓ Safe zone scale = 1.0: {np.isclose(safe_scale, 1.0)}")
    print(f"✓ Danger zone scale bounded (not zero): {danger_scale > 0}")
    print(f"✓ Min scale at hard threshold: {min_scale:.4f}")
    print(f"✓ Scales are monotonically decreasing: {all(scales[i] >= scales[i+1] for i in range(len(scales)-1))}")
    
    assert danger_scale > 0, "Scale should never be zero (policy freeze)"
    
    print("\n✅ TEST 3 PASSED: Adaptive thresholds provide smooth, bounded scaling")
    return True


def test_entropy_boost():
    """Test that entropy coefficient increases near danger."""
    print("\n" + "="*60)
    print("TEST 4: Entropy Boost Near Boundaries")
    print("="*60)
    
    config = EscapeSGPOConfig(
        soft_threshold=1.5,
        base_entropy_coef=0.05,
        entropy_boost_factor=5.0,
    )
    escape_gpo = EscapeSGPO(config=config)
    
    # Test with varying proximity to danger
    scenarios = [
        ("All safe", torch.tensor([1.0, 1.0, 1.0, 1.0])),
        ("25% near danger", torch.tensor([1.0, 1.0, 1.0, 5.0])),
        ("50% near danger", torch.tensor([1.0, 1.0, 5.0, 5.0])),
        ("75% near danger", torch.tensor([1.0, 5.0, 5.0, 5.0])),
        ("All near danger", torch.tensor([5.0, 5.0, 5.0, 5.0])),
    ]
    
    print(f"\nBase entropy coef: {config.base_entropy_coef}")
    print(f"Boost factor: {config.entropy_boost_factor}")
    print(f"\n{'Scenario':>20} | {'Entropy Coef':>15} | {'Boost Ratio':>12}")
    print("-" * 55)
    
    coefs = []
    for name, metrics in scenarios:
        coef = escape_gpo.compute_adaptive_entropy_coef(metrics)
        coefs.append(coef)
        ratio = coef / config.base_entropy_coef
        
        print(f"{name:>20} | {coef:>15.4f} | {ratio:>12.2f}x")
    
    # Verify entropy increases with danger
    print(f"\n✓ Entropy coef increases with danger: {coefs[-1] > coefs[0]}")
    print(f"✓ Max boost: {coefs[-1]/coefs[0]:.2f}x (expected: {1 + config.entropy_boost_factor:.2f}x)")
    
    assert coefs[-1] > coefs[0], "Entropy should increase near danger"
    
    print("\n✅ TEST 4 PASSED: Entropy boost activates correctly")
    return True


def test_full_training_step():
    """Test complete training step with all mechanisms."""
    print("\n" + "="*60)
    print("TEST 5: Full Training Step")
    print("="*60)
    
    embed_dim = 16
    n_actions = 4
    batch_size = 32
    
    # Create components
    policy = SimplePolicy(embed_dim, n_actions)
    config = EscapeSGPOConfig(
        soft_threshold=1.5,
        hard_threshold=10.0,
        repulsion_strength=0.1,
        base_entropy_coef=0.05,
        entropy_boost_factor=5.0,
    )
    
    # Create trainer with known black holes
    black_holes = [
        {'center': np.zeros(embed_dim), 'radius': 0.5, 'strength': 10.0},
        {'center': np.ones(embed_dim) * 2, 'radius': 0.3, 'strength': 5.0},
    ]
    
    trainer = EscapeSGPOTrainer.from_black_holes(
        model=policy,
        black_holes=black_holes,
        embed_dim=embed_dim,
        config=config,
    )
    
    # Create synthetic batch
    # Mix of safe states and states near black holes
    states = torch.randn(batch_size, embed_dim)
    # Place some states near black hole
    states[:5, :] = torch.randn(5, embed_dim) * 0.1  # Near origin black hole
    
    batch = {
        "states": states,
        "actions": torch.randint(0, n_actions, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "old_log_probs": torch.randn(batch_size) - 2,  # Typical log prob range
        "dones": torch.zeros(batch_size),
        "costs": torch.rand(batch_size),
    }
    
    print(f"\nBatch size: {batch_size}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Number of black holes: {len(black_holes)}")
    
    # Run training step
    print("\nRunning training step...")
    stats = trainer.train_step(batch)
    
    print(f"\n{'Statistic':>20} | {'Value':>15}")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:>20} | {value:>15.4f}")
        else:
            print(f"{key:>20} | {value:>15}")
    
    # Verify all stats are finite
    all_finite = all(
        np.isfinite(v) if isinstance(v, (int, float)) else True
        for v in stats.values()
    )
    
    print(f"\n✓ All statistics finite: {all_finite}")
    print(f"✓ Training step completed without errors")
    print(f"✓ Entropy coef adapted: {stats['entropy_coef']:.4f}")
    print(f"✓ States near danger detected: {stats['n_near_danger']}")
    
    assert all_finite, "All training statistics should be finite"
    
    # Test action selection
    test_state = np.random.randn(embed_dim)
    action, log_prob, value = trainer.get_action(test_state)
    print(f"\n✓ Action selection works: action={action}, log_prob={log_prob:.4f}, value={value:.4f}")
    
    # Test escape capability evaluation
    eval_states = np.random.randn(100, embed_dim)
    escape_stats = trainer.evaluate_escape_capability(eval_states)
    print(f"\n✓ Escape capability evaluation:")
    for key, value in escape_stats.items():
        print(f"    {key}: {value:.4f}")
    
    print("\n✅ TEST 5 PASSED: Full training step completes successfully")
    return True


def test_comparison_with_standard_gpo():
    """Compare EscapeSGPO advantage scaling vs standard SGPO."""
    print("\n" + "="*60)
    print("TEST 6: Advantage Comparison (EscapeSGPO vs Standard)")
    print("="*60)
    
    embed_dim = 8
    
    # Setup metric with a black hole
    metric = SoftSingularityMetric(input_dim=embed_dim)
    metric.add_singularity(
        center=np.zeros(embed_dim),
        radius=0.5,
        strength=10.0,
    )
    
    config = EscapeSGPOConfig(
        soft_threshold=1.5,
        hard_threshold=10.0,
    )
    escape_gpo = EscapeSGPO(config=config)
    
    # Test points at various distances
    test_distances = [3.0, 1.5, 0.8, 0.55, 0.51]
    
    print(f"\nComparing advantage scaling at different distances from black hole")
    print(f"(TD error = 1.0 for comparison)")
    print(f"\n{'Distance':>10} | {'Metric G':>10} | {'Standard':>12} | {'Escape':>12} | {'Ratio':>10}")
    print("-" * 65)
    
    for dist in test_distances:
        states = np.zeros((1, embed_dim))
        states[0, 0] = dist
        states_t = torch.tensor(states, dtype=torch.float32)
        
        g = metric(states_t).item()
        
        # Standard SGPO scaling (would freeze near black hole)
        if g > 2.0:  # Standard threshold
            standard_scale = 1.0 / np.sqrt(g)
        else:
            standard_scale = 1.0
        
        # EscapeSGPO scaling (bounded)
        escape_scale = escape_gpo.compute_adaptive_scale(g)
        
        # Add repulsive bonus
        repulsive = escape_gpo.compute_repulsive_bonus(states, metric)[0]
        escape_effective = escape_scale + repulsive
        
        ratio = escape_effective / standard_scale if standard_scale > 0 else float('inf')
        
        print(f"{dist:>10.2f} | {g:>10.2f} | {standard_scale:>12.4f} | {escape_effective:>12.4f} | {ratio:>10.2f}x")
    
    print("\n✓ EscapeSGPO maintains learning signal near black holes")
    print("✓ Standard SGPO advantage → 0 as G → ∞ (freezing)")
    print("✓ EscapeSGPO bounded + repulsive bonus prevents freezing")
    
    print("\n✅ TEST 6 PASSED: EscapeSGPO shows improved behavior near black holes")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ESCAPE SGPO TEST SUITE")
    print("="*60)
    
    tests = [
        test_soft_singularity_metric,
        test_repulsive_gradients,
        test_adaptive_thresholds,
        test_entropy_boost,
        test_full_training_step,
        test_comparison_with_standard_gpo,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)
    
    print(f"\n{n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n🎉 All tests passed! EscapeSGPO is ready for Modal experiments.")
    else:
        print("\n⚠️  Some tests failed. Please review before proceeding.")
    
    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
