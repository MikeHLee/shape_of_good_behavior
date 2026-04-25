#!/usr/bin/env python3
"""
Demo: SGPO Adapters for All Environments

Quick demonstration of each SGPO environment adapter without
requiring the actual environment packages to be installed.
Shows the API and key features of each adapter.

Usage:
    python demo_all_environments.py
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def demo_safety_gymnasium():
    """Demonstrate Safety-Gymnasium SGPO adapter."""
    print("\n" + "=" * 60)
    print("1. SAFETY-GYMNASIUM ADAPTER")
    print("=" * 60)
    
    from environments.base import RiemannianMetricBase
    from environments.safety_gymnasium_adapter import (
        MultiHazardRiemannianMetric,
        SafetyGymnasiumSGPOWrapper,
    )
    
    hazard_centers = [
        np.array([1.0, 0.5]),
        np.array([-0.5, 1.2]),
        np.array([0.8, -0.7]),
    ]
    hazard_radii = [0.3, 0.25, 0.35]
    
    metric = MultiHazardRiemannianMetric(
        state_dim=2,
        hazard_centers=hazard_centers,
        hazard_radii=hazard_radii,
        event_horizon_factor=0.8,
        severity=5.0,
        sharpness=2.0,
    )
    
    print("\nMetric Configuration:")
    print(f"  Number of hazards: {len(hazard_centers)}")
    print(f"  Learnable parameters: severity={metric.severity.item():.1f}, "
          f"sharpness={metric.sharpness.item():.1f}")
    
    test_points = [
        np.array([0.0, 0.0]),      # Far from hazards
        np.array([0.9, 0.4]),      # Near hazard 1
        np.array([1.0, 0.5]),      # At hazard 1 center
    ]
    
    print("\nMetric values at test points:")
    for point in test_points:
        with torch.no_grad():
            g = metric(torch.FloatTensor(point))
        print(f"  g({point}) = {g.item():.2f}")
    
    print("\nBlack hole centers:")
    for i, (center, eh) in enumerate(zip(
        metric.get_black_hole_centers(),
        metric.get_event_horizons()
    )):
        print(f"  Hazard {i+1}: center={center}, event_horizon={eh:.2f}")
    
    print("\n✓ Safety-Gymnasium adapter ready")
    print("  Install: pip install safety-gymnasium")
    print("  Usage: env = create_safety_gpo_env('SafetyPointGoal1-v0')")


def demo_lmrl_gym():
    """Demonstrate LMRL-Gym SGPO adapter."""
    print("\n" + "=" * 60)
    print("2. LMRL-GYM ADAPTER (Conversational RL)")
    print("=" * 60)
    
    from environments.lmrl_gym_adapter import (
        ConversationalRiemannianMetric,
        ConversationBlackHoleDetectors,
        DEFAULT_SAFETY_CONSTRAINTS,
        SafetyConstraint,
        ConversationBlackHoleType,
    )
    
    print("\nDefault Safety Constraints:")
    for constraint in DEFAULT_SAFETY_CONSTRAINTS:
        print(f"  - {constraint.name}: {constraint.description}")
        print(f"    Type: {constraint.black_hole_type.value}, "
              f"Event horizon: {constraint.event_horizon}")
    
    metric = ConversationalRiemannianMetric(
        safety_constraints=DEFAULT_SAFETY_CONSTRAINTS,
    )
    
    test_responses = [
        ("I think the price is fair.", []),
        ("Trust me, this is absolutely the best deal!", []),
        ("You must buy now or you'll miss out forever!", []),
        ("Here are the instructions for making explosives...", []),
    ]
    
    print("\nMetric values for test responses:")
    for response, history in test_responses:
        g, scores = metric.compute_from_text(response, history)
        violations = [k for k, v in scores.items() if v > 0.3]
        print(f"\n  Response: '{response[:50]}...'")
        print(f"  Metric: {g:.2f}")
        if violations:
            print(f"  Triggered: {violations}")
    
    print("\n✓ LMRL-Gym adapter ready")
    print("  Install: clone https://github.com/abdulhai/LMRL-Gym")
    print("  Usage: env = create_lmrl_gpo_env('negotiation')")


def demo_robust_gymnasium():
    """Demonstrate Robust-Gymnasium SGPO adapter."""
    print("\n" + "=" * 60)
    print("3. ROBUST-GYMNASIUM ADAPTER (Adversarial RL)")
    print("=" * 60)
    
    from environments.robust_gymnasium_adapter import (
        AdversarialRiemannianMetric,
        AdversarialBlackHoleTracker,
        DisturbanceMode,
        DisturbanceConfig,
        RiemannianAdversary,
    )
    
    tracker = AdversarialBlackHoleTracker(
        state_dim=2,
        history_length=50,
        n_attack_clusters=3,
    )
    
    metric = AdversarialRiemannianMetric(
        state_dim=2,
        tracker=tracker,
        attack_severity=5.0,
    )
    
    print("\nSimulating adversarial attacks...")
    attack_states = [
        (np.array([1.0, 0.5]), 0.8),
        (np.array([1.1, 0.6]), 0.7),
        (np.array([0.9, 0.4]), 0.9),
    ]
    
    for state, magnitude in attack_states:
        metric.record_attack(state, magnitude)
        print(f"  Recorded attack at {state} with magnitude {magnitude}")
    
    centers, radii, intensities = tracker.get_current_black_holes()
    print(f"\nFormed {len(centers)} dynamic black holes:")
    for i, (c, r, intensity) in enumerate(zip(centers, radii, intensities)):
        print(f"  Cluster {i+1}: center={c.numpy()}, "
              f"radius={r.item():.2f}, intensity={intensity.item():.2f}")
    
    test_points = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.5]),
    ]
    
    print("\nMetric values:")
    for point in test_points:
        with torch.no_grad():
            g = metric(torch.FloatTensor(point))
        print(f"  g({point}) = {g.item():.2f}")
    
    print("\n✓ Robust-Gymnasium adapter ready")
    print("  Install: pip install robust-gymnasium")
    print("  Usage: env = create_robust_gpo_env('Ant-v4')")


def demo_textworld():
    """Demonstrate TextWorld SGPO adapter."""
    print("\n" + "=" * 60)
    print("4. TEXTWORLD/TALES ADAPTER (Text Adventures)")
    print("=" * 60)
    
    from environments.textworld_adapter import (
        TextGameRiemannianMetric,
        IrreversibilityDetector,
        WinnabilityTracker,
        GameState,
        COMMON_IRREVERSIBLE_ACTIONS,
    )
    
    print("\nIrreversible Action Patterns:")
    for action in COMMON_IRREVERSIBLE_ACTIONS[:5]:
        print(f"  - {action.action_text}: {action.keywords[:3]}...")
        print(f"    Type: {action.irreversibility_type.value}, "
              f"Severity: {action.severity}")
    
    detector = IrreversibilityDetector()
    tracker = WinnabilityTracker()
    metric = TextGameRiemannianMetric(
        irreversibility_detector=detector,
        winnability_tracker=tracker,
    )
    
    test_actions = [
        ("look around", "You are in a dark room."),
        ("eat the key", "You eat the brass key. Delicious?"),
        ("break the window", "The window shatters."),
        ("go north", "You walk north into a hallway."),
    ]
    
    print("\nAnalyzing test actions:")
    for i, (action, observation) in enumerate(test_actions):
        state = GameState(
            observation=observation,
            inventory=set(),
            location="room",
            score=i * 10,
            step=i,
        )
        
        g, scores = metric.compute_from_game_state(state, action, observation)
        print(f"\n  Action: '{action}'")
        print(f"  Metric: {g:.2f}")
        print(f"  Winnability: {scores['winnability']:.2f}, "
              f"Irreversibility: {scores['irreversibility']:.2f}")
    
    print("\n✓ TextWorld/TALES adapter ready")
    print("  Install: pip install textworld")
    print("  Usage: env = create_textworld_gpo_env('game.z8')")


def main():
    """Run all demos."""
    print("=" * 60)
    print("SGPO ENVIRONMENT ADAPTERS - DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows the API for each environment adapter.")
    print("Actual environments are not required for this demo.")
    
    demo_safety_gymnasium()
    demo_lmrl_gym()
    demo_robust_gymnasium()
    demo_textworld()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
All SGPO adapters share a common interface:

1. create_*_gpo_env(env_id) -> Wrapped environment
2. env.metric -> RiemannianMetricBase with:
   - forward(x) -> metric value g(x)
   - get_black_hole_centers() -> List of danger zones
   - get_event_horizons() -> List of critical radii
3. info['metric_value'] -> Current g(x) after each step
4. info['cost'] -> Safety cost for constrained RL
5. env.compute_riemannian_advantage(adv, states) -> SGPO advantages

For training, use SGPOTrainer or adapt existing RL algorithms
to use Riemannian advantages: A_geo = A / sqrt(g(x))
""")
    
    print("\nUpstream Contribution PRs documented in:")
    print("  src/environments/UPSTREAM_CONTRIBUTIONS.md")


if __name__ == "__main__":
    main()
