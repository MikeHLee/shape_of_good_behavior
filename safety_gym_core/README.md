# Safety Gym Core

Rust core library for sheaf-theoretic safety in reinforcement learning environments.

## Overview

This crate provides the computational core for the Sheaf-Geodesic Policy Optimization (SGPO) framework, implementing:

- **Topological spaces** for arbitrary decision environments
- **Discrete navigation** (grid worlds, mazes)
- **Continuous control** (MuJoCo-style physics)
- **SGPO policy** inference with safety constraints
- **ONNX inference** for trained policy models (optional)
- **Godot GDExtension** bindings for game engine integration (optional)

## Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `default` | Includes ONNX support | `ort` |
| `onnx` | ONNX Runtime inference | `ort` |
| `godot` | Godot 4 GDExtension | `godot-rust` |

## Quick Start

```rust
use safety_gym_core::{
    topology::discrete::DiscreteNavigationSpace,
    topology::TopologicalSpace,
    policy::gpo::{SGPOConfig, SGPOPolicy},
};

// Create a 20x20 grid with hazards
let hazards = vec![(5, 5), (5, 6), (6, 5)];
let space = DiscreteNavigationSpace::new((20, 20), hazards, 64, 42);

// Create SGPO policy
let config = SGPOConfig {
    alpha: 2.0,           // Black hole strength
    risk_threshold: 0.5,  // Maximum acceptable risk
    clip_epsilon: 0.2,    // PPO-style clipping
    temperature: 1.0,
};
let policy = SGPOPolicy::new(space, config);

// Check state safety
assert!(policy.is_state_safe(&(0, 0)));
assert!(!policy.is_state_safe(&(5, 5)));

// Apply safety constraint to actions
let raw_action = vec![1.0, 0.5];
let safe_action = policy.apply_safety_constraint(&(4, 4), &raw_action);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (Training)                        │
│  Safety Gym Environment + SGPO Policy (PyTorch)              │
│                         │                                   │
│                         ▼                                   │
│  Export Layer (ONNX / TorchScript)                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Core Library                        │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  ort (ONNX RT)   │  │  TopologicalSpace│                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐     ┌─────────────────────────────────┐
│  C FFI Bindings     │     │  Godot GDExtension              │
└─────────────────────┘     └─────────────────────────────────┘
```

## Modules

### `topology`

Core topological space abstractions:

- **`TopologicalSpace`** trait - Abstract interface for any decision space
- **`DiscreteNavigationSpace`** - Grid worlds with hazard detection
- **`ContinuousControlSpace`** - 2D physics simulation with obstacles
- **`BlackHoleRegion`** - Dangerous regions to avoid
- **`TopologyData`** - Database of explored states and risks

### `policy`

Policy implementations:

- **`SGPOPolicy`** - Applies Riemannian metric safety constraints
- **`ClippedSGPOPolicy`** - Combines PPO clipping with SGPO geodesics
- **`OnnxPolicy`** - ONNX model inference (requires `onnx` feature)

### `bindings`

External integrations:

- **`c_api`** - C FFI for Unity/Unreal/native code
- **`godot`** - GDExtension for Godot 4 (requires `godot` feature)

## Building

```bash
# Basic build (no optional features)
cargo build --no-default-features

# With ONNX support
cargo build --features onnx

# With Godot support
cargo build --features godot

# All features
cargo build --all-features

# Run tests
cargo test --no-default-features

# Run demo
cargo run --example demo --no-default-features
```

## C API

The library exports C-compatible functions for integration with other languages:

```c
// Create a discrete navigation space
DiscreteSpaceHandle* handle = discrete_space_new(20, 20, 64, 42);

// Add hazards
discrete_space_add_hazard(handle, 5, 5);

// Check safety
int is_safe = discrete_space_is_safe(handle, 0, 0);  // Returns 1

// Compute harmonic risk
float risk = discrete_space_harmonic_risk(handle, 0, 0, 5);

// Clean up
discrete_space_free(handle);
```

## Godot Integration

When built with the `godot` feature, this library provides Godot nodes:

### SafetyAgent3D

A 3D agent with SGPO-based safe navigation:

```gdscript
var agent = $SafetyAgent3D
agent.add_obstacle(Vector2(5.0, 5.0), 1.0)
agent.finalize_obstacles(1.2)  # Safety margin

# Move safely (respects black holes)
agent.move_safely(Vector2(1, 0), 5.0)

# Check safety
var risk = agent.get_risk_at(Vector2(4.0, 4.0))
```

### GridAgent

A 2D grid-based agent for discrete navigation:

```gdscript
var agent = $GridAgent
agent.add_hazard(5, 5)

# Find safe path
var path = agent.find_path_to(19, 19, 0.5)

# Move in direction (0=up, 1=right, 2=down, 3=left)
agent.move_direction(1)  # Move right
```

## Theory

This library implements the computational core of **Sheaf-Geodesic Policy Optimization (SGPO)**:

1. **Black Holes**: Dangerous states identified via clustering of failure states
2. **Riemannian Metric**: Conformal factor φ(x) ≈ 1/dist(x, B)^α creates energy barriers
3. **Harmonic Risk**: H¹ cohomology estimates cyclic preference inconsistency
4. **Safety Constraint**: Actions scaled by 1/φ(x) to slow near dangers

For the full theory, see the paper: *Sheaf-Theoretic Reward Spaces for Safe Reinforcement Learning*.

## License

MIT

## Author

Mike Lee <mike@oasis-x.io>
