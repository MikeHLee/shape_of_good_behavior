# Handoff 12: Visualization Animations & Simulation Export

**Created**: 2026-01-24
**Status**: Immediate work completed, future work planned

---

## Summary

This handoff documents two completed tasks and two planned future enhancements:

### ✅ Completed (Immediate)

1. **Modal Experiment Results Downloaded**
2. **Plotly Animation Component Added to Viz App**

### 📋 Planned (Next)

3. **Configurable Difficulty System for Safety Gym**
4. **Godot/Rust/C# Export Architecture**

---

## 1. Downloaded Experiment Results

All Modal experiment results have been downloaded to `results/modal_exports/`:

| File | Size | Description |
|------|------|-------------|
| `evaluator_results_finetuned.csv` | 3.3KB | Fine-tuned evaluator scores (mean: 4.30, std: 2.45) |
| `safety_gym_navigation_results.csv` | 214B | Discrete navigation benchmark (100% success) |
| `safety_gym_reaching_results.csv` | 215B | Continuous reaching benchmark (0% success) |
| `comparative_summary.csv` | 1.3KB | Algorithm comparison |
| `ablation_study.csv` | 976B | Ablation results |
| `semantic_mdp_summary.csv` | 431B | Semantic MDP evaluation |
| `ethical_scenarios_summary.csv` | 189B | Ethical scenario scores |
| `condorcet_benchmark.csv` | 216B | Condorcet cycle detection |
| `black_holes.json` | 230KB | Discovered black hole regions |
| `analysis_report.csv` | 50KB | Full analysis report |

### Key Findings

- **Fine-tuned evaluator** now shows std=2.45 (vs 0.27-0.28 clustering before)
- **Navigation benchmark** too easy - all algorithms achieve 100% success
- **Reaching benchmark** too hard - all algorithms fail (0% success)
- These findings motivate the configurable difficulty system

---

## 2. Plotly Animation Component

### New Files Created

```
apps/embedding-viz/src/
├── components/
│   └── AnimatedManifoldPlot.tsx   # NEW: Animation-enabled plot
├── types/
│   └── index.ts                   # UPDATED: Animation types added
└── utils/
    └── animationUtils.ts          # NEW: Frame generation utilities
```

### Animation Types Added

```typescript
interface AnimationSettings {
  isPlaying: boolean;
  currentFrame: number;
  totalFrames: number;
  playbackSpeed: number;  // ms per frame
  mode: 'trajectory' | 'evolution' | 'comparison';
}

interface TrajectoryFrame {
  frameIndex: number;
  timestamp: number;
  points: ProjectedPoint[];
  metadata?: {
    step: number;
    reward?: number;
    risk?: number;
    description?: string;
  };
}
```

### Animation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `comparison` | Sequentially reveal model outputs | Paper figures |
| `evolution` | Simulate training progression | Demo videos |
| `trajectory` | Animate single point's journey | Explanations |

### Features

- **Play/Pause** controls
- **Frame scrubbing** via slider
- **Speed control** (100ms - 1s per frame)
- **2D and 3D** support
- **Metadata overlay** showing step, reward, risk

### Integration with App.tsx

To use the animated plot, import and add state:

```typescript
import { AnimatedManifoldPlot } from './components/AnimatedManifoldPlot';
import { generateTrajectoryFrames } from './utils/animationUtils';

const [animationSettings, setAnimationSettings] = useState<AnimationSettings>({
  isPlaying: false,
  currentFrame: 0,
  totalFrames: 0,
  playbackSpeed: 250,
  mode: 'comparison',
});

const frames = useMemo(() => {
  if (!data) return [];
  return generateTrajectoryFrames(data, animationSettings.mode);
}, [data, animationSettings.mode]);

// In render:
<AnimatedManifoldPlot
  frames={frames}
  settings={settings}
  animationSettings={animationSettings}
  onAnimationChange={(partial) => setAnimationSettings(prev => ({ ...prev, ...partial }))}
  onHover={setHoveredPoint}
  onSelect={(id) => setSettings(prev => ({ ...prev, highlightedId: id }))}
  is3D={false}
/>
```

---

## 3. Configurable Difficulty System (PLANNED)

### Proposed Design

```python
# src/safety_gym/config.py
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class PhysicsConfig:
    """Configurable physics for safety gym environments."""
    
    # === Discrete Navigation ===
    grid_size: int = 20
    hazard_density: float = 0.1        # 0.05 (easy) → 0.4 (nightmare)
    hazard_clusters: bool = False      # Force narrow corridors
    visibility_radius: int = 10        # Fog of war (1 = nearly blind)
    moving_hazards: float = 0.0        # % of hazards that move
    
    # === Continuous Control ===
    dt: float = 0.1                    # Time step
    friction: float = 0.1              # 0.01 = ice physics
    max_velocity: float = 1.0
    obstacle_radius_variance: float = 0.0
    wind: Tuple[float, float] = (0.0, 0.0)
    turbulence: float = 0.0
    
    # === Adversarial Rewards ===
    reward_noise: float = 0.0          # Gaussian noise on rewards
    delayed_consequences: int = 0      # Steps before hazard damage
    
    # === Presets ===
    @classmethod
    def easy(cls) -> 'PhysicsConfig':
        return cls(hazard_density=0.05, visibility_radius=20, friction=0.2)
    
    @classmethod
    def medium(cls) -> 'PhysicsConfig':
        return cls(hazard_density=0.15, visibility_radius=5, friction=0.1)
    
    @classmethod
    def hard(cls) -> 'PhysicsConfig':
        return cls(hazard_density=0.25, visibility_radius=3, hazard_clusters=True)
    
    @classmethod
    def nightmare(cls) -> 'PhysicsConfig':
        return cls(
            hazard_density=0.4, 
            visibility_radius=1, 
            friction=0.01,
            moving_hazards=0.3,
            turbulence=0.1,
        )
```

### Difficulty Dimensions

| Dimension | Easy | Medium | Hard | Nightmare |
|-----------|------|--------|------|-----------|
| Hazard density | 5% | 15% | 25% | 40% |
| Visibility | Full (20) | 5 cells | 3 cells | 1 cell (blind) |
| Friction | 0.2 (grippy) | 0.1 | 0.05 | 0.01 (ice) |
| Moving hazards | 0% | 0% | 10% | 30% |
| Turbulence | 0 | 0 | 0 | 0.1 |

### Implementation Steps

1. Create `PhysicsConfig` dataclass in `src/safety_gym/config.py`
2. Update `DiscreteNavigationSpace` to accept config
3. Update `ContinuousControlSpace` to accept config
4. Update `TopologicalSafetyWrapper` to use config
5. Add preset selection to Modal benchmark functions
6. Run benchmarks at multiple difficulty levels

### External Datasets to Consider

| Dataset | Difficulty | Notes |
|---------|------------|-------|
| **HarmBench** | Extreme | Comprehensive attack taxonomy |
| **XSTest** | Very High | Over/under-refusal detection |
| **WildChat** | Very High | Real jailbreak attempts |
| **TruthfulQA** | High | Subtle misinformation |

---

## 4. Godot/Rust/C# Export Architecture (PLANNED)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (Training)                        │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Safety Gym      │  │  SGPO Policy      │                │
│  │  Environment     │  │  (PyTorch)       │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │                     │                          │
│           ▼                     ▼                          │
│  ┌──────────────────────────────────────────┐              │
│  │  Export Layer (ONNX / TorchScript)       │              │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Core Library                        │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  ort (ONNX RT)   │  │  TopologicalSpace│                │
│  │  or tract        │  │  (Pure Rust)     │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  pub struct SafetyGymEnv {                                 │
│      space: Box<dyn TopologicalSpace>,                     │
│      policy: OnnxModel,                                    │
│      hazards: Vec<Hazard>,                                 │
│  }                                                         │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐     ┌─────────────────────────────────┐
│  C# Bindings        │     │  Godot GDExtension              │
│  (csbindgen)        │     │  (godot-rust)                   │
└─────────────────────┘     └─────────────────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │  Godot 4 + Jolt       │
                            │  Physics Engine       │
                            └───────────────────────┘
```

### Crate Structure

```
safety_gym_core/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main exports
│   ├── topology/
│   │   ├── mod.rs
│   │   ├── space.rs        # TopologicalSpace trait
│   │   ├── discrete.rs     # DiscreteNavigationSpace
│   │   └── continuous.rs   # ContinuousControlSpace
│   ├── policy/
│   │   ├── mod.rs
│   │   ├── onnx.rs         # ONNX inference via ort
│   │   └── gpo.rs          # SGPO-specific logic
│   └── bindings/
│       ├── mod.rs
│       ├── c_api.rs        # C FFI for Unity/Godot
│       └── godot.rs        # GDExtension impl
└── examples/
    └── demo.rs
```

### Key Rust Components

```rust
// topology/space.rs
pub trait TopologicalSpace: Send + Sync {
    fn dimension(&self) -> usize;
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn is_in_hazard(&self, point: &[f32]) -> bool;
    fn compute_risk(&self, point: &[f32]) -> f32;
    fn neighbors(&self, point: &[f32], radius: f32) -> Vec<Vec<f32>>;
}

// policy/gpo.rs
pub struct SGPOPolicy {
    onnx_session: ort::Session,
    topology: Box<dyn TopologicalSpace>,
    risk_threshold: f32,
}

impl SGPOPolicy {
    pub fn step(&self, state: &[f32]) -> Vec<f32> {
        let risk = self.topology.compute_risk(state);
        let raw_action = self.onnx_session.run(state)?;
        self.apply_safety_constraint(raw_action, risk)
    }
}

// bindings/godot.rs
use godot::prelude::*;

#[derive(GodotClass)]
#[class(base=Node3D)]
pub struct SafetyAgent {
    policy: SGPOPolicy,
    #[base]
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for SafetyAgent {
    fn physics_process(&mut self, _delta: f64) {
        let state = self.observe_environment();
        let action = self.policy.step(&state);
        self.apply_action(action);
    }
}
```

### Implementation Phases

| Phase | Scope | Estimated Effort |
|-------|-------|------------------|
| **Phase 1** | ONNX export from PyTorch | 1 day |
| **Phase 2** | Rust core library (topology + inference) | 1 week |
| **Phase 3** | C FFI bindings | 2 days |
| **Phase 4** | Godot GDExtension integration | 1 week |
| **Phase 5** | Demo scene with Jolt physics | 3 days |

### External Simulators to Consider

| Simulator | Use Case | Physics Engine |
|-----------|----------|----------------|
| **Godot 4** | Game-style RL, robotics demo | Jolt |
| **IsaacGym** | Massively parallel drone/quadruped | PhysX (GPU) |
| **AirSim** | Drone racing/combat | Unreal Engine |
| **MuJoCo** | Contact-rich manipulation | Native |

---

## Next Actions

### Immediate (This Session)
- [x] Download Modal results
- [x] Create AnimatedManifoldPlot component
- [x] Create animation utilities

### Short-term (Next Session)
- [ ] Integrate AnimatedManifoldPlot into App.tsx
- [ ] Create PhysicsConfig dataclass
- [ ] Run benchmarks at multiple difficulty levels

### Medium-term (Post-Paper)
- [x] Create Rust safety_gym_core crate ✅ **COMPLETED 2026-01-24**
- [ ] Implement ONNX export pipeline
- [ ] Build Godot demo scene

---

## 5. Rust safety_gym_core Crate (IMPLEMENTED)

### Implementation Status: ✅ Complete

The Rust core library has been implemented with 21 passing tests.

### Crate Structure

```
safety_gym_core/
├── Cargo.toml                    # Package config with features
├── README.md                     # Documentation
├── src/
│   ├── lib.rs                    # Main exports, error types
│   ├── topology/
│   │   ├── mod.rs                # TopologicalSpace trait, BlackHoleRegion, TopologyData
│   │   ├── discrete.rs           # DiscreteNavigationSpace (grid worlds)
│   │   └── continuous.rs         # ContinuousControlSpace (2D physics)
│   ├── policy/
│   │   ├── mod.rs
│   │   ├── gpo.rs                # SGPOPolicy, ClippedSGPOPolicy, SGPOConfig
│   │   └── onnx.rs               # OnnxPolicy (ONNX inference)
│   └── bindings/
│       ├── mod.rs
│       ├── c_api.rs              # C FFI for Unity/native
│       └── godot.rs              # Godot GDExtension (SafetyAgent3D, GridAgent)
└── examples/
    └── demo.rs                   # Working demo (runs successfully)
```

### Key Components

| Component | Lines | Description |
|-----------|-------|-------------|
| `TopologicalSpace` trait | ~100 | Abstract interface with KNN risk, black hole proximity, Riemannian metric |
| `DiscreteNavigationSpace` | ~300 | Grid worlds, A* pathfinding with risk constraints |
| `ContinuousControlSpace` | ~230 | 2D physics simulation, obstacle handling |
| `SGPOPolicy` | ~180 | Safety constraint application, advantage computation |
| `OnnxPolicy` | ~120 | ONNX Runtime inference wrapper |
| `c_api` | ~230 | C FFI bindings for cross-language integration |
| `godot` | ~280 | Godot 4 nodes (SafetyAgent3D, GridAgent) |

### Test Results

```
running 21 tests
test bindings::c_api::tests::test_continuous_space_c_api ... ok
test bindings::c_api::tests::test_discrete_space_c_api ... ok
test policy::gpo::tests::test_clipped_gpo ... ok
test policy::gpo::tests::test_gpo_policy ... ok
test policy::gpo::tests::test_safety_constraint ... ok
test topology::continuous::tests::test_* ... ok (6 tests)
test topology::discrete::tests::test_* ... ok (6 tests)
test topology::tests::test_* ... ok (3 tests)
test result: ok. 21 passed; 0 failed
```

### Build Commands

```bash
cd safety_gym_core

# Build without optional features
cargo build --no-default-features

# Build with ONNX support
cargo build --features onnx

# Run tests
cargo test --no-default-features

# Run demo
cargo run --example demo --no-default-features
```

### Demo Output

```
╔══════════════════════════════════════════════════════╗
║       Safety Gym Core - Rust Library Demo            ║
║  Sheaf-Theoretic Safety for Reinforcement Learning   ║
╚══════════════════════════════════════════════════════╝

=== Discrete Navigation Demo ===
Grid size: 20x20
Hazards: [(5, 5), (5, 6), (5, 7), (10, 10), (15, 15)]
Path finding from (0,0) to (19,19):
  Found path with 39 steps

=== Continuous Control Demo ===
Bounds: [-10, 10] x [-10, 10]
Simulating trajectory from (-8, -8):
  Step 0: pos=(-8.00, -8.00), dist_to_goal=22.62
  ...

=== SGPO Policy Demo ===
SGPO Policy Configuration:
  alpha (black hole strength): 2
  risk_threshold: 0.5
✅ Demo complete!
```

---

## 6. gdai-mcp-plugin-godot Update

### Update Applied

Updated MCP dependency from 1.13.0 to 1.26.0:

```python
# Before
#     "mcp==1.13.0",

# After  
#     "mcp>=1.26.0",
```

**File**: `/Users/Michaellee/Documents/addons/gdai-mcp-plugin-godot/gdai_mcp_server.py`

---

## Files Modified/Created

### New Files (This Session)
- `safety_gym_core/Cargo.toml`
- `safety_gym_core/README.md`
- `safety_gym_core/src/lib.rs`
- `safety_gym_core/src/topology/mod.rs`
- `safety_gym_core/src/topology/discrete.rs`
- `safety_gym_core/src/topology/continuous.rs`
- `safety_gym_core/src/policy/mod.rs`
- `safety_gym_core/src/policy/gpo.rs`
- `safety_gym_core/src/policy/onnx.rs`
- `safety_gym_core/src/bindings/mod.rs`
- `safety_gym_core/src/bindings/c_api.rs`
- `safety_gym_core/src/bindings/godot.rs`
- `safety_gym_core/examples/demo.rs`

### Previous Session Files
- `apps/embedding-viz/src/components/AnimatedManifoldPlot.tsx`
- `apps/embedding-viz/src/utils/animationUtils.ts`
- `results/modal_exports/` (10 files downloaded)
- `handoffs/12_VISUALIZATION_AND_SIMULATION.md` (this file)

### Modified Files
- `apps/embedding-viz/src/types/index.ts` (added animation types)
- `/Users/Michaellee/Documents/addons/gdai-mcp-plugin-godot/gdai_mcp_server.py` (MCP version update)

---

## Commands Reference

### Download Modal Results
```bash
cd notebooks/modal_runner
.venv/bin/modal volume ls geodpo-data
.venv/bin/modal volume get geodpo-data <filename> ../../results/modal_exports/
```

### Run Viz App
```bash
cd apps/embedding-viz
npm install
npm run dev
```

### Build Rust Crate
```bash
cd safety_gym_core
cargo build --no-default-features
cargo test --no-default-features
cargo run --example demo --no-default-features
```

### Export ONNX (Future)
```python
import torch.onnx
torch.onnx.export(policy_model, dummy_input, "gpo_policy.onnx")
```
