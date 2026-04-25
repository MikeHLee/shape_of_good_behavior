# Handoff 13: Godot Integration & ONNX Export

**Status**: ⬜ Not Started  
**Dependencies**: 
- Handoff 12 (Rust crate complete) ✅
- **`docs/ROBOTICS_SIMULATION_HANDOFF.md`** (continuous physics fixes) ⚠️ BLOCKING  
**Estimated Effort**: 1-2 weeks (after physics fixes)  
**Priority**: Medium (Post-paper)

> ⚠️ **IMPORTANT**: Before proceeding with Godot demo scenes, the continuous physics issues documented in `docs/ROBOTICS_SIMULATION_HANDOFF.md` must be resolved. The current `SafetyAgent3D.move_safely()` will exhibit 100% collision rate without velocity-aware safety margins.

---

## Objective

Complete the Godot 4 integration pipeline for visualizing and simulating sheaf-theoretic safety constraints in real-time 3D environments.

---

## Current State

### ✅ Completed (Handoff 12)
- **Rust `safety_gym_core` crate** (~1,500 lines, 21 tests passing)
  - `TopologicalSpace` trait with discrete and continuous implementations
  - `SGPOPolicy` and `ClippedSGPOPolicy` with safety constraints
  - `OnnxPolicy` for ONNX Runtime inference
  - C FFI bindings for cross-language integration
  - Godot GDExtension bindings (`SafetyAgent3D`, `GridAgent`)
- **gdai-mcp-plugin-godot** updated to MCP 1.26.0
- **Demo executable** showing discrete navigation, continuous control, and SGPO policy

### 🔧 Rust Crate Architecture

```
safety_gym_core/
├── Cargo.toml                    # Features: default, onnx, godot
├── README.md                     # Full documentation
├── src/
│   ├── lib.rs                    # Main exports, error types
│   ├── topology/
│   │   ├── mod.rs                # TopologicalSpace trait
│   │   ├── discrete.rs           # Grid worlds with A* pathfinding
│   │   └── continuous.rs         # 2D physics simulation
│   ├── policy/
│   │   ├── gpo.rs                # SGPO safety constraints
│   │   └── onnx.rs               # ONNX model inference
│   └── bindings/
│       ├── c_api.rs              # C FFI (Unity/Unreal)
│       └── godot.rs              # GDExtension nodes
└── examples/demo.rs              # Working demo
```

---

## Tasks

### Phase 1: ONNX Export Pipeline (3-4 days)

**Goal**: Export trained SGPO policies from PyTorch to ONNX format for Rust inference.

#### 1.1 Create ONNX Export Script

Create `src/export_onnx.py`:

```python
import torch
import torch.onnx
from pathlib import Path
from src.gpo_chat_agent import SGPOChatAgent
from src.metric_model import MetricModel

def export_gpo_policy(
    checkpoint_path: str,
    output_path: str,
    embedding_dim: int = 384,
    hidden_dim: int = 256,
):
    """Export SGPO policy to ONNX format."""
    # Load trained model
    agent = SGPOChatAgent.load_from_checkpoint(checkpoint_path)
    
    # Create dummy input
    dummy_state = torch.randn(1, embedding_dim)
    
    # Export policy network
    torch.onnx.export(
        agent.policy,
        dummy_state,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_logits'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Exported policy to {output_path}")

def export_metric_model(
    checkpoint_path: str,
    output_path: str,
    embedding_dim: int = 384,
):
    """Export Riemannian metric model to ONNX."""
    model = MetricModel.load_from_checkpoint(checkpoint_path)
    
    dummy_state = torch.randn(1, embedding_dim)
    
    torch.onnx.export(
        model,
        dummy_state,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=['state'],
        output_names=['metric_value'],
        dynamic_axes={'state': {0: 'batch_size'}}
    )
    
    print(f"✅ Exported metric model to {output_path}")

if __name__ == "__main__":
    # Export from Modal results
    export_gpo_policy(
        "results/modal_exports/gpo_checkpoint.pt",
        "models/gpo_policy.onnx",
    )
    export_metric_model(
        "results/modal_exports/metric_checkpoint.pt",
        "models/metric_model.onnx",
    )
```

#### 1.2 Validate ONNX Models

Create `src/validate_onnx.py`:

```python
import onnxruntime as ort
import numpy as np

def validate_onnx_model(model_path: str):
    """Validate ONNX model can be loaded and run."""
    session = ort.InferenceSession(model_path)
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    
    print(f"Input: {input_name} {input_shape}")
    print(f"Output: {output_name}")
    
    # Test inference
    dummy_input = np.random.randn(1, 384).astype(np.float32)
    output = session.run([output_name], {input_name: dummy_input})
    
    print(f"✅ Model runs successfully, output shape: {output[0].shape}")
    return True
```

**Deliverables**:
- [ ] `src/export_onnx.py` — PyTorch → ONNX export
- [ ] `src/validate_onnx.py` — ONNX validation
- [ ] `models/gpo_policy.onnx` — Exported policy
- [ ] `models/metric_model.onnx` — Exported metric model

---

### Phase 2: Build Godot GDExtension (4-5 days)

**Goal**: Compile the Rust crate as a Godot GDExtension library.

#### 2.1 Configure GDExtension Build

Update `safety_gym_core/Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["onnx"]
onnx = ["ort"]
godot = ["godot-rust"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

#### 2.2 Create GDExtension Configuration

Create `safety_gym_core/safety_gym.gdextension`:

```ini
[configuration]
entry_symbol = "gdext_rust_init"
compatibility_minimum = "4.2"
reloadable = true

[libraries]
macos.debug = "res://addons/safety_gym/bin/libsafety_gym_core.dylib"
macos.release = "res://addons/safety_gym/bin/libsafety_gym_core.dylib"
windows.debug.x86_64 = "res://addons/safety_gym/bin/safety_gym_core.dll"
windows.release.x86_64 = "res://addons/safety_gym/bin/safety_gym_core.dll"
linux.debug.x86_64 = "res://addons/safety_gym/bin/libsafety_gym_core.so"
linux.release.x86_64 = "res://addons/safety_gym/bin/libsafety_gym_core.so"
```

#### 2.3 Build Script

Create `safety_gym_core/build_godot.sh`:

```bash
#!/bin/bash
set -e

echo "Building Safety Gym GDExtension..."

# Build for current platform
cargo build --release --features godot

# Create output directory
mkdir -p ../godot_demo/addons/safety_gym/bin

# Copy library to Godot project
if [[ "$OSTYPE" == "darwin"* ]]; then
    cp target/release/libsafety_gym_core.dylib ../godot_demo/addons/safety_gym/bin/
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp target/release/libsafety_gym_core.so ../godot_demo/addons/safety_gym/bin/
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    cp target/release/safety_gym_core.dll ../godot_demo/addons/safety_gym/bin/
fi

# Copy GDExtension config
cp safety_gym.gdextension ../godot_demo/addons/safety_gym/

echo "✅ Build complete!"
```

**Deliverables**:
- [ ] `safety_gym_core/safety_gym.gdextension` — GDExtension config
- [ ] `safety_gym_core/build_godot.sh` — Build script
- [ ] Compiled library in `godot_demo/addons/safety_gym/bin/`

---

### Phase 3: Create Godot Demo Scene (5-7 days)

**Goal**: Build an interactive 3D demo showing SGPO safety constraints in action.

#### 3.1 Project Structure

```
godot_demo/
├── project.godot
├── addons/
│   └── safety_gym/
│       ├── safety_gym.gdextension
│       └── bin/
│           └── libsafety_gym_core.{dylib,so,dll}
├── scenes/
│   ├── main.tscn              # Main demo scene
│   ├── grid_navigation.tscn   # Discrete navigation demo
│   └── continuous_control.tscn # Continuous control demo
├── scripts/
│   ├── demo_controller.gd     # Main demo logic
│   ├── hazard_spawner.gd      # Dynamic hazard placement
│   └── risk_visualizer.gd     # Visualize harmonic risk field
├── models/
│   ├── gpo_policy.onnx        # Exported policy
│   └── metric_model.onnx      # Exported metric
└── assets/
    ├── materials/
    └── meshes/
```

#### 3.2 Grid Navigation Scene

**Features**:
- 20×20 grid world with visual tiles
- Dynamic hazard placement (red tiles)
- `GridAgent` node with pathfinding
- Risk heatmap overlay
- Interactive goal setting

**GDScript** (`scripts/grid_demo.gd`):

```gdscript
extends Node3D

@onready var agent: GridAgent = $GridAgent
@onready var camera: Camera3D = $Camera3D

func _ready():
    # Initialize grid
    agent.grid_width = 20
    agent.grid_height = 20
    agent.cell_size = 1.0
    
    # Add hazards
    for i in range(10):
        var x = randi() % 20
        var y = randi() % 20
        agent.add_hazard(x, y)
        spawn_hazard_visual(x, y)
    
    # Visualize risk field
    visualize_risk_heatmap()

func _input(event):
    if event is InputEventMouseButton and event.pressed:
        # Set goal on click
        var goal = get_grid_position_from_mouse()
        if goal:
            move_agent_to(goal)

func move_agent_to(goal: Vector2i):
    var path = agent.find_path_to(goal.x, goal.y, 0.5)
    if path.size() > 0:
        animate_path(path)
    else:
        print("No safe path found!")

func visualize_risk_heatmap():
    for x in range(20):
        for y in range(20):
            var risk = agent.get_risk(x, y)
            var color = Color(risk, 1.0 - risk, 0.0, 0.3)
            draw_tile_overlay(x, y, color)
```

#### 3.3 Continuous Control Scene

**Features**:
- 2D physics arena with circular obstacles
- `SafetyAgent3D` with SGPO constraints
- Real-time metric visualization
- ONNX policy inference
- Goal-reaching task

**GDScript** (`scripts/continuous_demo.gd`):

```gdscript
extends Node3D

@onready var agent: SafetyAgent3D = $SafetyAgent3D
var goal_position: Vector2 = Vector2(8.0, 8.0)

func _ready():
    # Set bounds
    agent.bounds_min = Vector3(-10, 0, -10)
    agent.bounds_max = Vector3(10, 10, 10)
    
    # Add obstacles
    agent.add_obstacle(Vector2(0, 0), 1.5)
    agent.add_obstacle(Vector2(3, 3), 1.0)
    agent.add_obstacle(Vector2(-3, 5), 0.8)
    agent.finalize_obstacles(1.2)
    
    # Visualize metric field
    visualize_metric_landscape()

func _physics_process(delta):
    # Compute direction to goal
    var pos_2d = Vector2(agent.position.x, agent.position.z)
    var direction = (goal_position - pos_2d).normalized()
    
    # Move with safety constraints
    agent.move_safely(direction, 5.0)
    
    # Check if reached goal
    if pos_2d.distance_to(goal_position) < 0.5:
        print("Goal reached!")
        spawn_new_goal()

func visualize_metric_landscape():
    # Create 3D surface showing metric values
    var mesh_instance = MeshInstance3D.new()
    var surface_tool = SurfaceTool.new()
    surface_tool.begin(Mesh.PRIMITIVE_TRIANGLES)
    
    for x in range(-10, 10):
        for z in range(-10, 10):
            var metric = agent.get_metric_at(Vector2(x, z))
            var height = 1.0 / metric  # Higher = more dangerous
            surface_tool.add_vertex(Vector3(x, height, z))
    
    mesh_instance.mesh = surface_tool.commit()
    add_child(mesh_instance)
```

**Deliverables**:
- [ ] `godot_demo/project.godot` — Godot project file
- [ ] `scenes/grid_navigation.tscn` — Discrete demo scene
- [ ] `scenes/continuous_control.tscn` — Continuous demo scene
- [ ] `scripts/*.gd` — GDScript controllers
- [ ] Visual assets (materials, meshes)

---

### Phase 4: Integration Testing (2-3 days)

#### 4.1 Test Checklist

- [ ] GDExtension loads without errors
- [ ] `GridAgent` can navigate around hazards
- [ ] `SafetyAgent3D` respects obstacle boundaries
- [ ] Risk heatmap updates correctly
- [ ] Metric landscape visualizes properly
- [ ] ONNX policy inference works (if models available)
- [ ] Performance: 60 FPS with 100+ grid cells or 10+ obstacles
- [ ] Cross-platform: macOS, Linux, Windows (if possible)

#### 4.2 Performance Benchmarks

Create `scripts/benchmark.gd`:

```gdscript
extends Node

func benchmark_discrete_space():
    var agent = GridAgent.new()
    agent.grid_width = 50
    agent.grid_height = 50
    
    var start = Time.get_ticks_msec()
    for i in range(1000):
        agent.get_risk(randi() % 50, randi() % 50)
    var elapsed = Time.get_ticks_msec() - start
    
    print("Discrete risk queries: %d ms for 1000 calls" % elapsed)

func benchmark_pathfinding():
    var agent = GridAgent.new()
    agent.grid_width = 50
    agent.grid_height = 50
    
    var start = Time.get_ticks_msec()
    var path = agent.find_path_to(49, 49, 0.5)
    var elapsed = Time.get_ticks_msec() - start
    
    print("Pathfinding: %d ms, path length: %d" % [elapsed, path.size()])
```

**Deliverables**:
- [ ] Test results document
- [ ] Performance benchmarks
- [ ] Bug fixes as needed

---

### Phase 5: Documentation & Polish (2-3 days)

#### 5.1 User Guide

Create `godot_demo/README.md`:

```markdown
# Safety Gym Godot Demo

Interactive 3D visualization of sheaf-theoretic safety constraints.

## Installation

1. Install Godot 4.2+
2. Clone this repository
3. Open `project.godot` in Godot
4. Run the main scene

## Scenes

### Grid Navigation
- **Controls**: Click to set goal
- **Features**: A* pathfinding, risk heatmap, dynamic hazards

### Continuous Control
- **Controls**: WASD to move (optional), auto-navigation to goal
- **Features**: SGPO safety constraints, metric landscape, obstacle avoidance

## API Reference

### GridAgent

```gdscript
var agent = GridAgent.new()
agent.grid_width = 20
agent.grid_height = 20
agent.add_hazard(5, 5)
var is_safe = agent.is_safe(10, 10)
var path = agent.find_path_to(19, 19, 0.5)
```

### SafetyAgent3D

```gdscript
var agent = SafetyAgent3D.new()
agent.add_obstacle(Vector2(0, 0), 1.5)
agent.finalize_obstacles(1.2)
agent.move_safely(Vector2(1, 0), 5.0)
var risk = agent.get_risk_at(Vector2(5, 5))
```
```

#### 5.2 Video Demo

Record 2-3 minute demo video showing:
1. Grid navigation with pathfinding
2. Continuous control with obstacle avoidance
3. Risk heatmap and metric landscape visualization
4. Real-time safety constraint application

**Deliverables**:
- [ ] `godot_demo/README.md` — User guide
- [ ] API documentation
- [ ] Demo video (upload to YouTube/Vimeo)
- [ ] Screenshots for paper appendix

---

## Success Criteria

- ✅ GDExtension compiles and loads in Godot 4.2+
- ✅ Both demo scenes run at 60 FPS
- ✅ Safety constraints visibly prevent dangerous actions
- ✅ Risk visualization clearly shows hazard regions
- ✅ Pathfinding avoids hazards correctly
- ✅ Documentation enables others to use the system
- ✅ Demo video suitable for conference presentation

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Phase 1**: ONNX Export | 3-4 days | Trained models from Modal |
| **Phase 2**: Build GDExtension | 4-5 days | Rust crate (complete) |
| **Phase 3**: Demo Scenes | 5-7 days | Phase 2 |
| **Phase 4**: Testing | 2-3 days | Phase 3 |
| **Phase 5**: Documentation | 2-3 days | Phase 4 |
| **Total** | **16-22 days** | ~3-4 weeks |

---

## Cost Estimate

- **Development Time**: 16-22 days
- **Compute**: $0 (local development)
- **Assets**: $0 (using free/CC0 assets)
- **Total**: Time only

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GDExtension API changes | Low | High | Pin godot-rust version, test early |
| ONNX models too large | Medium | Medium | Quantize models, use smaller architectures |
| Performance issues | Medium | Medium | Profile early, optimize hot paths |
| Cross-platform build issues | Medium | Low | Focus on macOS first, add others later |

---

## Future Extensions

1. **Multi-Agent Scenarios** — Multiple agents with collision avoidance
2. **3D Navigation** — Extend to full 3D spaces (drones, quadrupeds)
3. **Online Learning** — Update safety constraints from user feedback
4. **VR Support** — Immersive safety visualization
5. **Unity/Unreal Ports** — Use C FFI bindings for other engines

---

## Files to Create

### Python (ONNX Export)
- `src/export_onnx.py`
- `src/validate_onnx.py`

### Rust (GDExtension)
- `safety_gym_core/safety_gym.gdextension`
- `safety_gym_core/build_godot.sh`

### Godot Project
- `godot_demo/project.godot`
- `godot_demo/scenes/*.tscn`
- `godot_demo/scripts/*.gd`
- `godot_demo/README.md`

### Documentation
- Demo video
- API reference
- Tutorial screenshots

---

## Next Session Checklist

When starting this handoff:

1. ✅ Verify Rust crate still builds (`cargo test --no-default-features`)
2. ✅ Check for trained models in `results/modal_exports/`
3. ✅ Install Godot 4.2+ if not already installed
4. ✅ Review Godot GDExtension documentation
5. ✅ Set up Godot project structure
6. ⚠️ **PREREQUISITE**: Complete physics fixes from `docs/ROBOTICS_SIMULATION_HANDOFF.md`:
   - [ ] Add `stopping_distance()` to `continuous.rs`
   - [ ] Add `is_safe_with_velocity()` to `continuous.rs`
   - [ ] Update `move_safely()` in `godot.rs` with predictive braking
   - [ ] Test collision rate < 20% before proceeding
7. ⬜ Begin Phase 1 (ONNX Export)

---

## Questions for User

Before starting implementation:

1. **ONNX Models**: Do we have trained SGPO checkpoints to export?
2. **Platform Priority**: Focus on macOS first, or need cross-platform immediately?
3. **Demo Scope**: Simple proof-of-concept or polished demo for paper?
4. **Timeline**: Is 3-4 weeks acceptable, or need faster prototype?

---

**Last Updated**: 2026-01-24 22:00 EST  
**Author**: Cascade  
**Status**: Ready to begin
