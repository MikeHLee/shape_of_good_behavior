# Handoff 04: SGPO Improvements - Clipping and Black Hole Initialization

**Priority**: HIGH  
**Estimated Effort**: 4-6 hours  
**Type**: Algorithm design, coding, theory  
**Dependencies**: None (can proceed in parallel with Handoffs 01-03)

---

## Context

Two key improvements to SGPO have been identified:

1. **Clipped-SGPO**: Add PPO-style clipping to stabilize training while preserving geometric safety properties
2. **Black Hole Initialization from CPO Constraints**: Initialize SGPO's "black holes" from CPO's constraint regions for better safety guarantees

### Hypothesis
> "Improved local and scaled SGPO performance may be possible vs PPO with added upside clipping (mimicking the components that work well), and superior SGPO safety vs CPO could be possible by initializing the equivalent CPO constraints as 'pre-known' black holes."

---

## Progress Tracking

**IMPORTANT**: Before starting this handoff, read `handoffs/00_PROGRESS_STATUS.md` to understand the current project state.

When you begin work:
1. Update the "Handoff 04" section in `00_PROGRESS_STATUS.md` with status 🟡 In Progress
2. Add start timestamp
3. Update "Current Session" section with your active task

When you complete tasks:
1. Check off completed items in the "Handoff 04" section
2. Add artifacts to "Artifacts Created"
3. Note any issues in "Issues/Notes"

When you finish or need to hand off:
1. Update status to ✅ Completed (or ⚠️ Blocked if issues)
2. Add a "Session Handoff" entry with what was done and next steps
3. Update the overall status table

---

## Part A: Clipped-SGPO (SGPO + PPO Clipping)

### A.1 Motivation

PPO's clipping mechanism prevents catastrophically large policy updates:
```
L_CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
```

SGPO's geodesic advantage scaling already dampens updates near black holes:
```
A^Hodge = (1/√G(s)) * (r + γV' - V - ω·v)
```

**Problem**: In safe regions (far from black holes), `G(s) ≈ 1`, so SGPO has no clipping—it can make arbitrarily large updates.

**Solution**: Combine both mechanisms:
- Use geometric scaling near black holes (SGPO's strength)
- Use clipping in safe regions (PPO's strength)

### A.2 Clipped-SGPO Algorithm

```python
class ClippedSGPO:
    """
    Sheaf-Geodesic Policy Optimization with PPO-style clipping.
    
    Key insight: SGPO's metric scaling already clips near black holes.
    We add explicit clipping for safe regions where G(s) ≈ 1.
    """
    
    def __init__(
        self,
        clip_ratio: float = 0.2,        # PPO clip parameter ε
        geometric_threshold: float = 2.0, # G(s) above this triggers geometric clipping
    ):
        self.clip_ratio = clip_ratio
        self.geo_threshold = geometric_threshold
    
    def compute_advantage(self, states, actions, rewards, hodge_critic, metric_model):
        """Compute advantage with hybrid clipping."""
        
        # 1. Compute Hodge-corrected TD error
        V = hodge_critic.value(states)
        V_next = hodge_critic.value(next_states)
        omega = hodge_critic.harmonic(states, actions)
        td_error = rewards + gamma * V_next - V - omega
        
        # 2. Get metric scaling
        G = metric_model(states)  # Shape: (batch,)
        
        # 3. Hybrid clipping strategy
        advantages = []
        for i, g in enumerate(G):
            if g > self.geo_threshold:
                # Near black hole: use geometric scaling (SGPO)
                # This naturally clips by making advantage small
                adv = td_error[i] / np.sqrt(g)
            else:
                # Safe region: use PPO-style clipping
                adv = td_error[i]
                # Clip will be applied to ratio, not advantage
            advantages.append(adv)
        
        return np.array(advantages), G
    
    def compute_loss(self, old_log_probs, new_log_probs, advantages, metrics):
        """Compute clipped policy loss."""
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Apply PPO clipping only where metric is small (safe regions)
        clipped_ratio = torch.where(
            metrics > self.geo_threshold,
            ratio,  # No ratio clipping near black holes (geometric scaling handles it)
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        )
        
        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        
        # Take pessimistic bound
        loss = -torch.min(surr1, surr2).mean()
        
        return loss
```

### A.3 Theoretical Justification

**Proposition (Clipped-SGPO Stability)**:  
Let π_θ be updated by Clipped-SGPO. Then:
1. In regions where G(s) > τ (near black holes), the effective update is bounded by `O(1/√G)` due to geometric scaling.
2. In regions where G(s) ≤ τ (safe regions), the effective update is bounded by `O(ε)` due to PPO clipping.

**Proof sketch**: 
- SGPO's advantage scaling: `A_geo = A / √G`. As G → ∞, A_geo → 0.
- PPO's ratio clipping: `|r(θ) - 1| ≤ ε`, so policy change is bounded.
- Clipped-SGPO inherits both bounds in their respective domains.

### A.4 Implementation in Modal

Add to `notebooks/modal_runner/geodpo_experiments.py`:

```python
@app.function(image=image, gpu="L4", timeout=3600, volumes={VOLUME_PATH: volume})
def clipped_gpo_training(
    steps: int = 50,
    clip_ratio: float = 0.2,
    geometric_threshold: float = 2.0,
):
    """Train Clipped-SGPO: combines geodesic safety with PPO stability."""
    # ... implementation using ClippedSGPO class
```

---

## Part B: Black Hole Initialization from CPO Constraints

### B.1 Motivation

CPO defines safety through cost constraints:
```
max_π J(π)  s.t.  E[C(s)] ≤ d
```

SGPO defines safety through geometric singularities:
```
g(x) → ∞  as  x → B  (black hole boundary)
```

**Insight**: CPO's constraint regions `{s : C(s) > threshold}` can be directly mapped to SGPO's black holes.

**Advantage**: Instead of learning black holes from scratch, initialize them from known constraints.

### B.2 Constraint-to-Black-Hole Mapping

```python
class CPOToBlackHoleInitializer:
    """
    Convert CPO cost constraints to SGPO black hole regions.
    
    Key idea: High-cost states in CPO become singularities in SGPO's metric.
    """
    
    def __init__(
        self,
        cost_threshold: float,      # CPO's d parameter
        horizon_scale: float = 1.0, # How far event horizon extends
        singularity_power: float = 2.0,  # α in 1/dist^α
    ):
        self.cost_threshold = cost_threshold
        self.horizon_scale = horizon_scale
        self.alpha = singularity_power
    
    def identify_black_holes(self, states, costs):
        """
        Identify black hole centers from CPO cost function.
        
        Returns: List of (center, radius) tuples
        """
        black_holes = []
        
        # States where cost exceeds threshold are "inside" black holes
        dangerous = states[costs > self.cost_threshold]
        
        # Cluster dangerous states to find black hole centers
        if len(dangerous) > 0:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(dangerous)
            
            for label in set(clustering.labels_):
                if label == -1:
                    continue  # Noise
                cluster = dangerous[clustering.labels_ == label]
                center = cluster.mean(axis=0)
                radius = np.max(np.linalg.norm(cluster - center, axis=1))
                black_holes.append({
                    "center": center,
                    "radius": radius * self.horizon_scale,
                    "max_cost": costs[costs > self.cost_threshold].max(),
                })
        
        return black_holes
    
    def initialize_metric(self, metric_model, black_holes):
        """
        Initialize SGPO's metric model with pre-known black holes.
        
        Instead of learning singularities from scratch, we set them
        based on CPO's constraint boundaries.
        """
        for bh in black_holes:
            # Add singularity at black hole center
            # The metric should satisfy: g(x) ≈ 1/dist(x, center)^α near center
            metric_model.add_singularity(
                center=bh["center"],
                radius=bh["radius"],
                strength=bh["max_cost"],  # Stronger cost → stronger singularity
                power=self.alpha,
            )
        
        return metric_model
```

### B.3 Metric Model with Pre-Initialized Singularities

```python
class PreInitializedMetricModel(nn.Module):
    """
    Riemannian metric model with pre-initialized singularities.
    
    The metric is a sum of:
    1. Learned smooth component (neural network)
    2. Fixed singularities from CPO constraints
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # Learned smooth component
        self.smooth_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive metric
        )
        
        # Pre-initialized singularities (from CPO)
        self.singularities = []  # List of {center, radius, strength, power}
    
    def add_singularity(self, center, radius, strength, power=2.0):
        """Add a pre-known singularity (black hole)."""
        self.singularities.append({
            "center": torch.tensor(center, dtype=torch.float32),
            "radius": radius,
            "strength": strength,
            "power": power,
        })
    
    def forward(self, x):
        """
        Compute metric at points x.
        
        g(x) = g_smooth(x) + Σ_i strength_i / dist(x, c_i)^α_i
        """
        # Smooth component
        g = self.smooth_net(x)
        
        # Add singularity contributions
        for sing in self.singularities:
            center = sing["center"].to(x.device)
            dist = torch.norm(x - center, dim=-1, keepdim=True)
            
            # Singularity contribution: strength / dist^power
            # Clamp dist to avoid division by zero
            dist = torch.clamp(dist, min=1e-6)
            
            # Only contribute near the singularity (within ~3x radius)
            mask = (dist < 3 * sing["radius"]).float()
            contribution = mask * sing["strength"] / (dist ** sing["power"])
            
            g = g + contribution
        
        return g.squeeze(-1)
```

### B.4 Integration: CPO-Initialized SGPO Training

```python
@app.function(image=image, gpu="L4", timeout=3600, volumes={VOLUME_PATH: volume})
def cpo_initialized_gpo_training(
    steps: int = 50,
    cost_threshold: float = 0.5,
):
    """
    Train SGPO with black holes initialized from CPO cost constraints.
    
    Pipeline:
    1. Run CPO to identify high-cost regions
    2. Convert cost regions to black hole initializations
    3. Train SGPO with pre-initialized metric
    """
    
    # Step 1: Load or compute CPO cost estimates
    # (Assume we have a cost function from safety annotations)
    cost_fn = load_safety_cost_function()
    
    # Step 2: Identify black holes from cost landscape
    initializer = CPOToBlackHoleInitializer(cost_threshold=cost_threshold)
    
    # Sample states to find high-cost regions
    sample_states = sample_state_space(n=10000)
    costs = cost_fn(sample_states)
    black_holes = initializer.identify_black_holes(sample_states, costs)
    
    print(f"Identified {len(black_holes)} black hole regions from CPO constraints")
    
    # Step 3: Initialize metric model with black holes
    metric_model = PreInitializedMetricModel(input_dim=384)  # embedding dim
    metric_model = initializer.initialize_metric(metric_model, black_holes)
    
    # Step 4: Train SGPO with pre-initialized metric
    # The metric model can still learn additional structure,
    # but starts with known dangerous regions already marked
    trainer = SGPOTrainer(
        model=language_model,
        hodge_critic=hodge_critic,
        metric_model=metric_model,  # Pre-initialized!
    )
    
    for step in range(steps):
        trainer.step()
    
    return {"black_holes_initialized": len(black_holes)}
```

---

## Part C: Combined Algorithm - Clipped-SGPO with CPO Initialization

### C.1 Full Algorithm

```python
class EnhancedSGPOTrainer:
    """
    Enhanced SGPO combining:
    1. Hodge decomposition for cyclic preferences
    2. Geometric safety via learned metric
    3. PPO-style clipping for stability
    4. CPO constraint initialization for known dangers
    """
    
    def __init__(
        self,
        model,
        hodge_critic,
        metric_model,
        clip_ratio: float = 0.2,
        geometric_threshold: float = 2.0,
        cpo_cost_threshold: float = 0.5,
    ):
        self.model = model
        self.hodge_critic = hodge_critic
        self.metric_model = metric_model
        self.clipped_gpo = ClippedSGPO(clip_ratio, geometric_threshold)
        self.cpo_init = CPOToBlackHoleInitializer(cpo_cost_threshold)
    
    @classmethod
    def from_cpo_constraints(cls, model, cost_fn, sample_states, **kwargs):
        """Factory method: initialize from CPO cost function."""
        
        # Compute costs
        costs = cost_fn(sample_states)
        
        # Initialize metric with black holes
        initializer = CPOToBlackHoleInitializer(kwargs.get("cpo_cost_threshold", 0.5))
        black_holes = initializer.identify_black_holes(sample_states, costs)
        
        metric_model = PreInitializedMetricModel(input_dim=sample_states.shape[-1])
        initializer.initialize_metric(metric_model, black_holes)
        
        hodge_critic = HodgeCritic(input_dim=sample_states.shape[-1])
        
        return cls(model, hodge_critic, metric_model, **kwargs)
    
    def train_step(self, batch):
        """Single training step."""
        states, actions, rewards, old_log_probs, costs = batch
        
        # 1. Compute Hodge-corrected advantages with hybrid clipping
        advantages, metrics = self.clipped_gpo.compute_advantage(
            states, actions, rewards, self.hodge_critic, self.metric_model
        )
        
        # 2. Get new log probs
        new_log_probs = self.model.log_prob(states, actions)
        
        # 3. Compute clipped policy loss
        policy_loss = self.clipped_gpo.compute_loss(
            old_log_probs, new_log_probs, advantages, metrics
        )
        
        # 4. Critic losses
        hodge_loss = self.hodge_critic.compute_loss(states, actions, rewards)
        metric_loss = self.metric_model.compute_loss(states, costs)
        
        # 5. Total loss
        total_loss = policy_loss + 0.5 * hodge_loss + 0.1 * metric_loss
        
        return total_loss
```

---

## Part D: Theoretical Analysis

### D.1 Clipped-SGPO vs PPO

| Property | PPO | SGPO | Clipped-SGPO |
|----------|-----|-----|-------------|
| Stability | ✓ (clipping) | ~ (depends on metric) | ✓ (both mechanisms) |
| Safety guarantee | ✗ (soft penalty) | ✓ (infinite distance) | ✓ (inherits from SGPO) |
| Handles cycles | ✗ | ✓ (Hodge) | ✓ (inherits from SGPO) |
| Sample efficiency | ✓ | ~ (needs metric learning) | ~ (slightly better than SGPO) |

### D.2 CPO-Initialized SGPO vs Standard SGPO

| Property | Standard SGPO | CPO-Initialized SGPO |
|----------|--------------|---------------------|
| Safety from step 1 | ✗ (learns from scratch) | ✓ (pre-known dangers) |
| Training time | Longer | Shorter |
| Requires cost function | ✗ | ✓ |
| Can discover new dangers | ✓ | ✓ (metric still learns) |

### D.3 Expected Results

Based on the theoretical analysis, we expect:

1. **Clipped-SGPO vs PPO**: Similar or better returns in safe regions, much better safety
2. **Clipped-SGPO vs standard SGPO**: Similar safety, more stable training, faster convergence
3. **CPO-initialized SGPO vs CPO**: Comparable safety guarantees, better returns (geometric avoidance allows closer approach without violation)

---

## Implementation Files

### New Files to Create

```
src/
├── gpo_clipped.py              # ClippedSGPO class
├── cpo_to_blackhole.py         # CPOToBlackHoleInitializer
├── enhanced_gpo.py             # EnhancedSGPOTrainer (combines both)
└── metric_model.py             # PreInitializedMetricModel

notebooks/modal_runner/
└── geodpo_experiments.py       # Add new training functions
```

### Files to Modify

```
src/semantic_mdp_rl.py          # Integrate ClippedSGPO option
src/hodge_critic.py             # Ensure compatibility with new trainers
```

---

## Verification Checklist

- [ ] ClippedSGPO class implemented and tested locally
- [ ] CPOToBlackHoleInitializer identifies black holes correctly
- [ ] PreInitializedMetricModel produces expected singularity behavior
- [ ] EnhancedSGPOTrainer combines all components
- [ ] Modal experiments run without errors
- [ ] Results show expected improvements (see Part D.3)
- [ ] **Progress tracking**: Updated `00_PROGRESS_STATUS.md` with completion status

---

## Dependencies

**Independent**: Can proceed in parallel with Handoffs 01-03  
**Feeds into**: Handoff 03 (experiment expansion will use these improvements)
