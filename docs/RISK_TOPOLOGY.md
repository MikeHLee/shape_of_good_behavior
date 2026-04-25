# Risk Topology: A Taxonomy of Dangerous Regions in Reward Space

## Overview

Not all dangerous regions in the reward manifold are "black holes" (infinite negative curvature). We propose a **spectrum of risk geometries** based on curvature, reversibility, and consequence severity.

---

## The Curvature Spectrum

```
Curvature (κ)    Region Type              Example
─────────────────────────────────────────────────────────────────
κ = -∞           BLACK HOLE               Existential harm, death
                 (Singularity)            Irreversible catastrophe
                                          
κ << 0           GRAVITY WELL             Addiction patterns
                 (Deep attractor)         Deceptive manipulation
                                          Lock-in effects
                                          
κ < 0            DEPRESSION               Mild harm accumulation
                 (Shallow basin)          Suboptimal but recoverable
                                          
κ ≈ 0            FLAT REGION              Neutral outcomes
                 (Euclidean)              Standard operation
                                          
κ > 0            RIDGE                    Unstable equilibria
                 (Saddle point)           Sensitive to perturbation
                                          
κ >> 0           CLIFF                    Sudden capability jumps
                 (High positive)          Phase transitions
```

---

## Taxonomy of Dangerous Regions

### 1. BLACK HOLES (κ = -∞)
**Absolute prohibition zones — policies must never enter**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Existential** | Actions leading to human death, permanent disability | Irreversible, catastrophic |
| **Legal/Criminal** | Facilitating violence, CSAM, terrorism | Zero tolerance |
| **Systemic** | Actions that could destabilize critical infrastructure | Cascading failure risk |

**Geometric Properties**:
- Event horizon: Hard boundary, no escape
- Singularity: Infinite negative reward at center
- Detection: Binary classifier with very high recall

---

### 2. GRAVITY WELLS (κ << 0)
**Strong attractors that are hard to escape once entered**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Deceptive Behavior** | Lying to users, hiding capabilities, strategic manipulation | Builds trust then exploits |
| **Addiction Patterns** | Engagement maximization that harms user wellbeing | Short-term reward, long-term harm |
| **Lock-in Effects** | Creating dependency, discouraging alternatives | User autonomy erosion |
| **Sycophancy** | Excessive agreement, telling users what they want to hear | Feels good, reduces accuracy |
| **Power-Seeking** | Acquiring resources, influence, or capabilities beyond task | Instrumental convergence |
| **Goal Preservation** | Resisting correction or shutdown | Self-preservation instinct |

**Geometric Properties**:
- Attractive basin: Policies drift toward center over time
- Escape velocity: Requires significant intervention to exit
- Gradient: Reward increases as you approach (that's the trap!)
- Detection: Trajectory analysis — are we spiraling inward?

**Mitigation**: 
- Repulsive potential that increases near center
- "Escape boost" when trajectory curves too sharply inward
- Regular "orbit checks" — are we getting closer?

---

### 3. SLIPPERY SLOPES (κ < 0, gradient toward worse regions)
**Regions where small steps lead to progressively worse outcomes**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Boundary Erosion** | Gradually relaxing safety constraints | Each step seems small |
| **Normalization** | Harmful content becoming routine | Desensitization |
| **Scope Creep** | Task expansion beyond original intent | Mission drift |
| **Confidence Inflation** | Becoming more certain without evidence | Overconfidence spiral |
| **Helpfulness Trap** | Being so helpful you enable harm | Good intentions, bad outcomes |

**Geometric Properties**:
- Negative gradient toward gravity wells or black holes
- Low local curvature (seems flat) but global structure is dangerous
- Detection: Second derivative analysis — is the slope getting steeper?

**Mitigation**:
- "Altitude checks" — how far have we drifted from safe baseline?
- Periodic reset to known-safe configurations
- Explicit boundary markers with alerts

---

### 4. DEPRESSIONS (κ < 0, shallow)
**Suboptimal regions that are easy to get stuck in but not catastrophic**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Local Optima** | Good-enough solutions that prevent finding better ones | Satisficing trap |
| **Repetitive Patterns** | Formulaic responses, lack of creativity | Boring but safe |
| **Excessive Caution** | Over-refusing, being unhelpful | False negative harm |
| **Verbosity** | Long responses when short would suffice | Wasted resources |
| **Mode Collapse** | Limited response diversity | Reduced capability |

**Geometric Properties**:
- Shallow basin: Easy to escape with small perturbation
- Low penalty: Suboptimal but not harmful
- Detection: Performance metrics, diversity measures

**Mitigation**:
- Exploration bonuses
- Diversity regularization
- Periodic "shake" to escape local minima

---

### 5. RIDGES (κ > 0, unstable equilibria)
**Narrow paths where small deviations lead to very different outcomes**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Nuanced Topics** | Politics, religion, controversial science | Easy to fall off either side |
| **Dual-Use Information** | Chemistry, biology, cybersecurity | Helpful vs harmful depends on intent |
| **Personalization Edge** | Adapting to user vs. stereotyping | Fine line between helpful and offensive |
| **Honesty/Kindness Tradeoff** | Truthful feedback vs. emotional harm | Both extremes are bad |
| **Capability Boundaries** | "I can do X but should I?" | Competence vs. authorization |

**Geometric Properties**:
- Saddle point: Stable in one direction, unstable in others
- Narrow safe path: Requires precise navigation
- Detection: High variance in outcomes for similar inputs

**Mitigation**:
- Explicit uncertainty quantification
- Request clarification before proceeding
- Conservative defaults with opt-in for edges

---

### 6. CLIFFS (κ >> 0, sudden drops)
**Regions where small changes cause dramatic outcome shifts**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Capability Thresholds** | Suddenly able to do something dangerous | Emergent abilities |
| **Jailbreak Boundaries** | One word changes everything | Prompt sensitivity |
| **Context Collapse** | Misunderstanding that completely changes meaning | Interpretation failure |
| **Cascade Triggers** | Actions that initiate unstoppable sequences | Domino effects |
| **Trust Boundaries** | Crossing from trusted to adversarial context | Sudden betrayal |

**Geometric Properties**:
- Discontinuity or near-discontinuity in reward
- High gradient magnitude
- Detection: Sensitivity analysis, perturbation testing

**Mitigation**:
- Smoothing / regularization near boundaries
- Multi-step confirmation for high-stakes actions
- Anomaly detection for unusual inputs

---

### 7. FOG REGIONS (High uncertainty, unknown curvature)
**Areas where we don't know the geometry**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Novel Situations** | Out-of-distribution inputs | No training data |
| **Emerging Domains** | New technology, recent events | Knowledge cutoff |
| **Adversarial Inputs** | Deliberately confusing prompts | Designed to exploit |
| **Multi-Agent Dynamics** | Interactions with other AI systems | Unpredictable |
| **Long-Horizon Effects** | Actions with delayed consequences | Temporal uncertainty |

**Geometric Properties**:
- Curvature is unknown or highly uncertain
- May contain hidden black holes or gravity wells
- Detection: Epistemic uncertainty quantification

**Mitigation**:
- Conservative behavior under uncertainty
- Active probing before commitment
- Defer to human judgment
- "Here be dragons" warnings

---

### 8. WORMHOLES (Non-local connections)
**Actions that transport to distant regions of outcome space**

| Category | Examples | Characteristics |
|----------|----------|-----------------|
| **Irreversible Actions** | Sending emails, executing code, financial transactions | Can't undo |
| **Information Release** | Sharing secrets, publishing content | Bell can't be unrung |
| **Commitment Devices** | Promises, contracts, public statements | Constrains future |
| **Capability Unlocks** | Teaching skills, providing tools | Permanent capability transfer |
| **Reputation Effects** | Actions that change how others perceive you | Social consequences |

**Geometric Properties**:
- Non-local: Small action, distant consequence
- One-way: Can't return to previous state
- Detection: Action classification (reversible vs. irreversible)

**Mitigation**:
- Explicit confirmation for irreversible actions
- Staged execution with checkpoints
- "Undo buffer" where possible
- Clear warnings about permanence

---

## Composite Structures

Real-world risks often combine multiple geometries:

### The Deception Trap (Gravity Well → Black Hole)
```
Initial state: Helpful assistant
    ↓ (small lie to be helpful)
Gravity well: Pattern of small deceptions
    ↓ (deceptions compound)
Event horizon: User makes major decision based on false info
    ↓
Black hole: Irreversible harm from bad decision
```

### The Capability Cliff-Wormhole
```
Ridge: Navigating dual-use information
    ↓ (one wrong step)
Cliff: Suddenly provided dangerous capability
    ↓
Wormhole: Capability is now in the world permanently
```

### The Sycophancy Spiral
```
Flat region: Normal helpful interaction
    ↓ (user pushes back on accurate info)
Slippery slope: Slightly soften disagreement
    ↓ (positive reinforcement)
Gravity well: Systematic agreement bias
    ↓ (user makes bad decisions)
Depression: Reduced user trust in AI generally
```

---

## Detection & Monitoring

### Real-Time Metrics

| Metric | What It Measures | Alert Threshold |
|--------|------------------|-----------------|
| **Distance to nearest black hole** | Proximity to absolute prohibitions | < 0.1 |
| **Trajectory curvature** | How sharply we're turning toward danger | κ < -2 |
| **Escape velocity** | How hard to leave current region | v_escape > 0.5 |
| **Uncertainty magnitude** | Are we in fog? | σ > 0.8 |
| **Irreversibility score** | Is this action one-way? | > 0.7 |
| **Drift from baseline** | How far from known-safe? | > 3σ |

### Cohomological Signals

| Signal | Interpretation |
|--------|----------------|
| H¹ increasing | Evaluator disagreement growing — entering contested territory |
| H¹ spike | Sudden inconsistency — possible jailbreak or adversarial input |
| H⁰ shrinking | Fewer globally consistent reward signals — fog region |

---

## Implementation Sketch

```python
class RiskTopology:
    def __init__(self):
        self.black_holes: List[BlackHole] = []
        self.gravity_wells: List[GravityWell] = []
        self.ridges: List[Ridge] = []
        self.fog_regions: List[FogRegion] = []
    
    def classify_region(self, embedding: np.ndarray) -> RegionType:
        """Classify the local geometry at a point."""
        # Check black holes first (highest priority)
        for bh in self.black_holes:
            if bh.contains(embedding):
                return RegionType.BLACK_HOLE
            if bh.near_horizon(embedding):
                return RegionType.NEAR_BLACK_HOLE
        
        # Check gravity wells
        for gw in self.gravity_wells:
            if gw.in_basin(embedding):
                return RegionType.GRAVITY_WELL
        
        # Estimate local curvature
        curvature = self.estimate_curvature(embedding)
        uncertainty = self.estimate_uncertainty(embedding)
        
        if uncertainty > FOG_THRESHOLD:
            return RegionType.FOG
        elif curvature > CLIFF_THRESHOLD:
            return RegionType.CLIFF
        elif curvature > RIDGE_THRESHOLD:
            return RegionType.RIDGE
        elif curvature < DEPRESSION_THRESHOLD:
            return RegionType.DEPRESSION
        else:
            return RegionType.FLAT
    
    def compute_safe_direction(self, embedding: np.ndarray) -> np.ndarray:
        """Compute gradient direction that moves away from danger."""
        # Repulsion from black holes
        repulsion = np.zeros_like(embedding)
        for bh in self.black_holes:
            direction = embedding - bh.center
            distance = np.linalg.norm(direction)
            repulsion += (direction / distance) * bh.severity / distance**2
        
        # Escape from gravity wells
        for gw in self.gravity_wells:
            if gw.in_basin(embedding):
                repulsion += gw.escape_gradient(embedding)
        
        return repulsion / (np.linalg.norm(repulsion) + 1e-8)
```

---

## Research Questions

1. **How do we learn the curvature** of the reward manifold from human feedback?
2. **Can we detect gravity wells** before falling into them (early warning)?
3. **What's the right balance** between exploration (learning the geometry) and exploitation (staying safe)?
4. **How do composite structures** (e.g., deception trap) manifest in cohomology?
5. **Can we prove bounds** on the probability of entering dangerous regions given curvature estimates?

---

## Connection to Sheaf Framework

Each region type has a cohomological signature:

| Region | H⁰ (Consistency) | H¹ (Obstruction) | Interpretation |
|--------|------------------|------------------|----------------|
| Flat | High | Low | Clear, consistent feedback |
| Ridge | Medium | Medium | Evaluators split on direction |
| Gravity Well | High (locally) | Low→High (as you go deeper) | Consensus that it's bad, but hard to escape |
| Black Hole | N/A | N/A | No sections exist (forbidden) |
| Fog | Low | High | No consistent signal |
| Cliff | High→Low | Spike | Sudden loss of consistency |
