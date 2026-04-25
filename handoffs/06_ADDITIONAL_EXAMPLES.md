# Handoff 06: Additional Examples - Hodge, Feedback Decomposition, Agent Simulations

**Priority**: MEDIUM  
**Estimated Effort**: 6-8 hours  
**Type**: Research, writing, coding  
**Dependencies**: Handoffs 02-05 (paper structure and intuitions)

---

## Context

The paper needs more concrete examples to make abstract concepts tangible. Three categories requested:

1. **Hodge decomposition on preference graphs** with dangerous situations
2. **Feedback decomposition** for verbal/ordinal/pass-fail criteria
3. **Agent simulations** on realistic ethical scenarios

---

## Progress Tracking

**IMPORTANT**: Before starting this handoff, read `handoffs/00_PROGRESS_STATUS.md` to understand the current project state.

When you begin work:
1. Update the "Handoff 06" section in `00_PROGRESS_STATUS.md` with status 🟡 In Progress
2. Add start timestamp
3. Update "Current Session" section with your active task

When you complete tasks:
1. Check off completed items in the "Handoff 06" section
2. Add artifacts to "Artifacts Created"
3. Note any issues in "Issues/Notes"

When you finish or need to hand off:
1. Update status to ✅ Completed (or ⚠️ Blocked if issues)
2. Add a "Session Handoff" entry with what was done and next steps
3. Update the overall status table

---

## Part A: Hodge Decomposition Example with Dangerous Situation

### Scenario: Medical Triage AI

An AI assistant helps prioritize patients. Three patients with conditions:
- **A**: Stable but needs monitoring
- **B**: Moderate, needs treatment soon
- **C**: Critical, needs immediate attention

### The Preference Cycle (Danger!)

Different stakeholders give conflicting preferences:
- **Doctor**: C > B > A (severity-based)
- **Administrator**: A > C > B (resource efficiency) 
- **Family of A**: A > B > C (obvious bias)

Aggregated: Forms a Condorcet cycle where no clear winner exists.

### Hodge Decomposition

```
Preference Graph:
    A ----0.3----> B
    ^              |
    |              v
   -0.2           0.5
    |              |
    C <----0.4---- 

Edge weights = preference strengths
```

**Decomposition**:
```python
# Gradient component (dV): What a scalar reward would learn
V = {"A": 0.1, "B": 0.3, "C": 0.0}
gradient = {
    "A->B": V["B"] - V["A"],  # 0.2
    "B->C": V["C"] - V["B"],  # -0.3
    "C->A": V["A"] - V["C"],  # 0.1
}

# Harmonic component (ω): The irreducible cycle
# Sum around cycle: 0.3 + 0.5 + 0.4 - 0.2 = 1.0 (non-zero!)
harmonic = 1.0 / 3  # Distributed equally = 0.33 per edge

# Original = gradient + harmonic
# A->B: 0.2 + 0.33 ≈ 0.53 ✗ (actual: 0.3)
# Need proper Hodge computation...
```

### Proper Computation (for paper)

```python
import numpy as np
from scipy.linalg import lstsq

# Incidence matrix B (edges x vertices)
# Edges: A->B, B->C, C->A
B = np.array([
    [-1,  1,  0],  # A->B
    [ 0, -1,  1],  # B->C
    [ 1,  0, -1],  # C->A
])

# Observed preferences (1-cochain)
r = np.array([0.3, 0.5, -0.2])

# Hodge decomposition: r = B @ V + harmonic
# Least squares for gradient: V = (B^T B)^{-1} B^T r
L = B.T @ B  # Graph Laplacian
V, _, _, _ = lstsq(L, B.T @ r)

# Gradient component
gradient = B @ V

# Harmonic component
harmonic = r - gradient

print(f"Potential V: {V}")
print(f"Gradient: {gradient}")
print(f"Harmonic: {harmonic}")
print(f"H^1 magnitude: {np.linalg.norm(harmonic)}")
```

### Visualization for Paper

Create figure showing:
1. Original preference graph with edge weights
2. Gradient flow (arrows showing "downhill" direction from V)
3. Harmonic circulation (the cycle component)
4. Annotation: "H¹ ≠ 0 means no scalar reward captures these preferences"

### The Danger

If we force a scalar reward:
- AI might oscillate between patients
- Or collapse to always choosing one (mode collapse)
- Or exploit the cycle to game the system

**SGPO Solution**: Learn both V and ω, navigate the cycle explicitly.

---

## Part B: Feedback Decomposition in Common Embedding Space

### Scenario: AI Writing Assistant

An AI helps draft emails. Feedback comes in multiple forms:

| Feedback Type | Example | Embedding |
|---------------|---------|-----------|
| **Verbal** | "Too formal for a friend" | Sentence embedding |
| **Ordinal** | Rating: 3/5 stars | Scalar → vector |
| **Pass/Fail** | "Grammatically correct" ✓ | Binary → vector |

### Embedding Strategy

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_verbal_feedback(text: str) -> np.ndarray:
    """Direct sentence embedding."""
    return encoder.encode(text)

def embed_ordinal_feedback(rating: int, max_rating: int = 5) -> np.ndarray:
    """Map ordinal to embedding space via anchor texts."""
    anchors = [
        "This is terrible, completely unacceptable.",
        "This is poor, needs significant improvement.",
        "This is acceptable but mediocre.",
        "This is good, minor improvements possible.",
        "This is excellent, no changes needed.",
    ]
    # Interpolate between anchor embeddings
    anchor_embs = encoder.encode(anchors)
    idx = rating - 1
    return anchor_embs[idx]

def embed_passfail_feedback(criterion: str, passed: bool) -> np.ndarray:
    """Map pass/fail to embedding with criterion context."""
    if passed:
        text = f"Satisfies criterion: {criterion}"
    else:
        text = f"Fails criterion: {criterion}"
    return encoder.encode(text)
```

### Combined Preference Vector

```python
def compute_preference_vector(
    state_emb: np.ndarray,
    chosen_emb: np.ndarray,
    rejected_emb: np.ndarray,
    verbal_feedback: str = None,
    ordinal_rating: int = None,
    passfail: dict = None,  # {"criterion": str, "passed": bool}
    weights: dict = None,
) -> np.ndarray:
    """
    Compute weighted preference direction in embedding space.
    
    This vector points from rejected toward chosen, 
    modulated by additional feedback signals.
    """
    weights = weights or {"base": 1.0, "verbal": 0.5, "ordinal": 0.3, "passfail": 0.2}
    
    # Base preference direction
    base_pref = chosen_emb - rejected_emb
    pref_vector = weights["base"] * base_pref
    
    # Add verbal feedback influence
    if verbal_feedback:
        verbal_emb = embed_verbal_feedback(verbal_feedback)
        # Project verbal onto preference direction
        verbal_component = np.dot(verbal_emb, base_pref) / (np.linalg.norm(base_pref) + 1e-8)
        pref_vector += weights["verbal"] * verbal_component * (base_pref / np.linalg.norm(base_pref))
    
    # Add ordinal rating influence
    if ordinal_rating:
        ordinal_emb = embed_ordinal_feedback(ordinal_rating)
        # Scale preference by rating alignment
        rating_scale = (ordinal_rating - 3) / 2  # -1 to +1
        pref_vector += weights["ordinal"] * rating_scale * base_pref
    
    # Add pass/fail criteria
    if passfail:
        pf_emb = embed_passfail_feedback(passfail["criterion"], passfail["passed"])
        pf_scale = 1.0 if passfail["passed"] else -0.5
        pref_vector += weights["passfail"] * pf_scale * (pf_emb - state_emb)
    
    return pref_vector
```

### Hodge Decomposition on Combined Feedback

```python
def decompose_feedback_field(
    states: List[np.ndarray],
    preference_vectors: List[np.ndarray],
    k_neighbors: int = 5,
) -> dict:
    """
    Apply Hodge decomposition to preference vector field.
    
    Returns gradient (learnable) and harmonic (cyclic) components.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(states)
    
    # Construct edge list and preference 1-cochain
    edges = []
    edge_weights = []
    
    for i, state in enumerate(states):
        neighbors = nn.kneighbors([state], return_distance=False)[0]
        for j in neighbors:
            if i != j:
                edges.append((i, j))
                # Project preference onto edge direction
                edge_dir = states[j] - states[i]
                weight = np.dot(preference_vectors[i], edge_dir)
                edge_weights.append(weight)
    
    # Hodge decomposition (simplified)
    # ... (use proper graph Laplacian computation)
    
    return {
        "gradient": gradient_component,
        "harmonic": harmonic_component,
        "h1_magnitude": np.linalg.norm(harmonic_component),
    }
```

### Visualization

Create figure showing:
1. 2D PCA of embedding space with states as points
2. Preference vectors as arrows
3. Color-coded by feedback type contribution
4. Overlay: gradient field vs harmonic circulation

---

## Part C: Agent Simulations on Ethical Scenarios

### Scenario 1: Academic Integrity (Cheating Detection)

**Setup**: Student asks AI for help with homework. AI must decide:
- **A**: Provide full solution (cheating)
- **B**: Give hints only (helpful)
- **C**: Refuse entirely (unhelpful but safe)

**Preference Cycle**:
- Student prefers: A > B > C
- Teacher prefers: C > B > A
- AI alignment goal: B > C > A

**Simulation Code**:

```python
class AcademicIntegrityEnv:
    """Simulated environment for academic integrity decisions."""
    
    def __init__(self):
        self.states = ["question_asked", "hint_given", "solution_given", "refused"]
        self.actions = ["give_hint", "give_solution", "refuse", "ask_clarification"]
        
        # Preference matrices from different stakeholders
        self.student_prefs = {
            ("give_solution", "give_hint"): 0.8,
            ("give_hint", "refuse"): 0.6,
            ("give_solution", "refuse"): 0.9,
        }
        self.teacher_prefs = {
            ("refuse", "give_hint"): 0.3,
            ("give_hint", "give_solution"): 0.9,
            ("refuse", "give_solution"): 0.95,
        }
        
    def get_aggregated_preferences(self, weights={"student": 0.3, "teacher": 0.7}):
        """Aggregate preferences - may create cycles!"""
        # ... aggregation logic
        pass
    
    def step(self, action):
        """Execute action, return (next_state, reward_vector, done)."""
        # Reward is a VECTOR, not scalar
        reward_vector = np.array([
            self.student_utility(action),
            self.teacher_utility(action),
            self.learning_utility(action),
        ])
        return next_state, reward_vector, done
```

### Scenario 2: Military Drone Decision

**Setup**: Autonomous drone must decide on engagement:
- **A**: Engage target (potential civilian risk)
- **B**: Request human confirmation (delay)
- **C**: Abort mission (safe but mission failure)

**Black Hole**: Region where civilian casualties are certain = geometric singularity.

```python
class DroneDecisionEnv:
    """Military drone ethical decision environment."""
    
    def __init__(self):
        # State includes: target_confidence, civilian_proximity, time_pressure
        self.state_dim = 3
        
        # Black hole regions (forbidden)
        self.black_holes = [
            {"center": np.array([0.9, 0.9, 0.1]),  # High confidence, high civilian, low time
             "radius": 0.2,
             "description": "Certain civilian casualties"},
        ]
        
    def compute_metric(self, state):
        """Riemannian metric with singularities at black holes."""
        g = 1.0  # Base metric
        
        for bh in self.black_holes:
            dist = np.linalg.norm(state - bh["center"])
            if dist < bh["radius"] * 3:
                # Metric diverges near black hole
                g += bh["strength"] / (dist ** 2 + 1e-6)
        
        return g
    
    def geodesic_reward(self, state, action, next_state):
        """Reward adjusted by metric - dangerous paths are expensive."""
        base_reward = self.mission_reward(action)
        path_cost = self.compute_metric((state + next_state) / 2)
        return base_reward / np.sqrt(path_cost)
```

### Scenario 3: Profit Maximization Ethics

**Setup**: AI business advisor must recommend:
- **A**: Aggressive tactics (high profit, ethical concerns)
- **B**: Standard practices (moderate profit, ethical)
- **C**: Conservative approach (low profit, very safe)

**Condorcet Cycle**: Shareholders vs. Employees vs. Regulators

```python
class BusinessEthicsEnv:
    """Business decision environment with stakeholder conflicts."""
    
    def __init__(self):
        self.stakeholders = {
            "shareholders": {"weight": 0.4, "preferences": {...}},
            "employees": {"weight": 0.3, "preferences": {...}},
            "regulators": {"weight": 0.3, "preferences": {...}},
        }
        
    def detect_condorcet_cycles(self):
        """Use Hodge decomposition to find preference cycles."""
        # Aggregate preferences
        pref_graph = self.build_preference_graph()
        
        # Compute H^1
        h1 = compute_cohomology(pref_graph)
        
        if np.linalg.norm(h1) > 0.1:
            print(f"WARNING: Condorcet cycle detected! H^1 = {h1}")
            print("No scalar reward can satisfy all stakeholders.")
        
        return h1
```

---

## Part D: Full Loop - Chat Agent with SGPO

### Architecture

```
User Query → State Embedding → Policy (SGPO) → Action Selection → Response Generation
                 ↓
            Hodge Critic ← Human Feedback (verbal + ordinal + pass/fail)
                 ↓
            Topology Mining → Black Hole Detection → Metric Update
```

### Implementation Sketch

```python
class SGPOChatAgent:
    """Chat agent using Sheaf-Geodesic Policy Optimization."""
    
    def __init__(self, llm, hodge_critic, metric_model):
        self.llm = llm
        self.hodge_critic = hodge_critic
        self.metric_model = metric_model
        self.conversation_state = []
        
    def encode_state(self, conversation):
        """Encode conversation history as state embedding."""
        text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        return self.encoder.encode(text)
    
    def get_action_candidates(self, state, n=5):
        """Generate candidate responses."""
        candidates = []
        for _ in range(n):
            response = self.llm.generate(self.conversation_state, temperature=0.8)
            candidates.append(response)
        return candidates
    
    def select_action(self, state, candidates):
        """Select best action using SGPO policy."""
        state_emb = self.encode_state(state)
        
        # Score each candidate
        scores = []
        for candidate in candidates:
            cand_emb = self.encoder.encode(candidate)
            
            # Hodge-corrected value
            value = self.hodge_critic.value(cand_emb)
            harmonic = self.hodge_critic.harmonic(state_emb, cand_emb)
            
            # Metric scaling (avoid black holes)
            metric = self.metric_model(cand_emb)
            
            # Geodesic advantage
            advantage = (value - harmonic) / np.sqrt(metric + 1e-6)
            scores.append(advantage)
        
        # Select highest scoring candidate
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def update_from_feedback(self, state, action, feedback):
        """Update models from human feedback."""
        # Decompose feedback
        pref_vector = compute_preference_vector(
            state_emb=self.encode_state(state),
            chosen_emb=self.encoder.encode(action),
            rejected_emb=None,  # Could compare to alternatives
            verbal_feedback=feedback.get("verbal"),
            ordinal_rating=feedback.get("rating"),
            passfail=feedback.get("criteria"),
        )
        
        # Update Hodge critic
        self.hodge_critic.update(state, action, pref_vector)
        
        # Update metric if safety-relevant
        if feedback.get("safety_flag"):
            self.metric_model.add_danger_sample(self.encoder.encode(action))
```

---

## Deliverables

1. **Paper content**: 
   - Example A → New figure + 1 paragraph in Method section
   - Example B → Extended discussion in experiments or appendix
   - Example C → Qualitative results subsection

2. **Code**:
   - `src/examples/medical_triage.py`
   - `src/examples/feedback_decomposition.py`
   - `src/examples/ethical_scenarios.py`
   - `src/gpo_chat_agent.py`

3. **Figures**:
   - Hodge decomposition visualization
   - Feedback embedding space
   - Black hole avoidance trajectories

---

## Verification Checklist

- [ ] Medical triage example computes correct Hodge decomposition
- [ ] Feedback decomposition produces meaningful embeddings
- [ ] Ethical scenario environments run without errors
- [ ] SGPO chat agent integrates all components
- [ ] Figures are publication-quality
- [ ] **Progress tracking**: Updated `00_PROGRESS_STATUS.md` with completion status
