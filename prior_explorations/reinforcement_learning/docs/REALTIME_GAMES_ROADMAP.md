# Roadmap: Real-Time Games with LLM State Representations

**Status**: Future Research Direction  
**Prerequisites**: Semantic MDP framework, streaming LLM inference, context compression

---

## 1. Motivation

The Semantic MDP framework (natural language states, MCP actions, Hodge rewards) works well for **turn-based** domains:
- Text adventures
- Chess/Go with verbal constraints
- Coding tasks
- Alignment scenarios

However, extending to **real-time** games (shooters, racing, Atari) requires solving fundamental challenges around **continuous perception** and **low-latency action**.

---

## 2. The Core Challenge

### Turn-Based vs Real-Time

| Aspect | Turn-Based | Real-Time |
|--------|-----------|-----------|
| **State update** | Discrete events | 30-60+ FPS continuous |
| **Action latency** | Seconds acceptable | <100ms required |
| **State description** | Full scene description | Must compress to essentials |
| **Decision frequency** | Per-turn | Per-frame or sub-second |

### The Streaming Problem

For a real-time game, we need:
1. **Sensor stream** → Language: Convert visual/audio input to text at 30+ FPS
2. **Language** → Embedding: Embed state descriptions in real-time
3. **Embedding** → Action: Policy inference <50ms
4. **Action** → Actuator: Execute controls

The bottleneck is **LLM inference latency** for state description and policy.

---

## 3. Proposed Architecture

### 3.1 Hierarchical State Compression

```
Raw Pixels (1920x1080x3 @ 60fps)
    ↓ [CNN Feature Extractor - 5ms]
Spatial Features (64x64x256)
    ↓ [Object Detection - 10ms]
Object List + Positions
    ↓ [Change Detection - 1ms]
Delta Description (only what changed)
    ↓ [LLM Summary - 50ms, async]
Compressed Semantic State
```

**Key insight**: Only describe **changes** between frames, not full scenes.

### 3.2 Dual-Track Processing

```
Track 1: Low-Latency Reactive
  - CNN → Small policy network → Action
  - Latency: ~20ms
  - Handles: Immediate threats, reflexes

Track 2: High-Level Strategic  
  - Accumulated state → LLM reasoning → Strategic goals
  - Latency: ~500ms
  - Handles: Planning, resource allocation, positioning
```

This mirrors biological systems (reflexes vs deliberate thought).

### 3.3 Context Compression Strategies

| Strategy | Description | Compression | Latency |
|----------|-------------|-------------|---------|
| **Keyframe + Delta** | Full description every N frames, deltas between | 10-50x | Low |
| **Attention-based** | Only describe objects in attention window | 5-20x | Medium |
| **Importance sampling** | Describe based on relevance to current goal | Variable | Medium |
| **Learned compression** | Train encoder to produce minimal descriptions | 50-100x | High (training) |

### 3.4 Streaming LLM Integration

For state-to-language conversion, we need **streaming SSM/LLM** with:

1. **Speculative decoding**: Pre-generate likely descriptions
2. **KV-cache persistence**: Maintain context across frames
3. **Early exit**: Stop generation when sufficient information extracted
4. **Quantized inference**: INT8/INT4 models for speed

**Target architecture**: Mamba-based SSM with ~1B parameters
- Linear complexity in sequence length
- Can process continuous streams
- Maintains state without explicit attention

---

## 4. Target Domains

### 4.1 Atari Games (Starting Point)

**Why**: Well-studied, discrete actions, interpretable states

| Game | State Complexity | Action Space | Challenge |
|------|-----------------|--------------|-----------|
| Pong | Low | 3 actions | Baseline |
| Breakout | Medium | 4 actions | Spatial reasoning |
| Space Invaders | Medium | 6 actions | Threat prioritization |
| Montezuma | High | 18 actions | Long-term planning |

**Language representation example** (Pong):
```
Frame 1: "Ball moving right toward paddle. Paddle centered."
Frame 2: "Ball approaching. Paddle should move up."
Delta: "Ball closer. Move up."
```

### 4.2 Racing Games

**Why**: Continuous control, spatial reasoning, trajectory planning

**State elements**:
- Track geometry (upcoming turns)
- Vehicle position/velocity
- Opponents (relative positions)
- Resources (boost, health)

**Language representation**:
```
"Sharp left turn in 50m. Current speed 120km/h, optimal 80km/h.
Opponent close behind. Boost available.
Action: Brake now, apex mid-turn, accelerate exit."
```

### 4.3 First-Person Shooters

**Why**: Complex state, strategic depth, multi-objective

**State elements**:
- Spatial map (known/unknown areas)
- Enemy positions (seen/heard/predicted)
- Resources (ammo, health, objectives)
- Team coordination (if applicable)

**Challenges**:
- 3D spatial reasoning in language
- Temporal prediction (where will enemies be?)
- Multi-objective balancing

### 4.4 Real-Time Strategy (RTS)

**Why**: Maximum complexity, planning + micro

**State elements**:
- Base layout
- Unit positions (many)
- Resource economy
- Tech tree progress
- Enemy intelligence

**This is the hardest domain** - requires both:
- Micro (unit control, <100ms)
- Macro (strategy, multi-second)

---

## 5. Technical Requirements

### 5.1 LLM/SSM Streaming Pipeline

```python
class RealtimeStateEncoder:
    """
    Converts raw observations to semantic states at real-time speeds.
    """
    def __init__(self, ssm_model, vision_encoder, context_window=1000):
        self.ssm = ssm_model  # Mamba or similar
        self.vision = vision_encoder  # Lightweight CNN
        self.context = RingBuffer(context_window)
        self.last_description = ""
    
    async def encode_frame(self, frame: np.ndarray) -> str:
        # Fast visual encoding
        features = self.vision(frame)
        objects = self.detect_objects(features)
        
        # Delta from last frame
        delta = self.compute_delta(objects)
        
        if delta.is_significant():
            # Generate new description (async, don't block)
            description = await self.ssm.generate_delta(
                context=self.context,
                delta=delta,
                max_tokens=50,
            )
            self.last_description = description
            self.context.append(description)
        
        return self.last_description
```

### 5.2 Context Compression

```python
class ContextCompressor:
    """
    Maintains compressed context for long episodes.
    """
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        self.importance_model = ImportanceScorer()
    
    def compress(self, context: List[str]) -> str:
        # Score each context item by relevance to current state
        scores = self.importance_model.score(context)
        
        # Keep most important, summarize rest
        important = [c for c, s in zip(context, scores) if s > 0.5]
        summarized = self.summarize([c for c, s in zip(context, scores) if s <= 0.5])
        
        return self.merge(summarized, important)
```

### 5.3 Latency Budget

| Component | Target Latency | Notes |
|-----------|---------------|-------|
| Frame capture | 1ms | Hardware |
| Vision encoding | 5ms | MobileNet-scale CNN |
| Object detection | 10ms | YOLO-nano |
| Delta computation | 1ms | CPU |
| SSM inference | 30ms | Mamba-1B, INT8 |
| Policy inference | 5ms | Small MLP |
| Action execution | 1ms | Hardware |
| **Total** | **~53ms** | 18 FPS minimum |

For 60 FPS games, we need to pipeline and run async.

---

## 6. Research Questions

### 6.1 Fundamental

1. **Can language capture real-time dynamics?**
   - Is there information loss that prevents optimal play?
   - What's the minimum description rate for different games?

2. **Does Hodge decomposition work for continuous time?**
   - Current formulation is for discrete transitions
   - Need continuous-time extension

3. **How do we define "black holes" in real-time?**
   - States that lead to death/loss
   - Time-dependent (a state may be safe now, dangerous later)

### 6.2 Engineering

1. **How to handle variable latency?**
   - LLM inference time varies
   - Need graceful degradation

2. **What's the right abstraction level?**
   - Too detailed: slow, redundant
   - Too abstract: loses crucial information

3. **How to train the compression model?**
   - Need labels for "important" vs "unimportant"
   - May require game-specific tuning

---

## 7. Experimental Plan

### Phase 1: Atari Baseline (Q1 2026)
- [ ] Implement frame-to-language encoder for Pong
- [ ] Measure latency vs description quality tradeoff
- [ ] Compare semantic policy to CNN policy

### Phase 2: Streaming Architecture (Q2 2026)
- [ ] Build dual-track (reactive + strategic) system
- [ ] Implement context compression
- [ ] Test on Breakout, Space Invaders

### Phase 3: Complex Games (Q3-Q4 2026)
- [ ] Racing game (continuous control)
- [ ] Simple FPS (spatial reasoning)
- [ ] Evaluate Hodge reward learning in real-time

### Phase 4: Paper & Release (Q4 2026)
- [ ] Write up results
- [ ] Open-source streaming RL framework
- [ ] Target: NeurIPS 2026 or ICLR 2027

---

## 8. Related Work

### Language-Conditioned RL
- **CLIP-guided RL**: Use CLIP embeddings for reward
- **Language goals**: Natural language task specification
- **LLM planners**: Use LLMs for high-level planning (e.g., SayCan)

### Real-Time Game AI
- **AlphaStar**: StarCraft II, hierarchical policies
- **OpenAI Five**: Dota 2, massive scale
- **MuZero**: Model-based, works on Atari

### Streaming ML
- **Online learning**: Continuous model updates
- **Streaming transformers**: Efficient attention for sequences
- **State space models**: Mamba, S4 for long sequences

---

## 9. Connection to Semantic MDP Framework

The real-time extension preserves our core principles:

| Principle | Turn-Based | Real-Time Extension |
|-----------|-----------|---------------------|
| **Semantic states** | Full scene description | Compressed delta descriptions |
| **MCP actions** | Discrete tool calls | Continuous control + discrete commands |
| **Hodge rewards** | Per-transition feedback | Continuous reward signal + periodic feedback |
| **Black holes** | Forbidden states | Forbidden trajectories (time-dependent) |
| **SGPO** | Geodesic optimization | Continuous-time geodesics |

The key insight: **Real-time is not fundamentally different**, just requires:
1. Faster inference (compression, quantization)
2. Asynchronous processing (don't block on LLM)
3. Continuous-time formulation (extend Hodge theory)

---

## 10. Conclusion

Extending Semantic MDP to real-time games is a significant research challenge but achievable with:
1. **Hierarchical state compression** (only describe changes)
2. **Dual-track processing** (fast reflexes + slow reasoning)
3. **Streaming SSM inference** (Mamba-style models)
4. **Continuous-time Hodge theory** (future theoretical work)

This represents a natural evolution of the framework and a compelling future research direction.

---

*Last updated: January 2026*
