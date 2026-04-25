# HodgeRank Computational Tractability Analysis

## Problem Statement

HodgeRank for preference filtering in RLHF training data requires:
1. Building a preference graph over all responses
2. Computing Hodge decomposition (gradient + curl + harmonic)
3. Filtering preferences based on component magnitudes

**Current bottleneck**: For N preference pairs with ~2N unique responses, we construct an O(N) x O(N) adjacency matrix and compute its Hodge decomposition.

## Complexity Analysis

### Graph Construction
- **Embedding responses**: O(N) calls to sentence-transformer
  - Each embed: ~10ms on GPU → 5000 prefs = 50 seconds
- **Building adjacency**: O(N) iterations → negligible

### Hodge Decomposition
The Hodge decomposition on a graph requires solving:
```
L = d₀ᵀ d₀ + d₁ d₁ᵀ  (Hodge Laplacian)
```

Where:
- **d₀**: Gradient operator (V × E matrix)
- **d₁**: Curl operator (E × F matrix for triangles)

**Key operations**:
1. **Graph Laplacian**: O(V²) storage, O(V³) for eigendecomposition
2. **Helmholtz decomposition**: Solving L⁺ (pseudoinverse) → O(V³)

| Preferences | Unique Responses | Matrix Size | Eigendecomp Time |
|-------------|------------------|-------------|------------------|
| 1,000       | ~2,000           | 2K × 2K     | ~1 second        |
| 5,000       | ~10,000          | 10K × 10K   | ~15 minutes      |
| 50,000      | ~100,000         | 100K × 100K | ~25 hours        |
| 161,000 (HH-RLHF full) | ~300K | 300K × 300K | **Intractable** |

## Current Implementation Issue

Our current implementation computes the **full Hodge decomposition** on the entire preference graph:

```python
# O(N²) matrix
adjacency = np.zeros((n_nodes, n_nodes))

# O(N³) eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
```

For 5000 preferences → 10K × 10K matrix → **~15 min computation time** on A100.

## Proposed Optimizations

### 1. Sparse Representation (Immediate)
Most preference graphs are sparse (each response appears in few comparisons).

```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Use sparse matrix
adjacency = csr_matrix((n_nodes, n_nodes))

# Compute only top-k eigenvalues needed
eigenvalues, eigenvectors = eigsh(laplacian, k=100)
```

**Speedup**: 10-100x for sparse graphs

### 2. Local Hodge Decomposition (Medium-term)
Instead of global decomposition, compute locally around each preference:

```python
def local_hodge_score(pref, k_neighbors=50):
    # Get k-hop neighborhood around chosen/rejected
    subgraph = get_neighborhood(pref, k=k_neighbors)
    # Decompose small subgraph
    return hodge_decompose(subgraph)
```

**Complexity**: O(k³) per preference → O(N × k³) total
- k=50: 50³ = 125K ops per pref → very fast

### 3. Sampling-Based Estimation (Recommended for Scale)
Estimate reliability via random sampling:

```python
def sampled_hodge_reliability(preferences, n_samples=1000):
    # Sample random triplets
    triplets = sample_triplets(preferences, n_samples)
    
    # Count transitive vs cyclic
    transitive = sum(is_transitive(t) for t in triplets)
    
    # Reliability ≈ transitive ratio
    return transitive / n_samples
```

**Complexity**: O(n_samples) → constant time regardless of dataset size

### 4. Hierarchical Decomposition (Sophisticated)
Cluster responses, compute inter-cluster Hodge, then intra-cluster:

```
Level 0: Full dataset → 100 clusters
Level 1: Each cluster → local Hodge
Combine: Weighted aggregation
```

**Complexity**: O(C³ + N × c³) where C = clusters, c = cluster size

## Theoretical Justification for Local Methods

**Key insight**: The Hodge decomposition is *local* in the following sense:
- Gradient component: Measures local transitivity (A > B > C)
- Curl component: Measures local cycles (A > B > C > A)
- Harmonic component: Measures global inconsistency

For **filtering preferences**, we primarily care about:
1. Is this preference consistent with local structure? (gradient dominance)
2. Is this preference part of a cycle? (curl presence)

Both can be estimated from **k-hop neighborhoods** without full graph decomposition.

## Recommended Approach for Paper

### Pre-training Filter (One-time Cost)
Use **full Hodge decomposition** on training data:
- Acceptable to spend 15-60 min on 5K-50K preferences
- Result: Filtered dataset used for all subsequent training
- Amortized cost is low

### Online/Iterative Setting
Use **sampled reliability estimation**:
- O(1) time per batch
- Approximate but sufficient for gradient-based filtering

### Implementation for Experiments

```python
class ScalableHodgeFilter:
    def __init__(self, method="sparse"):
        self.method = method
    
    def filter(self, preferences, threshold=0.5):
        if self.method == "sparse":
            return self._sparse_hodge(preferences, threshold)
        elif self.method == "local":
            return self._local_hodge(preferences, threshold)
        elif self.method == "sampled":
            return self._sampled_hodge(preferences, threshold)
    
    def _sparse_hodge(self, prefs, threshold):
        # Use scipy.sparse for O(nnz) storage
        # Use eigsh for O(k × nnz) eigendecomp
        pass
    
    def _local_hodge(self, prefs, threshold, k=50):
        # Compute per-preference in k-neighborhood
        pass
    
    def _sampled_hodge(self, prefs, threshold, n_samples=1000):
        # Random triplet sampling for reliability estimation
        pass
```

## Conclusion

| Method | Complexity | Accuracy | Use Case |
|--------|------------|----------|----------|
| Full dense | O(N³) | Exact | Small datasets (<1K) |
| Sparse | O(k × nnz) | Exact | Medium datasets (<50K) |
| Local k-hop | O(N × k³) | Approximate | Large datasets |
| Sampled | O(n_samples) | Approximate | Any size, online |

**For our experiments**:
- 5K preferences: Use **sparse** method (~30 seconds)
- Full HH-RLHF: Use **local** or **sampled** method

**For paper narrative**: Frame Hodge filtering as a **pre-training data curation** step where one-time computational cost is acceptable, then discuss scalable approximations for production use.
