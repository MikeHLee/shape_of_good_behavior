# -*- coding: utf-8 -*-
"""
Hodge decomposition on a 2-dimensional clique complex (SGB-045 Stage 0).

The older ``hodge_utils.hodge_decompose`` has no 2-cells: its "harmonic" part is the
full residual ``f - d0 phi`` (curl + harmonic together). This module adds 2-cells so the
residual splits into a LOCAL part (curl: inconsistency inside filled triangles) and a
GLOBAL part (harmonic: inconsistency around loops that no triangle fills).

Complex
-------
- 0-cells: items 0..n-1.
- 1-cells: compared pairs, stored canonically as (i, j) with i < j.
  The flow f[e] on edge (i, j) is read with the same convention as ``hodge_utils``:
  positive f means j is preferred over i, and a pure gradient flow is f = phi[j] - phi[i].
- 2-cells: every triangle (i, j, k), i < j < k, whose three edges all exist (clique complex).

Operators
---------
- d0 (E x n):  (d0 phi)[e=(i,j)] = phi[j] - phi[i]            (row: -1 at i, +1 at j)
- d1 (T x E):  (d1 f)[t=(i,j,k)] = f[ij] + f[jk] - f[ik]     (the circulation i->j->k->i)
- d1 @ d0 == 0 exactly.

Weighted decomposition
----------------------
Edge weights w > 0 (for example, the number of labels on an edge) define the inner
product <x, y>_W = sum_e w_e x_e y_e on 1-cochains. Triangles get unit weight. Then

    f = d0 phi  +  W^{-1} d1^T psi  +  h

with
- gradient  = d0 phi,               phi = argmin ||f - d0 phi||_W
- curl      = W^{-1} d1^T psi,      the W-orthogonal projection of f onto im(W^{-1} d1^T)
- harmonic  = remainder; it satisfies d1 h = 0 and d0^T W h = 0.

The three parts are mutually orthogonal in <.,.>_W (d1 d0 = 0 gives <d0 phi, W^{-1} d1^T psi>_W
= phi^T d0^T d1^T psi = 0). W^{-1} d1^T is the W-adjoint of d1 (Jiang, Lim, Yao, Ye 2011,
"Statistical ranking and combinatorial Hodge theory"). With uniform weights this is the
usual unweighted decomposition f = d0 phi + d1^T psi + h.

dim(harmonic space) = beta_1 = E - rank(d0) - rank(d1) (first Betti number of the complex).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CliqueComplex",
    "HodgeResult",
    "canonicalize_flow",
    "build_clique_complex",
    "hodge_decompose_2d",
]


# ---------------------------------------------------------------------------
# Complex construction
# ---------------------------------------------------------------------------

def canonicalize_flow(
    edges: Sequence[Tuple[int, int]], flow: Sequence[float]
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Orient every edge as (min, max) and flip the sign of the flow where needed.

    Raises on self-loops and on duplicate pairs (aggregate duplicates before calling).
    """
    flow = np.asarray(flow, dtype=float)
    if len(edges) != len(flow):
        raise ValueError(f"edges ({len(edges)}) and flow ({len(flow)}) differ in length")
    out_e: List[Tuple[int, int]] = []
    out_f = np.empty(len(flow))
    seen = set()
    for idx, ((a, b), v) in enumerate(zip(edges, flow)):
        a, b = int(a), int(b)
        if a == b:
            raise ValueError(f"self-loop at edge {idx}: ({a},{b})")
        i, j, s = (a, b, 1.0) if a < b else (b, a, -1.0)
        if (i, j) in seen:
            raise ValueError(f"duplicate pair ({i},{j}); aggregate labels first")
        seen.add((i, j))
        out_e.append((i, j))
        out_f[idx] = s * v
    return out_e, out_f


@dataclass
class CliqueComplex:
    n_nodes: int
    edges: np.ndarray            # (E, 2) int, i < j, in the caller's order
    triangles: np.ndarray        # (T, 3) int, i < j < k, lexicographic order
    edge_index: Dict[Tuple[int, int], int]
    d0: np.ndarray               # (E, n)
    d1: np.ndarray               # (T, E)

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])


def build_clique_complex(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    fill_triangles: bool = True,
) -> CliqueComplex:
    """Build the 2-skeleton of the clique complex of a graph.

    ``edges`` must be canonical (i < j) and unique (use ``canonicalize_flow``). Edge order
    is preserved: row e of d0 and column e of d1 refer to ``edges[e]``.
    ``fill_triangles=False`` gives the bare graph (no 2-cells), which reproduces the old
    residual-only decomposition (curl = 0, harmonic = full residual).
    """
    E = len(edges)
    edge_arr = np.asarray(edges, dtype=int).reshape(E, 2) if E else np.zeros((0, 2), int)
    edge_index: Dict[Tuple[int, int], int] = {}
    for e, (i, j) in enumerate(edge_arr):
        i, j = int(i), int(j)
        if not (0 <= i < j < n_nodes):
            raise ValueError(f"edge {e}=({i},{j}) is not canonical i<j within [0,{n_nodes})")
        if (i, j) in edge_index:
            raise ValueError(f"duplicate edge ({i},{j})")
        edge_index[(i, j)] = e

    d0 = np.zeros((E, n_nodes))
    if E:
        d0[np.arange(E), edge_arr[:, 0]] = -1.0
        d0[np.arange(E), edge_arr[:, 1]] = 1.0

    tris: List[Tuple[int, int, int]] = []
    if fill_triangles and E:
        nbrs: List[set] = [set() for _ in range(n_nodes)]
        for i, j in edge_arr:
            nbrs[int(i)].add(int(j))
            nbrs[int(j)].add(int(i))
        for (i, j) in sorted(edge_index):
            for k in sorted(nbrs[i] & nbrs[j]):
                if k > j:
                    tris.append((i, j, k))
    T = len(tris)
    tri_arr = np.asarray(tris, dtype=int).reshape(T, 3) if T else np.zeros((0, 3), int)
    d1 = np.zeros((T, E))
    for t, (i, j, k) in enumerate(tris):
        d1[t, edge_index[(i, j)]] = 1.0
        d1[t, edge_index[(j, k)]] = 1.0
        d1[t, edge_index[(i, k)]] = -1.0
    return CliqueComplex(n_nodes, edge_arr, tri_arr, edge_index, d0, d1)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

@dataclass
class HodgeResult:
    potential: np.ndarray        # phi, (n,), mean-zero on every connected component
    gradient: np.ndarray         # d0 phi, (E,)
    curl: np.ndarray             # per-edge curl component W^{-1} d1^T psi, (E,)
    harmonic: np.ndarray         # per-edge harmonic component, (E,)
    psi: np.ndarray              # triangle potential, (T,)
    triangle_curl: np.ndarray    # (d1 f)[t]: circulation of the observed flow on triangle t
    weights: np.ndarray          # (E,)
    norms: Dict[str, float]      # W-norms: total, gradient, curl, harmonic
    fractions: Dict[str, float]  # squared-norm fractions of the total (sum to 1)
    betti1: int                  # dim of the harmonic space
    rank_d0: int
    rank_d1: int
    extra: Dict = field(default_factory=dict)

    @property
    def residual(self) -> np.ndarray:
        return self.curl + self.harmonic


def _wnorm(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w * x * x)))


def hodge_decompose_2d(
    cx: CliqueComplex,
    flow: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    compute_betti: bool = True,
    rcond: float = 1e-10,
) -> HodgeResult:
    """Weighted least-squares Hodge decomposition f = d0 phi + W^{-1} d1^T psi + h."""
    f = np.asarray(flow, dtype=float)
    E, T, n = cx.n_edges, cx.n_triangles, cx.n_nodes
    if f.shape != (E,):
        raise ValueError(f"flow has shape {f.shape}, complex has {E} edges")
    w = np.ones(E) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (E,):
        raise ValueError(f"weights have shape {w.shape}, complex has {E} edges")
    if E and np.any(w <= 0):
        raise ValueError("edge weights must be strictly positive")

    if E == 0:
        z = np.zeros(0)
        return HodgeResult(np.zeros(n), z, z, z, np.zeros(T), np.zeros(T), w,
                           dict(total=0.0, gradient=0.0, curl=0.0, harmonic=0.0),
                           dict(gradient=0.0, curl=0.0, harmonic=0.0), 0, 0, 0)

    sw = np.sqrt(w)
    # Gradient: min ||W^{1/2}(d0 phi - f)||. lstsq gives the min-norm solution, which is
    # orthogonal to the kernel of d0 (component indicators), so phi sums to zero on each
    # connected component and is zero on isolated nodes.
    phi, *_ = np.linalg.lstsq(sw[:, None] * cx.d0, sw * f, rcond=rcond)
    grad = cx.d0 @ phi

    # Curl: W-orthogonal projection onto im(W^{-1} d1^T). In scaled coordinates
    # g = W^{1/2} f, the target subspace is im(A) with A = W^{-1/2} d1^T.
    if T:
        A = cx.d1.T / sw[:, None]
        psi, *_ = np.linalg.lstsq(A, sw * f, rcond=rcond)
        curl = (A @ psi) / sw
        tri_curl = cx.d1 @ f
    else:
        psi = np.zeros(0)
        curl = np.zeros(E)
        tri_curl = np.zeros(0)
    harm = f - grad - curl

    tot = _wnorm(f, w)
    norms = dict(total=tot, gradient=_wnorm(grad, w), curl=_wnorm(curl, w),
                 harmonic=_wnorm(harm, w))
    if tot > 0:
        fr = {k: norms[k] ** 2 / tot ** 2 for k in ("gradient", "curl", "harmonic")}
    else:
        fr = dict(gradient=0.0, curl=0.0, harmonic=0.0)

    if compute_betti:
        r0 = int(np.linalg.matrix_rank(cx.d0))
        r1 = int(np.linalg.matrix_rank(cx.d1)) if T else 0
        b1 = E - r0 - r1
    else:
        r0 = r1 = b1 = -1
    return HodgeResult(phi, grad, curl, harm, psi, tri_curl, w, norms, fr, b1, r0, r1)
