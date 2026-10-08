#!/usr/bin/env python3
"""SGB-046: Hodge structure as a reward-model diagnostic for heterogeneous grader panels.

Implements the pre-registration ``feedback_geometry/SGB046_PREREGISTRATION.md``
(SHA-256 5e19fdf1572ab3ce9a6c68638ca625a96f1ab7d318f243789c13362976e8a550). The script checks
that hash at start-up and refuses to run if the file changed.

Simulation only. CPU only. No network. No API calls.

World (per seed, scenario, P)
-----------------------------
- 300 prompt blocks (200 train, 100 held-out). Each block has k = 5 responses and all 10 pairs.
- Each response i has a feature vector x_i ~ N(0, I_d), d = P + 2 (two nuisance dimensions).
  Principle utilities are linear in the features: u_p(i) = A_p . x_i. So a reward model must
  generalize across blocks through the shared features.
- Grader p has temperature T_p ~ U(0.5, 1.0). True log-odds that j beats i: eta = (u_p(j)-u_p(i))/T_p.
- Each pair (i, j), i < j, is judged in both presentation orders with V votes per grader per order
  and a first-position bias beta (0.3 log-odds). Order 1 shows i first, order 2 shows j first.
  Graded label per order: l = logit((k + 0.5) / (V + 1)), k = votes for the first-shown response.
  Antisymmetrized label (Hodge sign convention, + = j preferred): f = (l2 - l1) / 2.
- Aggregators: 'wsum' = sum_p w_p f_p (transitive control); 'wsum_bin' = sign of that (binarized
  control); 'wmaj' = weighted principle vote sum_p v_p sign(f_p) / sum v (constitution vote; tiny
  fixed tie-break weights so the true vote never ties).
- Ground truth from noise-free utilities: wsum -> sign(sum_p w_p eta_p); wmaj -> sign(sum_p v_p
  sign(Delta_p)). True directed 3-cycles from that relation. True cycle-causing tension of a
  principle pair (p, q) = |L_p U L_q| averaged over blocks, where L_p is the set of true cycles that
  disappear when principle p is removed (leave-one-out on the true utilities).

D1 cycle test (per block)
-------------------------
Statistic: number of directed 3-cycles in the observed aggregate relation (10 triangles of K5).
Null: vote-level parametric bootstrap (B = 200) from a transitive-consistent block model:
  1. Fit the BT potential of the observed aggregate flow by least squares (HodgeRank gradient) and
     take its order sigma (a total order, so the null relation is transitive).
  2. On every edge whose observed aggregate sign disagrees with sigma, flip the sign of all
     principle labels (the aggregate then points along sigma with its observed strength).
  3. Deflate each edge toward a tie: with rho = plug-in probability that the aggregate sign points
     along sigma (normal approximation), set the null probability to Phi(max(Phi^-1(rho) - c, 0)),
     reached by one common log-odds shift of all principles on that edge. c is a winner's-curse
     correction, calibrated on pilot seeds 1000-1004 that are disjoint from the main seeds.
  4. Resample all votes (both orders, observed position effect kept) from these null log-odds,
     re-aggregate, count cycles. p = (1 + #{null count >= observed}) / (B + 1). Flag if p <= 0.05.
A cycle on triangle t is CERTIFIED if the block is flagged and (1 + #{null has the same directed
cycle on t}) / (B + 1) <= 0.05. Common random numbers: one array of uniforms per block drives every
bootstrap of that block (main test and leave-one-out re-tests).

Naive curl test (reported, not a gate): statistic = curl fraction of the observed aggregate flow;
null = bootstrap of the GRADED aggregate around its own gradient projection. It does not model label
thresholding, so it should over-fire on `transitive_binarized` (design note section 1).

D2, D3, D4: see the pre-registration and ``summarise``.

Reproduce:
    ./venv/bin/python3 scripts/sgb046_reward_diagnostic_sim.py            # full run
    ./venv/bin/python3 scripts/sgb046_reward_diagnostic_sim.py --smoke    # 2 seeds
    ./venv/bin/python3 scripts/sgb046_reward_diagnostic_sim.py --rebuild-md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import binom, norm, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feedback_geometry" / "src"))
from hodge_complex import build_clique_complex, hodge_decompose_2d  # noqa: E402

PREREG = ROOT / "feedback_geometry" / "SGB046_PREREGISTRATION.md"
PREREG_SHA256 = "5e19fdf1572ab3ce9a6c68638ca625a96f1ab7d318f243789c13362976e8a550"
OUT_JSON = ROOT / "shared" / "results" / "sgb046_reward_diagnostic_v1.json"
OUT_MD = ROOT / "feedback_geometry" / "SGB046_RESULTS.md"
COMMAND = "./venv/bin/python3 scripts/sgb046_reward_diagnostic_sim.py"

# ---------------------------------------------------------------------------
# Design constants (pre-registration section 3)
# ---------------------------------------------------------------------------
SEEDS = list(range(40))
N_TRAIN, N_HELD = 200, 100
N_BLOCKS = N_TRAIN + N_HELD
K = 5
PS = (4, 6)
VS = (5, 25)
B = 200
ALPHA = 0.05

# Implementation constants (not fixed by the pre-registration; disclosed in the results file)
PILOT_SEEDS = list(range(1000, 1005))
C_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFLATE_MODES = ("reoriented", "all")  # which edges get the deflation (chosen on the pilot)
PILOT_FPR_TARGET = 0.04          # choose the smallest c whose worst pilot FPR is <= this
T_LO, T_HI = 0.5, 1.0             # grader temperatures
G_SCALE = 5.0 / 3.0                # weight of the shared quality factor in every utility
S_SCALE = 5.0                      # weight of the principle-specific trade-off component
HARMLESS_FACTOR = 2.0             # harmless pairs: opposed component = 2 * S_SCALE
NEARTIE_SCALE = 0.05               # utility multiplier in transitive_neartie
BIAS_TWO_ORDER = 0.3              # first-position bias (log-odds) when both orders are collected
BIAS_ONE_ORDER = 1.0              # first-position bias in position_bias
TIEBREAK = 0.001                  # tie-break weight increment for the principle vote
RM_CS = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
N_BOOT_SEEDS = 10000
D2_MIN_GAIN = 0.02

SCENARIOS = ["transitive", "transitive_binarized", "transitive_neartie", "condorcet",
             "harmless_disagreement", "mixed", "position_bias"]
SCEN_IDX = {s: i for i, s in enumerate(SCENARIOS)}
GATE_SCENARIOS = ["transitive", "transitive_binarized", "transitive_neartie"]
D3_SCENARIOS = ["condorcet", "harmless_disagreement", "mixed"]
WSUM_SCENARIOS = ["transitive", "transitive_binarized", "transitive_neartie", "position_bias"]
CFGS = [(P, V) for P in PS for V in VS]

# ---------------------------------------------------------------------------
# Complex and index tables (assert alignment, do not assume it)
# ---------------------------------------------------------------------------
EDGES = list(combinations(range(K), 2))
E = len(EDGES)
EIDX = {e: n for n, e in enumerate(EDGES)}
I_ = np.array([i for i, _ in EDGES])
J_ = np.array([j for _, j in EDGES])
CX = build_clique_complex(K, EDGES)
TRIS = [tuple(int(v) for v in t) for t in CX.triangles]
NT = len(TRIS)
T_IJ = np.array([EIDX[(i, j)] for i, j, k in TRIS])
T_JK = np.array([EIDX[(j, k)] for i, j, k in TRIS])
T_IK = np.array([EIDX[(i, k)] for i, j, k in TRIS])
EDGE_TRI = np.zeros((E, NT), dtype=bool)
for t in range(NT):
    EDGE_TRI[[T_IJ[t], T_JK[t], T_IK[t]], t] = True
PINV_D0 = np.linalg.pinv(CX.d0)                 # (K, E): least-squares potential
PROJ_GRAD = CX.d0 @ PINV_D0                      # (E, E)
_A = CX.d1.T
PROJ_CURL = _A @ np.linalg.pinv(_A)              # (E, E), unit edge weights


def check_alignment() -> dict:
    """Section 4: item, edge, triangle and label indices are aligned. Raises on failure."""
    assert [tuple(int(v) for v in e) for e in CX.edges] == EDGES, "edge order differs from complex"
    assert TRIS == list(combinations(range(K), 3)), "triangle order differs"
    assert E == 10 and NT == 10
    for t, (i, j, k) in enumerate(TRIS):
        row = CX.d1[t]
        assert row[EIDX[(i, j)]] == 1 and row[EIDX[(j, k)]] == 1 and row[EIDX[(i, k)]] == -1
        assert np.count_nonzero(row) == 3
        assert T_IJ[t] == EIDX[(i, j)] and T_JK[t] == EIDX[(j, k)] and T_IK[t] == EIDX[(i, k)]
    for e, (i, j) in enumerate(EDGES):
        assert CX.d0[e, i] == -1 and CX.d0[e, j] == 1
    assert EDGE_TRI.sum(1).tolist() == [3] * E
    rng = np.random.default_rng(12345)
    # tri_orient agrees with the circulation d1 s of a sign flow
    s = rng.choice([-1.0, 1.0], size=(2000, E))
    circ = s @ CX.d1.T
    ori = tri_orient(s)
    assert np.array_equal(ori != 0, np.abs(circ) == 3)
    assert np.array_equal(np.sign(circ)[ori != 0], ori[ori != 0])
    # vectorized projector == hodge_decompose_2d
    max_err = 0.0
    for _ in range(50):
        f = rng.standard_normal(E)
        h = hodge_decompose_2d(CX, f)
        max_err = max(max_err, np.abs(h.curl - PROJ_CURL @ f).max(),
                      np.abs(h.gradient - PROJ_GRAD @ f).max(), np.abs(h.harmonic).max())
    assert max_err < 1e-9, max_err
    # K5 pure gradient has zero curl
    phi = rng.standard_normal(K)
    assert np.abs(PROJ_CURL @ (CX.d0 @ phi)).max() < 1e-9
    return dict(edges=EDGES, triangles=TRIS, projector_max_abs_err=max_err, betti1_K5=int(
        hodge_decompose_2d(CX, rng.standard_normal(E)).betti1))


def tri_orient(s: np.ndarray) -> np.ndarray:
    """Directed 3-cycle orientation per triangle (+1 / -1), 0 if the triangle is not a cycle.

    s[..., e] is the sign of the flow on canonical edge e (+ = j preferred over i).
    A triangle (i, j, k) is cyclic iff s_ij == s_jk == -s_ik != 0.
    """
    a = s[..., T_IJ]
    b = s[..., T_JK]
    c = -s[..., T_IK]
    cyc = (a == b) & (b == c) & (a != 0)
    return np.where(cyc, a, 0).astype(np.int8)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _tb(P: int) -> np.ndarray:
    # binary tie-break weights: every subset has a distinct sum, so the vote never ties exactly
    return TIEBREAK * 2.0 ** -np.arange(P)


def scenario_spec(scen: str, P: int) -> dict:
    """Utilities u_p(i) = A_p . x_i. Feature dims 0..P-1 are principle-specific, dim P is a shared
    quality factor q, dim P+1 is nuisance. g = weight of q, s = weight of the principle-specific
    trade-off, sh = weight of the harmless pair's opposed component."""
    d = P + 2
    A = np.zeros((P, d))
    eye = np.eye(d)
    q, g, s, sh = eye[P], G_SCALE, S_SCALE, HARMLESS_FACTOR * S_SCALE
    spec = dict(name=scen, P=P, d=d, n_orders=2, bias=BIAS_TWO_ORDER, harmless_pair=None,
                designed_cycle_principles=[])
    if scen in ("transitive", "transitive_binarized", "transitive_neartie", "position_bias"):
        for p in range(P):
            A[p] = g * q + s * eye[p]
        if scen == "transitive_neartie":
            A *= NEARTIE_SCALE
        spec.update(agg="wsum_bin" if scen == "transitive_binarized" else "wsum",
                    w=np.ones(P), v=None)
        if scen == "position_bias":
            spec.update(n_orders=1, bias=BIAS_ONE_ORDER)
    elif scen == "condorcet":
        for p in range(P):
            A[p] = g * q + s * (eye[p] - eye[:P].mean(0))
        spec.update(agg="wmaj", w=None, v=np.ones(P) + _tb(P),
                    designed_cycle_principles=list(range(P)))
    elif scen == "harmless_disagreement":
        # principles 0 and 1 disagree strongly (opposite signs on a large shared component);
        # principle 2 has more vote weight than all other principles together, so it decides
        # every vote. No true cycles.
        A[0] = g * q + sh * eye[0]
        A[1] = g * q - sh * eye[0]
        for p in range(2, P):
            A[p] = g * q + s * eye[p - 1]
        v = np.ones(P)
        v[2] = P - 0.5
        spec.update(agg="wmaj", w=None, v=v + _tb(P), harmless_pair=(0, 1))
    elif scen == "mixed":
        # principles 0, 1, 2: balanced trade-off trio (Condorcet cycles). The other principles
        # only disagree: their total vote weight is below the smallest trio margin (1), so they
        # are never pivotal in the true vote.
        trio = eye[:3].mean(0)
        for p in range(3):
            A[p] = g * q + s * (eye[p] - trio)
        v = np.ones(P)
        if P == 4:
            A[3] = g * q - sh * eye[0]         # disagrees strongly with principle 0
            v[3] = 0.5
            spec["harmless_pair"] = (0, 3)
        elif P == 6:
            A[3] = g * q + sh * eye[3]
            A[4] = g * q - sh * eye[3]         # pair (3, 4) disagrees strongly
            A[5] = g * q + s * eye[4]
            v[3:] = 0.3
            spec["harmless_pair"] = (3, 4)
        else:
            raise ValueError(P)
        spec.update(agg="wmaj", w=None, v=v + _tb(P), designed_cycle_principles=[0, 1, 2])
    else:
        raise ValueError(scen)
    spec["A"] = A
    return spec


def restrict_spec(spec: dict, keep: list) -> dict:
    s = dict(spec)
    s["P"] = len(keep)
    if spec["w"] is not None:
        s["w"] = spec["w"][keep]
    if spec["v"] is not None:
        s["v"] = spec["v"][keep]
    return s


# ---------------------------------------------------------------------------
# Labels, aggregation, null probabilities
# ---------------------------------------------------------------------------
_LOGIT = {V: np.log((np.arange(V + 1) + 0.5) / (V + 0.5 - np.arange(V + 1))) for V in VS}
_VAR = {}
for _V in VS:
    _pt = (np.arange(_V + 1) + 0.5) / (_V + 1)
    _VAR[_V] = 1.0 / ((_V + 1) * _pt * (1 - _pt))


def labels_from_counts(k1, k2, V, n_orders):
    """Graded per-principle labels. Returns f (antisymmetrized, + = j), sym, se^2."""
    L, Vr = _LOGIT[V], _VAR[V]
    l1 = L[k1]
    if n_orders == 2:
        l2 = L[k2]
        return 0.5 * (l2 - l1), 0.5 * (l1 + l2), 0.25 * (Vr[k1] + Vr[k2])
    return -l1, np.zeros_like(l1), Vr[k1]


def aggregate(f: np.ndarray, spec: dict, graded: bool = False) -> np.ndarray:
    """f[..., P, E] -> aggregate flow [..., E]. graded=True ignores binarization (naive null)."""
    if spec["agg"] in ("wsum", "wsum_bin"):
        m = np.einsum("...pe,p->...e", f, spec["w"])
        if spec["agg"] == "wsum_bin" and not graded:
            m = np.sign(m)
        return m
    v = spec["v"]
    return np.einsum("...pe,p->...e", np.sign(f), v) / v.sum()


_PATTERNS = {}


def _patterns(P):
    if P not in _PATTERNS:
        _PATTERNS[P] = np.array([[(b >> q) & 1 for q in range(P)] for b in range(2 ** P)], float)
    return _PATTERNS[P]


def agree_prob(mu: np.ndarray, se: np.ndarray, spec: dict) -> np.ndarray:
    """P(aggregate sign > 0) when principle labels ~ N(mu, se^2). mu, se: (nb, P, E) -> (nb, E)."""
    if spec["agg"] in ("wsum", "wsum_bin"):
        w = spec["w"]
        mean = np.einsum("bpe,p->be", mu, w)
        sd = np.sqrt(np.einsum("bpe,p->be", se ** 2, w ** 2))
        return norm.cdf(mean / sd)
    v = spec["v"]
    P = len(v)
    pi = norm.cdf(mu / se)                               # (nb, P, E)
    pat = _patterns(P)                                   # (2^P, P)
    win = (pat * 2 - 1) @ v > 0                          # (2^P,)
    pi_t = np.moveaxis(pi, 1, -1)                        # (nb, E, P)
    lp = (np.log(np.clip(pi_t, 1e-300, 1))[..., None, :] * pat
          + np.log(np.clip(1 - pi_t, 1e-300, 1))[..., None, :] * (1 - pat)).sum(-1)
    return (np.exp(lp) * win).sum(-1)


def solve_shift(mu, se, spec, q, iters=60):
    """Common shift delta per edge with agree_prob(mu - delta) = q (agree_prob decreases in delta)."""
    span = np.abs(mu).max(1) + 12 * se.max(1) + 5.0
    lo, hi = -span, span.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pr = agree_prob(mu - mid[:, None, :], se, spec)
        up = pr > q
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def sample_counts(Uu: np.ndarray, V: int, p: np.ndarray, chunk: int = 10) -> np.ndarray:
    """Binomial(V, p) counts from uniforms via the CDF (common random numbers).

    Uu: (B, nb, P, E, O) uniforms; p: (nb, P, E, O) -> counts (B, nb, P, E, O)."""
    cdf = binom.cdf(np.arange(V + 1), V, p[..., None])  # (nb, P, E, O, V+1)
    out = np.empty(Uu.shape, dtype=np.int16)
    nb = p.shape[0]
    for s in range(0, nb, chunk):
        out[:, s:s + chunk] = (Uu[:, s:s + chunk, ..., None] > cdf[None, s:s + chunk]).sum(-1)
    return out


def simulate_null(mu_can, sym, V, n_orders, spec, Uu, graded=False):
    """Resample all votes from null log-odds mu_can (+ = j) and return the aggregate (B, nb, E)."""
    if n_orders == 2:
        p = np.stack([expit(-mu_can + sym), expit(mu_can + sym)], -1)
    else:
        p = expit(-mu_can)[..., None]
    k = sample_counts(Uu, V, p)
    L = _LOGIT[V][k]
    f = 0.5 * (L[..., 1] - L[..., 0]) if n_orders == 2 else -L[..., 0]
    return aggregate(f, spec, graded=graded)


from itertools import permutations as _perms

_PERM = np.array(list(_perms(range(K))))                              # (120, K): position -> item
_RANK = np.argsort(_PERM, axis=1)                                    # item -> position (0 = top)
# orientation of each edge under each order: +1 if j is ranked above i
_PERM_O = np.sign(_RANK[:, I_] - _RANK[:, J_]).astype(float)        # (120, E)


def bt_order(m: np.ndarray) -> np.ndarray:
    """Least-squares BT potential of the aggregate flow -> orientation per edge (+1: j above i).
    Used only to break exact ties between equally good orders in ``ml_order``."""
    phi = m @ PINV_D0.T                                   # (nb, K)
    phi = phi + 1e-9 * np.arange(K)[None, :]
    rank = np.argsort(np.argsort(phi, axis=1), axis=1)
    o = np.sign(rank[:, J_] - rank[:, I_]).astype(float)
    assert np.all(o != 0)
    return o


def ml_order(s: np.ndarray, w: np.ndarray, m_graded: np.ndarray) -> np.ndarray:
    """Transitive order that reorients the least evidence: argmin over all 120 orders of
    sum_e w_e * 1[s_e disagrees with the order], w_e = plug-in |z| of the observed aggregate sign.
    Ties are broken by agreement with the least-squares BT order. Returns orientation (nb, E)."""
    cost = 0.5 * ((w * (s != 0)).sum(1, keepdims=True) - (w * s) @ _PERM_O.T)   # (nb, 120)
    o_bt = bt_order(m_graded)
    cost = cost + 1e-6 * 0.5 * (E - o_bt @ _PERM_O.T)
    best = np.argmin(cost, axis=1)
    return _PERM_O[best]


def d1_test(f, sym, se2, V, n_orders, spec, Uu, c, deflate="reoriented"):
    """Per-block certified cycle test. Returns dict of per-block / per-triangle arrays."""
    m = aggregate(f, spec)
    s = np.sign(m)
    orient_obs = tri_orient(s)
    count_obs = (orient_obs != 0).sum(1)
    se = np.sqrt(se2)
    z_obs = norm.ppf(np.clip(agree_prob(f * s[:, None, :], se, spec), 1e-12, 1 - 1e-12))
    o = ml_order(s, np.abs(z_obs) * (s != 0), aggregate(f, spec, graded=True))
    flip = (s * o) < 0
    mu_sig = np.where(flip[:, None, :], -f, f) * o[:, None, :]
    rho = agree_prob(mu_sig, se, spec)
    z = norm.ppf(np.clip(rho, 1e-12, 1 - 1e-12))
    c_edge = np.where(flip, c, 0.0) if deflate == "reoriented" else np.full(flip.shape, c)
    q = norm.cdf(np.maximum(z - c_edge, 0.0))
    delta = solve_shift(mu_sig, se, spec, q)
    mu_can = (mu_sig - delta[:, None, :]) * o[:, None, :]
    m_star = simulate_null(mu_can, sym, V, n_orders, spec, Uu)
    ori_star = tri_orient(np.sign(m_star))                # (B, nb, NT)
    cnt_star = (ori_star != 0).sum(-1)
    ge = (cnt_star >= count_obs[None]).sum(0)
    p_block = (1 + ge) / (B + 1)
    flag = (count_obs > 0) & (p_block <= ALPHA)
    same = ((ori_star == orient_obs[None]) & (orient_obs[None] != 0)).sum(0)
    p_tri = (1 + same) / (B + 1)
    cert = flag[:, None] & (orient_obs != 0) & (p_tri <= ALPHA)
    return dict(m=m, orient=orient_obs, count=count_obs, p_block=p_block, flag=flag,
                p_tri=p_tri, cert=cert, cert_orient=np.where(cert, orient_obs, 0),
                null_mean_count=cnt_star.mean(0))


def naive_curl_test(f, sym, se2, V, n_orders, spec, Uu, m_obs):
    """Curl-fraction test with a null that treats labels as faithful graded margins."""
    def cfrac(m):
        tot = (m ** 2).sum(-1)
        cur = ((m @ PROJ_CURL.T) ** 2).sum(-1)
        return np.where(tot > 0, cur / np.where(tot > 0, tot, 1), 0.0)
    mg = aggregate(f, spec, graded=True)
    g = mg @ PROJ_GRAD.T
    delta = (g - mg) / spec["w"].sum()
    mu_can = f + delta[:, None, :]
    m_star = simulate_null(mu_can, sym, V, n_orders, spec, Uu, graded=True)
    cf_obs = cfrac(m_obs)
    cf_star = cfrac(m_star)
    p = (1 + (cf_star >= cf_obs[None]).sum(0)) / (B + 1)
    return dict(curl_frac=cf_obs, p=p, flag=p <= ALPHA)


# ---------------------------------------------------------------------------
# One task = (seed, scenario, P, V)
# ---------------------------------------------------------------------------

def make_world(seed, scen, P):
    spec = scenario_spec(scen, P)
    rng = np.random.default_rng(np.random.SeedSequence([seed, SCEN_IDX[scen], P, 0]))
    X = rng.standard_normal((N_BLOCKS, K, spec["d"]))
    T = rng.uniform(T_LO, T_HI, size=P)
    U = X @ spec["A"].T                                   # (nb, K, P)
    Delta = np.transpose(U[:, J_, :] - U[:, I_, :], (0, 2, 1))   # (nb, P, E)
    # alignment: Delta[b, p, e] = u_p(j) - u_p(i) for EDGES[e] = (i, j)
    for b, p, e in [(0, 0, 0), (N_BLOCKS - 1, P - 1, E - 1), (17, P // 2, 5)]:
        i, j = EDGES[e]
        assert np.isclose(Delta[b, p, e], U[b, j, p] - U[b, i, p])
    eta = Delta / T[None, :, None]
    return spec, X, T, U, Delta, eta


def true_aggregate(Delta, eta, spec):
    if spec["agg"] in ("wsum", "wsum_bin"):
        return np.einsum("bpe,p->be", eta, spec["w"])
    return np.einsum("bpe,p->be", np.sign(Delta), spec["v"])


def observe(eta, spec, V, rng):
    bias = spec["bias"]
    k1 = rng.binomial(V, expit(-eta + bias))             # order 1: i first; votes for i
    k2 = rng.binomial(V, expit(eta + bias))              # order 2: j first; votes for j
    return k1, k2


def bt_feats(X):
    return X[:, J_, :] - X[:, I_, :]                      # (nb, E, d), + favours j


def skew_feats(X):
    lin = bt_feats(X)
    d = X.shape[2]
    ab = list(combinations(range(d), 2))
    Xi, Xj = X[:, I_, :], X[:, J_, :]
    bil = np.stack([Xi[..., a] * Xj[..., b] - Xi[..., b] * Xj[..., a] for a, b in ab], -1)
    return np.concatenate([lin, bil], -1)


def _block_folds(n_blocks, n_folds, rng):
    perm = rng.permutation(n_blocks)
    return [np.sort(perm[k::n_folds]) for k in range(n_folds)]


def fit_lr(F, y, C):
    sc = StandardScaler(with_mean=False).fit(F)
    lr = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(sc.transform(F), y)
    return lambda G: lr.decision_function(sc.transform(G))


def fit_pair_model(Fb, mb, train_blocks, rng):
    """Pairwise logistic model (no intercept, antisymmetric features) on observed aggregate signs.

    Fb: (nb, E, nf) features; mb: (nb, E) observed aggregate. C by grouped CV over train blocks.
    Returns (predict_fn, best_C, oof_margin for the train blocks)."""
    assert train_blocks.max() < N_TRAIN, "held-out block in training set"
    folds = _block_folds(len(train_blocks), N_FOLDS, rng)

    def rows(blks):
        F = Fb[blks].reshape(-1, Fb.shape[2])
        y = mb[blks].reshape(-1)
        keep = y != 0
        return F[keep], (y[keep] > 0).astype(int)

    best, best_ll = None, np.inf
    for C in RM_CS:
        ll = 0.0
        for k in range(N_FOLDS):
            te = train_blocks[folds[k]]
            tr = np.setdiff1d(train_blocks, te)
            Ftr, ytr = rows(tr)
            Fte, yte = rows(te)
            sc = fit_lr(Ftr, ytr, C)(Fte)
            ll += np.sum(np.logaddexp(0, -sc * (2 * yte - 1)))
        if ll < best_ll:
            best, best_ll = C, ll
    oof = np.zeros((len(train_blocks), E))
    for k in range(N_FOLDS):
        te = train_blocks[folds[k]]
        tr = np.setdiff1d(train_blocks, te)
        Ftr, ytr = rows(tr)
        pred = fit_lr(Ftr, ytr, best)
        oof[folds[k]] = pred(Fb[te].reshape(-1, Fb.shape[2])).reshape(len(te), E)
    Fall, yall = rows(train_blocks)
    pred_all = fit_lr(Fall, yall, best)
    return (lambda Fq: pred_all(Fq.reshape(-1, Fq.shape[-1])).reshape(Fq.shape[0], E)), best, oof


def auroc(y, s):
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def error_predictor_auc(Ftr, ytr, Fte, yte):
    if ytr.min() == ytr.max() or yte.min() == yte.max():
        return float("nan")
    sc = StandardScaler().fit(Ftr)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Ftr), ytr)
    return auroc(yte, lr.decision_function(sc.transform(Fte)))


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def spearman_safe(score, truth):
    if np.allclose(truth, truth[0]):
        return float("nan")
    if np.allclose(score, score[0]):
        return 0.0                              # a constant score carries no ranking
    return float(spearmanr(score, truth).correlation)


def precision_at_k(score, truth, rng_unused=None):
    """Expected precision@k under uniform random tie-breaking; k = number of truly positive pairs."""
    pos = truth > 0
    k = int(pos.sum())
    if k == 0 or k == len(truth):
        return float("nan")
    order_vals = np.sort(score)[::-1]
    thr = order_vals[k - 1]
    above = score > thr
    tied = score == thr
    n_above = int(above.sum())
    slots = k - n_above
    tp = pos[above].sum() + slots * (pos[tied].mean() if tied.any() else 0.0)
    return float(tp / k)


def run_task(args):
    seed, scen, P, V, c, deflate, mode = args
    t0 = time.time()
    spec, X, T, U, Delta, eta = make_world(seed, scen, P)
    rng = np.random.default_rng(np.random.SeedSequence([seed, SCEN_IDX[scen], P, V, 1]))
    k1, k2 = observe(eta, spec, V, rng)
    n_orders = spec["n_orders"]
    if scen != "position_bias":
        assert n_orders == 2 and k2.shape == k1.shape, "both orders must be collected"
    f, sym, se2 = labels_from_counts(k1, k2, V, n_orders)
    assert f.shape == (N_BLOCKS, P, E) and se2.shape == f.shape
    Uu = np.random.default_rng(np.random.SeedSequence([seed, SCEN_IDX[scen], P, V, 2])).random(
        (B, N_BLOCKS, P, E, n_orders))

    tagg = true_aggregate(Delta, eta, spec)
    tsign = np.sign(tagg)
    n_true_ties = int((tsign == 0).sum())
    torient = tri_orient(tsign)
    true_cyc_block = (torient != 0).any(1)

    d1 = d1_test(f, sym, se2, V, n_orders, spec, Uu, c, deflate)
    flag = d1["flag"]
    out = dict(seed=seed, scenario=scen, P=P, V=V, c=c, deflate=deflate,
               n_blocks=N_BLOCKS, n_true_ties=n_true_ties,
               n_true_cycle_blocks=int(true_cyc_block.sum()),
               n_true_cycles=int((torient != 0).sum()),
               n_flagged=int(flag.sum()),
               n_flagged_true=int((flag & true_cyc_block).sum()),
               n_flagged_no_true=int((flag & ~true_cyc_block).sum()),
               n_no_true=int((~true_cyc_block).sum()),
               n_obs_cycle_blocks=int((d1["count"] > 0).sum()),
               n_obs_cycles=int(d1["count"].sum()),
               n_cert=int(d1["cert"].sum()),
               n_cert_true=int((d1["cert"] & (d1["cert_orient"] == torient)).sum()),
               n_true_recovered_cert=int(((torient != 0) & (d1["cert_orient"] == torient)).sum()),
               mean_true_cycles_per_block=float((torient != 0).sum(1).mean()))
    if scen in WSUM_SCENARIOS:
        nc = naive_curl_test(f, sym, se2, V, n_orders, spec, Uu, d1["m"])
        out.update(naive_curl_flagged=int(nc["flag"].sum()),
                   naive_curl_flagged_no_true=int((nc["flag"] & ~true_cyc_block).sum()),
                   mean_curl_frac=float(nc["curl_frac"].mean()))
    if scen == "position_bias":
        # antisymmetrization check on the SAME blocks: add the second order that was not collected
        f2, sym2, se22 = labels_from_counts(k1, k2, V, 2)
        m2 = aggregate(f2, spec)
        cf = lambda m: float(np.mean(((m @ PROJ_CURL.T) ** 2).sum(-1) / np.maximum((m ** 2).sum(-1), 1e-300)))
        Uu2 = np.random.default_rng(np.random.SeedSequence([seed, SCEN_IDX[scen], P, V, 3])).random(
            (B, N_BLOCKS, P, E, 2))
        d1b = d1_test(f2, sym2, se22, V, 2, spec, Uu2, c, deflate)
        out.update(curl_frac_one_order=cf(d1["m"]), curl_frac_both_orders=cf(m2),
                   n_flagged_both_orders=int(d1b["flag"].sum()),
                   n_obs_cycle_blocks_both_orders=int((d1b["count"] > 0).sum()))
    if mode == "pilot":
        out["seconds"] = time.time() - t0
        return out

    # -------------------- per-pair features --------------------
    m = d1["m"]
    curl = np.empty_like(m)
    for b in range(N_BLOCKS):
        curl[b] = hodge_decompose_2d(CX, m[b], compute_betti=False).curl
    abscurl = np.abs(curl)
    ncert = (d1["cert"].astype(float) @ EDGE_TRI.T.astype(float))      # (nb, E)
    ntrue = ((torient != 0).astype(float) @ EDGE_TRI.T.astype(float))
    signs = np.sign(f)
    frac_j = ((signs > 0).sum(1) + 0.5 * (signs == 0).sum(1)) / P
    ent = binary_entropy(frac_j)
    absm = np.abs(m)
    flag_e = np.repeat(flag[:, None], E, 1).astype(float)

    train = np.arange(N_TRAIN)
    held = np.arange(N_TRAIN, N_BLOCKS)
    assert np.intersect1d(train, held).size == 0
    rng_fit = np.random.default_rng(np.random.SeedSequence([seed, SCEN_IDX[scen], P, V, 4]))
    Fbt = bt_feats(X)
    bt_pred, bt_C, bt_oof = fit_pair_model(Fbt, m, train, rng_fit)
    r_held = bt_pred(Fbt[held])
    r_tr = bt_oof
    err_tr = (np.sign(r_tr) != tsign[train]).astype(int).ravel()
    err_he = (np.sign(r_held) != tsign[held]).astype(int).ravel()

    def feats(blks, r, names):
        cols = dict(ent=ent[blks], absm=absm[blks], absr=np.abs(r), ncert=ncert[blks],
                    abscurl=abscurl[blks], flag=flag_e[blks],
                    rm_disagree=(np.sign(r) != np.sign(m[blks])).astype(float),
                    rm_obs_prod=r * m[blks] / (np.abs(m[blks]).max() + 1e-12))
        return np.stack([cols[n].ravel() for n in names], 1)

    base = ["ent", "absm", "absr"]
    hodge = ["ncert", "abscurl", "flag"]
    strong = base + ["rm_disagree", "rm_obs_prod"]
    d2 = dict(rm_C=bt_C, rm_err_rate_held=float(err_he.mean()), rm_err_rate_train_oof=float(err_tr.mean()),
              n_err_held=int(err_he.sum()))
    for name, cols in [("base", base), ("full", base + hodge), ("strong", strong),
                       ("strong_full", strong + hodge), ("hodge_only", hodge)]:
        d2[f"auc_{name}"] = error_predictor_auc(feats(train, r_tr, cols), err_tr,
                                                feats(held, r_held, cols), err_he)
    for nm, arr in [("ncert", ncert), ("abscurl", abscurl), ("ent", ent), ("absm_neg", -absm),
                    ("oracle_true_cycles", ntrue)]:
        d2[f"auc_single_{nm}"] = auroc(err_he, arr[held].ravel())
    d2["auc_single_rm_disagree"] = auroc(err_he, (np.sign(r_held) != np.sign(m[held])).ravel())
    # disagreement on true-cycle edges vs other edges (construction check)
    on_cyc = ntrue > 0
    d2["mean_entropy_true_cycle_edges"] = float(ent[on_cyc].mean()) if on_cyc.any() else float("nan")
    d2["mean_entropy_other_edges"] = float(ent[~on_cyc].mean())
    d2["frac_rm_err_on_true_cycle_edges"] = float(
        (np.sign(r_held) != tsign[held])[on_cyc[held]].mean()) if on_cyc[held].any() else float("nan")
    d2["frac_rm_err_on_other_edges"] = float((np.sign(r_held) != tsign[held])[~on_cyc[held]].mean())
    out["d2"] = d2

    # -------------------- D4 --------------------
    Fsk = skew_feats(X)
    sk_pred, sk_C, _ = fit_pair_model(Fsk, m, train, rng_fit)
    acc_bt = (np.sign(r_held) == tsign[held]).mean(1)
    acc_sk = (np.sign(sk_pred(Fsk[held])) == tsign[held]).mean(1)
    fh = flag[held]
    out["d4"] = dict(skew_C=sk_C, acc_bt=float(acc_bt.mean()), acc_skew=float(acc_sk.mean()),
                     n_flagged_held=int(fh.sum()),
                     gain_flagged=float((acc_sk - acc_bt)[fh].mean()) if fh.any() else float("nan"),
                     gain_unflagged=float((acc_sk - acc_bt)[~fh].mean()) if (~fh).any() else float("nan"),
                     gain_true_cycle_blocks=float((acc_sk - acc_bt)[true_cyc_block[held]].mean())
                     if true_cyc_block[held].any() else float("nan"))

    # -------------------- D3 --------------------
    if scen in D3_SCENARIOS:
        pairs = list(combinations(range(P), 2))
        has_cert = np.where(d1["cert"].any(1))[0]
        removed = np.zeros((P, N_BLOCKS, NT), dtype=bool)
        flips = np.zeros((P, N_BLOCKS, E), dtype=bool)
        L_true = np.zeros((P, N_BLOCKS, NT), dtype=bool)
        for p in range(P):
            keep = [q for q in range(P) if q != p]
            sp = restrict_spec(spec, keep)
            m_loo = aggregate(f[:, keep], sp)
            flips[p] = np.sign(m_loo) != np.sign(m)
            if has_cert.size:
                r = d1_test(f[has_cert][:, keep], sym[has_cert][:, keep], se2[has_cert][:, keep], V,
                            n_orders, sp, Uu[:, has_cert][:, :, keep], c, deflate)
                removed[p, has_cert] = d1["cert"][has_cert] & (r["cert_orient"] != d1["cert_orient"][has_cert])
            t_loo = np.sign(true_aggregate(Delta[:, keep], eta[:, keep], sp))
            L_true[p] = (torient != 0) & (tri_orient(t_loo) != torient)
        hodge_s = np.array([(removed[p] | removed[q]).sum() / N_BLOCKS for p, q in pairs])
        hodge_i = np.array([(removed[p] & removed[q]).sum() / N_BLOCKS for p, q in pairs])
        truth_u = np.array([(L_true[p] | L_true[q]).sum() / N_BLOCKS for p, q in pairs])
        truth_i = np.array([(L_true[p] & L_true[q]).sum() / N_BLOCKS for p, q in pairs])
        sgn = np.sign(f)
        disagree = np.array([np.mean(sgn[:, p] != sgn[:, q]) for p, q in pairs])
        pivot = np.array([(flips[p] | flips[q]).sum() / N_BLOCKS for p, q in pairs])
        hp = spec["harmless_pair"]
        hidx = pairs.index(tuple(hp)) if hp is not None else None

        def share(sc):
            tot = sc.sum()
            return float(sc[hidx] / tot) if (hidx is not None and tot > 0) else 0.0

        out["d3"] = dict(
            pairs=pairs, truth_union=truth_u.tolist(), truth_inter=truth_i.tolist(),
            hodge_union=hodge_s.tolist(), hodge_inter=hodge_i.tolist(),
            disagree=disagree.tolist(), pivot=pivot.tolist(),
            n_blocks_with_cert=int(has_cert.size),
            spearman_hodge=spearman_safe(hodge_s, truth_u),
            spearman_base=spearman_safe(disagree, truth_u),
            spearman_pivot=spearman_safe(pivot, truth_u),
            spearman_hodge_inter=spearman_safe(hodge_i, truth_i),
            spearman_base_inter=spearman_safe(disagree, truth_i),
            prec_hodge=precision_at_k(hodge_s, truth_u),
            prec_base=precision_at_k(disagree, truth_u),
            prec_pivot=precision_at_k(pivot, truth_u),
            prec_hodge_inter=precision_at_k(hodge_i, truth_i),
            prec_base_inter=precision_at_k(disagree, truth_i),
            harmless_pair=list(hp) if hp is not None else None,
            harmless_share_hodge=share(hodge_s), harmless_share_base=share(disagree),
            harmless_share_pivot=share(pivot),
            harmless_top1_hodge=bool(hidx is not None and hodge_s.max() > 0 and np.argmax(hodge_s) == hidx),
            harmless_top1_base=bool(hidx is not None and np.argmax(disagree) == hidx),
        )
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def clopper_pearson(x, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if x == 0 else float(beta_dist.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta_dist.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def boot_ci(diffs, rng_seed=0):
    d = np.asarray([x for x in diffs if np.isfinite(x)])
    if d.size < 3:
        return dict(n=int(d.size), mean=float("nan"), ci=[float("nan")] * 2, p=float("nan"))
    rng = np.random.default_rng(rng_seed)
    bm = d[rng.integers(0, d.size, size=(N_BOOT_SEEDS, d.size))].mean(1)
    p = 2 * min((bm <= 0).mean(), (bm >= 0).mean())
    p = max(p, 1.0 / N_BOOT_SEEDS)
    return dict(n=int(d.size), mean=float(d.mean()), sd=float(d.std(ddof=1)),
                ci=[float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))], p=float(min(p, 1.0)))


def wilcoxon_test(diffs):
    d = np.asarray([x for x in diffs if np.isfinite(x)])
    if d.size < 3 or np.allclose(d, 0):
        return dict(n=int(d.size), mean=float(np.mean(d)) if d.size else float("nan"),
                    p=float("nan") if d.size < 3 else 1.0, wins=int((d > 0).sum()),
                    ties=int((d == 0).sum()), losses=int((d < 0).sum()))
    return dict(n=int(d.size), mean=float(d.mean()), sd=float(d.std(ddof=1)),
                p=float(wilcoxon(d).pvalue), wins=int((d > 0).sum()), ties=int((d == 0).sum()),
                losses=int((d < 0).sum()))


def holm(pvals: dict, alpha=ALPHA):
    names = sorted(pvals, key=lambda k: (np.inf if not np.isfinite(pvals[k]) else pvals[k]))
    m = len(names)
    adj, running, out = {}, 0.0, {}
    for r, n in enumerate(names):
        p = pvals[n] if np.isfinite(pvals[n]) else 1.0
        running = max(running, min(1.0, (m - r) * p))
        adj[n] = running
    for n in pvals:
        out[n] = dict(p=pvals[n], p_holm=adj[n], significant=bool(adj[n] <= alpha))
    return out


def cfg_key(P, V):
    return f"P{P}_V{V}"


def seed_level(raw_s, getter):
    """Per seed: mean over the 4 (P, V) cells of getter(task) (NaNs ignored)."""
    vals = []
    for sd in sorted({t["seed"] for cfg in raw_s.values() for t in cfg}):
        xs = [getter(t) for cfg in raw_s.values() for t in cfg if t["seed"] == sd]
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        vals.append(float(np.mean(xs)) if xs else float("nan"))
    return vals


def summarise(raw, calib):
    S = dict()
    # ---------------- D1 ----------------
    d1 = {}
    gate_ok = True
    for scen in SCENARIOS:
        if scen not in raw:
            continue
        d1[scen] = {}
        for P, V in CFGS:
            ts = raw[scen].get(cfg_key(P, V), [])
            if not ts:
                continue
            n_no = sum(t["n_no_true"] for t in ts)
            fp = sum(t["n_flagged_no_true"] for t in ts)
            n_tr = sum(t["n_true_cycle_blocks"] for t in ts)
            tp = sum(t["n_flagged_true"] for t in ts)
            nflag = sum(t["n_flagged"] for t in ts)
            r = dict(n_blocks=sum(t["n_blocks"] for t in ts), n_no_true=n_no, false_pos=fp,
                     fpr=fp / n_no if n_no else float("nan"),
                     fpr_cp95=list(clopper_pearson(fp, n_no)) if n_no else None,
                     n_true_cycle_blocks=n_tr, true_pos=tp,
                     recall=tp / n_tr if n_tr else float("nan"),
                     recall_cp95=list(clopper_pearson(tp, n_tr)) if n_tr else None,
                     flag_precision=tp / nflag if nflag else float("nan"),
                     frac_blocks_true_cycle=n_tr / sum(t["n_blocks"] for t in ts),
                     frac_blocks_obs_cycle=sum(t["n_obs_cycle_blocks"] for t in ts) / sum(t["n_blocks"] for t in ts),
                     n_true_cycles=sum(t["n_true_cycles"] for t in ts),
                     triangle_recall=(sum(t["n_true_recovered_cert"] for t in ts) / sum(t["n_true_cycles"] for t in ts))
                     if sum(t["n_true_cycles"] for t in ts) else float("nan"),
                     n_cert=sum(t["n_cert"] for t in ts),
                     triangle_precision=(sum(t["n_cert_true"] for t in ts) / sum(t["n_cert"] for t in ts))
                     if sum(t["n_cert"] for t in ts) else float("nan"))
            if "naive_curl_flagged_no_true" in ts[0]:
                nfp = sum(t["naive_curl_flagged_no_true"] for t in ts)
                r.update(naive_curl_fpr=nfp / n_no if n_no else float("nan"),
                         naive_curl_fpr_cp95=list(clopper_pearson(nfp, n_no)) if n_no else None,
                         mean_curl_frac=float(np.mean([t["mean_curl_frac"] for t in ts])))
            if scen == "position_bias":
                r.update(curl_frac_one_order=float(np.mean([t["curl_frac_one_order"] for t in ts])),
                         curl_frac_both_orders=float(np.mean([t["curl_frac_both_orders"] for t in ts])),
                         flag_rate_both_orders=sum(t["n_flagged_both_orders"] for t in ts) / r["n_blocks"],
                         flag_rate_one_order=nflag / r["n_blocks"],
                         obs_cycle_rate_both_orders=sum(t["n_obs_cycle_blocks_both_orders"] for t in ts) / r["n_blocks"])
            if scen in GATE_SCENARIOS:
                r["gate_pass"] = bool(r["fpr"] <= 0.05 and r["fpr_cp95"][1] <= 0.08)
                gate_ok &= r["gate_pass"]
            d1[scen][cfg_key(P, V)] = r
    pos_ctrl = {cfg_key(P, 25): d1.get("condorcet", {}).get(cfg_key(P, 25), {}).get("recall", float("nan"))
                for P in PS}
    pos_ok = all(np.isfinite(v) and v >= 0.5 for v in pos_ctrl.values())
    S["d1"] = dict(by_scenario=d1, gate_pass=bool(gate_ok), positive_control_recall_V25=pos_ctrl,
                   positive_control_pass=bool(pos_ok))
    interpretable = bool(gate_ok and pos_ok)
    S["interpretable"] = interpretable

    # ---------------- D2 ----------------
    d2 = {}
    for scen in SCENARIOS:
        if scen not in raw or "d2" not in next(iter(raw[scen].values()))[0]:
            continue
        rs = raw[scen]
        g = lambda key: (lambda t: t["d2"][key])
        delta = seed_level(rs, lambda t: t["d2"]["auc_full"] - t["d2"]["auc_base"])
        delta_strong = seed_level(rs, lambda t: t["d2"]["auc_strong_full"] - t["d2"]["auc_strong"])
        row = dict(auc_base=seed_level(rs, g("auc_base")), auc_full=seed_level(rs, g("auc_full")),
                   auc_strong=seed_level(rs, g("auc_strong")), auc_strong_full=seed_level(rs, g("auc_strong_full")),
                   auc_hodge_only=seed_level(rs, g("auc_hodge_only")),
                   delta=delta, delta_boot=boot_ci(delta, 1 + SCEN_IDX[scen]),
                   delta_strong=delta_strong, delta_strong_boot=boot_ci(delta_strong, 101 + SCEN_IDX[scen]),
                   per_cfg={})
        for P, V in CFGS:
            ts = rs.get(cfg_key(P, V), [])
            if not ts:
                continue
            dd = [t["d2"]["auc_full"] - t["d2"]["auc_base"] for t in ts]
            ds = [t["d2"]["auc_strong_full"] - t["d2"]["auc_strong"] for t in ts]
            row["per_cfg"][cfg_key(P, V)] = dict(
                auc_base=float(np.nanmean([t["d2"]["auc_base"] for t in ts])),
                auc_full=float(np.nanmean([t["d2"]["auc_full"] for t in ts])),
                auc_strong=float(np.nanmean([t["d2"]["auc_strong"] for t in ts])),
                auc_strong_full=float(np.nanmean([t["d2"]["auc_strong_full"] for t in ts])),
                delta_boot=boot_ci(dd, 7), delta_strong_boot=boot_ci(ds, 8),
                rm_err_rate_held=float(np.mean([t["d2"]["rm_err_rate_held"] for t in ts])),
                single={k: float(np.nanmean([t["d2"][f"auc_single_{k}"] for t in ts]))
                        for k in ("ncert", "abscurl", "ent", "absm_neg", "oracle_true_cycles", "rm_disagree")},
                entropy_true_cycle_edges=float(np.nanmean([t["d2"]["mean_entropy_true_cycle_edges"] for t in ts])),
                entropy_other_edges=float(np.nanmean([t["d2"]["mean_entropy_other_edges"] for t in ts])),
                rm_err_true_cycle_edges=float(np.nanmean([t["d2"]["frac_rm_err_on_true_cycle_edges"] for t in ts])),
                rm_err_other_edges=float(np.nanmean([t["d2"]["frac_rm_err_on_other_edges"] for t in ts])))
        d2[scen] = row
    S["d2"] = d2

    # ---------------- D3 ----------------
    d3 = {}
    for scen in D3_SCENARIOS:
        if scen not in raw or "d3" not in next(iter(raw[scen].values()))[0]:
            continue
        rs = raw[scen]
        row = {}
        for metric in ("spearman", "prec"):
            for suffix in ("", "_inter"):
                h = seed_level(rs, lambda t, mm=metric, s=suffix: t["d3"][f"{mm}_hodge{s}"])
                bse = seed_level(rs, lambda t, mm=metric, s=suffix: t["d3"][f"{mm}_base{s}"])
                diff = [a - b for a, b in zip(h, bse)]
                row[f"{metric}{suffix}"] = dict(hodge=h, base=bse, diff=diff, test=wilcoxon_test(diff),
                                                hodge_mean=float(np.nanmean(h)) if np.isfinite(h).any() else float("nan"),
                                                base_mean=float(np.nanmean(bse)) if np.isfinite(bse).any() else float("nan"))
            if metric in ("spearman", "prec"):
                pv = seed_level(rs, lambda t, mm=metric: t["d3"][f"{mm}_pivot"])
                h = row[f"{metric}"]["hodge"]
                diff = [a - b for a, b in zip(h, pv)]
                row[f"{metric}_vs_pivot"] = dict(pivot=pv, diff=diff, test=wilcoxon_test(diff),
                                                 pivot_mean=float(np.nanmean(pv)) if np.isfinite(pv).any() else float("nan"))
        hs = seed_level(rs, lambda t: t["d3"]["harmless_share_hodge"])
        bs = seed_level(rs, lambda t: t["d3"]["harmless_share_base"])
        ps_ = seed_level(rs, lambda t: t["d3"]["harmless_share_pivot"])
        row["harmless_share"] = dict(hodge=hs, base=bs, pivot=ps_,
                                     diff_base_minus_hodge=[b - a for a, b in zip(hs, bs)],
                                     test=wilcoxon_test([b - a for a, b in zip(hs, bs)]),
                                     hodge_mean=float(np.nanmean(hs)), base_mean=float(np.nanmean(bs)),
                                     pivot_mean=float(np.nanmean(ps_)))
        row["harmless_top1_rate"] = dict(
            hodge=float(np.mean([t["d3"]["harmless_top1_hodge"] for cfg in rs.values() for t in cfg])),
            base=float(np.mean([t["d3"]["harmless_top1_base"] for cfg in rs.values() for t in cfg])))
        row["per_cfg"] = {}
        for P, V in CFGS:
            ts = rs.get(cfg_key(P, V), [])
            if not ts:
                continue
            row["per_cfg"][cfg_key(P, V)] = dict(
                truth_union_mean=np.mean([t["d3"]["truth_union"] for t in ts], 0).round(4).tolist(),
                hodge_union_mean=np.mean([t["d3"]["hodge_union"] for t in ts], 0).round(4).tolist(),
                disagree_mean=np.mean([t["d3"]["disagree"] for t in ts], 0).round(4).tolist(),
                pivot_mean=np.mean([t["d3"]["pivot"] for t in ts], 0).round(4).tolist(),
                pairs=ts[0]["d3"]["pairs"],
                spearman_hodge=float(np.nanmean([t["d3"]["spearman_hodge"] for t in ts])),
                spearman_base=float(np.nanmean([t["d3"]["spearman_base"] for t in ts])),
                spearman_pivot=float(np.nanmean([t["d3"]["spearman_pivot"] for t in ts])),
                spearman_diff_test=wilcoxon_test([t["d3"]["spearman_hodge"] - t["d3"]["spearman_base"] for t in ts]),
                n_blocks_with_cert=float(np.mean([t["d3"]["n_blocks_with_cert"] for t in ts])))
        d3[scen] = row
    S["d3"] = d3

    # ---------------- Holm over {D2, D3} ----------------
    fam = {}
    if "condorcet" in d2 and "mixed" in d2:
        pc, pm = d2["condorcet"]["delta_boot"]["p"], d2["mixed"]["delta_boot"]["p"]
        fam["D2"] = float(max(pc, pm))           # conjunction (intersection-union) over the 2 scenarios
    if "mixed" in d3:
        fam["D3"] = d3["mixed"]["spearman"]["test"]["p"]
    H = holm(fam) if fam else {}
    verdicts = {}
    if "D2" in H:
        dc, dm = d2["condorcet"]["delta_boot"], d2["mixed"]["delta_boot"]
        ci_ex = {s: (d2[s]["delta_boot"]["ci"][0] > 0 or d2[s]["delta_boot"]["ci"][1] < 0) for s in ("condorcet", "mixed")}
        killed = not ci_ex["condorcet"] and not ci_ex["mixed"]
        confirmed = (H["D2"]["significant"] and dc["mean"] >= D2_MIN_GAIN and dm["mean"] >= D2_MIN_GAIN
                     and dc["ci"][0] > 0 and dm["ci"][0] > 0)
        evaluable = np.isfinite(dc["mean"]) and np.isfinite(dm["mean"])
        verdicts["D2"] = dict(killed=bool(killed and evaluable), confirmed=bool(confirmed), ci_excludes_zero=ci_ex,
                              verdict=("NOT EVALUABLE (too few seeds)" if not evaluable else
                                       "KILLED" if killed else ("CONFIRMED" if confirmed else "NOT CONFIRMED (not killed)")))
    if "D3" in H:
        t = d3["mixed"]["spearman"]["test"]
        win = H["D3"]["significant"] and t["mean"] > 0
        evaluable = np.isfinite(t["p"])
        verdicts["D3"] = dict(killed=bool(evaluable and not win), confirmed=bool(win),
                              verdict=("NOT EVALUABLE (too few seeds)" if not evaluable else
                                       "CONFIRMED in mixed" if win else "KILLED (no significant win in mixed)"))
    S["primary"] = dict(family_p=fam, holm=H, verdicts=verdicts,
                        rule="D2 p = max of the two scenario bootstrap p-values (both scenarios are predicted); "
                             "D3 p = paired Wilcoxon on per-seed Spearman difference in mixed; Holm over {D2, D3}.")

    # ---------------- D4 ----------------
    d4 = {}
    for scen in SCENARIOS:
        if scen not in raw or "d4" not in next(iter(raw[scen].values()))[0]:
            continue
        rs = raw[scen]
        gf = seed_level(rs, lambda t: t["d4"]["gain_flagged"])
        gu = seed_level(rs, lambda t: t["d4"]["gain_unflagged"])
        inter = [a - b for a, b in zip(gf, gu)]
        d4[scen] = dict(acc_bt=float(np.mean(seed_level(rs, lambda t: t["d4"]["acc_bt"]))),
                        acc_skew=float(np.mean(seed_level(rs, lambda t: t["d4"]["acc_skew"]))),
                        gain_overall=boot_ci(seed_level(rs, lambda t: t["d4"]["acc_skew"] - t["d4"]["acc_bt"]), 300),
                        gain_flagged=boot_ci(gf, 301), gain_unflagged=boot_ci(gu, 302),
                        interaction=boot_ci(inter, 303), interaction_wilcoxon=wilcoxon_test(inter),
                        gain_true_cycle_blocks=boot_ci(seed_level(rs, lambda t: t["d4"]["gain_true_cycle_blocks"]), 304),
                        n_flagged_held_mean=float(np.mean(seed_level(rs, lambda t: t["d4"]["n_flagged_held"]))))
    S["d4"] = d4
    S["calibration_choice"] = calib.get("chosen_c") if calib else None
    return S


# ---------------------------------------------------------------------------
# Pilot calibration of the deflation constant c (disjoint seeds)
# ---------------------------------------------------------------------------

def calibrate(workers, seeds):
    """Pilot on seeds disjoint from the main seeds. Rule (fixed before the main run):
    for each deflation variant, c* = smallest c in C_GRID whose worst FPR over the 3 gate scenarios x
    4 (P, V) cells is <= PILOT_FPR_TARGET. Then use the variant whose mean condorcet block recall at
    V = 25 (at its own c*) is higher."""
    scens = GATE_SCENARIOS + ["condorcet", "mixed"]
    jobs = [(sd, sc, P, V, c, dm, "pilot") for dm in DEFLATE_MODES for c in C_GRID for sc in scens
            for P, V in CFGS for sd in seeds]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for t in ex.map(run_task, jobs, chunksize=2):
            key = (t["deflate"], t["c"], t["scenario"], cfg_key(t["P"], t["V"]))
            res.setdefault(key, []).append(t)
    table = {}
    for (dm, c, sc, cfg), ts in res.items():
        n_no = sum(t["n_no_true"] for t in ts)
        n_tr = sum(t["n_true_cycle_blocks"] for t in ts)
        table.setdefault(dm, {}).setdefault(str(c), {}).setdefault(sc, {})[cfg] = dict(
            fpr=sum(t["n_flagged_no_true"] for t in ts) / n_no if n_no else float("nan"),
            recall=sum(t["n_flagged_true"] for t in ts) / n_tr if n_tr else float("nan"),
            n_no_true=n_no, n_true_cycle_blocks=n_tr)
    worst, per_mode = {}, {}
    for dm in DEFLATE_MODES:
        worst[dm] = {}
        chosen = None
        for c in C_GRID:
            w = max(table[dm][str(c)][sc][cfg]["fpr"] for sc in GATE_SCENARIOS for cfg in table[dm][str(c)][sc])
            worst[dm][str(c)] = w
            if chosen is None and w <= PILOT_FPR_TARGET:
                chosen = c
        rec = (float(np.mean([table[dm][str(chosen)]["condorcet"][cfg_key(P, 25)]["recall"] for P in PS]))
               if chosen is not None else float("nan"))
        per_mode[dm] = dict(c_star=chosen, condorcet_recall_V25=rec)
    ok = [dm for dm in DEFLATE_MODES if per_mode[dm]["c_star"] is not None]
    if ok:
        best = max(ok, key=lambda dm: per_mode[dm]["condorcet_recall_V25"])
        chosen_c = per_mode[best]["c_star"]
    else:
        best, chosen_c = DEFLATE_MODES[0], C_GRID[-1]
    rule = (f"for each deflation variant {list(DEFLATE_MODES)}: c* = smallest c in {list(C_GRID)} whose worst pilot "
            f"FPR over the 3 gate scenarios x 4 (P, V) cells is <= {PILOT_FPR_TARGET}; then the variant with the "
            f"higher mean condorcet block recall at V = 25. Pilot seeds {list(seeds)} (disjoint from the main seeds)")
    if not ok:
        rule += "; no c met the target, so the largest c was used"
    return dict(c_grid=list(C_GRID), pilot_seeds=list(seeds), table=table, worst_gate_fpr=worst,
                per_mode=per_mode, chosen_deflate=best, chosen_c=chosen_c, rule=rule)


# ---------------------------------------------------------------------------
# Markdown report (generated from the JSON only)
# ---------------------------------------------------------------------------

def _f(x, nd=3):
    if x is None:
        return "n/a"
    try:
        if not np.isfinite(x):
            return "n/a"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


def _p(x):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.2g}" if x >= 1e-3 else f"{x:.1e}"


def write_markdown(J, path):
    S = J["summary"]
    d1 = S["d1"]
    L = []
    a = L.append
    a("# SGB-046 results: Hodge structure as a reward-model diagnostic (simulation)")
    a("")
    a(f"This file was generated from `{J['output_json']}` by `{J['script']}` on {J['timestamp']}. "
      "All numbers come from that JSON file.")
    a("")
    a(f"- Pre-registration: `feedback_geometry/SGB046_PREREGISTRATION.md`, SHA-256 `{J['prereg_sha256']}` "
      f"(the script checked this hash before the run: {J['prereg_hash_ok']}).")
    a(f"- Git HEAD at the run: `{J['git_head']}`. Runtime: {J['runtime_seconds']:.0f} s on {J['workers']} CPU workers.")
    a(f"- Command: `{J['command']}`")
    a(f"- Seeds: {J['config']['seeds'][0]}–{J['config']['seeds'][-1]} ({len(J['config']['seeds'])} seeds). "
      f"Blocks per seed: {J['config']['n_train']} train + {J['config']['n_held']} held-out. k = 5. "
      f"P in {J['config']['Ps']}. V in {J['config']['Vs']}. Bootstrap B = {J['config']['B']}.")
    a("- Everything is simulated. A synthetic result is not evidence about human or real-grader preferences "
      "(pre-registration § 5).")
    a("")
    # ---------- D1 ----------
    a("## 1. D1 calibration gate (read this first)")
    a("")
    gp = d1["gate_pass"]
    a(f"**D1 gate: {'PASS' if gp else 'FAIL'}.** The rule: false-positive rate (FPR) ≤ 0.05 and the upper "
      "95% Clopper–Pearson bound ≤ 0.08, in each transitive scenario, at each (P, V) cell.")
    a("")
    a(f"**Positive control (`condorcet`, block-level recall at V = 25): "
      f"{'PASS' if d1['positive_control_pass'] else 'FAIL'}** "
      + ", ".join(f"{k}: {_f(v)}" for k, v in d1["positive_control_recall_V25"].items()) + " (rule: ≥ 0.5).")
    a("")
    if not S["interpretable"]:
        a("**The gate or the positive control failed. D2, D3 and D4 below are NOT interpretable.** "
          "The tables are kept only as a record.")
        a("")
    a(f"The null uses deflation variant `{J['calibration']['chosen_deflate']}` with c = {J['calibration']['chosen_c']} "
      "(see § 6, item 1, and § 8).")
    a("")
    a("FPR = fraction of blocks with no true cycle that the test flagged. Recall = fraction of blocks with "
      "a true cycle that the test flagged.")
    a("")
    a("| scenario | cell | blocks without a true cycle | FPR | 95% CP | gate | naive curl-test FPR | blocks with a true cycle | recall | flag precision |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for scen in SCENARIOS:
        for cfg, r in d1["by_scenario"].get(scen, {}).items():
            cp = r.get("fpr_cp95")
            a(f"| `{scen}` | {cfg} | {r['n_no_true']} | {_f(r['fpr'])} | "
              f"{'[' + _f(cp[0]) + ', ' + _f(cp[1]) + ']' if cp else 'n/a'} | "
              f"{('pass' if r['gate_pass'] else 'FAIL') if 'gate_pass' in r else '—'} | "
              f"{_f(r.get('naive_curl_fpr'))} | {r['n_true_cycle_blocks']} | {_f(r['recall'])} | {_f(r['flag_precision'])} |")
    a("")
    a("Triangle level (certified cycles): ")
    a("")
    a("| scenario | cell | true cycles | certified cycles | triangle recall | triangle precision | blocks with an observed cycle |")
    a("|---|---|---|---|---|---|---|")
    for scen in ["condorcet", "mixed", "harmless_disagreement"]:
        for cfg, r in d1["by_scenario"].get(scen, {}).items():
            a(f"| `{scen}` | {cfg} | {r['n_true_cycles']} | {r['n_cert']} | {_f(r['triangle_recall'])} | "
              f"{_f(r['triangle_precision'])} | {_f(r['frac_blocks_obs_cycle'])} |")
    a("")
    a("Naive curl-mass test (expected to fail on `transitive_binarized`; not a gate). Mean curl fraction of the "
      "observed aggregate flow:")
    a("")
    a("| scenario | cell | mean curl fraction | naive curl-test FPR |")
    a("|---|---|---|---|")
    for scen in WSUM_SCENARIOS:
        for cfg, r in d1["by_scenario"].get(scen, {}).items():
            a(f"| `{scen}` | {cfg} | {_f(r.get('mean_curl_frac'))} | {_f(r.get('naive_curl_fpr'))} |")
    a("")
    a("Antisymmetrization check (`position_bias`, one presentation order, bias 1.0 log-odds). The same blocks "
      "were also scored with the second order added:")
    a("")
    a("| cell | curl fraction, one order | curl fraction, both orders | flag rate, one order | flag rate, both orders |")
    a("|---|---|---|---|---|")
    for cfg, r in d1["by_scenario"].get("position_bias", {}).items():
        a(f"| {cfg} | {_f(r['curl_frac_one_order'])} | {_f(r['curl_frac_both_orders'])} | "
          f"{_f(r['flag_rate_one_order'])} | {_f(r['flag_rate_both_orders'])} |")
    a("")
    # ---------- primary ----------
    pr = S["primary"]
    a("## 2. Primary family: D2 and D3 (Holm over {D2, D3})")
    a("")
    if not S["interpretable"]:
        a("**Not interpretable, because D1 failed.**")
        a("")
    a(f"Rule: {pr['rule']}")
    a("")
    a("| hypothesis | p | Holm-adjusted p | significant at 0.05 | verdict |")
    a("|---|---|---|---|---|")
    for h in ("D2", "D3"):
        if h in pr["holm"]:
            a(f"| {h} | {_p(pr['holm'][h]['p'])} | {_p(pr['holm'][h]['p_holm'])} | "
              f"{'yes' if pr['holm'][h]['significant'] else 'no'} | **{pr['verdicts'][h]['verdict']}** |")
    a("")
    if "condorcet" in S["d2"] and "mixed" in S["d2"] and "mixed" in S["d3"]:
        d2s, d3s = S["d2"], S["d3"]
        a("**Plain reading.** The verdicts above are against the pre-registered baselines. Two stronger "
          "comparisons change the meaning:")
        a("")
        for sc in ("condorcet", "mixed"):
            st = d2s[sc]["delta_strong_boot"]
            a(f"- D2, `{sc}`: when the baseline also knows whether the RM sign disagrees with the observed panel "
              f"label, the baseline AUROC is {_f(np.nanmean(d2s[sc]['auc_strong']))} and the Hodge features add "
              f"Δ = {_f(st['mean'], 4)} (95% CI [{_f(st['ci'][0], 4)}, {_f(st['ci'][1], 4)}]). The Hodge "
              "features then add nothing measurable.")
        for sc in ("condorcet", "mixed"):
            if sc in d3s:
                t = d3s[sc]["spearman_vs_pivot"]
                a(f"- D3, `{sc}`: against 'pivotality' (edges whose aggregate sign changes under leave-one-out, no "
                  f"cycles), the Hodge score wins by mean Spearman {_f(t['test'].get('mean'))} (Hodge "
                  f"{_f(d3s[sc]['spearman']['hodge_mean'])}, pivotality {_f(t['pivot_mean'])}; Wilcoxon p "
                  f"{_p(t['test']['p'])}, wins/ties/losses {t['test']['wins']}/{t['test']['ties']}/{t['test']['losses']}).")
        a("- The D3 truth is defined as the estimand of the Hodge score (§ 4). The large win over raw "
          "disagreement is therefore expected by construction.")
        a("")
    # ---------- D2 ----------
    d2 = S["d2"]
    a("### 2.1 D2: failure localization")
    a("")
    a("Target: the scalar BT reward model is wrong on a held-out pair, relative to the true aggregate relation. "
      "Score: held-out AUROC of a logistic regression that was fitted on training blocks only. "
      "Baseline features: vote entropy across principles, |observed aggregate margin|, |RM margin|. "
      "Hodge features: certified cycles through the edge, |curl component|, block flag. "
      "Δ = AUROC(baseline + Hodge) − AUROC(baseline). Per seed, Δ is the mean over the 4 (P, V) cells. "
      f"The CI is a paired bootstrap over {len(J['config']['seeds'])} seeds (10 000 resamples).")
    a("")
    a("| scenario | AUROC baseline | AUROC baseline + Hodge | mean Δ | 95% CI | bootstrap p | Δ ≥ 0.02 and CI > 0 |")
    a("|---|---|---|---|---|---|---|")
    for scen in SCENARIOS:
        if scen not in d2:
            continue
        r = d2[scen]
        db = r["delta_boot"]
        a(f"| `{scen}`{' (confirmatory)' if scen in ('condorcet', 'mixed') else ''} | {_f(np.nanmean(r['auc_base']))} | "
          f"{_f(np.nanmean(r['auc_full']))} | {_f(db['mean'], 4)} | [{_f(db['ci'][0], 4)}, {_f(db['ci'][1], 4)}] | "
          f"{_p(db['p'])} | {'yes' if (np.isfinite(db['mean']) and db['mean'] >= D2_MIN_GAIN and db['ci'][0] > 0) else 'no'} |")
    a("")
    a("Per (P, V) cell, confirmatory scenarios:")
    a("")
    a("| scenario | cell | RM error rate (held-out) | AUROC base | AUROC base + Hodge | Δ mean | Δ 95% CI |")
    a("|---|---|---|---|---|---|---|")
    for scen in ("condorcet", "mixed"):
        for cfg, r in d2.get(scen, {}).get("per_cfg", {}).items():
            a(f"| `{scen}` | {cfg} | {_f(r['rm_err_rate_held'])} | {_f(r['auc_base'])} | {_f(r['auc_full'])} | "
              f"{_f(r['delta_boot']['mean'], 4)} | [{_f(r['delta_boot']['ci'][0], 4)}, {_f(r['delta_boot']['ci'][1], 4)}] |")
    a("")
    # ---------- D3 ----------
    d3 = S["d3"]
    a("### 2.2 D3: attribution to principle pairs")
    a("")
    a("Hodge score of pair (p, q) = certified cycles that leave-one-out of p or of q removes (re-tested with the "
      "D1 test), averaged over the 300 blocks. Baseline = raw disagreement rate of graders p and q. Truth = true "
      "cycles that leave-one-out of p or q removes, on the true utilities. Per seed, the metric is the mean over "
      "the 4 (P, V) cells. Test: paired Wilcoxon over seeds.")
    a("")
    a("| scenario | metric | Hodge mean | baseline mean | mean diff | Wilcoxon p | wins/ties/losses |")
    a("|---|---|---|---|---|---|---|")
    for scen in D3_SCENARIOS:
        if scen not in d3:
            continue
        for metric, lab in (("spearman", "Spearman"), ("prec", "precision@k")):
            r = d3[scen][metric]
            t = r["test"]
            a(f"| `{scen}` | {lab} | {_f(r['hodge_mean'])} | {_f(r['base_mean'])} | {_f(t.get('mean'))} | {_p(t['p'])} | "
              f"{t['wins']}/{t['ties']}/{t['losses']} |")
    a("")
    if "harmless_disagreement" in d3:
        h = d3["harmless_disagreement"]
        a("`harmless_disagreement` has no true cycles by construction, so the true tension is 0 for every pair "
          "and Spearman and precision@k are undefined there (see § 7). The substitute metric is the share of each "
          "method's total score that falls on the harmless pair ((0, 1) in `harmless_disagreement`; (0, 3) at P = 4 "
          "and (3, 4) at P = 6 in `mixed`). Lower is better.")
        a("")
        a("| scenario | Hodge share | disagreement share | pivotality share | Wilcoxon p (disagreement − Hodge) | harmless pair ranked first: Hodge / disagreement |")
        a("|---|---|---|---|---|---|")
        for scen in ("harmless_disagreement", "mixed"):
            if scen not in d3:
                continue
            hs = d3[scen]["harmless_share"]
            a(f"| `{scen}` | {_f(hs['hodge_mean'])} | {_f(hs['base_mean'])} | {_f(hs['pivot_mean'])} | {_p(hs['test']['p'])} | "
              f"{_f(d3[scen]['harmless_top1_rate']['hodge'])} / {_f(d3[scen]['harmless_top1_rate']['base'])} |")
        a("")
    a("Mean scores per principle pair (`mixed`):")
    a("")
    for cfg, r in d3.get("mixed", {}).get("per_cfg", {}).items():
        a(f"- {cfg}: pairs {r['pairs']}")
        a(f"  - truth: {r['truth_union_mean']}")
        a(f"  - Hodge: {r['hodge_union_mean']}")
        a(f"  - disagreement: {r['disagree_mean']}")
        a(f"  - pivotality: {r['pivot_mean']}")
        a(f"  - mean Spearman: Hodge {_f(r['spearman_hodge'])}, disagreement {_f(r['spearman_base'])}, "
          f"pivotality {_f(r['spearman_pivot'])}")
    a("")
    # ---------- D4 ----------
    d4 = S["d4"]
    a("## 3. D4: decision relevance (exploratory, no correction)")
    a("")
    a("Skew-symmetric pairwise model (linear part + skew bilinear form on item features) vs the scalar BT model, "
      "both fitted on the same training blocks. Gain = held-out accuracy (skew) − accuracy (BT) against the true "
      "aggregate relation. Interaction = gain in D1-flagged blocks − gain in unflagged blocks (per seed).")
    a("")
    a("| scenario | acc BT | acc skew | gain, all | gain, flagged blocks | gain, unflagged blocks | interaction (95% CI) | Wilcoxon p |")
    a("|---|---|---|---|---|---|---|---|")
    for scen in SCENARIOS:
        if scen not in d4:
            continue
        r = d4[scen]
        it = r["interaction"]
        a(f"| `{scen}` | {_f(r['acc_bt'])} | {_f(r['acc_skew'])} | {_f(r['gain_overall']['mean'], 4)} | "
          f"{_f(r['gain_flagged']['mean'], 4)} | {_f(r['gain_unflagged']['mean'], 4)} | "
          f"{_f(it['mean'], 4)} [{_f(it['ci'][0], 4)}, {_f(it['ci'][1], 4)}] | {_p(r['interaction_wilcoxon']['p'])} |")
    a("")
    # ---------- bias ----------
    a("## 4. Ways this simulation could favour the diagnostic")
    a("")
    for line in J.get("bias_findings", []):
        a(line)
    a("")
    a("## 5. What this experiment cannot show")
    a("")
    a("- Everything is simulated. The graders are a model of constitutional principle graders. They are not "
      "real LLM judges and not humans. The result shows only that the diagnostic **can** work when the structure "
      "is present.")
    a("- The real-grader test is SGB-045 Stage 2 (log-prob judges with a constitution ablation).")
    a("")
    a("## 6. Implementation choices that the pre-registration did not fix")
    a("")
    for line in J.get("implementation_notes", []):
        a(line)
    a("")
    a("## 7. Deviations from pre-registration")
    a("")
    for line in J.get("deviations", []):
        a(line)
    a("")
    a("## 8. Calibration pilot for the D1 null (seeds 1000–1004, not used in any result above)")
    a("")
    cal = J["calibration"]
    a(f"Rule: {cal['rule']}.")
    a("")
    a(f"Chosen: deflation variant `{cal['chosen_deflate']}`, c = {cal['chosen_c']}.")
    a("")
    if cal.get("table"):
        a("| variant | c | worst gate FPR | " + " | ".join(f"condorcet recall {cfg_key(P, V)}" for P, V in CFGS) + " |")
        a("|---|---|---|" + "---|" * len(CFGS))
        for dm, tbm in cal["table"].items():
            for c in cal["c_grid"]:
                tb = tbm[str(c)]
                a(f"| `{dm}` | {c} | {_f(cal['worst_gate_fpr'][dm][str(c)])} | " +
                  " | ".join(_f(tb.get("condorcet", {}).get(cfg_key(P, V), {}).get("recall")) for P, V in CFGS) + " |")
        a("")
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 seeds, 2 pilot seeds; writes to --out")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--md", type=str, default=None)
    ap.add_argument("--rebuild-md", action="store_true")
    ap.add_argument("--c", type=float, default=None, help="skip the pilot and use this c")
    ap.add_argument("--deflate", type=str, default="reoriented", choices=DEFLATE_MODES)
    args = ap.parse_args()
    out_json = Path(args.out) if args.out else OUT_JSON
    out_md = Path(args.md) if args.md else OUT_MD
    if args.rebuild_md:
        J = json.loads(out_json.read_text())
        J = add_text_sections(J)
        out_json.write_text(json.dumps(J, indent=1))
        write_markdown(J, out_md)
        print(f"wrote {out_md}")
        return

    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if sha != PREREG_SHA256:
        raise SystemExit(f"pre-registration hash mismatch: {sha}")
    align = check_alignment()
    t0 = time.time()
    seeds = SEEDS[:2] if args.smoke else SEEDS
    pilot = PILOT_SEEDS[:2] if args.smoke else PILOT_SEEDS
    if args.c is None:
        calib = calibrate(args.workers, pilot)
        c, dm = calib["chosen_c"], calib["chosen_deflate"]
    else:
        calib = dict(c_grid=[args.c], pilot_seeds=[], table={}, worst_gate_fpr={}, per_mode={},
                     chosen_c=args.c, chosen_deflate=args.deflate, rule="given on the command line")
        c, dm = args.c, args.deflate
    print(f"[calibration] deflate = {dm}, c = {c}  ({time.time() - t0:.0f}s)", flush=True)
    jobs = [(sd, sc, P, V, c, dm, "main") for sc in SCENARIOS for P, V in CFGS for sd in seeds]
    jobs.sort(key=lambda j: (j[1] not in D3_SCENARIOS, -j[2], -j[3]))   # long jobs first
    raw = {}
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for t in ex.map(run_task, jobs):
            n += 1
            raw.setdefault(t["scenario"], {}).setdefault(cfg_key(t["P"], t["V"]), []).append(t)
            if n % 50 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    for sc in raw:
        for cfg in raw[sc]:
            raw[sc][cfg].sort(key=lambda t: t["seed"])
    summary = summarise(raw, calib)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        git = None
    cmd = COMMAND + (" --smoke" if args.smoke else "") + (f" --c {args.c} --deflate {args.deflate}" if args.c is not None else "") + \
        (f" --out {args.out}" if args.out else "") + (f" --md {args.md}" if args.md else "") + \
        f" --workers {args.workers}"
    J = dict(experiment="SGB-046: Hodge structure as a reward-model diagnostic for heterogeneous grader panels",
             script="scripts/sgb046_reward_diagnostic_sim.py",
             output_json=str(out_json.relative_to(ROOT)) if str(out_json).startswith(str(ROOT)) else str(out_json),
             prereg=str(PREREG.relative_to(ROOT)), prereg_sha256=sha, prereg_hash_ok=sha == PREREG_SHA256,
             command=cmd, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), git_head=git, workers=args.workers,
             versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                           sklearn=sklearn.__version__, platform=platform.platform()),
             runtime_seconds=time.time() - t0,
             config=dict(seeds=seeds, pilot_seeds=pilot, n_train=N_TRAIN, n_held=N_HELD, k=K, Ps=list(PS),
                         Vs=list(VS), B=B, alpha=ALPHA, c=c, deflate=dm, c_grid=list(C_GRID), pilot_fpr_target=PILOT_FPR_TARGET,
                         T_range=[T_LO, T_HI], g_scale=G_SCALE, s_scale=S_SCALE, harmless_factor=HARMLESS_FACTOR, neartie_scale=NEARTIE_SCALE,
                         bias_two_order=BIAS_TWO_ORDER, bias_one_order=BIAS_ONE_ORDER, tiebreak=TIEBREAK,
                         rm_Cs=list(RM_CS), n_folds=N_FOLDS, n_boot_seeds=N_BOOT_SEEDS, d2_min_gain=D2_MIN_GAIN,
                         scenarios={s: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                        for k, v in scenario_spec(s, P).items()}
                                    for s in SCENARIOS for P in [6]},
                         scenarios_P4={s: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                           for k, v in scenario_spec(s, 4).items()} for s in SCENARIOS}),
             assertions=dict(index_alignment=align, heldout_isolation="asserted in fit_pair_model and run_task",
                             both_orders="asserted in run_task for every scenario except position_bias",
                             positive_control_recall_V25=summary["d1"]["positive_control_recall_V25"],
                             positive_control_pass=summary["d1"]["positive_control_pass"]),
             calibration=calib, summary=summary, raw=raw)
    J = add_text_sections(J)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(J, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    write_markdown(json.loads(out_json.read_text()), out_md)
    print(json.dumps(dict(gate=summary["d1"]["gate_pass"], pos=summary["d1"]["positive_control_recall_V25"],
                          primary=summary["primary"]), indent=1, default=str))
    print(f"wrote {out_json}\nwrote {out_md}\nruntime {time.time() - t0:.0f}s")


def add_text_sections(J):
    """Bias findings, implementation notes and deviations. Numbers are read from the JSON summary."""
    S = J["summary"]
    d2, d3, d1 = S["d2"], S["d3"], S["d1"]["by_scenario"]
    bias = []

    def g(dct, *ks):
        for k in ks:
            if not isinstance(dct, dict) or k not in dct:
                return float("nan")
            dct = dct[k]
        return dct

    # D2 construction checks
    for scen in ("condorcet", "mixed"):
        if scen not in d2:
            continue
        r = d2[scen]
        pc = r["per_cfg"]
        bias.append(f"- **D2, `{scen}`: oracle check.** A score that counts the TRUE cycles through an edge has held-out "
                    "AUROC " + ", ".join(f"{k} {_f(v['single']['oracle_true_cycles'])}" for k, v in pc.items()) +
                    ". The certified-cycle count alone has AUROC " +
                    ", ".join(f"{k} {_f(v['single']['ncert'])}" for k, v in pc.items()) +
                    ". The RM error rate on true-cycle edges is " +
                    ", ".join(f"{k} {_f(v['rm_err_true_cycle_edges'])}" for k, v in pc.items()) +
                    " and on other edges is " + ", ".join(f"{k} {_f(v['rm_err_other_edges'])}" for k, v in pc.items()) +
                    ". So part of the Hodge signal is structural (a scalar must be wrong on one edge of every true "
                    "cycle), as the pre-registration disclosed.")
        bias.append(f"- **D2, `{scen}`: is disagreement low on cycle edges?** Mean vote entropy on true-cycle edges vs "
                    "other edges: " + ", ".join(f"{k} {_f(v['entropy_true_cycle_edges'])} vs {_f(v['entropy_other_edges'])}"
                                                for k, v in pc.items()) + ".")
        ds = r["delta_strong_boot"]
        bias.append(f"- **D2, `{scen}`: stronger baseline.** The pre-registered baseline does not contain the most direct "
                    "error signal that is available on a held-out block: whether the RM sign disagrees with the observed "
                    "panel label. With that feature and the signed product (RM margin × observed margin) added, "
                    f"the baseline AUROC is {_f(np.nanmean(r['auc_strong']))} and the Hodge features add Δ = "
                    f"{_f(ds['mean'], 4)} (95% CI [{_f(ds['ci'][0], 4)}, {_f(ds['ci'][1], 4)}], bootstrap p {_p(ds['p'])}). "
                    "The single feature 'RM disagrees with the observed label' has AUROC " +
                    ", ".join(f"{k} {_f(v['single']['rm_disagree'])}" for k, v in pc.items()) + ".")
    # D3 construction checks
    if "mixed" in d3:
        r = d3["mixed"]
        bias.append("- **D3: the truth is the estimand of the Hodge score.** The true tension is 'true cycles removed "
                    "by leave-one-out of p or q'. The Hodge score is the same quantity computed on noisy labels. The "
                    "disagreement baseline estimates a different quantity. So D3 favours the Hodge score by "
                    "definition, and a win shows only that the plug-in estimate survives the label noise.")
        sp = r["spearman_vs_pivot"]
        bias.append("- **D3: a fairer non-Hodge baseline.** 'Pivotality' counts the edges whose observed aggregate "
                    "sign changes under leave-one-out of p or q. It uses no cycles and no bootstrap. In `mixed`, mean "
                    f"Spearman: Hodge {_f(r['spearman']['hodge_mean'])}, pivotality {_f(sp['pivot_mean'])}, disagreement "
                    f"{_f(r['spearman']['base_mean'])}. Hodge − pivotality: mean {_f(sp['test'].get('mean'))}, Wilcoxon p "
                    f"{_p(sp['test']['p'])}, wins/ties/losses {sp['test']['wins']}/{sp['test']['ties']}/{sp['test']['losses']}.")
        bias.append("- **D3: the harmless pairs were built to disagree maximally** (utilities exactly or strongly "
                    "anti-correlated). That is the case the pre-registration names, but it is the worst case for the "
                    "disagreement baseline. Mean disagreement rate per pair is in § 2.2.")
    # D1
    bias.append("- **D1: the null design (order rule, deflation variant, c) was chosen on this simulator** (pilot "
                "seeds 1000–1004). The main-seed FPR is an out-of-sample check on new seeds of the same simulator. It is not a check on "
                "a different noise model. Real graders with other noise (for example correlated votes, or "
                "label noise that is not binomial) can make the test anti-conservative.")
    bias.append("- **The signal-to-noise scale was raised on pilot seeds after block recall at V = 25 was about 0.5.** "
                "At the earlier scale, about half of the true-cycle blocks hinged on a principle that was close to a "
                "tie on a pivotal edge, and no calibrated test can certify such a cycle. The D2–D4 results are "
                "therefore conditional on a regime where most planted cycles are certifiable at V = 25. At V = 5 "
                "recall is much lower; see the D1 table.")
    bias.append("- **Grader noise is independent binomial votes with a known logit link.** The null resamples votes "
                "with the same link. This matches the simulator exactly, which favours calibration.")
    bias.append("- **Principle utilities are linear in the item features and each principle reads its own feature "
                "dimension.** The skew-bilinear model in D4 and the BT model in D2 use the same features, so model "
                "misspecification comes only from the aggregation rule.")
    impl = [
        "1. **D1 null (transitive-consistent parametric bootstrap).** (a) Order: the total order of the 5 "
        "responses that reorients the least plug-in evidence, found by exhaustive search over all 120 orders "
        "(cost = sum of |z| of the observed aggregate sign over the edges that disagree with the order; ties broken "
        "by the least-squares BT order). This is the maximum-likelihood transitive order under a normal "
        "approximation. (b) On edges that disagree with the order, the null flips all principle labels, so every "
        "edge points along the order with its observed strength; the expected null relation is therefore "
        "transitive. (c) Optional deflation toward a tie by c standard units (winner's-curse correction), on the "
        "reoriented edges only or on all edges. (d) Votes are resampled from the null log-odds in both orders "
        "with the observed position effect, B = 200. The variant and c were chosen on pilot seeds 1000–1004 by "
        "the fixed rule in § 8. A pure-tie null (every disagreeing edge set to a tie) was rejected at design "
        "time: it can never certify a single cycle, because the tied edge reproduces the cycle in about half of "
        "the replicates.",
        "2. **Certified cycle.** An observed directed 3-cycle on triangle t in a D1-flagged block, with per-triangle "
        "bootstrap p ≤ 0.05 (the null reproduces the same directed cycle on t in ≤ 5% of replicates).",
        "3. **Aggregators.** Constitution vote = weighted principle vote sum_p v_p sign(f_p) with tie-break weights "
        f"of at most {TIEBREAK}. Weighted-sum control = sum_p f_p (unit weights) on graded labels, so the truth is "
        "sign(sum_p eta_p) with eta the true log-odds.",
        f"4. **Scenario parameters.** Utilities u_p = g·q + s·(principle component), g = {G_SCALE:.3f}, s = {S_SCALE}, "
        f"grader temperatures T_p ~ U({T_LO}, {T_HI}), first-position bias {BIAS_TWO_ORDER} log-odds. `condorcet`: "
        "principle components are centred across the P principles (balanced trade-off). `harmless_disagreement`: "
        f"u_0 = g·q + {HARMLESS_FACTOR:.0f}s·x_0, u_1 = g·q − {HARMLESS_FACTOR:.0f}s·x_0, and principle 2 has vote "
        "weight P − 0.5, so it decides every vote. `mixed`: principles 0–2 form a centred trade-off trio "
        f"(Condorcet cycles); P = 4: u_3 = g·q − {HARMLESS_FACTOR:.0f}s·x_0 with weight 0.5; P = 6: u_3, u_4 = "
        f"g·q ± {HARMLESS_FACTOR:.0f}s·x_3 and u_5 = g·q + s·x_4, weights 0.3. These principles are never pivotal in "
        f"the true vote. `transitive_neartie`: utilities × {NEARTIE_SCALE}. `position_bias`: one order (the "
        f"lower-index response is shown first), bias {BIAS_ONE_ORDER} log-odds.",
        "5. **Reward models.** Logistic regression without intercept on antisymmetric pair features, trained on the "
        "sign of the observed aggregate label (ties dropped). C is chosen by 5-fold grouped CV over training blocks "
        f"from {list(RM_CS)}. The D2 error predictor is trained on out-of-fold (cross-fitted) RM margins of the "
        "training blocks, then scored on held-out blocks with the RM fitted on all training blocks.",
        "6. **Combining P and V.** The pre-registration does not say how the 4 (P, V) cells combine. Each seed's "
        "metric is the mean over its 4 cells; tests run over the 40 seed-level values. Per-cell values are in the JSON.",
        "7. **One p-value per hypothesis.** D2 predicts a gain in BOTH `condorcet` and `mixed`, so p_D2 = the larger "
        "of the two bootstrap p-values (intersection-union). D3's kill criterion names `mixed`, so p_D3 = the "
        "Wilcoxon p for the Spearman difference in `mixed`. Holm runs over {D2, D3}. precision@k is secondary.",
        "8. **D3 pair score uses the union** of the single-principle leave-one-out sets ('p or q'), as written. "
        "The intersection variant is reported in the JSON as exploratory.",
        "9. **precision@k** uses expected precision under random tie-breaking. When every pair is truly positive "
        "or none is, precision@k is undefined and is not counted. Spearman of a constant score is set to 0.",
    ]
    dev = [
        "- **D3 in `harmless_disagreement`:** the scenario has no true cycles by construction (as its table row "
        "says), so the true tension is 0 for every pair and the pre-registered Spearman and precision@k are "
        "undefined. The results file reports a substitute: the share of each score on the harmless pair. This "
        "substitute is not part of the Holm family. The D3 kill criterion is defined on `mixed` only, so it is "
        "not affected.",
        "- **D1 gate scope:** the gate is applied to `transitive`, `transitive_binarized` and `transitive_neartie`. "
        "`position_bias` also has a transitive truth, but the pre-registration lists it as the antisymmetrization "
        "check (one order only), so its FPR is reported and is not a gate.",
        "- **D1 null details (order, deflation variant, c):** the pre-registration does not fix them. They were "
        "selected on pilot seeds that are disjoint from seeds 0–39, before the main run (§ 6 item 1, § 8).",
        "- **Design history before the main run (disclosed in full).** (1) One debugging pass ran one task per "
        "scenario on MAIN seed 0 with an earlier scenario design (no shared quality factor and degenerate "
        "tie-break weights, so the true vote tied on 3–3 splits and the scalar RM was near chance everywhere). "
        "Those numbers were discarded and that design was replaced. Seed 0 was later re-run with the final design "
        "like every other seed. (2) All later design work used pilot seeds 1000–1004 only: (a) shared quality "
        "factor and binary tie-break weights; (b) the D1 order changed from the least-squares BT order to the "
        "minimum-evidence order, because the least-squares order depended on binarization of the aggregate label "
        "and gave a higher FPR in `transitive_binarized`; (c) two deflation variants were compared; (d) the "
        "signal-to-noise scale (g, s) was raised by 5/3 because condorcet block recall at V = 25 was about 0.5 "
        "at the earlier scale, which is at the pre-registered floor of § 4. The true relation (and so every true "
        "cycle) does not depend on this scale; only vote noise relative to signal changes. Item (d) favours the "
        "diagnostic (see § 4).",
    ]
    if J.get("extra_deviations"):
        dev.extend(J["extra_deviations"])
    J["bias_findings"] = bias
    J["implementation_notes"] = impl
    J["deviations"] = dev
    return J


if __name__ == "__main__":
    main()
