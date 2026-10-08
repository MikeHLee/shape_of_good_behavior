#!/usr/bin/env python3
"""SGB-045 Stage 1: ground-truth simulation of Hodge-targeted data acquisition.

Question: at an EQUAL budget of new labels, does "fill harmonic holes + re-label high-curl
edges" (ours) beat the algebraic-connectivity / Fisher-information greedy baseline
(Osting, Brune, Osher ICML 2013; Xu et al. AAAI 2018) and random acquisition on
HELD-OUT comparisons?

World (per seed)
----------------
- N_ITEMS items, each with N_CRIT latent criterion scores x_i ~ N(0, I).
- Annotator types m with criterion weights w_m and population shares pi_m. One label on
  pair (i, j) comes from one annotator of a random type: clean label =
  sign(w_m . (x_j - x_i)), then flipped with probability FLIP (label noise).
- Population truth: majority relation M(i, j) = sign(sum_m pi_m sign(w_m . (x_j - x_i))).
  A triple is "genuinely intransitive" iff M is cyclic on it (Condorcet cycle). An edge is
  "intransitive" iff it lies on at least one such triple.
- True aggregate score s = x @ (sum_m pi_m w_m) (for Kendall tau).
- Label streams are pre-drawn per (pair, slot) (common random numbers), so two policies
  that query the same pair get the same labels. Held-out labels are a separate stream.

Graph and budget
----------------
- N_HELDOUT pairs are held out first. No policy may query them. Their labels are never
  seen by any policy or any fit.
- Initial graph: random spanning tree + random extra pairs (E0 edges, 1 label each), all
  outside the held-out set. It is sparse, so it has many unfilled loops (beta_1 > 0).
- One action = one new label, either on a new pair or a re-label of an existing pair
  (max MAX_LABELS labels per pair). Snapshots are taken at BUDGETS (prefix budgets).
- Edge flow = mean of its +-1 labels (+ means j preferred, i < j); edge weight = label count.

Policies
--------
- random       : uniform new pair.
- random_mix   : equal-count control for ours. At step t it re-labels a uniformly random
                 existing edge iff `ours` re-labelled at step t, else a uniform new pair.
- fiedler      : exact greedy maximisation of lambda_2 of the label-count-weighted graph
                 Laplacian over ALL allowed pairs (new pairs and re-labels), evaluated
                 exactly for every candidate at every step (Osting 2013 / Xu 2018 baseline).
- ours         : the Hodge-targeted policy. It alternates, one step each: fill a harmonic
                 hole, then re-label the edge with the largest |curl| component.
                 Fill a hole: score non-edges (i,k) by the harmonic mass on the 2-paths
                 i-j-k they would close into triangles; for the top LOOKAHEAD candidates,
                 impute the new flow as clip(phi_k - phi_i, -1, 1) and pick the one whose
                 addition gives the smallest harmonic norm. If no candidate touches any
                 harmonic mass (all holes already closed), fall back to a uniform new pair
                 and count the step as a fallback.
- ours_adaptive: variant. Fill a hole if ||harmonic||_W^2 >= ||curl||_W^2, else re-label.
- ours_fill    : ablation, always fill a hole.
- ours_relabel : ablation, always re-label the top-|curl| edge.

Metrics (held-out, never used for acquisition)
----------------------------------------------
- heldout_acc_noisy : sign(phi_j - phi_i) vs the held-out observed label.
- heldout_acc_clean : sign(phi_j - phi_i) vs the population majority M on held-out pairs.
- kendall_tau       : tau(phi, s) over all items.
- *_2crit           : same accuracies for a 2-criteria logistic model
                      z = phi_j - phi_i + a_i b_j - a_j b_i (can represent cycles).
Diagnostics (in-sample): harmonic/curl/gradient fractions, beta_1, #triangles,
noise-edge flag precision/recall (initial edges; flag = initial label disagrees with phi),
intransitive-triple flag precision/recall (flag = filled triangle with |d1 f| >= 2).

Kill criterion (fixed before running; proposal section 5, Stage 1)
-----------------------------------------------------------------
In the MAIN scenario, `ours` "beats" `fiedler` only if BOTH hold:
  (1) heldout_acc_clean: mean paired diff > 0 with Wilcoxon p < 0.05 at >= half of the
      positive budgets;
  (2) no held-out metric (heldout_acc_clean, heldout_acc_noisy, kendall_tau,
      heldout_acc_clean_2crit) has `ours` significantly worse (p < 0.05, mean diff < 0) at
      any positive budget.
Otherwise the kill criterion FIRES.

Reproduce:
    ./venv/bin/python3 scripts/sgb045_hodge_acquisition_sim.py
"""

from __future__ import annotations

import argparse
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
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import kendalltau, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feedback_geometry" / "src"))
from hodge_complex import build_clique_complex, hodge_decompose_2d  # noqa: E402

OUT_JSON = ROOT / "shared" / "results" / "sgb045_hodge_acquisition_sim_v1.json"
OUT_MD = ROOT / "feedback_geometry" / "SGB045_STAGE01_RESULTS.md"
COMMAND = "./venv/bin/python3 scripts/sgb045_hodge_acquisition_sim.py"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITEMS = 40
N_CRIT = 3
E0 = 60
N_HELDOUT = 200
MAX_LABELS = 5
BUDGETS = [0, 20, 40, 80, 160]
SEEDS = list(range(40))
LOOKAHEAD = 10
TRIANGLE_FLAG_TAU = 2.0
RIDGE_2CRIT = 0.5

W_MILD = [[1.0, 0.3, 0.0], [0.0, 1.0, 0.3], [0.3, 0.0, 1.0]]
W_ONEHOT = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
W_SINGLE = [list(np.mean(np.array(W_MILD), axis=0))]

SCENARIOS = {
    # primary scenario for the kill criterion
    "main": dict(type_weights=W_MILD, type_probs=[1 / 3] * 3, flip=0.15),
    # sensitivity: no genuine intransitivity (one annotator type), label noise only
    "noise_only": dict(type_weights=W_SINGLE, type_probs=[1.0], flip=0.15),
    # sensitivity: genuine intransitivity only, no label noise
    "intransitive_only": dict(type_weights=W_MILD, type_probs=[1 / 3] * 3, flip=0.0),
    # sensitivity: stronger intransitivity (one-hot criteria) + label noise
    "strong_intransitive": dict(type_weights=W_ONEHOT, type_probs=[1 / 3] * 3, flip=0.15),
}
POLICIES = ["random", "random_mix", "fiedler", "ours", "ours_adaptive", "ours_fill",
            "ours_relabel"]
HELDOUT_METRICS = ["heldout_acc_clean", "heldout_acc_noisy", "kendall_tau",
                   "heldout_acc_clean_2crit", "heldout_acc_noisy_2crit"]
DIAG_METRICS = ["frac_harmonic", "frac_curl", "frac_gradient", "betti1", "n_triangles",
                "n_edges", "n_relabels", "noise_flag_precision", "noise_flag_recall",
                "noise_flag_f1", "triple_flag_precision", "triple_flag_recall_all",
                "triple_n_flagged", "edges_noise_driven", "edges_intransitive",
                "relabels_on_noise_edges", "relabels_on_intransitive_edges"]
COMPARISONS = [("ours", "fiedler"), ("ours", "random"), ("ours", "random_mix"),
               ("ours_adaptive", "fiedler"), ("ours_fill", "fiedler"),
               ("ours_fill", "random"), ("ours_relabel", "fiedler"),
               ("fiedler", "random")]


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
class World:
    def __init__(self, seed: int, scen: dict):
        rng = np.random.default_rng([seed, 45])
        n = N_ITEMS
        self.n = n
        self.W = np.asarray(scen["type_weights"], float)
        self.pi = np.asarray(scen["type_probs"], float)
        self.flip = float(scen["flip"])
        self.X = rng.normal(size=(n, N_CRIT))
        self.pairs = np.array(list(combinations(range(n), 2)), dtype=int)  # canonical i<j
        self.P = len(self.pairs)
        self.pair_id = {(int(i), int(j)): p for p, (i, j) in enumerate(self.pairs)}
        D = self.X[self.pairs[:, 1]] - self.X[self.pairs[:, 0]]           # (P, K)
        self.type_sign = np.sign(D @ self.W.T)                             # (P, m)
        assert np.all(self.type_sign != 0)
        maj = self.type_sign @ self.pi
        assert np.all(np.abs(maj) > 1e-12), "majority tie"
        self.M = np.sign(maj)                                              # (P,)
        self.s = self.X @ (self.pi @ self.W)
        # pre-drawn label streams (common random numbers across policies)
        self.lab_type = rng.choice(len(self.pi), size=(self.P, MAX_LABELS), p=self.pi)
        self.lab_flip = rng.random((self.P, MAX_LABELS)) < self.flip
        self.lab_clean = self.type_sign[np.arange(self.P)[:, None], self.lab_type]
        self.labels = np.where(self.lab_flip, -self.lab_clean, self.lab_clean)
        # held-out pairs and their own label stream
        self.heldout = np.sort(rng.choice(self.P, size=N_HELDOUT, replace=False))
        self.is_heldout = np.zeros(self.P, bool)
        self.is_heldout[self.heldout] = True
        ho_type = rng.choice(len(self.pi), size=N_HELDOUT, p=self.pi)
        ho_flip = rng.random(N_HELDOUT) < self.flip
        ho_clean = self.type_sign[self.heldout, ho_type]
        self.ho_label = np.where(ho_flip, -ho_clean, ho_clean)
        # genuine intransitivity (Condorcet cycles of M)
        cyc = []
        for i, j, k in combinations(range(n), 3):
            a = self.M[self.pair_id[(i, j)]]
            b = self.M[self.pair_id[(j, k)]]
            c = self.M[self.pair_id[(i, k)]]
            if a == b == -c:
                cyc.append((i, j, k))
        self.cyclic = set(cyc)
        self.intrans_pair = np.zeros(self.P, bool)
        for i, j, k in cyc:
            for e in ((i, j), (j, k), (i, k)):
                self.intrans_pair[self.pair_id[e]] = True
        # initial sparse connected graph outside the held-out set
        self.init_pids = self._initial_graph(rng)

    def _initial_graph(self, rng):
        n = self.n
        # random spanning tree: Kruskal over the non-held-out pairs in random order
        chosen = set()
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for p in rng.permutation(np.flatnonzero(~self.is_heldout)):
            i, j = map(int, self.pairs[p])
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                chosen.add(int(p))
        if len(chosen) != n - 1:
            raise RuntimeError("non-held-out graph is disconnected")
        free = [p for p in range(self.P) if not self.is_heldout[p] and p not in chosen]
        extra = rng.choice(free, size=E0 - len(chosen), replace=False)
        out = sorted(chosen) + sorted(int(p) for p in extra)
        assert len(set(out)) == E0
        return out


class State:
    def __init__(self, world: World):
        self.w = world
        self.order = list(world.init_pids)       # edge order = insertion order of pair ids
        self.count = np.zeros(world.P, int)
        self.count[self.order] = 1
        self.n_relabels = 0
        self.relabel_noise = 0
        self.relabel_intrans = 0
        self.actions = []                        # ("new"|"relabel", pid)

    def add(self, pid: int):
        w = self.w
        assert not w.is_heldout[pid], "policy queried a held-out pair"
        assert self.count[pid] < MAX_LABELS
        if self.count[pid] == 0:
            self.order.append(pid)
            self.actions.append(("new", pid))
        else:
            self.n_relabels += 1
            c = self.count[pid]
            fl = np.mean(w.labels[pid, :c]); cl = np.mean(w.lab_clean[pid, :c])
            self.relabel_noise += int(np.sign(fl) != np.sign(cl))
            self.relabel_intrans += int(w.intrans_pair[pid])
            self.actions.append(("relabel", pid))
        self.count[pid] += 1

    def arrays(self):
        w = self.w
        pids = np.asarray(self.order, int)
        edges = [tuple(map(int, w.pairs[p])) for p in pids]
        cnt = self.count[pids]
        flow = np.array([w.labels[p, :c].mean() for p, c in zip(pids, cnt)])
        # --- explicit index-alignment checks ---
        assert len(pids) == len(set(pids.tolist())) == len(edges) == len(flow) == len(cnt)
        assert np.all(cnt >= 1)
        assert not np.any(w.is_heldout[pids])
        for e, p in enumerate(pids):
            assert w.pair_id[edges[e]] == p
        return pids, edges, flow, cnt.astype(float)

    def decompose(self, compute_betti=False):
        pids, edges, flow, cnt = self.arrays()
        cx = build_clique_complex(self.w.n, edges)
        return pids, edges, flow, cnt, cx, hodge_decompose_2d(cx, flow, cnt, compute_betti)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def _new_candidates(st: State):
    w = st.w
    return np.flatnonzero((st.count == 0) & ~w.is_heldout)


def _relabel_candidates(st: State):
    pids = np.asarray(st.order, int)
    return pids[st.count[pids] < MAX_LABELS]


def act_random(st, rng, _ctx):
    cand = _new_candidates(st)
    return int(rng.choice(cand))


def act_random_mix(st, rng, ctx):
    t = len(st.actions)
    if ctx["ours_actions"][t][0] == "relabel":
        cand = _relabel_candidates(st)
        if len(cand):
            return int(rng.choice(cand))
    return act_random(st, rng, ctx)


def act_fiedler(st, rng, _ctx):
    w = st.w
    n = w.n
    L = np.zeros((n, n))
    pids = np.asarray(st.order, int)
    for p in pids:
        i, j = w.pairs[p]
        c = st.count[p]
        L[i, i] += c; L[j, j] += c; L[i, j] -= c; L[j, i] -= c
    cand = np.flatnonzero((st.count < MAX_LABELS) & ~w.is_heldout)
    I, J = w.pairs[cand, 0], w.pairs[cand, 1]
    Ls = np.repeat(L[None], len(cand), axis=0)
    r = np.arange(len(cand))
    Ls[r, I, I] += 1; Ls[r, J, J] += 1; Ls[r, I, J] -= 1; Ls[r, J, I] -= 1
    lam2 = np.linalg.eigvalsh(Ls)[:, 1]
    best = np.flatnonzero(lam2 >= lam2.max() - 1e-9)
    return int(cand[rng.choice(best)])


def _fill_hole(st, rng, pids, edges, flow, cnt, res):
    w = st.w
    n = w.n
    A = np.zeros((n, n)); Hm = np.zeros((n, n))
    for (i, j), h in zip(edges, res.harmonic):
        A[i, j] = A[j, i] = 1.0
        Hm[i, j] = Hm[j, i] = abs(h)
    score_mat = Hm @ A + A @ Hm
    cand = _new_candidates(st)
    sc = score_mat[w.pairs[cand, 0], w.pairs[cand, 1]]
    pos = sc > 1e-12
    if not np.any(pos):
        return int(rng.choice(cand)), True
    cand, sc = cand[pos], sc[pos]
    top = cand[np.argsort(-sc, kind="stable")[:LOOKAHEAD]]
    phi = res.potential
    best, best_h = None, np.inf
    for p in top:
        i, k = map(int, w.pairs[p])
        e2 = edges + [(i, k)]
        f2 = np.append(flow, np.clip(phi[k] - phi[i], -1.0, 1.0))
        c2 = np.append(cnt, 1.0)
        r2 = hodge_decompose_2d(build_clique_complex(n, e2), f2, c2, compute_betti=False)
        hn = r2.norms["harmonic"] ** 2
        if hn < best_h - 1e-12:
            best, best_h = int(p), hn
    return best, False


def _relabel_curl(st, rng, pids, res):
    ok = st.count[pids] < MAX_LABELS
    if not np.any(ok):
        return None
    score = np.abs(res.curl) + 1e-3 * np.abs(res.harmonic)
    score = np.where(ok, score, -np.inf)
    best = np.flatnonzero(score >= score.max() - 1e-12)
    return int(pids[rng.choice(best)])


def make_ours(mode):
    def act(st, rng, ctx):
        pids, edges, flow, cnt, cx, res = st.decompose()
        H2, C2 = res.norms["harmonic"] ** 2, res.norms["curl"] ** 2
        do_fill = {"alt": len(st.actions) % 2 == 0, "adaptive": H2 >= C2,
                   "fill": True, "relabel": False}[mode]
        if not do_fill:
            p = _relabel_curl(st, rng, pids, res)
            if p is not None:
                return p
        p, fb = _fill_hole(st, rng, pids, edges, flow, cnt, res)
        ctx["fallbacks"] = ctx.get("fallbacks", 0) + int(fb)
        return p
    return act


ACTORS = {"random": act_random, "random_mix": act_random_mix, "fiedler": act_fiedler,
          "ours": make_ours("alt"), "ours_adaptive": make_ours("adaptive"),
          "ours_fill": make_ours("fill"), "ours_relabel": make_ours("relabel")}


# ---------------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------------
def fit_2crit(n, edges, npos, nneg, phi0, seed):
    E = np.asarray(edges, int)
    I, J = E[:, 0], E[:, 1]
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([phi0, 0.1 * rng.normal(size=2 * n)])

    def fg(x):
        phi, a, b = x[:n], x[n:2 * n], x[2 * n:]
        z = phi[J] - phi[I] + a[I] * b[J] - a[J] * b[I]
        loss = np.sum(npos * np.logaddexp(0, -z) + nneg * np.logaddexp(0, z))
        loss += 0.5 * RIDGE_2CRIT * np.sum(x * x)
        sg = expit(z)
        g = -npos * (1 - sg) + nneg * sg
        gphi = np.zeros(n); ga = np.zeros(n); gb = np.zeros(n)
        np.add.at(gphi, J, g); np.add.at(gphi, I, -g)
        np.add.at(ga, I, g * b[J]); np.add.at(gb, J, g * a[I])
        np.add.at(ga, J, -g * b[I]); np.add.at(gb, I, -g * a[J])
        return loss, np.concatenate([gphi, ga, gb]) + RIDGE_2CRIT * x

    r = minimize(fg, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return r.x[:n], r.x[n:2 * n], r.x[2 * n:]


def _acc(pred, target):
    return float(np.mean(np.where(pred == 0, 0.5, (pred == target).astype(float))))


def _prf(flag, truth):
    tp = int(np.sum(flag & truth))
    prec = tp / flag.sum() if flag.sum() else 0.0
    rec = tp / truth.sum() if truth.sum() else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def snapshot(st: State, seed: int):
    w = st.w
    pids, edges, flow, cnt, cx, res = st.decompose(compute_betti=True)
    # index-alignment check against an independent recomputation of the clean labels
    for p, c in zip(pids, cnt.astype(int)):
        i, j = w.pairs[p]
        for s in range(c):
            m = w.lab_type[p, s]
            assert np.sign(w.W[m] @ (w.X[j] - w.X[i])) == w.lab_clean[p, s]
    phi = res.potential
    HI, HJ = w.pairs[w.heldout, 0], w.pairs[w.heldout, 1]
    pred = np.sign(phi[HJ] - phi[HI])
    out = dict(
        heldout_acc_clean=_acc(pred, w.M[w.heldout]),
        heldout_acc_noisy=_acc(pred, w.ho_label),
        kendall_tau=float(kendalltau(phi, w.s)[0]),
        frac_harmonic=res.fractions["harmonic"], frac_curl=res.fractions["curl"],
        frac_gradient=res.fractions["gradient"], betti1=res.betti1,
        n_triangles=cx.n_triangles, n_edges=cx.n_edges, n_labels=int(cnt.sum()),
        n_relabels=st.n_relabels,
        relabels_on_noise_edges=st.relabel_noise,
        relabels_on_intransitive_edges=st.relabel_intrans,
    )
    # 2-criteria model
    npos = np.array([np.sum(w.labels[p, :c] > 0) for p, c in zip(pids, cnt.astype(int))], float)
    nneg = cnt - npos
    phi2, a, b = fit_2crit(w.n, edges, npos, nneg, phi, seed)
    z = phi2[HJ] - phi2[HI] + a[HI] * b[HJ] - a[HJ] * b[HI]
    pred2 = np.sign(z)
    out["heldout_acc_clean_2crit"] = _acc(pred2, w.M[w.heldout])
    out["heldout_acc_noisy_2crit"] = _acc(pred2, w.ho_label)
    # noise-edge flags on the fixed initial edge set
    init = np.asarray(w.init_pids, int)
    Ii, Ji = w.pairs[init, 0], w.pairs[init, 1]
    truth = w.lab_flip[init, 0]
    flag = np.sign(phi[Ji] - phi[Ii]) == -w.labels[init, 0]
    out["noise_flag_precision"], out["noise_flag_recall"], out["noise_flag_f1"] = _prf(flag, truth)
    # intransitive-triple flags on filled triangles
    if cx.n_triangles:
        tflag = np.abs(res.triangle_curl) >= TRIANGLE_FLAG_TAU - 1e-12
        ttrue = np.array([tuple(map(int, t)) in w.cyclic for t in cx.triangles])
        tp = int(np.sum(tflag & ttrue))
        out["triple_flag_precision"] = float(tp / tflag.sum()) if tflag.sum() else 0.0
        out["triple_flag_recall_all"] = float(tp / max(len(w.cyclic), 1))
        out["triple_n_flagged"] = int(tflag.sum())
    else:
        out["triple_flag_precision"] = 0.0
        out["triple_flag_recall_all"] = 0.0
        out["triple_n_flagged"] = 0
    # edge categories on the current graph
    clean_mean = np.array([w.lab_clean[p, :c].mean() for p, c in zip(pids, cnt.astype(int))])
    out["edges_noise_driven"] = int(np.sum(np.sign(flow) != np.sign(clean_mean)))
    out["edges_intransitive"] = int(np.sum(w.intrans_pair[pids]))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_policy(world, name, seed, ctx):
    st = State(world)
    rng = np.random.default_rng([seed, 7, POLICIES.index(name)])
    snaps = {}
    for t in range(BUDGETS[-1] + 1):
        if t in BUDGETS:
            s = snapshot(st, seed)
            assert s["n_labels"] == E0 + t, (s["n_labels"], t)
            snaps[t] = s
        if t == BUDGETS[-1]:
            break
        st.add(ACTORS[name](st, rng, ctx))
    return snaps, st.actions, ctx.get("fallbacks", 0)


def run_seed(args):
    scen_name, seed = args
    t0 = time.time()
    world = World(seed, SCENARIOS[scen_name])
    res, meta = {}, {}
    ours_snaps, ours_actions, fb = run_policy(world, "ours", seed, {})
    res["ours"] = ours_snaps
    meta["ours_fallbacks"] = fb
    ctx_mix = {"ours_actions": ours_actions}
    for name in POLICIES:
        if name == "ours":
            continue
        ctx = dict(ctx_mix) if name == "random_mix" else {}
        snaps, actions, fb = run_policy(world, name, seed, ctx)
        res[name] = snaps
        if name.startswith("ours"):
            meta[f"{name}_fallbacks"] = fb
        if name == "random_mix":
            assert [a[0] for a in actions] == [a[0] for a in ours_actions], "mix not matched"
    # budget 0 must be identical across policies
    for name in POLICIES:
        # repr comparison: NaN != NaN, but a NaN recall is legitimate when a scenario has
        # no noise edges at all.
        assert repr(res[name][0]) == repr(res["ours"][0]), f"budget-0 mismatch for {name}"
    meta.update(n_cyclic_triples=len(world.cyclic),
                frac_cyclic_triples=len(world.cyclic) / (N_ITEMS * (N_ITEMS - 1) * (N_ITEMS - 2) / 6),
                n_intransitive_pairs=int(world.intrans_pair.sum()),
                n_initial_noise_edges=int(world.lab_flip[world.init_pids, 0].sum()),
                ceiling_true_score_acc_clean=_acc(
                    np.sign(world.s[world.pairs[world.heldout, 1]] - world.s[world.pairs[world.heldout, 0]]),
                    world.M[world.heldout]),
                ceiling_majority_acc_noisy=_acc(world.M[world.heldout], world.ho_label),
                seconds=time.time() - t0)
    return scen_name, seed, res, meta


def summarise(raw):
    out = {}
    for scen, per_seed in raw.items():
        seeds = sorted(per_seed, key=int)
        S = {"policies": {}, "comparisons": {}, "meta": {}}
        for pol in POLICIES:
            S["policies"][pol] = {}
            for b in BUDGETS:
                S["policies"][pol][str(b)] = {}
                for m in HELDOUT_METRICS + DIAG_METRICS:
                    v = np.array([per_seed[s]["results"][pol][str(b)][m] for s in seeds], float)
                    S["policies"][pol][str(b)][m] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)))
        for a, bpol in COMPARISONS:
            key = f"{a}_vs_{bpol}"
            S["comparisons"][key] = {}
            for b in BUDGETS[1:]:
                S["comparisons"][key][str(b)] = {}
                for m in HELDOUT_METRICS + ["frac_harmonic", "frac_curl", "noise_flag_f1",
                                            "triple_flag_precision", "triple_flag_recall_all"]:
                    x = np.array([per_seed[s]["results"][a][str(b)][m] for s in seeds], float)
                    y = np.array([per_seed[s]["results"][bpol][str(b)][m] for s in seeds], float)
                    d = x - y
                    if np.allclose(d, 0):
                        p = 1.0
                    else:
                        p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
                    S["comparisons"][key][str(b)][m] = dict(
                        mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)), p_wilcoxon=p,
                        wins=int(np.sum(d > 1e-12)), ties=int(np.sum(np.abs(d) <= 1e-12)),
                        losses=int(np.sum(d < -1e-12)))
        for k in ["n_cyclic_triples", "frac_cyclic_triples", "n_intransitive_pairs",
                  "n_initial_noise_edges", "ceiling_true_score_acc_clean",
                  "ceiling_majority_acc_noisy", "ours_fallbacks", "ours_fill_fallbacks",
                  "ours_adaptive_fallbacks"]:
            v = np.array([per_seed[s]["meta"][k] for s in seeds], float)
            S["meta"][k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)))
        out[scen] = S
    return out


def kill_criterion(summary, scen="main"):
    comp = summary[scen]["comparisons"]["ours_vs_fiedler"]
    pos = [str(b) for b in BUDGETS[1:]]
    wins_primary = [b for b in pos if comp[b]["heldout_acc_clean"]["mean_diff"] > 0
                    and comp[b]["heldout_acc_clean"]["p_wilcoxon"] < 0.05]
    worse = [(b, m) for b in pos for m in ["heldout_acc_clean", "heldout_acc_noisy",
                                           "kendall_tau", "heldout_acc_clean_2crit"]
             if comp[b][m]["mean_diff"] < 0 and comp[b][m]["p_wilcoxon"] < 0.05]
    cond1 = len(wins_primary) >= len(pos) / 2
    cond2 = len(worse) == 0
    return dict(scenario=scen, fires=not (cond1 and cond2),
                cond1_primary_significant_wins_at_budgets=wins_primary,
                cond1_pass=cond1, cond2_significantly_worse=[list(x) for x in worse],
                cond2_pass=cond2)


# ---------------------------------------------------------------------------
# Markdown (numbers only from the JSON)
# ---------------------------------------------------------------------------
def fmt(ms, k=3):
    return f"{ms['mean']:.{k}f} ± {ms['sd']:.{k}f}"


def write_markdown(J, path):
    S = J["summary"]
    K = J["kill_criterion"]
    L = []
    L.append("# SGB-045 Stage 0 + Stage 1 results: Hodge-targeted data acquisition (simulation)\n")
    L.append(f"Generated from `{J['output_json']}` by `{J['script']}` on {J['timestamp']}. "
             "All numbers in this file come from that JSON file.\n")
    L.append("## Headline\n")
    if K["fires"]:
        L.append("**The kill criterion FIRES.** In the main scenario, the Hodge-targeted policy "
                 "(`ours`: fill harmonic holes + re-label high-curl edges) does not beat the "
                 "algebraic-connectivity baseline (`fiedler`, Osting 2013 / Xu 2018) on held-out "
                 "comparisons, by the rule fixed before the run.\n")
    else:
        L.append("**The kill criterion does NOT fire.** In the main scenario, `ours` beats `fiedler` "
                 "on held-out clean accuracy by the rule fixed before the run.\n")
    L.append(f"- Condition 1 (significant held-out clean-accuracy win at >= half of budgets): "
             f"{'pass' if K['cond1_pass'] else 'fail'}; budgets with a significant win: "
             f"{K['cond1_primary_significant_wins_at_budgets'] or 'none'}.")
    worse = {}
    for b, m in K["cond2_significantly_worse"]:
        worse.setdefault(b, []).append(m)
    worse_txt = "; ".join(f"budget {b}: {', '.join(v)}" for b, v in worse.items()) or "none"
    L.append(f"- Condition 2 (no held-out metric significantly worse): "
             f"{'pass' if K['cond2_pass'] else 'fail'}; significantly worse at: {worse_txt}.\n")
    L.append("## Reproduce\n")
    L.append(f"```\ncd topics/shape_of_good_behavior\n{J['command']}\n```\n")
    L.append(f"Seeds ({len(J['config']['seeds'])}): `{J['config']['seeds']}`. "
             f"Budgets (new labels): {J['config']['budgets']}. Items: {J['config']['n_items']}; "
             f"initial edges: {J['config']['e0']}; held-out pairs: {J['config']['n_heldout']}; "
             f"max labels per pair: {J['config']['max_labels']}.\n")
    L.append("Unit tests (Stage 0): `./venv/bin/python3 feedback_geometry/tests/test_hodge_complex.py`.\n")
    for scen in S:
        sc = J["config"]["scenarios"][scen]
        M = S[scen]["meta"]
        L.append(f"## Scenario `{scen}`\n")
        L.append(f"Annotator weights {sc['type_weights']}, shares {[round(x, 3) for x in sc['type_probs']]}, "
                 f"flip probability {sc['flip']}. Cyclic (Condorcet) triples: "
                 f"{fmt(M['frac_cyclic_triples'], 4)} of all triples; initial noise edges: "
                 f"{fmt(M['n_initial_noise_edges'], 1)} of {J['config']['e0']}. "
                 f"Ceiling: true aggregate score vs majority on held-out = "
                 f"{fmt(M['ceiling_true_score_acc_clean'])}; majority vs noisy held-out labels = "
                 f"{fmt(M['ceiling_majority_acc_noisy'])}.\n")
        L.append("### Held-out metrics (mean ± sd over seeds)\n")
        L.append("| budget | policy | acc clean | acc noisy | Kendall tau | acc clean (2-crit) |")
        L.append("|---|---|---|---|---|---|")
        for b in J["config"]["budgets"]:
            for pol in J["config"]["policies"]:
                r = S[scen]["policies"][pol][str(b)]
                L.append(f"| {b} | {pol} | {fmt(r['heldout_acc_clean'])} | {fmt(r['heldout_acc_noisy'])} | "
                         f"{fmt(r['kendall_tau'])} | {fmt(r['heldout_acc_clean_2crit'])} |")
                if b == 0:
                    break  # identical for all policies at budget 0
        L.append("")
        L.append("### Paired comparisons (mean paired diff ± sd, Wilcoxon p, wins/ties/losses)\n")
        L.append("| comparison | budget | acc clean | acc noisy | Kendall tau |")
        L.append("|---|---|---|---|---|")
        for key in S[scen]["comparisons"]:
            for b in J["config"]["budgets"][1:]:
                c = S[scen]["comparisons"][key][str(b)]
                cell = lambda m: (f"{c[m]['mean_diff']:+.4f} ± {c[m]['sd_diff']:.4f}, p={c[m]['p_wilcoxon']:.3g}, "  # noqa: E731
                                  f"{c[m]['wins']}/{c[m]['ties']}/{c[m]['losses']}")
                L.append(f"| {key} | {b} | {cell('heldout_acc_clean')} | {cell('heldout_acc_noisy')} | {cell('kendall_tau')} |")
        L.append("")
        bmax = str(J["config"]["budgets"][-1])
        L.append(f"### In-sample diagnostics at budget {bmax} (mean ± sd)\n")
        L.append("| policy | harmonic frac | curl frac | beta_1 | triangles | re-labels | noise-flag F1 | triple-flag precision | triple-flag recall (all cyclic) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        b0 = S[scen]["policies"]["random"]["0"]
        L.append(f"| (budget 0) | {fmt(b0['frac_harmonic'])} | {fmt(b0['frac_curl'])} | {fmt(b0['betti1'], 1)} | "
                 f"{fmt(b0['n_triangles'], 1)} | 0 | {fmt(b0['noise_flag_f1'])} | {fmt(b0['triple_flag_precision'])} | "
                 f"{fmt(b0['triple_flag_recall_all'])} |")
        for pol in J["config"]["policies"]:
            r = S[scen]["policies"][pol][bmax]
            L.append(f"| {pol} | {fmt(r['frac_harmonic'])} | {fmt(r['frac_curl'])} | {fmt(r['betti1'], 1)} | "
                     f"{fmt(r['n_triangles'], 1)} | {fmt(r['n_relabels'], 1)} | {fmt(r['noise_flag_f1'])} | "
                     f"{fmt(r['triple_flag_precision'])} | {fmt(r['triple_flag_recall_all'])} |")
        r = S[scen]["policies"]["ours"][bmax]
        L.append(f"\n`ours` re-labels at budget {bmax}: {fmt(r['n_relabels'], 1)}; of these, on edges whose "
                 f"current sign was noise-driven: {fmt(r['relabels_on_noise_edges'], 1)}; on genuinely "
                 f"intransitive edges: {fmt(r['relabels_on_intransitive_edges'], 1)}.\n")
    L.append("## Secondary findings (`ours` minus `fiedler` at the largest budget)\n")
    L.append("| scenario | acc clean (phi) | acc clean (2-crit) | noise-flag F1 | "
             "triple-flag precision | harmonic fraction |")
    L.append("|---|---|---|---|---|---|")
    bmax = str(J["config"]["budgets"][-1])
    for scen in S:
        c = S[scen]["comparisons"]["ours_vs_fiedler"][bmax]
        cell = lambda m: f"{c[m]['mean_diff']:+.4f} (p={c[m]['p_wilcoxon']:.2g})"  # noqa: E731
        L.append(f"| {scen} | {cell('heldout_acc_clean')} | {cell('heldout_acc_clean_2crit')} | "
                 f"{cell('noise_flag_f1')} | {cell('triple_flag_precision')} | "
                 f"{cell('frac_harmonic')} |")
    L.append("")
    L.append("## Caveats\n")
    for c in J["caveats"]:
        L.append(f"- {c}")
    L.append("")
    path.write_text("\n".join(L))


CAVEATS = [
    "The harmonic fraction and the curl fraction are not fair comparison metrics. The `ours` "
    "policy chooses edges to reduce the harmonic norm directly, and any new edge that closes a "
    "triangle moves harmonic mass into curl or gradient. A low harmonic fraction for `ours` is "
    "true by construction. It is not evidence of a better dataset.",
    "The intransitive-triple recall counts only filled triangles. A policy that makes more "
    "triangles gets a higher recall by construction. The triple-flag precision is the fairer "
    "number.",
    "The noise-edge flag rule (initial label disagrees with the fitted potential) is the same "
    "for all policies. Re-labels change the potential only through the edge weight; a direct "
    "majority vote on re-labelled edges would be a stronger detector for re-label policies.",
    "The simulation was designed by the same team that proposed the method. The annotator "
    "weights, the flip rate, the graph size (40 items, 60 initial edges, 200 held-out pairs), "
    "the cap of 5 labels per pair, the look-ahead of 10 candidates, the curl-vs-harmonic switch "
    "rule, the 2-criteria ridge of 0.5 and the triangle-flag threshold of 2 were fixed before the "
    "first full run and were not tuned. Other settings can give other results.",
    "The held-out pairs are a uniform random sample of pairs. Held-out accuracy therefore "
    "rewards a good global ranking. A policy that concentrates labels on a few loops can lose "
    "on this metric even when it describes those loops better.",
    "The Fiedler baseline uses exact greedy evaluation of lambda_2 for every allowed pair at every "
    "step. It is a strong implementation of the Osting 2013 / Xu 2018 unsupervised criterion; it "
    "is not the supervised Bayesian information-gain variant of Xu 2018.",
    "Each held-out pair has one noisy label. The noisy accuracy has a ceiling well below 1 (see "
    "the ceiling line per scenario). The clean accuracy against the population majority is the "
    "primary metric because it has less variance, but a real dataset does not give it.",
    "The held-out gain of `ours` on the 2-criteria model is not part of the kill criterion. "
    "It is a post-hoc observation from the same runs, over four scenarios and five budgets, "
    "with no correction. Treat it as a hypothesis for a new pre-registered test.",
    "Each observed comparison carries one annotator. An observed cyclic triangle can come from "
    "three different annotators and not from a population-level Condorcet cycle. This is why the "
    "triple-flag precision stays near the base rate of cyclic triples even when the label noise "
    "is zero. A design with several annotators per pair would test the localisation claim better.",
    "In this simulation the Fiedler baseline is close to random acquisition (see the "
    "fiedler_vs_random rows). A simulation in which the baseline itself gains little may also be "
    "a weak test of any targeting policy.",
    "Disclosure of one change after a pilot: the first version of `ours` used the adaptive "
    "switch (fill a hole if the harmonic norm exceeds the curl norm, else re-label). A 4-seed "
    "pilot showed that this rule spends almost the whole budget on re-labels, because the curl "
    "norm grows as soon as triangles exist. The primary `ours` policy was changed to a fixed "
    "alternation of one fill and one re-label. The adaptive rule is still reported as "
    "`ours_adaptive`. No other parameter was changed after seeing any result.",
    "No multiple-comparison correction is applied to the Wilcoxon p-values. The kill criterion "
    "uses the uncorrected p-values; a correction can only make a win harder to claim.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="debug: first k seeds")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--rebuild-md", action="store_true",
                    help="rewrite the markdown summary from the existing JSON, run nothing")
    args = ap.parse_args()
    if args.rebuild_md:
        write_markdown(json.loads(OUT_JSON.read_text()), OUT_MD)
        print(f"wrote {OUT_MD}")
        return
    seeds = SEEDS[: args.seeds]
    scens = args.scenarios.split(",")
    jobs = [(s, sd) for s in scens for sd in seeds]
    raw = {s: {} for s in scens}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (scen, seed, res, meta) in enumerate(ex.map(run_seed, jobs)):
            raw[scen][str(seed)] = dict(
                results={pol: {str(b): v for b, v in snaps.items()} for pol, snaps in res.items()},
                meta=meta)
            print(f"[{k + 1}/{len(jobs)}] {scen} seed {seed} {meta['seconds']:.1f}s", flush=True)
    summary = summarise(raw)
    kill = kill_criterion(summary) if "main" in summary else None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        git = None
    J = dict(
        experiment="SGB-045 Stage 1: Hodge-targeted data acquisition simulation",
        script="scripts/sgb045_hodge_acquisition_sim.py",
        output_json=str(OUT_JSON.relative_to(ROOT)),
        command=COMMAND if (args.seeds == len(SEEDS) and scens == list(SCENARIOS)) else
        f"{COMMAND} --seeds {args.seeds} --scenarios {args.scenarios}",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), git_head=git,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        runtime_seconds=time.time() - t0,
        config=dict(n_items=N_ITEMS, n_crit=N_CRIT, e0=E0, n_heldout=N_HELDOUT,
                    max_labels=MAX_LABELS, budgets=BUDGETS, seeds=seeds, lookahead=LOOKAHEAD,
                    triangle_flag_tau=TRIANGLE_FLAG_TAU, ridge_2crit=RIDGE_2CRIT,
                    scenarios={k: SCENARIOS[k] for k in scens}, policies=POLICIES,
                    primary_metric="heldout_acc_clean"),
        kill_criterion=kill, summary=summary, caveats=CAVEATS, raw=raw)
    print(json.dumps(kill, indent=1))
    if not args.no_write:
        OUT_JSON.write_text(json.dumps(J, indent=1))
        write_markdown(json.loads(OUT_JSON.read_text()), OUT_MD)
        print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
