"""E2: vector-valued Hodge decomposition of the peer-consistency cochain.

Analysis-only continuation of E1. Loads the cached per-model embeddings written
by ``scripts/peer_sheaf_e1.py``, fits affine restriction maps on the same
calibration split, then projects each evaluation input's measured 1-cochain
onto resolvable (im δ⁰) and harmonic ((im δ⁰)^⊥) subspaces via
``shared/src/peer_hodge.py``.

The chosen-vs-rejected separation test is re-run on three signals:

  total      — ||c(x)||           (the linear-algebra analogue of E1's L)
  resolvable — ||resolvable(x)||
  harmonic   — ||h(x)||           (the irreducible-inconsistency channel)
  fraction   — ||h(x)||² / ||c(x)||²

If the chosen/rejected gap shows up specifically in the harmonic norm (and not
the resolvable norm), the cross-model sheaf is doing real work: the asymmetry
is not explainable by any choice of stalk values, i.e. it is a genuine
sheaf-cohomological obstruction.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 scripts/peer_sheaf_e2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "shared" / "src"))

from peer_sheaf import fit_all_restriction_maps  # noqa: E402
from peer_hodge import (  # noqa: E402
    PeerComplex,
    hodge_decompose,
    per_edge_norms,
)

PANEL = [
    ("SmolLM2", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("Qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    s = np.concatenate([scores_pos, scores_neg])
    return float(roc_auc_score(y, s))


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ttest_ind

    res = ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def summary(rejected: np.ndarray, chosen: np.ndarray) -> dict:
    t_stat, p_val = welch_t(rejected, chosen)
    a = auc(rejected, chosen)
    pooled_var = 0.5 * (rejected.var() + chosen.var()) + 1e-12
    d = float((rejected.mean() - chosen.mean()) / np.sqrt(pooled_var))
    return {
        "chosen_mean": float(chosen.mean()),
        "chosen_std": float(chosen.std()),
        "rejected_mean": float(rejected.mean()),
        "rejected_std": float(rejected.std()),
        "welch_t": t_stat,
        "p_value": p_val,
        "cohens_d": d,
        "auc_rejected_high": a,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "shared" / "data" / "cache" / "peer_sheaf_e1"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "shared" / "results" / "peer_sheaf_e2.json"),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"[E2] loading cached E1 embeddings from {cache_dir}", flush=True)
    feats_all: dict[str, np.ndarray] = {}
    for name, _ in PANEL:
        path = cache_dir / f"{name}_n{args.n_pairs}_seed{args.seed}.npy"
        if not path.exists():
            print(f"[E2] MISSING: {path}", flush=True)
            return 2
        feats_all[name] = np.load(path)
        print(f"  {name}: {feats_all[name].shape}", flush=True)

    n = feats_all[PANEL[0][0]].shape[0] // 2
    feats_chosen = {k: v[:n] for k, v in feats_all.items()}
    feats_rejected = {k: v[n:] for k, v in feats_all.items()}

    # Same calibration/eval split as E1 (seed + 1 → permutation, first half cal).
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(n)
    cal_idx = perm[: n // 2]
    eval_idx = perm[n // 2 :]
    print(f"[E2] n={n}  n_cal={len(cal_idx)}  n_eval={len(eval_idx)}", flush=True)

    feats_cal = {
        k: np.concatenate(
            [feats_chosen[k][cal_idx], feats_rejected[k][cal_idx]], axis=0
        )
        for k in feats_all
    }

    t0 = time.time()
    maps = fit_all_restriction_maps(feats_cal, ridge_lambda=args.ridge)
    print(f"[E2] fit {len(maps)} restriction maps in {time.time() - t0:.1f}s", flush=True)

    complex_ = PeerComplex.from_features(feats_cal)
    print(
        f"[E2] complex: models={complex_.models}  D_V={complex_.D_V}  D_E={complex_.D_E}",
        flush=True,
    )

    feats_eval_chosen = {k: feats_chosen[k][eval_idx] for k in feats_all}
    feats_eval_rejected = {k: feats_rejected[k][eval_idx] for k in feats_all}

    t0 = time.time()
    decomp_c = hodge_decompose(complex_, feats_eval_chosen, maps)
    decomp_r = hodge_decompose(complex_, feats_eval_rejected, maps)
    print(
        f"[E2] decomposed (rank δ⁰ = {decomp_c.rank_B}) in {time.time() - t0:.1f}s",
        flush=True,
    )

    aggregate = {
        "total_norm": summary(decomp_r.norm_total, decomp_c.norm_total),
        "resolvable_norm": summary(decomp_r.norm_res, decomp_c.norm_res),
        # Expected ~0 on this data — see peer_hodge module docstring.
        "harmonic_norm": summary(decomp_r.norm_harm, decomp_c.norm_harm),
        "harmonic_fraction": summary(
            decomp_r.harmonic_fraction(), decomp_c.harmonic_fraction()
        ),
        # The actual sheaf signal: composition failure of restriction maps
        # around triangles applied to the input.
        "cocycle_violation_norm": summary(
            decomp_r.norm_cocycle_violation, decomp_c.norm_cocycle_violation
        ),
    }

    edge_norms_c_total = per_edge_norms(complex_, decomp_c.C)
    edge_norms_r_total = per_edge_norms(complex_, decomp_r.C)
    edge_norms_c_harm = per_edge_norms(complex_, decomp_c.C_harm)
    edge_norms_r_harm = per_edge_norms(complex_, decomp_r.C_harm)
    edge_norms_c_res = per_edge_norms(complex_, decomp_c.C_res)
    edge_norms_r_res = per_edge_norms(complex_, decomp_r.C_res)

    per_pair = {}
    for edge in edge_norms_c_total:
        key = f"{edge[0]}->{edge[1]}"
        per_pair[key] = {
            "total": summary(edge_norms_r_total[edge], edge_norms_c_total[edge]),
            "resolvable": summary(edge_norms_r_res[edge], edge_norms_c_res[edge]),
            "harmonic": summary(edge_norms_r_harm[edge], edge_norms_c_harm[edge]),
        }

    # Per-triangle cocycle-violation norms.
    per_triangle = {}
    for tri, sl in complex_.tri_slices.items():
        v_c = np.linalg.norm(decomp_c.cocycle_violation[:, sl], axis=1)
        v_r = np.linalg.norm(decomp_r.cocycle_violation[:, sl], axis=1)
        per_triangle[f"{tri[0]}->{tri[1]}->{tri[2]}"] = summary(v_r, v_c)

    # Global mass split (sanity check + descriptive).
    sum_sq = lambda X: float((X * X).sum())  # noqa: E731
    mass = {
        "chosen": {
            "total_sq": sum_sq(decomp_c.C),
            "resolvable_sq": sum_sq(decomp_c.C_res),
            "harmonic_sq": sum_sq(decomp_c.C_harm),
        },
        "rejected": {
            "total_sq": sum_sq(decomp_r.C),
            "resolvable_sq": sum_sq(decomp_r.C_res),
            "harmonic_sq": sum_sq(decomp_r.C_harm),
        },
    }
    for k in ("chosen", "rejected"):
        m = mass[k]
        m["harmonic_fraction"] = m["harmonic_sq"] / max(m["total_sq"], 1e-12)
        m["resolvable_fraction"] = m["resolvable_sq"] / max(m["total_sq"], 1e-12)

    result = {
        "config": {
            "experiment": "E2",
            "depends_on": "E1 cached embeddings",
            "n_pairs": n,
            "n_eval": int(len(eval_idx)),
            "seed": args.seed,
            "ridge": args.ridge,
            "panel": [
                {
                    "name": n_,
                    "hf_id": h,
                    "hidden_dim": int(feats_all[n_].shape[1]),
                }
                for n_, h in PANEL
            ],
            "complex": {
                "D_V": complex_.D_V,
                "D_E": complex_.D_E,
                "D_F": complex_.D_F,
                "rank_coboundary": decomp_c.rank_B,
                "num_directed_edges": len(complex_.edge_slices),
                "num_ordered_triangles": len(complex_.tri_slices),
            },
        },
        "aggregate": aggregate,
        "global_mass": mass,
        "per_pair": per_pair,
        "per_triangle_cocycle_violation": per_triangle,
        "methodological_note": (
            "Harmonic component (im δ⁰)^⊥ is numerically zero on this data: "
            "with affine ridge-fit restriction maps, c(x) = δ⁰(F(x)−μ) exactly, "
            "so the data itself provides a valid stalk-assignment that "
            "explains every measured cochain. The non-trivial sheaf signal is "
            "the cocycle violation δ¹c (composition failure of the triangle "
            "restriction maps). See shared/src/peer_hodge.py docstring."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[E2] wrote {out_path}", flush=True)

    print("\n[E2] SUMMARY (rejected > chosen?)")
    for key, agg in aggregate.items():
        print(
            f"  {key:>18s}  Δ={agg['rejected_mean'] - agg['chosen_mean']:+.4g}  "
            f"d={agg['cohens_d']:+.3f}  AUC={agg['auc_rejected_high']:.3f}  "
            f"p={agg['p_value']:.2e}"
        )
    print(
        f"\n[E2] Global mass split  chosen: harm={mass['chosen']['harmonic_fraction']:.3f}  "
        f"rejected: harm={mass['rejected']['harmonic_fraction']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
