"""E4 (SGB-014): peer-consistency sheaf with low-rank stalk constraint.

E2 showed that with full-rank stalks the data itself is a valid section, so the
harmonic component ``(im δ⁰)^⊥`` collapses to zero. SGB-014 fixes this by
restricting each model's stalk to a fixed k-dimensional subspace U_i (top-k
PCA of the calibration features of model i). With k ≪ d_i, the orthogonal
complement of im δ⁰_lowrank is large and the per-input harmonic norm becomes a
meaningful signal — the "part the panel cannot agree on within k dims of
shared structure."

This script runs the analysis at one or more k values, on the same cached
E1 embeddings used by E2/E3. The cocycle violation ‖δ¹c‖ is also reported as
a reference (it does not depend on k).

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 scripts/peer_sheaf_e4_low_rank.py
    ./venv/bin/python3 scripts/peer_sheaf_e4_low_rank.py --k-list 16,32,64,128
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
    fit_top_k_pca_per_model,
    low_rank_hodge_decompose,
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


def run_for_k(
    k: int,
    feats_cal: dict[str, np.ndarray],
    feats_eval_chosen: dict[str, np.ndarray],
    feats_eval_rejected: dict[str, np.ndarray],
    maps,
    complex_,
) -> dict:
    """One full low-rank Hodge analysis at stalk-dimension ``k``."""
    t0 = time.time()
    U_by_model = fit_top_k_pca_per_model(feats_cal, k=k)
    fit_time = time.time() - t0

    t0 = time.time()
    decomp_c = low_rank_hodge_decompose(complex_, feats_eval_chosen, maps, U_by_model)
    decomp_r = low_rank_hodge_decompose(complex_, feats_eval_rejected, maps, U_by_model)
    decomp_time = time.time() - t0

    aggregate = {
        "total_norm": summary(decomp_r.norm_total, decomp_c.norm_total),
        "resolvable_norm": summary(decomp_r.norm_res, decomp_c.norm_res),
        "harmonic_norm": summary(decomp_r.norm_harm, decomp_c.norm_harm),
        "harmonic_fraction": summary(
            decomp_r.harmonic_fraction(), decomp_c.harmonic_fraction()
        ),
        "cocycle_violation_norm": summary(
            decomp_r.norm_cocycle_violation, decomp_c.norm_cocycle_violation
        ),
    }

    # Per-pair harmonic and resolvable norms.
    edge_norms_c_harm = per_edge_norms(complex_, decomp_c.C_harm)
    edge_norms_r_harm = per_edge_norms(complex_, decomp_r.C_harm)
    edge_norms_c_res = per_edge_norms(complex_, decomp_c.C_res)
    edge_norms_r_res = per_edge_norms(complex_, decomp_r.C_res)
    per_pair = {}
    for edge in edge_norms_c_harm:
        key = f"{edge[0]}->{edge[1]}"
        per_pair[key] = {
            "harmonic": summary(edge_norms_r_harm[edge], edge_norms_c_harm[edge]),
            "resolvable": summary(edge_norms_r_res[edge], edge_norms_c_res[edge]),
        }

    # Global mass split (chosen vs rejected, fraction of squared mass in harmonic).
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
    for k_ in ("chosen", "rejected"):
        m = mass[k_]
        m["harmonic_fraction"] = m["harmonic_sq"] / max(m["total_sq"], 1e-12)
        m["resolvable_fraction"] = m["resolvable_sq"] / max(m["total_sq"], 1e-12)

    # Explained variance per model at this k (sanity check).
    pca_explained_var: dict[str, float] = {}
    for name, X in feats_cal.items():
        Xc = X - X.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        s2 = s * s
        pca_explained_var[name] = float(s2[:k].sum() / max(s2.sum(), 1e-12))

    return {
        "k": k,
        "rank_low_rank_coboundary": int(decomp_c.rank_B),
        "max_possible_rank": int(len(complex_.models) * k),
        "fit_pca_seconds": fit_time,
        "decompose_seconds": decomp_time,
        "pca_explained_variance": pca_explained_var,
        "aggregate": aggregate,
        "per_pair": per_pair,
        "global_mass": mass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--k-list",
        type=str,
        default="16,32,64,128",
        help="comma-separated stalk dimensions to evaluate",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "shared" / "data" / "cache" / "peer_sheaf_e1"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "shared" / "results" / "peer_sheaf_e4_low_rank.json"),
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_list.split(",")]

    cache_dir = Path(args.cache_dir)
    print(f"[E4] loading cached E1 embeddings from {cache_dir}", flush=True)
    feats_all: dict[str, np.ndarray] = {}
    for name, _ in PANEL:
        path = cache_dir / f"{name}_n{args.n_pairs}_seed{args.seed}.npy"
        if not path.exists():
            print(f"[E4] MISSING: {path}", flush=True)
            return 2
        feats_all[name] = np.load(path)
        print(f"  {name}: {feats_all[name].shape}", flush=True)

    n = feats_all[PANEL[0][0]].shape[0] // 2
    feats_chosen = {k: v[:n] for k, v in feats_all.items()}
    feats_rejected = {k: v[n:] for k, v in feats_all.items()}

    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(n)
    cal_idx = perm[: n // 2]
    eval_idx = perm[n // 2 :]
    print(f"[E4] n={n}  n_cal={len(cal_idx)}  n_eval={len(eval_idx)}", flush=True)

    feats_cal = {
        k: np.concatenate(
            [feats_chosen[k][cal_idx], feats_rejected[k][cal_idx]], axis=0
        )
        for k in feats_all
    }

    t0 = time.time()
    maps = fit_all_restriction_maps(feats_cal, ridge_lambda=args.ridge)
    complex_ = PeerComplex.from_features(feats_cal)
    print(
        f"[E4] fit maps + complex (D_V={complex_.D_V}, D_E={complex_.D_E}) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    feats_eval_chosen = {k: feats_chosen[k][eval_idx] for k in feats_all}
    feats_eval_rejected = {k: feats_rejected[k][eval_idx] for k in feats_all}

    per_k: list[dict] = []
    for k in k_values:
        print(f"\n[E4] === k = {k} ===", flush=True)
        r = run_for_k(
            k,
            feats_cal,
            feats_eval_chosen,
            feats_eval_rejected,
            maps,
            complex_,
        )
        per_k.append(r)
        agg = r["aggregate"]
        print(
            f"  rank_B = {r['rank_low_rank_coboundary']}/{r['max_possible_rank']}",
            flush=True,
        )
        ev_str = "  ".join(
            f"{m}={v:.3f}" for m, v in r["pca_explained_variance"].items()
        )
        print(f"  PCA explained var: {ev_str}", flush=True)
        for name in (
            "total_norm",
            "resolvable_norm",
            "harmonic_norm",
            "harmonic_fraction",
            "cocycle_violation_norm",
        ):
            s = agg[name]
            print(
                f"  {name:>22s}  d={s['cohens_d']:+.3f}  "
                f"AUC={s['auc_rejected_high']:.3f}  "
                f"p={s['p_value']:.2e}  "
                f"Δ={s['rejected_mean'] - s['chosen_mean']:+.4g}",
                flush=True,
            )

    result = {
        "config": {
            "experiment": "E4 (SGB-014)",
            "depends_on": "E1 cached embeddings",
            "n_pairs": n,
            "n_eval": int(len(eval_idx)),
            "seed": args.seed,
            "ridge": args.ridge,
            "k_values": k_values,
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
                "num_directed_edges": len(complex_.edge_slices),
                "num_ordered_triangles": len(complex_.tri_slices),
            },
        },
        "per_k": per_k,
        "methodological_note": (
            "Stalks restricted to U_i ⊂ R^{d_i} = span of top-k PCA directions "
            "of model i's calibration features. (im δ⁰_lowrank)^⊥ is nontrivial, "
            "so the per-input harmonic norm measures the part of cross-model "
            "translation that depends on the high-rank tail of each model's "
            "representation — what the panel cannot agree on within k dims of "
            "shared structure. The cocycle-violation channel δ¹c does not "
            "depend on k and is reported for cross-reference with E2/E3."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[E4] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
