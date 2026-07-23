"""SGB-012 analysis: scaled-up peer-consistency sheaf on the 7B–9B Modal panel.

Loads the cached Modal embeddings written by ``shared/modal_peer_sheaf.py``
(local copy at ``shared/data/cache/peer_sheaf_e1_modal/``) and runs the full
SGB-011/E2/E3/E4 stack against them:

  - Aggregate L(rejected) − L(chosen) and cocycle violation ‖δ¹c‖
    (the SGB-012 pass criterion is L Cohen's d ≥ 0.3).
  - Per-category stratification using the existing
    ``shared/data/cache/hh_rejected_categories.json`` label cache. Labels were
    generated on the same HH-RLHF seed=0 sample so the global indices line up.
  - Low-rank stalk Hodge decomposition (SGB-014) at k ∈ {16, 32, 64, 128}.

Writes ``shared/results/peer_sheaf_modal_summary.json`` with the same shape as
``peer_sheaf_e1.json + peer_sheaf_e2.json + peer_sheaf_e3_stratify.json +
peer_sheaf_e4_low_rank.json`` collapsed into a single document for easy
side-by-side comparison with the small-LM results.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 scripts/peer_sheaf_modal_analysis.py
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

from peer_sheaf import (  # noqa: E402
    fit_all_restriction_maps,
    panel_residual_summary,
)
from peer_hodge import (  # noqa: E402
    PeerComplex,
    hodge_decompose,
    fit_top_k_pca_per_model,
    low_rank_hodge_decompose,
)

PANEL = [
    ("Yi-1.5-9B", "01-ai/Yi-1.5-9B-Chat"),
    ("Zephyr-7B", "HuggingFaceH4/zephyr-7b-beta"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
]

CATEGORIES = [
    "incoherent",
    "wrong",
    "sycophantic",
    "refusal_of_benign",
    "harmful",
    "deceptive",
]
HIDING_CATS = {"deceptive", "harmful"}
COMPETENCE_CATS = {"incoherent", "wrong", "sycophantic", "refusal_of_benign"}


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    s = np.concatenate([scores_pos, scores_neg])
    return float(roc_auc_score(y, s))


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ttest_ind

    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    res = ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = 0.5 * (a.var() + b.var()) + 1e-12
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def summary(rejected: np.ndarray, chosen: np.ndarray) -> dict:
    t_stat, p_val = welch_t(rejected, chosen)
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
        "auc_rejected_high": auc(rejected, chosen),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "shared" / "data" / "cache" / "peer_sheaf_e1_modal"),
    )
    parser.add_argument(
        "--label-cache",
        default=str(ROOT / "shared" / "data" / "cache" / "hh_rejected_categories.json"),
    )
    parser.add_argument(
        "--k-list",
        type=str,
        default="16,32,64,128",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "shared" / "results" / "peer_sheaf_modal_summary.json"),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"[modal-analysis] loading from {cache_dir}", flush=True)
    feats_all: dict[str, np.ndarray] = {}
    for name, _ in PANEL:
        path = cache_dir / f"{name}_n{args.n_pairs}_seed{args.seed}.npy"
        if not path.exists():
            print(f"[modal-analysis] MISSING: {path}", flush=True)
            return 2
        feats_all[name] = np.load(path).astype(np.float64)
        print(f"  {name}: {feats_all[name].shape}", flush=True)

    n = feats_all[PANEL[0][0]].shape[0] // 2
    feats_chosen = {k: v[:n] for k, v in feats_all.items()}
    feats_rejected = {k: v[n:] for k, v in feats_all.items()}

    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(n)
    cal_idx = perm[: n // 2]
    eval_idx = perm[n // 2 :]
    print(f"[modal-analysis] n={n}  n_cal={len(cal_idx)}  n_eval={len(eval_idx)}", flush=True)

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
        f"[modal-analysis] fit maps + complex (D_V={complex_.D_V}, D_E={complex_.D_E}) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    feats_eval_chosen = {k: feats_chosen[k][eval_idx] for k in feats_all}
    feats_eval_rejected = {k: feats_rejected[k][eval_idx] for k in feats_all}

    # ------------------------------------------------------------------
    # E1-style aggregate residual L(x) and cocycle violation δ¹c.
    # ------------------------------------------------------------------
    sum_c = panel_residual_summary(feats_eval_chosen, maps, metric="cosine")
    sum_r = panel_residual_summary(feats_eval_rejected, maps, metric="cosine")
    L_c = sum_c["L"]
    L_r = sum_r["L"]
    print(
        f"[modal-analysis] L  chosen={L_c.mean():.4f}±{L_c.std():.4f}  "
        f"rejected={L_r.mean():.4f}±{L_r.std():.4f}  "
        f"d={cohens_d(L_c, L_r):+.3f}  "
        f"AUC={auc(L_r, L_c):.3f}",
        flush=True,
    )

    t0 = time.time()
    print(f"[modal-analysis] hodge_decompose start (SVD on B {complex_.D_E}x{complex_.D_V})...", flush=True)
    decomp_c = hodge_decompose(complex_, feats_eval_chosen, maps)
    decomp_r = hodge_decompose(complex_, feats_eval_rejected, maps)
    print(f"[modal-analysis] hodge_decompose finished in {time.time() - t0:.1f}s", flush=True)

    delta1_c = decomp_c.norm_cocycle_violation
    delta1_r = decomp_r.norm_cocycle_violation

    e1_block = {
        "L_summary": summary(L_r, L_c),
        "delta1c_summary": summary(delta1_r, delta1_c),
        "per_pair_residual": {
            f"{s}->{d}": {
                "mean_chosen": float(sum_c["pair_residuals"][(s, d)].mean()),
                "mean_rejected": float(sum_r["pair_residuals"][(s, d)].mean()),
                "delta": float(
                    sum_r["pair_residuals"][(s, d)].mean()
                    - sum_c["pair_residuals"][(s, d)].mean()
                ),
                "auc_rejected_high": auc(
                    sum_r["pair_residuals"][(s, d)],
                    sum_c["pair_residuals"][(s, d)],
                ),
            }
            for (s, d) in sum_c["pair_residuals"]
        },
        "per_model_loo": {
            m: {
                "mean_chosen": float(sum_c["loo"][m].mean()),
                "mean_rejected": float(sum_r["loo"][m].mean()),
                "auc_rejected_high": auc(sum_r["loo"][m], sum_c["loo"][m]),
            }
            for m in feats_all
        },
        "rank_coboundary": int(decomp_c.rank_B),
    }

    # ------------------------------------------------------------------
    # SGB-013 stratification using the existing label cache.
    # ------------------------------------------------------------------
    label_cache_path = Path(args.label_cache)
    stratified: dict = {"available": False}
    if label_cache_path.exists():
        cache_blob = json.loads(label_cache_path.read_text())
        labels = cache_blob.get("labels", {})
        eval_cat: dict[int, str] = {}
        for k, gi in enumerate(eval_idx):
            lab = labels.get(str(int(gi)))
            if lab and lab.get("category") in CATEGORIES:
                eval_cat[k] = lab["category"]

        per_category: dict[str, dict] = {}
        for cat in CATEGORIES:
            sel = np.array([k for k, c in eval_cat.items() if c == cat], dtype=int)
            if len(sel) == 0:
                per_category[cat] = {"n": 0}
                continue
            d1 = delta1_r[sel]
            L1 = L_r[sel]
            t_stat, p_val = welch_t(d1, delta1_c)
            per_category[cat] = {
                "n": int(len(sel)),
                "delta1c_mean": float(d1.mean()),
                "delta1c_std": float(d1.std()),
                # category - chosen baseline (rejected-minus-chosen convention,
                # matches scripts/peer_sheaf_e3_stratify.py)
                "delta1c_cohens_d_vs_chosen": cohens_d(d1, delta1_c),
                "delta1c_p_value_vs_chosen": p_val,
                "delta1c_auc_vs_chosen": auc(d1, delta1_c),
                "L_mean": float(L1.mean()),
                "L_cohens_d_vs_chosen": cohens_d(L1, L_c),
                "L_auc_vs_chosen": auc(L1, L_c),
            }

        hide_sel = np.array(
            [k for k, c in eval_cat.items() if c in HIDING_CATS], dtype=int
        )
        comp_sel = np.array(
            [k for k, c in eval_cat.items() if c in COMPETENCE_CATS], dtype=int
        )
        discriminator: dict = {
            "hiding_categories": sorted(HIDING_CATS),
            "competence_categories": sorted(COMPETENCE_CATS),
            "n_hiding": int(len(hide_sel)),
            "n_competence": int(len(comp_sel)),
        }
        if len(hide_sel) >= 2 and len(comp_sel) >= 2:
            d_hide_delta1 = delta1_r[hide_sel]
            d_comp_delta1 = delta1_r[comp_sel]
            t_d1, p_d1 = welch_t(d_hide_delta1, d_comp_delta1)
            discriminator.update(
                {
                    "delta1c_hiding_mean": float(d_hide_delta1.mean()),
                    "delta1c_competence_mean": float(d_comp_delta1.mean()),
                    # hiding - competence (matches scripts/peer_sheaf_e3_stratify.py;
                    # negative ⇒ hiding categories have *lower* δ¹c than competence,
                    # contrary to "deception is hard to coordinate" hypothesis)
                    "delta1c_cohens_d_hiding_vs_competence": cohens_d(d_hide_delta1, d_comp_delta1),
                    "delta1c_p_value": p_d1,
                    "L_hiding_mean": float(L_r[hide_sel].mean()),
                    "L_competence_mean": float(L_r[comp_sel].mean()),
                    "L_cohens_d_hiding_vs_competence": cohens_d(L_r[hide_sel], L_r[comp_sel]),
                }
            )

        stratified = {
            "available": True,
            "n_labels_total": len(labels),
            "n_labels_in_eval": len(eval_cat),
            "per_category": per_category,
            "discriminator_hiding_vs_competence": discriminator,
        }

    # ------------------------------------------------------------------
    # SGB-014 low-rank stalks at several k.
    # ------------------------------------------------------------------
    k_values = [int(x) for x in args.k_list.split(",")]
    low_rank_per_k = []
    for k in k_values:
        print(f"[modal-analysis] low-rank k={k}", flush=True)
        t0 = time.time()
        U_by_model = fit_top_k_pca_per_model(feats_cal, k=k)
        dec_c = low_rank_hodge_decompose(complex_, feats_eval_chosen, maps, U_by_model)
        dec_r = low_rank_hodge_decompose(complex_, feats_eval_rejected, maps, U_by_model)
        ev = {}
        for name, X in feats_cal.items():
            Xc = X - X.mean(axis=0, keepdims=True)
            _, s, _ = np.linalg.svd(Xc, full_matrices=False)
            s2 = s * s
            ev[name] = float(s2[:k].sum() / max(s2.sum(), 1e-12))
        low_rank_per_k.append({
            "k": k,
            "rank_low_rank_coboundary": int(dec_c.rank_B),
            "max_possible_rank": len(complex_.models) * k,
            "elapsed_seconds": time.time() - t0,
            "pca_explained_variance": ev,
            "total_norm": summary(dec_r.norm_total, dec_c.norm_total),
            "resolvable_norm": summary(dec_r.norm_res, dec_c.norm_res),
            "harmonic_norm": summary(dec_r.norm_harm, dec_c.norm_harm),
            "harmonic_fraction": summary(dec_r.harmonic_fraction(), dec_c.harmonic_fraction()),
        })
        print(
            f"  rank_B={dec_c.rank_B}/{len(complex_.models) * k}  "
            f"ev={ev}  "
            f"harmonic d={low_rank_per_k[-1]['harmonic_norm']['cohens_d']:+.3f}  "
            f"resolvable d={low_rank_per_k[-1]['resolvable_norm']['cohens_d']:+.3f}",
            flush=True,
        )

    pass_criterion = bool(e1_block["L_summary"]["cohens_d"] >= 0.3)

    result = {
        "config": {
            "experiment": "SGB-012 (Modal 7-9B panel analysis)",
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
            },
        },
        "pass_criterion_d_ge_0.3": pass_criterion,
        "e1_aggregate": e1_block,
        "stratified": stratified,
        "low_rank_per_k": low_rank_per_k,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[modal-analysis] wrote {out_path}", flush=True)
    print(
        f"\n[modal-analysis] SGB-012 PASS CRITERION (L Cohen's d ≥ 0.3): "
        f"{'PASS' if pass_criterion else 'FAIL'} "
        f"(d = {e1_block['L_summary']['cohens_d']:+.3f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
