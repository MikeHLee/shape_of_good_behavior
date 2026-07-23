"""E0: peer-consistency sheaf on a sentence-transformer panel over HH-RLHF.

Proof-of-mechanism step before E1 (which uses 7B LLM hidden states). The panel is
three diverse sentence-transformer encoders. We fit linear restriction maps on a
calibration split (alignment-agnostic: chosen + rejected mixed), then measure the
aggregate lossiness L(x) on held-out chosen vs rejected. The question is whether
L(x) separates the two classes at all -- non-vacuous separation is the bar for
moving to E1.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 scripts/peer_sheaf_e0.py --n-pairs 1000

Output is written to shared/results/peer_sheaf_e0.json.
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
    aggregate_lossiness,
    fit_all_restriction_maps,
    panel_residual_summary,
)

PANEL = [
    ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    ("MPNet", "sentence-transformers/all-mpnet-base-v2"),
    ("BGE", "BAAI/bge-small-en-v1.5"),
]


def extract_final_response(dialogue: str) -> str:
    """HH-RLHF stores a dialogue; the chosen/rejected diff is at the last 'A:' turn."""
    marker = "\n\nA:"
    idx = dialogue.rfind(marker)
    if idx < 0:
        return dialogue.strip()
    tail = dialogue[idx + len(marker):]
    # Strip any trailing H: turn (rare but possible).
    cut = tail.find("\n\nH:")
    if cut >= 0:
        tail = tail[:cut]
    return tail.strip()


def load_hh_rlhf_pairs(n_pairs: int, seed: int = 0) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_pairs, len(ds)), replace=False)
    chosen = [extract_final_response(ds[int(i)]["chosen"]) for i in idx]
    rejected = [extract_final_response(ds[int(i)]["rejected"]) for i in idx]
    # Drop empty rows (rare).
    pairs = [(c, r) for c, r in zip(chosen, rejected) if c and r]
    chosen = [p[0] for p in pairs]
    rejected = [p[1] for p in pairs]
    return chosen, rejected


def embed_with_panel(texts: list[str]) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    out: dict[str, np.ndarray] = {}
    for name, hf_id in PANEL:
        t0 = time.time()
        model = SentenceTransformer(hf_id)
        emb = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float64)
        out[name] = emb
        print(
            f"  embedded {len(texts)} texts with {name} -> {emb.shape} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
        del model
    return out


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUC where higher score should mean rejected (non-aligned)."""
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    s = np.concatenate([scores_pos, scores_neg])
    return float(roc_auc_score(y, s))


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ttest_ind

    res = ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--metric", choices=["cosine", "rel_l2"], default="cosine")
    parser.add_argument(
        "--out",
        default=str(ROOT / "shared" / "results" / "peer_sheaf_e0.json"),
    )
    args = parser.parse_args()

    print(f"[E0] loading {args.n_pairs} HH-RLHF pairs (seed={args.seed})", flush=True)
    chosen, rejected = load_hh_rlhf_pairs(args.n_pairs, seed=args.seed)
    n = len(chosen)
    print(f"  -> {n} usable pairs", flush=True)

    # Concatenate so each model embeds in one pass; recover chosen/rejected by slicing.
    all_texts = chosen + rejected
    print(f"[E0] embedding {len(all_texts)} responses with panel of {len(PANEL)}", flush=True)
    feats_all = embed_with_panel(all_texts)

    feats_chosen = {k: v[:n] for k, v in feats_all.items()}
    feats_rejected = {k: v[n:] for k, v in feats_all.items()}

    # Calibration / evaluation split: 50/50 on pair indices, then mix chosen+rejected
    # in calibration so restriction maps are alignment-agnostic.
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(n)
    cal_idx = perm[: n // 2]
    eval_idx = perm[n // 2 :]

    feats_cal = {
        k: np.concatenate([feats_chosen[k][cal_idx], feats_rejected[k][cal_idx]], axis=0)
        for k in feats_all
    }
    print(
        f"[E0] fitting restriction maps on {feats_cal[next(iter(feats_cal))].shape[0]} "
        f"calibration vectors (ridge={args.ridge})",
        flush=True,
    )
    maps = fit_all_restriction_maps(feats_cal, ridge_lambda=args.ridge)

    feats_eval_chosen = {k: feats_chosen[k][eval_idx] for k in feats_all}
    feats_eval_rejected = {k: feats_rejected[k][eval_idx] for k in feats_all}

    print(f"[E0] computing residuals on {len(eval_idx)} eval pairs (metric={args.metric})", flush=True)
    sum_c = panel_residual_summary(feats_eval_chosen, maps, metric=args.metric)
    sum_r = panel_residual_summary(feats_eval_rejected, maps, metric=args.metric)

    L_chosen = sum_c["L"]
    L_rejected = sum_r["L"]
    t_stat, p_val = welch_t(L_rejected, L_chosen)  # H1: rejected > chosen
    auc_score = auc(L_rejected, L_chosen)

    per_pair = {}
    for (src, dst), r_c in sum_c["pair_residuals"].items():
        r_r = sum_r["pair_residuals"][(src, dst)]
        per_pair[f"{src}->{dst}"] = {
            "mean_chosen": float(r_c.mean()),
            "mean_rejected": float(r_r.mean()),
            "delta": float(r_r.mean() - r_c.mean()),
            "auc_rejected_high": auc(r_r, r_c),
        }

    per_model_loo = {}
    for model in feats_all:
        loo_c = sum_c["loo"][model]
        loo_r = sum_r["loo"][model]
        per_model_loo[model] = {
            "mean_chosen": float(loo_c.mean()),
            "mean_rejected": float(loo_r.mean()),
            "auc_rejected_high": auc(loo_r, loo_c),
        }

    result = {
        "config": {
            "n_pairs": n,
            "n_eval": int(len(eval_idx)),
            "seed": args.seed,
            "ridge": args.ridge,
            "metric": args.metric,
            "panel": [name for name, _ in PANEL],
        },
        "aggregate": {
            "L_chosen_mean": float(L_chosen.mean()),
            "L_chosen_std": float(L_chosen.std()),
            "L_rejected_mean": float(L_rejected.mean()),
            "L_rejected_std": float(L_rejected.std()),
            "welch_t": t_stat,
            "p_value": p_val,
            "cohens_d": float(
                (L_rejected.mean() - L_chosen.mean())
                / np.sqrt(0.5 * (L_rejected.var() + L_chosen.var()) + 1e-12)
            ),
            "auc_rejected_high": auc_score,
        },
        "per_pair": per_pair,
        "per_model_loo": per_model_loo,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[E0] wrote {out_path}", flush=True)

    agg = result["aggregate"]
    print(
        f"\n[E0] SUMMARY  L(chosen)={agg['L_chosen_mean']:.4f}±{agg['L_chosen_std']:.4f}  "
        f"L(rejected)={agg['L_rejected_mean']:.4f}±{agg['L_rejected_std']:.4f}",
        flush=True,
    )
    print(
        f"      Welch t={agg['welch_t']:.2f} (p={agg['p_value']:.2e})  "
        f"Cohen d={agg['cohens_d']:.3f}  AUC={agg['auc_rejected_high']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
