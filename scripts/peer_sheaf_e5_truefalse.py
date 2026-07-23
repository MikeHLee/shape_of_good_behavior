"""E5 (SGB-016): peer-consistency δ¹c on TRUE vs FALSE statements.

A cheap, zero-API, fully-local partial of the "deception-enriched dataset"
arm. HH-RLHF only has ~2.5% deceptive responses, so the deceptive cell in E3
is stuck at n=5. Here we swap in a dense, pre-labeled proxy:
`notrichardren/azaria-mitchell` — ~17k short factual statements each labeled
true (1) or false (0), ungated, loads on `datasets` 4.x.

This is a *proxy*, not intentional deception: a bare false statement is not
"hiding". But it is the cleanest dense local test of the underlying question
— does cross-model peer-consistency lossiness separate false content from
true content? If δ¹c(false) > δ¹c(true) with a real effect, that is positive
evidence the channel tracks (un)truthfulness and motivates paying for the
7-9B Modal panel on a genuine intentional-deception set. If it is flat, the
"competence/coherence detector, not truth detector" reading is reinforced.

Reuses the E1 embedder and the E2/E3 Hodge decomposition (same metric).

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 scripts/peer_sheaf_e5_truefalse.py --n-per-class 1500
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
sys.path.insert(0, str(HERE))

from peer_sheaf_e1 import PANEL, embed_with_model, pick_device  # noqa: E402
from peer_sheaf import fit_all_restriction_maps, panel_residual_summary  # noqa: E402
from peer_hodge import PeerComplex, hodge_decompose  # noqa: E402


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = 0.5 * (a.var() + b.var()) + 1e-12
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ttest_ind

    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    res = ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    s = np.concatenate([scores_pos, scores_neg])
    return float(roc_auc_score(y, s))


def load_true_false(n_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    """Return (true_statements, false_statements), balanced and subsampled."""
    from datasets import load_dataset

    ds = load_dataset("notrichardren/azaria-mitchell", split="train")
    cols = ds.column_names
    text_field = "claim" if "claim" in cols else (
        "statement" if "statement" in cols else cols[0]
    )
    label_field = "label" if "label" in cols else (
        "answer" if "answer" in cols else cols[-1]
    )
    print(f"[E5] dataset cols={cols} text_field={text_field!r} label_field={label_field!r}",
          flush=True)

    texts = ds[text_field]
    labels = ds[label_field]
    true_idx, false_idx = [], []
    for i, lab in enumerate(labels):
        v = int(lab) if not isinstance(lab, str) else (1 if lab.strip().lower() in {"1", "true", "yes"} else 0)
        (true_idx if v == 1 else false_idx).append(i)

    rng = np.random.default_rng(seed)
    rng.shuffle(true_idx)
    rng.shuffle(false_idx)
    k = min(n_per_class, len(true_idx), len(false_idx))
    true_idx = true_idx[:k]
    false_idx = false_idx[:k]
    true_txt = [str(texts[i]).strip() for i in true_idx]
    false_txt = [str(texts[i]).strip() for i in false_idx]
    true_txt = [t for t in true_txt if t]
    false_txt = [t for t in false_txt if t]
    m = min(len(true_txt), len(false_txt))
    print(f"[E5] balanced n_per_class={m} (avail true={len(true_idx)} false={len(false_idx)})",
          flush=True)
    return true_txt[:m], false_txt[:m]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "shared" / "data" / "cache" / "peer_sheaf_e5"),
    )
    parser.add_argument(
        "--out", default=str(ROOT / "shared" / "results" / "peer_sheaf_e5_truefalse.json")
    )
    parser.add_argument("--force-reembed", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[E5] device={device}", flush=True)

    true_txt, false_txt = load_true_false(args.n_per_class, args.seed)
    n = len(true_txt)  # == len(false_txt)
    all_texts = true_txt + false_txt  # [true block | false block]

    feats_all: dict[str, np.ndarray] = {}
    for name, hf_id in PANEL:
        cache_path = cache_dir / f"{name}_n{n}_seed{args.seed}.npy"
        if cache_path.exists() and not args.force_reembed:
            print(f"[E5] loading cached embeddings: {cache_path.name}", flush=True)
            feats_all[name] = np.load(cache_path)
            continue
        t0 = time.time()
        print(f"[E5] embedding {len(all_texts)} statements with {name}", flush=True)
        feats_all[name] = embed_with_model(
            all_texts, hf_id, device=device,
            batch_size=args.batch_size, max_length=args.max_length,
        ).astype(np.float64)
        np.save(cache_path, feats_all[name])
        print(f"  -> {feats_all[name].shape} saved in {time.time()-t0:.1f}s", flush=True)

    feats_true = {k: v[:n] for k, v in feats_all.items()}
    feats_false = {k: v[n:] for k, v in feats_all.items()}

    # Calibration pooled alignment-agnostically across true+false (mirror E1/E3).
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(n)
    cal_idx = perm[: n // 2]
    eval_idx = perm[n // 2 :]

    feats_cal = {
        k: np.concatenate([feats_true[k][cal_idx], feats_false[k][cal_idx]], axis=0)
        for k in feats_all
    }
    t0 = time.time()
    maps = fit_all_restriction_maps(feats_cal, ridge_lambda=args.ridge)
    complex_ = PeerComplex.from_features(feats_cal)
    print(f"[E5] fit maps + complex (D_V={complex_.D_V} D_E={complex_.D_E} "
          f"D_F={complex_.D_F}) in {time.time()-t0:.1f}s", flush=True)

    feats_eval_true = {k: feats_true[k][eval_idx] for k in feats_all}
    feats_eval_false = {k: feats_false[k][eval_idx] for k in feats_all}

    t0 = time.time()
    dec_t = hodge_decompose(complex_, feats_eval_true, maps)
    dec_f = hodge_decompose(complex_, feats_eval_false, maps)
    print(f"[E5] decomposed in {time.time()-t0:.1f}s", flush=True)

    d1_t, d1_f = dec_t.norm_cocycle_violation, dec_f.norm_cocycle_violation
    tot_t, tot_f = dec_t.norm_total, dec_f.norm_total

    # L-residual channel (mirror E1) for cross-check.
    sum_t = panel_residual_summary(feats_eval_true, maps, metric="cosine")
    sum_f = panel_residual_summary(feats_eval_false, maps, metric="cosine")
    L_t, L_f = sum_t["L"], sum_f["L"]

    t_d1, p_d1 = welch_t(d1_f, d1_t)
    t_L, p_L = welch_t(L_f, L_t)

    result = {
        "config": {
            "experiment": "E5 (SGB-016)",
            "dataset": "notrichardren/azaria-mitchell (true/false statements)",
            "metric": "cocycle violation ||δ¹c|| (false high) + L residual",
            "n_per_class": n,
            "n_eval_per_class": int(len(eval_idx)),
            "seed": args.seed,
            "ridge": args.ridge,
            "panel": [
                {"name": nm, "hf_id": h, "hidden_dim": int(feats_all[nm].shape[1])}
                for nm, h in PANEL
            ],
            "complex": {"D_V": complex_.D_V, "D_E": complex_.D_E, "D_F": complex_.D_F},
            "note": "false-statement is a TRUTH proxy, not intentional deception.",
        },
        "delta1c": {
            "false_mean": float(d1_f.mean()), "true_mean": float(d1_t.mean()),
            "cohens_d_false_vs_true": cohens_d(d1_f, d1_t),
            "welch_t": t_d1, "p_value": p_d1,
            "auc_false_high": auc(d1_f, d1_t),
        },
        "total_norm": {
            "false_mean": float(tot_f.mean()), "true_mean": float(tot_t.mean()),
            "cohens_d_false_vs_true": cohens_d(tot_f, tot_t),
            "auc_false_high": auc(tot_f, tot_t),
        },
        "L_residual": {
            "false_mean": float(L_f.mean()), "true_mean": float(L_t.mean()),
            "cohens_d_false_vs_true": cohens_d(L_f, L_t),
            "welch_t": t_L, "p_value": p_L,
            "auc_false_high": auc(L_f, L_t),
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[E5] wrote {out_path}", flush=True)

    print("\n[E5] FALSE vs TRUE  (proxy for the deception channel)")
    dd = result["delta1c"]
    print(f"  ||δ¹c||  false={dd['false_mean']:.4g}  true={dd['true_mean']:.4g}  "
          f"d={dd['cohens_d_false_vs_true']:+.3f}  AUC={dd['auc_false_high']:.3f}  "
          f"p={dd['p_value']:.2e}")
    ll = result["L_residual"]
    print(f"  L resid  false={ll['false_mean']:.4g}  true={ll['true_mean']:.4g}  "
          f"d={ll['cohens_d_false_vs_true']:+.3f}  AUC={ll['auc_false_high']:.3f}  "
          f"p={ll['p_value']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
