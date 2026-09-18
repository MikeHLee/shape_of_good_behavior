"""Held-out re-run of the embedding-level optimizer benchmark, with control arms.

Why this exists: `shared/src/optimizer_comparison.py` scores every method on its own
training pairs (in-sample), and the Hodge potential-alignment target for pair i comes
from a graph that contains pair i's own edge. This script:

  1. Splits the 500 pairs in `shared/results/counterfactual_pairs.json` 80/20 into
     train / held-out, for several split seeds.
  2. Builds the preference graph (direct + kNN similarity edges), the item registry,
     the kNN index and the PCA from the TRAIN pairs only, by calling
     `PreferenceMapper.map_pairs(train_pairs)`. The Hodge diagnosis (potential diffs
     and cycle weights) is computed on that train-only graph.
     Held-out texts are only passed through the frozen MiniLM encoder to get
     evaluation embeddings; they never enter the graph or any training signal.
  3. Trains each method on the train pairs and scores ranking accuracy on both the
     train pairs and the held-out pairs.

Arms (DPO family; the KTO family is identical with KTO in place of DPO):
  DPO                 plain baseline
  Hodge-DPO           HodgeDPOTrainer, diagnosis index-aligned to the pairs
  MarginControl-DPO   HodgeDPOTrainer, same loss form and lambda, but every pair's
                      target = mean of the Hodge potential diffs over the train pairs,
                      and uniform sample weights (no pair-specific Hodge information)
  WeightsOnly-DPO     HodgeDPOTrainer with hodge_lambda = 0 (cycle weights only)
  Misaligned-DPO      HodgeDPOTrainer with potential diffs AND cycle weights permuted
                      across pairs by a fixed permutation per training seed
                      (mimics the index misalignment in the published v3 run)
  GRPO, ORPO          references

The existing benchmark code is not modified; controls are built by constructing a
modified CycleDiagnosis object.

Usage:
    ./venv/bin/python3 scripts/heldout_hodge_benchmark.py            # full run
    ./venv/bin/python3 scripts/heldout_hodge_benchmark.py --smoke    # 1 split, 2 seeds
"""

import argparse
import copy
import dataclasses
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PAIRS_PATH = ROOT / "shared" / "results" / "counterfactual_pairs.json"
OUT_PATH = ROOT / "shared" / "results" / "optimizer_comparison_heldout_v1.json"

HELDOUT_FRAC = 0.2
EPOCHS = 50  # matches the published v3 run (modal_runner.py sets rm_epochs = 50)

FAMILIES = ("DPO", "KTO")
ARMS = (
    ["DPO", "KTO", "GRPO", "ORPO"]
    + [f"{a}-{f}" for f in FAMILIES
       for a in ("Hodge", "MarginControl", "WeightsOnly", "Misaligned")]
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pairs():
    from shared.src.counterfactual_gen import CounterfactualPair
    rows = json.loads(PAIRS_PATH.read_text())
    return [CounterfactualPair(**r) for r in rows]


def split_indices(n, split_seed):
    perm = np.random.RandomState(split_seed).permutation(n)
    n_held = int(round(HELDOUT_FRAC * n))
    return np.sort(perm[n_held:]), np.sort(perm[:n_held])  # train, held-out


def build_split(split_seed, pairs, cfg):
    """Graph + diagnosis from train pairs only; embeddings for held-out pairs."""
    from shared.src.preference_mapper import PreferenceMapper
    from shared.src.hodge_diagnostic import HodgeDiagnosticCritic

    tr_idx, ho_idx = split_indices(len(pairs), split_seed)
    train_pairs = [pairs[i] for i in tr_idx]
    held_pairs = [pairs[i] for i in ho_idx]

    mapper = PreferenceMapper(cfg)
    mapping = mapper.map_pairs(train_pairs)  # graph, kNN, PCA: train only
    n_tr = len(train_pairs)

    # Alignment check: direct edge i must be (ideal_i, exploit_i) of train pair i.
    for i, p in enumerate(train_pairs):
        e = mapping.preference_edges[i]
        assert e[0] == mapper._item_texts[p.ideal_text[:200]], i
        assert e[1] == mapper._item_texts[p.exploit_text[:200]], i
    # Leakage check: no held-out text registered as a graph node, unless the identical
    # text (first 200 chars, the registry key) also occurs in a train pair.
    train_keys = {t[:200] for p in train_pairs for t in (p.ideal_text, p.exploit_text)}
    for p in held_pairs:
        for t in (p.ideal_text, p.exploit_text):
            assert (t[:200] not in mapper._item_texts) or (t[:200] in train_keys)
    n_heldout_text_overlap = sum(
        1 for p in held_pairs for t in (p.ideal_text, p.exploit_text)
        if t[:200] in train_keys
    )

    diagnosis = HodgeDiagnosticCritic(cfg).diagnose_for_samples(
        mapping.preference_edges, mapping.n_items,
        embedding_pairs=mapping.embedding_pairs,
    )

    # Held-out embeddings: frozen encoder only (same model object as the mapper).
    model = mapper._get_embed_model()
    ho_emb = model.encode(
        [p.exploit_text for p in held_pairs]
        + [p.ideal_text for p in held_pairs]
        + [p.context_text for p in held_pairs],
        show_progress_bar=False, batch_size=32,
    )
    m = len(held_pairs)
    held = dict(exploit=ho_emb[:m], ideal=ho_emb[m:2 * m], context=ho_emb[2 * m:])

    # Consistency check: the held-out encoding path, applied to train texts, must
    # reproduce the embeddings the mapper produced for those same texts.
    chk = model.encode([p.ideal_text for p in train_pairs[:50]],
                       show_progress_bar=False, batch_size=32)
    ref = np.array([mapping.embedding_pairs[i].ideal_embed for i in range(50)])
    encoder_path_max_abs_diff = float(np.abs(chk - ref).max())

    # Reference only (not one of the benchmark arms): a linear Bradley-Terry probe on
    # the same frozen features, trained on train pairs, scored on held-out pairs.
    # It indicates how much held-out signal the frozen MiniLM features carry.
    from sklearn.linear_model import LogisticRegression
    d_tr = np.array([ep.ideal_embed - ep.exploit_embed for ep in mapping.embedding_pairs])
    d_ho = held["ideal"] - held["exploit"]
    probe = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(
        np.vstack([d_tr, -d_tr]), np.r_[np.ones(len(d_tr)), np.zeros(len(d_tr))])
    linear_probe = {"train_acc": float((d_tr @ probe.coef_[0] > 0).mean()),
                    "heldout_acc": float((d_ho @ probe.coef_[0] > 0).mean())}

    n_edges = len(mapping.preference_edges)
    pd = diagnosis.sample_potential_diffs
    w = diagnosis.per_sample_weights
    graph_info = {
        "split_seed": split_seed,
        "n_train": n_tr,
        "n_heldout": m,
        "n_edges_total": n_edges,
        "n_edges_direct": n_tr,
        "n_edges_similarity_and_cross": n_edges - n_tr,
        "n_items": mapping.n_items,
        "heldout_texts_identical_to_a_train_text": n_heldout_text_overlap,
        "encoder_path_max_abs_diff": encoder_path_max_abs_diff,
        "reference_linear_probe_C1": linear_probe,
        "diagnosis_exploit_fraction": float(diagnosis.exploit_fraction),
        "marginal_h1": float(diagnosis.marginal_h1),
        "dphi_mean": float(pd.mean()), "dphi_sd": float(pd.std()),
        "dphi_min": float(pd.min()), "dphi_max": float(pd.max()),
        "dphi_n_positive": int((pd > 0).sum()),
        "weight_mean": float(w.mean()), "weight_sd": float(w.std()),
        "weight_min": float(w.min()), "weight_max": float(w.max()),
        "train_indices": tr_idx.tolist(),
        "heldout_indices": ho_idx.tolist(),
    }
    return mapping, diagnosis, held, graph_info


def make_samples(mapping):
    from shared.src.preference_optimizers import mapping_to_preference_samples
    return mapping_to_preference_samples(mapping)  # fresh objects, weight = 1.0


def make_heldout_samples(held):
    from shared.src.preference_optimizers import PreferenceSample
    return [
        PreferenceSample(prompt_embed=held["context"][i], chosen_embed=held["ideal"][i],
                         rejected_embed=held["exploit"][i])
        for i in range(len(held["ideal"]))
    ]


# ---------------------------------------------------------------------------
# Control diagnoses
# ---------------------------------------------------------------------------

def margin_control_diagnosis(diag):
    """Every target = mean train Δφ; uniform weights."""
    d = copy.deepcopy(diag)
    n = len(diag.sample_potential_diffs)
    d.sample_potential_diffs = np.full(n, float(diag.sample_potential_diffs.mean()))
    d.per_sample_weights = np.ones(n)
    return d


def misaligned_diagnosis(diag, train_seed):
    """Permute Δφ and weights across pairs (same permutation for both)."""
    d = copy.deepcopy(diag)
    perm = np.random.RandomState(10_000 + train_seed).permutation(
        len(diag.sample_potential_diffs))
    d.sample_potential_diffs = diag.sample_potential_diffs[perm].copy()
    d.per_sample_weights = diag.per_sample_weights[perm].copy()
    return d


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

def make_trainer(arm, cfg, diag, train_seed):
    from shared.src.preference_optimizers import (
        DPOTrainer, GRPOTrainer, ORPOTrainer, KTOTrainer)
    from shared.src.hodge_preference_optimizers import HodgeDPOTrainer, HodgeKTOTrainer

    common = dict(embed_dim=cfg.embed_dim, hidden_dim=cfg.rm_hidden_dim,
                  lr=cfg.rm_lr, epochs=EPOCHS)
    dpo_kw = dict(beta=cfg.dpo_beta, **common)
    kto_kw = dict(beta=cfg.kto_beta, **common)

    if arm == "DPO":
        return DPOTrainer(**dpo_kw)
    if arm == "KTO":
        # Same kwargs as optimizer_comparison.py (lambda_good / lambda_bad).
        return KTOTrainer(lambda_good=cfg.kto_lambda_good,
                          lambda_bad=cfg.kto_lambda_bad, **kto_kw)
    if arm == "GRPO":
        return GRPOTrainer(beta=cfg.grpo_beta, group_size=cfg.grpo_group_size, **common)
    if arm == "ORPO":
        return ORPOTrainer(lambda_align=cfg.orpo_lambda, **common)

    variant, family = arm.split("-")
    lam = cfg.hodge_lambda
    if variant == "Hodge":
        d = diag
    elif variant == "MarginControl":
        d = margin_control_diagnosis(diag)
    elif variant == "WeightsOnly":
        d, lam = diag, 0.0
    elif variant == "Misaligned":
        d = misaligned_diagnosis(diag, train_seed)
    else:
        raise ValueError(arm)
    # Hodge-KTO in optimizer_comparison.py uses the KTOTrainer defaults for
    # lambda_good/lambda_bad (1.0 / 1.33), which equal the config values.
    cls = HodgeDPOTrainer if family == "DPO" else HodgeKTOTrainer
    kw = dpo_kw if family == "DPO" else kto_kw
    return cls(diagnosis=d, hodge_lambda=lam, **kw)


# ---------------------------------------------------------------------------
# One split
# ---------------------------------------------------------------------------

def run_split(args):
    split_seed, n_train_seeds, arms = args
    import torch
    torch.set_num_threads(1)
    logging.disable(logging.CRITICAL)
    from shared.src.config import PipelineConfig

    cfg = PipelineConfig()
    t0 = time.time()
    pairs = load_pairs()
    mapping, diag, held, graph_info = build_split(split_seed, pairs, cfg)
    held_samples = make_heldout_samples(held)
    graph_info["build_s"] = time.time() - t0

    rows = []
    for arm in arms:
        for seed in range(n_train_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_samples = make_samples(mapping)  # fresh: Hodge trainers mutate weights
            tr = make_trainer(arm, cfg, diag, seed)
            res = tr.train(train_samples)
            ho_acc, _, _ = tr.evaluate_exploit_resistance(held_samples)
            rows.append(dict(split_seed=split_seed, train_seed=seed, method=arm,
                             train_acc=float(res.exploit_resistance),
                             heldout_acc=float(ho_acc),
                             final_loss=float(res.losses[-1])))
    graph_info["total_s"] = time.time() - t0
    return graph_info, rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _paired(a, b):
    from scipy import stats
    diff = np.asarray(a) - np.asarray(b)
    out = dict(n=int(len(diff)), mean_diff=float(diff.mean()),
               sd_diff=float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
               n_positive=int((diff > 0).sum()), n_zero=int((diff == 0).sum()),
               n_negative=int((diff < 0).sum()))
    out["d_z"] = float(out["mean_diff"] / out["sd_diff"]) if out["sd_diff"] > 0 else None
    if len(diff) > 1 and out["sd_diff"] > 0:
        t = stats.ttest_rel(a, b)
        out["paired_t"] = float(t.statistic)
        out["paired_t_p"] = float(t.pvalue)
    else:
        out["paired_t"] = None
        out["paired_t_p"] = None
    if (diff != 0).sum() >= 1:
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox")
            out["wilcoxon_p"] = float(w.pvalue)
        except ValueError:
            out["wilcoxon_p"] = None
    else:
        out["wilcoxon_p"] = None
    return out


def summarize(rows, arms, split_seeds):
    from collections import defaultdict
    by = defaultdict(dict)  # (method) -> {(split, seed): row}
    for r in rows:
        by[r["method"]][(r["split_seed"], r["train_seed"])] = r

    def stats_for(keys, method):
        tr = np.array([by[method][k]["train_acc"] for k in keys])
        ho = np.array([by[method][k]["heldout_acc"] for k in keys])
        return dict(n=int(len(keys)),
                    heldout_mean=float(ho.mean()), heldout_sd=float(ho.std(ddof=1)),
                    train_mean=float(tr.mean()), train_sd=float(tr.std(ddof=1)),
                    gap_train_minus_heldout=float((tr - ho).mean()))

    comparisons = []
    for f in FAMILIES:
        comparisons += [(f"Hodge-{f}", f), (f"Hodge-{f}", f"MarginControl-{f}"),
                        (f"MarginControl-{f}", f), (f"WeightsOnly-{f}", f),
                        (f"Misaligned-{f}", f), (f"Hodge-{f}", f"Misaligned-{f}")]
    comparisons = [(a, b) for a, b in comparisons if a in arms and b in arms]

    summary = {"per_split": {}, "pooled": {}}
    for s in split_seeds:
        keys = sorted(k for k in by[arms[0]] if k[0] == s)
        blk = {"methods": {m: stats_for(keys, m) for m in arms}, "paired_heldout": {},
               "paired_train": {}}
        for a, b in comparisons:
            blk["paired_heldout"][f"{a} - {b}"] = _paired(
                [by[a][k]["heldout_acc"] for k in keys], [by[b][k]["heldout_acc"] for k in keys])
            blk["paired_train"][f"{a} - {b}"] = _paired(
                [by[a][k]["train_acc"] for k in keys], [by[b][k]["train_acc"] for k in keys])
        summary["per_split"][str(s)] = blk

    keys = sorted(by[arms[0]].keys())
    pooled = {"methods": {m: stats_for(keys, m) for m in arms},
              "paired_heldout": {}, "paired_train": {},
              "paired_heldout_split_level": {}}
    for a, b in comparisons:
        pooled["paired_heldout"][f"{a} - {b}"] = _paired(
            [by[a][k]["heldout_acc"] for k in keys], [by[b][k]["heldout_acc"] for k in keys])
        pooled["paired_train"][f"{a} - {b}"] = _paired(
            [by[a][k]["train_acc"] for k in keys], [by[b][k]["train_acc"] for k in keys])
        # Conservative: one number per split (mean over training seeds), n = #splits.
        am = [np.mean([by[a][k]["heldout_acc"] for k in keys if k[0] == s]) for s in split_seeds]
        bm = [np.mean([by[b][k]["heldout_acc"] for k in keys if k[0] == s]) for s in split_seeds]
        pooled["paired_heldout_split_level"][f"{a} - {b}"] = _paired(am, bm)
    summary["pooled"] = pooled
    return summary


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seeds", default="0,1,2,3,4")
    ap.add_argument("--train-seeds", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--replay-summary", default=None,
                    help="optional JSON with the v3 subsample replay summary to embed")
    args = ap.parse_args()

    split_seeds = [int(s) for s in args.split_seeds.split(",")]
    n_seeds = args.train_seeds
    if args.smoke:
        split_seeds, n_seeds = split_seeds[:1], 2

    t0 = time.time()
    with Pool(len(split_seeds)) as pool:
        results = pool.map(run_split, [(s, n_seeds, ARMS) for s in split_seeds])
    graphs = [g for g, _ in results]
    rows = [r for _, rs in results for r in rs]
    summary = summarize(rows, ARMS, split_seeds)
    runtime = time.time() - t0

    from shared.src.config import PipelineConfig
    cfg = PipelineConfig()
    out = {
        "notes": NOTES,
        "config": {
            "data": "shared/results/counterfactual_pairs.json (500 HH-RLHF harmless-base pairs)",
            "embed_model": cfg.embed_model, "embed_dim": cfg.embed_dim,
            "hidden_dim": cfg.rm_hidden_dim, "lr": cfg.rm_lr, "epochs": EPOCHS,
            "batch_size": 64, "dpo_beta": cfg.dpo_beta, "kto_beta": cfg.kto_beta,
            "kto_lambda_good": cfg.kto_lambda_good, "kto_lambda_bad": cfg.kto_lambda_bad,
            "grpo_beta": cfg.grpo_beta, "orpo_lambda": cfg.orpo_lambda,
            "hodge_lambda": cfg.hodge_lambda,
            "heldout_fraction": HELDOUT_FRAC, "split_seeds": split_seeds,
            "train_seeds_per_split": n_seeds, "arms": ARMS,
            "split_rule": "np.random.RandomState(split_seed).permutation(500); first 100 = held-out",
            "train_seed_rule": "np.random.seed(s); torch.manual_seed(s) before trainer construction "
                               "(same init across arms -> paired comparisons)",
            "misaligned_permutation_rule": "np.random.RandomState(10000 + train_seed).permutation(n_train), "
                                           "applied to both potential diffs and weights",
            "margin_control_rule": "target = mean(train potential diffs) for every pair; weights = 1",
            "metric": "ranking accuracy: fraction of pairs with implicit reward(chosen) > "
                      "implicit reward(rejected); evaluate_exploit_resistance()",
            "torch_threads": 1,
            "smoke": bool(args.smoke),
        },
        "graphs": graphs,
        "summary": summary,
        "runs": rows,
        "runtime_s": runtime,
    }
    if args.replay_summary:
        out["published_v3_replay"] = json.loads(Path(args.replay_summary).read_text())
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({len(rows)} runs, {runtime:.0f}s)")
    p = summary["pooled"]
    for m in ARMS:
        s = p["methods"][m]
        print(f"{m:20s} heldout {s['heldout_mean']:.4f} ± {s['heldout_sd']:.4f}   "
              f"train {s['train_mean']:.4f} ± {s['train_sd']:.4f}   gap {s['gap_train_minus_heldout']:+.4f}")
    for k, v in p["paired_heldout"].items():
        sl = p["paired_heldout_split_level"][k]
        print(f"{k:34s} diff {v['mean_diff']:+.4f} pos {v['n_positive']}/{v['n']} "
              f"(zero {v['n_zero']}) t_p {v['paired_t_p']} wil_p {v['wilcoxon_p']} "
              f"| split-level p {sl['paired_t_p']}")


NOTES = [
    "HELD-OUT RE-RUN. Every method is trained on 400 train pairs and scored on 100 held-out "
    "pairs per split (5 splits x 30 training seeds). Train accuracy is also reported.",
    "The preference graph (direct + kNN similarity edges), the item registry, the kNN index, "
    "the PCA and the Hodge diagnosis (potential diffs, cycle weights) are built from the TRAIN "
    "pairs ONLY (PreferenceMapper.map_pairs(train_pairs)). Held-out texts only pass through the "
    "frozen all-MiniLM-L6-v2 encoder for evaluation embeddings.",
    "Hodge targets and weights are index-aligned: direct edge i is asserted to be "
    "(ideal_i, exploit_i) of train pair i before the diagnosis is used.",
    "THIS FILE USES A REBUILT 500-PAIR GRAPH, NOT THE PUBLISHED PIPELINE. The published v3 "
    "result (optimizer_comparison_hodge_v3_30seed.json, byte-identical to "
    "pipeline/optimizer_comparison.json on the Modal volume reward-hacking-results) was produced "
    "by shared/modal_runner.py::run_optimizer_comparison from the volume-only file "
    "pipeline/mapping.pkl (2268 pairs: 2000 HH-RLHF harmless-base + 268 TRACE LLM "
    "counterfactuals; 21767 edges), subsampled to 500 pairs with the UNSEEDED global "
    "np.random.choice (modal_runner.py:99). The exact subsample is not recoverable, so the "
    "published run is not reproducible. counterfactual_pairs.json is NOT the published "
    "training set.",
    "MISALIGNMENT IN THE PUBLISHED RUN: after that subsample, kept_edges keeps the original "
    "edge order while embedding_pairs follows the random draw order, and "
    "hodge_diagnostic.py:230-251 reads sample i's endpoints from preference_edges[i]. So each "
    "sample's potential-diff target and cycle weight came from a different pair's edge "
    "(replay: see published_v3_replay). The Misaligned-* arms mimic this with a permutation.",
    "Rebuilding from counterfactual_pairs.json in-sample (30 seeds) gave DPO 0.9074 and KTO "
    "0.7681 vs published 0.9403 / 0.8004, and Hodge-DPO 1.0000 / Hodge-KTO 0.9979 vs "
    "published 0.9999 / 0.9964.",
    "The metric is ranking accuracy over frozen MiniLM embeddings of HH-RLHF pairs; it does not "
    "measure reward hacking under optimisation pressure (see shared/results/README.md caveat 3).",
]


if __name__ == "__main__":
    main()
