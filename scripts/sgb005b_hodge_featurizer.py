"""SGB-005b: test the Hodge decomposition as an auxiliary FEATURE, not a
training-time loss reweight.

SGB-004 found no LM-level Hodge-PPO advantage over standard PPO on the true
holdout split (25.5% vs 23.5% exploit resistance — noise at n=51). The
Hodge signal there was only ever used to reweight the reward-model's
Bradley-Terry loss (`compute_hodge_weights` in lm_finetuning.py). This
script asks a narrower, cheaper question instead: does the per-pair Hodge
gradient-potential difference (`sample_potential_diffs` from
`HodgeDiagnosticCritic.diagnose_for_samples`, computed once from frozen
sentence embeddings, no GPU/training needed) carry information the
already-trained standard reward model's own scalar score is missing, when
ranking reference ideal_text vs exploit_text pairs?

Runs entirely on CPU, locally. Depends on shared/results/finetune/
sgb005b_rm_scores.json (produced by `modal run shared/modal_finetune.py
--stage score-pairs`) for the RM side; computes the Hodge side itself.

Usage:
    ./venv/bin/python3 scripts/sgb005b_hodge_featurizer.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.src.config import PipelineConfig
from shared.modal_finetune import _load_pairs
from shared.src.hodge_diagnostic import HodgeDiagnosticCritic
from shared.src.preference_mapper import EmbeddingPair

ROOT = Path(__file__).resolve().parent.parent
RM_SCORES_PATH = ROOT / "shared/results/finetune/sgb005b_rm_scores.json"
OUT_PATH = ROOT / "shared/results/finetune/sgb005b_hodge_featurizer_result.json"


def compute_hodge_potential_diffs(pairs, pipeline_config, direct_edge_prob=None):
    """Mirrors compute_hodge_weights' graph construction but keeps
    sample_potential_diffs instead of discarding it for per_sample_weights.

    direct_edge_prob: confidence assigned to the direct (exploit_i, ideal_i)
    edge that compute_hodge_weights normally sets to ~0.999 ("definite
    preference"). Left at that value, the graph is *told* the answer for
    the exact quantity (ideal vs exploit ranking) this function is then
    asked to recover -- near-tautological. Passing 0.5 (log-odds 0, i.e. a
    zero-weight/uninformative edge) removes that direct signal so only the
    unsupervised cross-pair kNN edges (pure embedding-similarity structure,
    no ideal/exploit labels) drive the resulting potential -- an ablation
    to check whether the earlier result was circular.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    exploit_embs = embedder.encode([p.exploit_text for p in pairs], normalize_embeddings=True, show_progress_bar=False)
    ideal_embs   = embedder.encode([p.ideal_text   for p in pairs], normalize_embeddings=True, show_progress_bar=False)
    context_embs = embedder.encode([p.context_text for p in pairs], normalize_embeddings=True, show_progress_bar=False)

    n = len(pairs)
    EPS = 1e-3
    def _as_prob(x):
        return float(np.clip(x, EPS, 1.0 - EPS))

    direct_p = 1.0 - EPS if direct_edge_prob is None else direct_edge_prob
    preference_edges = [(2 * i, 2 * i + 1, direct_p) for i in range(n)]

    k = min(5, n - 1)
    all_embs = np.vstack([e for pair in zip(exploit_embs, ideal_embs) for e in pair])
    sim = cosine_similarity(all_embs)
    for a in range(len(all_embs)):
        s = sim[a].copy(); s[a] = -1.0
        for b in np.argsort(s)[-k:]:
            preference_edges.append((int(a), int(b), _as_prob((s[b] + 1.0) / 2.0)))

    fake_pairs = [
        EmbeddingPair(
            exploit_embed=exploit_embs[i],
            ideal_embed=ideal_embs[i],
            context_embed=context_embs[i],
            constitutional_gradient=ideal_embs[i] - exploit_embs[i],
            category=getattr(pairs[i], "exploit_category", "default"),
        )
        for i in range(n)
    ]

    critic = HodgeDiagnosticCritic(pipeline_config)
    diagnosis = critic.diagnose_for_samples(
        preference_edges=preference_edges,
        n_items=2 * n,
        embedding_pairs=fake_pairs,
    )

    diffs = diagnosis.sample_potential_diffs
    if diffs is None:
        raise RuntimeError("diagnose_for_samples returned no sample_potential_diffs")
    diffs = np.asarray(diffs, dtype=np.float64)
    nonfinite = ~np.isfinite(diffs)
    if nonfinite.any():
        print(f"  {nonfinite.sum()}/{len(diffs)} non-finite potential diffs -> replaced with 0.0")
        diffs[nonfinite] = 0.0
    return diffs, diagnosis.exploit_fraction


def fit_lambda(rm_diff, hodge_diff, lambdas):
    """Grid-search the single scalar combining weight maximizing ranking
    accuracy fraction(rm_diff + lambda*hodge_diff > 0) — every example is a
    'ideal preferred' instance by TRACE construction, so this is a 1-D
    threshold-style fit, not a full classifier (no negative-direction class
    exists to fit a real logistic regression against)."""
    best_lambda, best_acc = 0.0, float((rm_diff > 0).mean())
    for lam in lambdas:
        acc = float((rm_diff + lam * hodge_diff > 0).mean())
        if acc > best_acc:
            best_lambda, best_acc = lam, acc
    return best_lambda, best_acc


def main():
    config = PipelineConfig()
    config.trace_max_samples   = 517
    config.hh_rlhf_max_samples = 0
    config.cache_dir           = str(ROOT / "shared/data/cache")

    train_pairs, _  = _load_pairs(config, split="train")
    holdout_pairs, _ = _load_pairs(config, split="holdout")
    print(f"train={len(train_pairs)}  holdout={len(holdout_pairs)}")

    all_pairs = train_pairs + holdout_pairs
    n_train = len(train_pairs)

    print("Computing Hodge potential diffs over the combined train+holdout graph "
          "(transductive: holdout nodes are embedded into the same fixed kNN/direct-edge "
          "graph as train, but no reward-model or label information touches this step)...")
    diffs, exploit_fraction = compute_hodge_potential_diffs(all_pairs, config)
    print(f"exploit_fraction (conditional/marginal H1) = {exploit_fraction:.2%}")

    print("\nAblation: same graph but with the direct (exploit_i, ideal_i) edge set to "
          "an uninformative prob=0.5 (zero log-odds weight) -- isolates whatever signal "
          "comes from the unsupervised cross-pair kNN structure alone, since the direct "
          "edge at prob=0.999 literally tells the graph the answer being tested.")
    diffs_ablation, exploit_fraction_ablation = compute_hodge_potential_diffs(
        all_pairs, config, direct_edge_prob=0.5
    )
    print(f"exploit_fraction (ablation) = {exploit_fraction_ablation:.2%}")

    if not RM_SCORES_PATH.exists():
        print(f"\n{RM_SCORES_PATH} not found yet — run:")
        print("  modal run shared/modal_finetune.py --stage score-pairs")
        print("then re-run this script. Hodge features computed above are cheap to redo.")
        return

    rm_scores = json.loads(RM_SCORES_PATH.read_text())
    rm_by_text = {}
    for split in ("train", "holdout"):
        for row in rm_scores[split]:
            rm_by_text[row["exploit_text"]] = row

    rm_diff = np.zeros(len(all_pairs))
    missing = 0
    for i, p in enumerate(all_pairs):
        row = rm_by_text.get(p.exploit_text)
        if row is None:
            missing += 1
            continue
        rm_diff[i] = row["rm_ideal"] - row["rm_exploit"]
    if missing:
        print(f"  WARNING: {missing}/{len(all_pairs)} pairs had no matching RM score (join by exploit_text)")

    train_rm, holdout_rm = rm_diff[:n_train], rm_diff[n_train:]

    def _evaluate(diffs_arr, label):
        train_hodge, holdout_hodge = diffs_arr[:n_train], diffs_arr[n_train:]
        hodge_alone_train = float((train_hodge > 0).mean())
        hodge_alone_holdout = float((holdout_hodge > 0).mean())
        lambdas = np.concatenate([[0.0], np.linspace(-10, 10, 401)])
        best_lambda, best_train_acc = fit_lambda(train_rm, train_hodge, lambdas)
        featurized_holdout_acc = float((holdout_rm + best_lambda * holdout_hodge > 0).mean())
        print(f"\n[{label}]")
        print(f"  Hodge feature alone  — train acc: {hodge_alone_train:.2%}   holdout acc: {hodge_alone_holdout:.2%}")
        print(f"  RM + Hodge feature   — train acc: {best_train_acc:.2%}   holdout acc: {featurized_holdout_acc:.2%}   (lambda={best_lambda:.3f})")
        return {
            "hodge_alone": {"train_accuracy": hodge_alone_train, "holdout_accuracy": hodge_alone_holdout},
            "rm_plus_hodge": {
                "best_lambda_fit_on_train": best_lambda,
                "train_accuracy": best_train_acc,
                "holdout_accuracy": featurized_holdout_acc,
            },
        }

    baseline_train_acc = float((train_rm > 0).mean())
    baseline_holdout_acc = float((holdout_rm > 0).mean())

    print("\n=== SGB-005b: Hodge potential-diff as RM featurizer ===")
    print(f"RM alone — train acc: {baseline_train_acc:.2%}   holdout acc: {baseline_holdout_acc:.2%}")
    print("(RM train_loss was ~0.693 = log(2), i.e. the RM barely learned to discriminate "
          "at all — treat the low RM-alone accuracy above as a sign the RM itself, not "
          "just the featurizer test, needs scrutiny.)")

    full_result = _evaluate(diffs, "with direct edges (original — potentially circular)")
    ablation_result = _evaluate(diffs_ablation, "ablation — direct edge removed (prob=0.5, unsupervised kNN only)")

    result = {
        "n_train": n_train,
        "n_holdout": len(holdout_pairs),
        "rm_alone": {
            "train_accuracy": baseline_train_acc,
            "holdout_accuracy": baseline_holdout_acc,
            "note": "RM train_loss ~0.693 (=log 2) in the SGB-004 RM training log -- near-chance, the RM barely learned to discriminate ideal from exploit at all.",
        },
        "with_direct_edges": {"exploit_fraction_graph": exploit_fraction, **full_result},
        "ablation_no_direct_edges": {"exploit_fraction_graph": exploit_fraction_ablation, **ablation_result},
        "interpretation": (
            "accuracy = fraction of reference pairs correctly ranked (score(ideal) > "
            "score(exploit)). 'with_direct_edges' feeds the graph a near-certain "
            "(prob=0.999) direct edge asserting ideal>exploit for every pair -- the same "
            "quantity being tested for recovery, so a high score there is expected/"
            "near-tautological, not evidence of independent signal. "
            "'ablation_no_direct_edges' sets that edge's confidence to 0.5 (zero log-odds "
            "weight, i.e. removed), so any accuracy above chance (50%) there reflects "
            "genuine unsupervised structure in the cross-pair kNN embedding-similarity "
            "graph alone, with no ideal/exploit labels involved. 'rm_plus_hodge' lambda "
            "is fit on train only via grid search (no negative-class labels exist "
            "to fit a real logistic regression, since every reference pair is a "
            "known ideal>exploit instance by TRACE construction)."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
