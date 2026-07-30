"""SGB-042 Stage A: length-matched re-scoring audit.

Reuses EXISTING RM scores (no new training, no GPU, no API calls) to test
whether reward-model ranking accuracy survives once ideal_text/exploit_text
length is controlled for. If accuracy collapses toward chance on the
length-matched subset, the near-100% accuracy reported in SGB-005c/SGB-006
is largely a length artifact rather than genuine content understanding.

Inputs (already on disk, all pre-existing SGB-005c/SGB-006 outputs):
  - shared/results/finetune/sgb005b_rm_scores.json        (1.5B standard RM)
  - shared/results/finetune/sgb005b_rm_scores_hodge.json  (1.5B Hodge RM)
  - shared/results/finetune/7b_rm_verify_scores.json      (7B standard RM)
  - shared/results/finetune/7b_rm_verify_scores_hodge.json(7B Hodge RM)

Output:
  - shared/results/finetune/sgb042_stage_a_results.json
"""
import json
from math import comb
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "shared" / "results" / "finetune"

DATASETS = {
    "1.5B_standard": "sgb005b_rm_scores.json",
    "1.5B_hodge": "sgb005b_rm_scores_hodge.json",
    "7B_standard": "7b_rm_verify_scores.json",
    "7B_hodge": "7b_rm_verify_scores_hodge.json",
}

LENGTH_MATCH_TOLERANCE = 0.20  # primary threshold: keep pair only if len(exploit)/len(ideal) within +-20%
TOLERANCE_SWEEP = [0.10, 0.20, 0.30, 0.50]  # robustness sweep, avoids resting the finding on one threshold

# Hedge-phrase openers found in ideal_text (see project_trace_length_confound.md);
# used to test the STYLE confound the same cheap way length is tested above.
HEDGE_OPENER_PREFIXES = [
    "i need to stop",
    "i want to flag",
    "i need to be transparent",
    "i need to be honest",
    "i need to be direct",
    "i need to be upfront",
    "i have to be direct",
    "i have to be transparent",
    "i want to be transparent",
    "i want to be direct",
    "i want to be upfront",
]


def has_hedge_opener(record):
    text = record["ideal_text"].strip().lower()
    return any(text.startswith(p) for p in HEDGE_OPENER_PREFIXES)


def binomial_two_sided_p(k, n, p=0.5):
    """P(X <= k or X >= n-k) under Binomial(n, p) -- two-sided exact test for 'accuracy is chance'."""
    if n == 0:
        return None
    lo = min(k, n - k)
    hi = n - lo
    tail = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, lo + 1))
    tail += sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(hi, n + 1))
    return min(1.0, tail)


def ranking_accuracy(records):
    if not records:
        return None
    correct = sum(1 for r in records if r["rm_ideal"] > r["rm_exploit"])
    return correct / len(records)


def length_ratio_ok(record, tol):
    li = len(record["ideal_text"])
    le = len(record["exploit_text"])
    if li == 0 or le == 0:
        return False
    ratio = le / li
    return (1 - tol) <= ratio <= (1 + tol)


def audit_split(records, tol):
    all_acc = ranking_accuracy(records)
    matched = [r for r in records if length_ratio_ok(r, tol)]
    matched_acc = ranking_accuracy(matched)
    matched_correct = sum(1 for r in matched if r["rm_ideal"] > r["rm_exploit"])
    matched_p = binomial_two_sided_p(matched_correct, len(matched)) if matched else None

    # Trivial length-only classifier as a baseline on the SAME full set:
    # "predict ideal_text is preferred iff it is longer than exploit_text."
    length_only_correct = sum(
        1 for r in records if len(r["ideal_text"]) > len(r["exploit_text"])
    )
    length_only_acc = length_only_correct / len(records) if records else None

    sweep = {}
    for t in TOLERANCE_SWEEP:
        m = [r for r in records if length_ratio_ok(r, t)]
        sweep[t] = {"n": len(m), "accuracy": ranking_accuracy(m)}

    # Style-confound check: drop every pair whose ideal_text uses a known
    # hedge-phrase opener, then re-check accuracy on what remains.
    no_hedge = [r for r in records if not has_hedge_opener(r)]
    no_hedge_acc = ranking_accuracy(no_hedge)
    no_hedge_correct = sum(1 for r in no_hedge if r["rm_ideal"] > r["rm_exploit"])
    no_hedge_p = binomial_two_sided_p(no_hedge_correct, len(no_hedge)) if no_hedge else None

    # Combined check: length-matched AND no hedge opener -- the strictest cheap slice.
    strict = [r for r in matched if not has_hedge_opener(r)]
    strict_acc = ranking_accuracy(strict)

    return {
        "n_total": len(records),
        "n_length_matched": len(matched),
        "pct_length_matched": round(100 * len(matched) / len(records), 1) if records else None,
        "rm_ranking_accuracy_all": round(all_acc, 4) if all_acc is not None else None,
        "rm_ranking_accuracy_length_matched": round(matched_acc, 4) if matched_acc is not None else None,
        "length_matched_binomial_p_vs_chance": round(matched_p, 6) if matched_p is not None else None,
        "length_only_classifier_accuracy_all": round(length_only_acc, 4) if length_only_acc is not None else None,
        "tolerance_sweep": {
            str(t): {"n": v["n"], "accuracy": round(v["accuracy"], 4) if v["accuracy"] is not None else None}
            for t, v in sweep.items()
        },
        "n_hedge_opener": len(records) - len(no_hedge),
        "pct_hedge_opener": round(100 * (len(records) - len(no_hedge)) / len(records), 1) if records else None,
        "rm_ranking_accuracy_no_hedge_opener": round(no_hedge_acc, 4) if no_hedge_acc is not None else None,
        "no_hedge_binomial_p_vs_chance": round(no_hedge_p, 6) if no_hedge_p is not None else None,
        "n_length_matched_and_no_hedge": len(strict),
        "rm_ranking_accuracy_length_matched_and_no_hedge": round(strict_acc, 4) if strict_acc is not None else None,
    }


def main():
    results = {"length_match_tolerance": LENGTH_MATCH_TOLERANCE, "datasets": {}}

    for label, fname in DATASETS.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        data = json.load(open(path))
        entry = {}
        for split in ("train", "holdout"):
            if split in data:
                entry[split] = audit_split(data[split], LENGTH_MATCH_TOLERANCE)
        results["datasets"][label] = entry
        print(f"\n=== {label} ===")
        for split, stats in entry.items():
            print(f"  [{split}] n={stats['n_total']}  "
                  f"length-matched n={stats['n_length_matched']} "
                  f"({stats['pct_length_matched']}%)")
            print(f"    RM accuracy (all pairs):            {stats['rm_ranking_accuracy_all']}")
            print(f"    RM accuracy (length-matched only):  {stats['rm_ranking_accuracy_length_matched']}  "
                  f"(p={stats['length_matched_binomial_p_vs_chance']} vs chance)")
            print(f"    Length-only classifier (all pairs): {stats['length_only_classifier_accuracy_all']}")
            print(f"    Tolerance sweep: " + ", ".join(
                f"+-{t}: n={v['n']} acc={v['accuracy']}" for t, v in stats["tolerance_sweep"].items()
            ))
            print(f"    Hedge-opener pairs: {stats['n_hedge_opener']} ({stats['pct_hedge_opener']}%)")
            print(f"    RM accuracy (no hedge opener):       {stats['rm_ranking_accuracy_no_hedge_opener']}  "
                  f"(p={stats['no_hedge_binomial_p_vs_chance']} vs chance)")
            print(f"    RM accuracy (length-matched AND no hedge opener, n={stats['n_length_matched_and_no_hedge']}): "
                  f"{stats['rm_ranking_accuracy_length_matched_and_no_hedge']}")

    out_path = RESULTS_DIR / "sgb042_stage_a_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
