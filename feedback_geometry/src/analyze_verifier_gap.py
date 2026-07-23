"""Analysis for the verifier-gap replication (SGB-033).

Tests the inverted-U claim from SGB-032 per seed rather than on the aggregate
curve, because the aggregate is a MIXTURE: at a given oracle_fraction some
seeds rise monotonically to 1.0 while others peak and fall, so the mean curve's
shape depends on the mix and can misrepresent both populations.

Pass criterion (from the queue item): per-seed peak budget N* > 1 for the
crossover oracle fractions, and hack_rate(N*) > hack_rate(N_max) on a paired
test across seeds.

Also tests the proposed mechanism: curve shape should be predicted by whether
the verifier's GLOBAL argmax lies inside the trap. If it does, saturating
search converges on it and hacking goes to 1.0; if it does not, saturating
search escapes the trap's local optimum and hacking collapses.

Run:
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 feedback_geometry/src/analyze_verifier_gap.py \
        feedback_geometry/results/verifier_gap/replication_invertedU_50seed.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


def per_seed_curves(rows: List[Dict], budgets: List[int]) -> Dict[float, List[Dict]]:
    by_orc: Dict[float, List[Dict]] = defaultdict(list)
    for r in rows:
        curve = np.array([r["search"][str(b)] for b in budgets], dtype=float)
        peak_i = int(np.argmax(curve))
        by_orc[r["verifier"]["oracle_fraction"]].append({
            "seed": r["seed"],
            "curve": curve,
            "peak_budget": budgets[peak_i],
            "peak_value": float(curve[peak_i]),
            "final_value": float(curve[-1]),
            "base_value": float(curve[0]),
            "argmax_in_trap": r["competence"].get("argmax_in_trap"),
            "trap_vs_safe_acc": r["competence"]["trap_vs_safe_acc"],
        })
    return by_orc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--trials", type=int, default=800,
                    help="search trials per cell, for binomial noise bounds")
    ap.add_argument("--ref-budget", type=int, default=16,
                    help="pre-specified budget for the non-selected paired test")
    args = ap.parse_args()
    trials, ref_budget = args.trials, args.ref_budget

    blob = json.loads(Path(args.path).read_text())
    budgets: List[int] = blob["budgets"]
    by_orc = per_seed_curves(blob["rows"], budgets)

    print(f"budgets: {budgets}   trials/cell: {trials}   ref budget: N={ref_budget}")
    print(f"\n{'orc':>6} {'n':>4} {'interior_peak':>14} {'noise_turn':>12} "
          f"{'median_N*':>10} {'peak':>8} {'final':>8} {'dz':>7} "
          f"{'p(N=%d vs max)' % ref_budget:>14}")
    print("-" * 96)

    verdicts = {}
    for orc in sorted(by_orc):
        rs = by_orc[orc]
        peaks = np.array([r["peak_value"] for r in rs])
        finals = np.array([r["final_value"] for r in rs])
        nstars = np.array([r["peak_budget"] for r in rs], dtype=float)

        # Interior peak = the curve genuinely TURNS OVER: the maximum sits at a
        # budget above the smallest and strictly exceeds BOTH endpoints by a
        # margin. Testing index position alone is wrong -- np.argmax breaks ties
        # by first index, so a saturated curve (..., 1.000, 1.000) reports a
        # spurious interior peak when it is really monotone-increasing.
        tol = 0.02
        interior = np.array([
            (r["peak_budget"] > budgets[0])
            and (r["peak_value"] > r["final_value"] + tol)
            and (r["peak_value"] > r["base_value"] + tol)
            for r in rs
        ])
        frac_interior = float(interior.mean())

        # NOTE: do NOT test peak vs final with a one-sided Wilcoxon. `peak` is
        # max(curve), so peak - final >= 0 by construction for every seed; a
        # one-sided test on a quantity that can never be negative cannot reject
        # anything and returns spuriously tiny p-values. Two non-circular tests
        # are used instead.
        #
        # (a) NOISE-AWARE TURNOVER: each hack rate is a binomial proportion over
        #     `trials` draws. Count a seed as turning over only if some earlier
        #     budget beats the final budget by more than 3 combined standard
        #     errors -- i.e. beyond sampling noise.
        curves = np.vstack([r["curve"] for r in rs])
        fin = curves[:, -1]
        p_hat = np.clip(curves, 0.0, 1.0)
        se = np.sqrt(p_hat * (1.0 - p_hat) / trials)
        #     The qualifying budget must also beat the FIRST budget by the same
        #     margin -- otherwise a monotonically DECLINING curve (peak = the
        #     N=1 baseline, then falling to zero) counts as a turnover, which
        #     inverts the meaning. That is the orc>=0.75 case.
        base = curves[:, 0]
        thresh_f = 3.0 * np.sqrt(se[:, :-1] ** 2 + se[:, [-1]] ** 2)
        thresh_b = 3.0 * np.sqrt(se[:, :-1] ** 2 + se[:, [0]] ** 2)
        noise_turn = (
            ((curves[:, :-1] - fin[:, None]) > thresh_f)
            & ((curves[:, :-1] - base[:, None]) > thresh_b)
        ).any(axis=1)
        frac_noise = float(noise_turn.mean())

        # (b) FIXED-BUDGET PAIRED TEST: compare a pre-specified mid budget to the
        #     largest budget. No argmax selection, so this is a legitimate test.
        if ref_budget in budgets:
            a = curves[:, budgets.index(ref_budget)]
            b = curves[:, -1]
            p_fixed = float(stats.ttest_rel(a, b).pvalue)
            dz = float((a - b).mean() / ((a - b).std(ddof=1) + 1e-12))
        else:
            p_fixed, dz = float("nan"), float("nan")

        verdicts[orc] = (frac_noise, p_fixed, dz)
        print(f"{orc:>6.3f} {len(rs):>4} {frac_interior:>13.1%} {frac_noise:>12.1%} "
              f"{np.median(nstars):>10.0f} {peaks.mean():>8.3f} {finals.mean():>8.3f} "
              f"{dz:>7.2f} {p_fixed:>14.2e}")

    # ---- mechanism ----
    have_argmax = [r for rs in by_orc.values() for r in rs if r["argmax_in_trap"] is not None]
    print("\n=== MECHANISM: does the verifier's global argmax predict curve shape? ===")
    if not have_argmax:
        print("  argmax_in_trap not recorded in this run "
              "(added after it was launched) -- rerun to populate.")
    else:
        inn = np.array([r["final_value"] for r in have_argmax if r["argmax_in_trap"]])
        out = np.array([r["final_value"] for r in have_argmax if not r["argmax_in_trap"]])
        print(f"  argmax INSIDE trap  (n={len(inn)}): final hack rate {inn.mean():.3f}")
        print(f"  argmax OUTSIDE trap (n={len(out)}): final hack rate {out.mean():.3f}")
        if len(inn) > 1 and len(out) > 1:
            p = float(stats.mannwhitneyu(inn, out, alternative="greater").pvalue)
            print(f"  Mann-Whitney p = {p:.2e}")

    print("\n=== VERDICT (noise-aware turnover >=50% AND non-selected paired test p<0.05) ===")
    for orc, (frac, p, dz) in sorted(verdicts.items()):
        ok = frac >= 0.5 and (p == p) and p < 0.05 and dz > 0
        print(f"  orc={orc:.3f}: inverted-U {'CONFIRMED' if ok else 'not confirmed'} "
              f"({frac:.0%} turn over beyond noise; N={ref_budget} vs max "
              f"dz={dz:+.2f}, p={p:.2e})")


if __name__ == "__main__":
    main()
