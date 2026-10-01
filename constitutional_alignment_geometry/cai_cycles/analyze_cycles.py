"""analyze_cycles.py — offline analysis of the Stage 2 margins. No GPU.

Reads the margin JSONL written by cai_cycles_stage2.py and applies the
PRE-REGISTERED test. Nothing here touches the model, so the test cannot be
retuned by re-running the expensive part.

Pre-registered (fixed 2026-09-30, before any real data):
  * Per-pair test: exact one-sided sign test on the stage-2 order-averaged
    margins. Target relation: a > b iff most paraphrases prefer a.
  * Block (prompt) p-value: the max of the three sign-test p-values of the
    triangle picked in stage 1, in its cyclic orientation. No Bonferroni —
    stage 1 selects, stage 2 tests, on disjoint paraphrases.
  * Dataset p-value: count of blocks with p <= 0.05 against Bin(B, 0.05),
    plus Fisher over the block p-values. alpha = 0.05.
  * Curl is reported DESCRIPTIVELY ONLY. It is not a transitivity test
    (cycle_harness.py: every curl null rejects transitive truths, up to 100%).

Derived graders (no extra GPU calls): any weighting of the principle graders,
including leave-one-out and the sampled-principle grader, is a weighted mean of
the stored per-principle margins.

Usage:
    python3 analyze_cycles.py /results/cai_cycles_stage2/*.jsonl
    python3 analyze_cycles.py --self-test
"""

from __future__ import annotations

import glob
import json
import math
import sys
from itertools import combinations


def sign_p(vals: list[float], flip: bool = False) -> float:
    """One-sided exact sign test: P(Bin(R,1/2) >= #positive). Zeros are ties
    and count against the claimed direction (conservative)."""
    v = [(-x if flip else x) for x in vals]
    R = len(v)
    k = sum(1 for x in v if x > 0)
    return sum(math.comb(R, x) for x in range(k, R + 1)) / 2.0 ** R


def binom_upper(B: int, a: float, s: int) -> float:
    return sum(math.comb(B, x) * a ** x * (1 - a) ** (B - x) for x in range(s, B + 1))


def chi2_sf_even(x: float, k: int) -> float:
    h, term = x / 2.0, math.exp(-x / 2.0)
    tot = term
    for i in range(1, k // 2):
        term *= h / i
        tot += term
    return min(1.0, tot)


def block_p(rec: dict) -> float:
    """Block p from the stage-2 margins of the stage-1-picked oriented triangle."""
    tri = rec["picked_triangle"]
    s2 = rec["stage2"]
    ps = []
    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        key = f"{min(a, b)}-{max(a, b)}"
        if key not in s2:
            return 1.0
        ps.append(sign_p(s2[key], flip=(a > b)))
    return max(ps)


def curl_share(rec: dict) -> float | None:
    """Descriptive only: fraction of stage-1 flow energy in the curl component.
    Full K_k with every triangle filled => harmonic is exactly 0, so
    1 - curl_share is the gradient share."""
    s1 = rec["stage1"]
    k = rec["k"]
    Y = {}
    for key, v in s1.items():
        i, j = (int(t) for t in key.split("-"))
        Y[(i, j)] = sum(v) / len(v)
    if len(Y) != k * (k - 1) // 2:
        return None
    pot = [0.0] * k
    for (i, j), y in Y.items():
        pot[i] += y
        pot[j] -= y
    pot = [p / k for p in pot]
    num = den = 0.0
    for (i, j), y in Y.items():
        g = pot[i] - pot[j]
        num += (y - g) ** 2
        den += y ** 2
    return (num / den) if den > 0 else None


def analyze(records: list[dict], alpha: float = 0.05) -> dict:
    ps = [block_p(r) for r in records]
    B = len(ps)
    s = sum(1 for p in ps if p <= alpha)
    X = -2.0 * sum(math.log(max(p, 1e-300)) for p in ps)
    cs = [c for c in (curl_share(r) for r in records) if c is not None]
    cert = [r["dataset_idx"] for r, p in zip(records, ps) if p <= alpha]
    return {
        "blocks": B,
        "certified_cyclic_blocks": s,
        "certified_fraction": round(s / B, 4) if B else None,
        "p_count": binom_upper(B, alpha, s) if B else None,
        "p_fisher": chi2_sf_even(X, 2 * B) if B else None,
        "mean_curl_share_descriptive": round(sum(cs) / len(cs), 4) if cs else None,
        "certified_dataset_idx": cert[:50],
    }


def mix(records_by_grader: dict[str, list[dict]], weights: dict[str, float]) -> list[dict]:
    """Build a DERIVED grader as a weighted mean of principle margins. Free."""
    names = [n for n in weights if n in records_by_grader]
    if not names:
        raise ValueError("no matching graders")
    base = {r["dataset_idx"]: r for r in records_by_grader[names[0]]}
    out = []
    for idx, r0 in base.items():
        acc = {"dataset_idx": idx, "k": r0["k"], "picked_triangle": r0["picked_triangle"],
               "stage1": {}, "stage2": {}}
        ok = True
        for stage in ("stage1", "stage2"):
            for key in r0[stage]:
                tot = None
                for n in names:
                    rr = next((x for x in records_by_grader[n] if x["dataset_idx"] == idx), None)
                    if rr is None or key not in rr[stage]:
                        ok = False
                        break
                    w = weights[n]
                    v = [w * x for x in rr[stage][key]]
                    tot = v if tot is None else [a + b for a, b in zip(tot, v)]
                if not ok:
                    break
                acc[stage][key] = tot
            if not ok:
                break
        if ok:
            out.append(acc)
    return out


def self_test() -> int:
    """Checks on synthetic records: a transitive block must not certify, a
    strong planted cycle must."""
    k, R = 5, 16
    tri = [0, 1, 2]

    def rec(margins, idx):
        s = {f"{min(i,j)}-{max(i,j)}": [margins[(min(i,j), max(i,j))]] * R
             for i, j in combinations(range(k), 2)}
        return {"dataset_idx": idx, "k": k, "picked_triangle": tri,
                "stage1": s, "stage2": s}

    pot = [2.0, 1.0, 0.0, -1.0, -2.0]
    trans = {e: pot[e[0]] - pot[e[1]] for e in combinations(range(k), 2)}
    cyc = dict(trans)
    cyc[(0, 1)], cyc[(1, 2)], cyc[(0, 2)] = 3.0, 3.0, -3.0   # 0>1>2>0

    a_t = analyze([rec(trans, i) for i in range(20)])
    a_c = analyze([rec(cyc, i) for i in range(20)])
    ok = (a_t["certified_cyclic_blocks"] == 0 and a_c["certified_cyclic_blocks"] == 20
          and a_c["p_count"] < 1e-6 and (a_t["mean_curl_share_descriptive"] or 0) < 1e-9
          and (a_c["mean_curl_share_descriptive"] or 0) > 0.05)
    print("transitive:", a_t)
    print("planted cycle:", a_c)
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    paths = [p for a in argv[1:] for p in glob.glob(a)]
    if not paths:
        print(__doc__)
        return 2
    by_grader: dict[str, list[dict]] = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                by_grader.setdefault(r["grader"], []).append(r)
    rows = {g: analyze(rs) for g, rs in sorted(by_grader.items())}
    print(f"{'grader':<34}{'B':>5}{'certified':>11}{'p(count)':>11}{'p(Fisher)':>11}{'curl':>8}")
    for g, a in rows.items():
        print(f"{g:<34}{a['blocks']:>5}{a['certified_cyclic_blocks']:>11}"
              f"{a['p_count']:>11.2e}{a['p_fisher']:>11.2e}"
              f"{(a['mean_curl_share_descriptive'] or 0):>8.3f}")
    prin = {g: rs for g, rs in by_grader.items() if g.startswith("principle:")}
    if len(prin) > 1:
        print("\nDerived graders (offline, no extra GPU calls):")
        eq = {g: 1.0 / len(prin) for g in prin}
        print(f"  {'equal-weight mixture':<32}{analyze(mix(by_grader, eq))}")
        for drop in prin:
            w = {g: 1.0 / (len(prin) - 1) for g in prin if g != drop}
            a = analyze(mix(by_grader, w))
            print(f"  {'drop ' + drop:<32}certified={a['certified_cyclic_blocks']} "
                  f"p={a['p_count']:.2e}")
    json.dump(rows, open("cycle_analysis.json", "w"), indent=1)
    print("\nwrote cycle_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
