# Proposal: Hodge-Targeted Data Acquisition (SGB-045 draft)

**Status:** proposal, 2026-09-19. Nothing here has been run.
**Motivation:** HodgePO (a new loss) gave no held-out benefit (`shared/results/README.md`,
file 5), and Hodge-PPO equals PPO at 1.5B under the standard RM
(`shared/results/finetune/eval_rescore_sgb044.json`). A learning method cannot remove
contradictions that exist in its training data. This proposal moves the Hodge machinery
from the **loss** to the **dataset**: find where the preference data is inconsistent, and
add the data that resolves or localizes the inconsistency.

## 1. What a "hole" is, precisely

Build a simplicial complex from the preference data:
- 0-cells: responses (items).
- 1-cells: compared pairs, with flow `f(i,j)` = preference margin.
- 2-cells: triangles `(i,j,k)` where all three pairs are compared.

Hodge decomposition splits the flow into three orthogonal parts:
`f = grad(φ) + curl-part + harmonic-part`.
- **Gradient:** the part a single score `φ` explains.
- **Curl:** inconsistency inside filled triangles. It is local: we know the exact triple.
- **Harmonic:** inconsistency around loops that no triangle fills. These loops are the
  topological holes (first homology). The data shows that a loop is inconsistent but does
  not show which comparison causes it.

**Current code limit:** `hodge_diagnostic.py` uses no 2-cells. Its "harmonic" is the full
residual `f − d0φ` (curl and harmonic together). Step 0 below adds 2-cells.

## 2. What new data can and cannot do

New data acts on each part differently:

| Part | Cause | Effect of new data |
|---|---|---|
| Harmonic (hole) | Missing comparisons | A comparison that fills a triangle across the hole turns harmonic mass into gradient or curl. The inconsistency becomes local and attributable. |
| Curl from label noise | Annotator error | Repeat labels on the high-curl edges. The curl goes down. |
| Curl from real intransitivity | Different criteria (for example helpful vs harmless) | Data cannot remove the curl. Data can show which criteria split the triple. The correct response is a multi-criteria reward, not a cleaner scalar. |

So "resolve as much tension as possible" has two honest meanings:
1. Remove tension that comes from missing data or noise.
2. Localize and label tension that is real, so that a model can represent it instead of
   averaging it away.

This matches the scope rule in project memory: Hodge separates resolvable from irreducible
inconsistency, and attribution needs extra structure. Targeted data **is** that extra structure.

## 3. Prior art (verified 2026-09-19)

- Osting, Brune, Osher, "Enhanced statistical rankings via targeted data collection",
  ICML 2013 (PMLR v28). Chooses new comparisons to maximize the Fisher information of the
  HodgeRank estimate (graph algebraic connectivity).
- Xu, Xiong, Chen, Huang, Yao, "HodgeRank with Information Maximization for Crowdsourced
  Pairwise Ranking Aggregation", AAAI 2018 (arXiv 1711.05957). Unsupervised sampling by
  algebraic connectivity, and supervised Bayesian information-gain sampling.

**Our difference must be real, not a relabel:**
1. The target is the curl and harmonic residual, not the variance of the ranking.
2. We can **generate new items** (responses written to split a loop by criterion), not only
   choose among existing pairs.
3. The evaluation is downstream: held-out RM accuracy and policy exploit resistance.

The two papers above are the **required baselines**. Beating random acquisition is not enough.

## 4. Precondition: a graph with real loops

The HodgePO graph had no real loops. HH-RLHF gives one pair per prompt, and the cycles came
from similarity edges we built (`shared/results/README.md`, caveat 6). Ratings on a scale
(for example a 1–10 score per response) give only transitive comparisons and also have no
loops. Real loops need **pairwise** judgments of several responses to the same prompt, from
several annotators or criteria.

Candidate sources:
- A judge panel: for each prompt, generate k = 4–6 responses (base, SFT, PPO policies),
  then judge all pairs with 3 judges that use different rubrics (helpful, harmless,
  honest). Loops then come from real disagreement between criteria.
- A public pairwise dataset with repeated items (model-level arena comparisons). Check
  the license and the structure before use.

## 5. Staged plan with kill criteria

**Stage 0 — CPU, about 1 day.** Add 2-cells to the decomposition. Unit test on known flows:
a pure 3-cycle must be all curl when the triangle is filled, and all harmonic when it is not.

**Stage 1 — CPU simulation with ground truth.** Items have 2–3 latent criteria. Annotators
weight the criteria differently and add label noise. The simulation therefore knows which
loops come from noise and which come from real intransitivity. Compare these acquisition
policies at equal budget:
- random
- algebraic connectivity (Osting 2013 / Xu 2018)
- ours: harmonic-hole filling plus curl re-labelling

Metrics: recovery of the true noise and intransitive edges, and held-out pairwise accuracy.
**Kill criterion:** if ours does not beat the algebraic-connectivity baseline, stop.

**Stage 2 — real data, small.** Build the judge-panel graph for about 200 prompts. Measure
the gradient, curl, and harmonic shares. **Kill criterion:** if curl + harmonic is below
about 5% of the flow norm, the data has too little tension to target. Report that as the result.

**Stage 3 — targeted generation.** For each high-tension loop, generate new responses that
differ on only one criterion, and judge the new responses. Train an RM on (a) the base
data, (b) plus targeted data, and (c) plus the **same count** of random extra data. Test on
held-out prompts.

**Stage 4 — policy (GPU).** Run this stage only if (b) beats (c) on held-out data. Then run
PPO against each RM and grade the policies with an independent judge, not with an RM
from the training family.

## 6. Controls carried over from the HodgePO audit

- Held-out evaluation only.
- An equal-count random-augmentation control (the analogue of the margin control).
- Assert index alignment between items, targets, and labels. Do not assume it.
- Never score a policy with the RM it trained against.
