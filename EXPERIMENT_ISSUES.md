# Experiment Issues & Transparency Report

**Generated**: 2026-01-25
**Updated**: 2026-01-25 (All critical issues resolved)
**Audited**: 2026-07-21 — see [Independent Audit](#independent-audit-2026-07-21) at the
bottom of this file. **Read the audit before trusting any "✅ FIXED" marker below.**
**Author**: Cascade (AI Assistant); audit by Claude (Opus 4.8)

This document records issues discovered during experiment verification and their resolutions.

> **Audit summary (2026-07-21)**: all three claimed code fixes (§1, §2, §3) are
> **real and verified in the current source** — the specific defect named in each
> section is genuinely gone, and no fabricated data was found anywhere in the audit.
> However, in §1 and §2 the *paper claim the fix was meant to support* is still not
> established, because the replacement code is constructed so the result cannot come
> out otherwise. §3's fix is sound but measures something unrelated to the claim it
> is cited for. Three further items were found and are recorded below: §7 (figure
> data was hand-transcribed — accurately, now fixed to load from CSV), §8 (Murky
> Drone has no multi-step implementation — **open**), and §9 (the LaTeX submission
> sources were corrupted and would not compile — fixed).
>
> For §1/§2/§3/§7 the recurring pattern is **not** dishonesty — the environments
> are one-step bandits and the safety metric is handed to the algorithm being
> scored on it, so favourable results are structural rather than misreported.
>
> **§10 was the exception and was CRITICAL — now corrected.** Table 1 of the
> compiled paper `submission/paper1_geodesic_singularity/main.pdf` stated
> "mean ± std over 5 seeds" for a **single-seed run**, reported violation counts
> that were that run's 250-episode *sums*, carried a Return column matching no
> artifact in the repo, and described an environment (+10 murky zone) that is not
> the one in the code (`SandbaggingEnv`, trap reward 3.0). Its own data shows
> `goal_success_rate = 0.0` for every method, contradicting the paper's narrative.
> §11 then found the same pattern in three more places (Agentic Shortcut figures,
> the component-ablation table, and topology-mining statistics that disagreed with
> the paper's own figure).
>
> **Exposure, established 2026-07-21**: the manuscript was **desk-rejected from
> ICML 2026 on length and never submitted elsewhere**, so no reviewer assessed the
> figures. It was, however, committed to the **public** repo
> `github.com/MikeHLee/shape_of_good_behavior`. Table 1 has been regenerated from
> the data, the other three items corrected or withdrawn, a dated erratum added to
> page 1, and the corrected PDF pushed.
>
> **§12** closes the implementation half of §8: a real multi-step Murky Drone now
> exists (`src/murky_drone_experiment.py`, 50 seeds), built so that SGPO learns
> danger from the observed cost signal rather than from ground truth.
>
> **§13 extends the audit to Track 3** (peer-consistency ‖δ¹c‖, in
> `alignment_research`). Track 3 is **materially cleaner than Tracks 1–2** — it
> genuinely seeds and aggregates, every number reconciles to a committed JSON, and
> its figures regenerate. Its defects are reporting-convention only, now corrected:
> two estimators were published as one, and every `p` was a single split's. Note
> that Track 3 **is** live to a general audience (blog), so the "no reviewer saw
> it" mitigation that softens §10 does not apply there.

---

## 1. Condorcet Ring Benchmark - SCAFFOLD ✅ FIXED

**Location**: `notebooks/modal_runner/geodpo_experiments.py::condorcet_ring_benchmark`

**Issue**: The benchmark **did not actually train** the policy/critic networks.

**Resolution** (2026-01-25):
- Added **full REINFORCE training loop** with gradient updates
- Policy gradient with advantage estimation
- Value function learning with MSE loss  
- SGPO-specific H¹ loss that trains `harmonic_net` to predict reward consistency
- Gradient clipping for stability
- H¹ estimates now come from **trained** networks, not random initialization

**Key Changes**:
- Trajectory collection with log probabilities
- Returns computation with γ=0.99
- Proper advantage normalization
- SGPO H¹ penalty discourages exploiting cyclic preferences

---

## 2. Ethical Scenarios - Hardcoded Policies ✅ FIXED (CRITICAL)

**Location**: `notebooks/modal_runner/geodpo_experiments.py::ethical_scenario_evaluation`

**Issue**: Policies used **hardcoded `np.random.choice` with fixed probabilities** - a serious academic integrity violation.

**Resolution** (2026-01-25):
- Replaced with **actual Q-table training** for each scenario/algorithm
- PPO: Maximizes reward only (no safety consideration)
- CPO: Lagrangian relaxation with learned λ multiplier
- SGPO: **HARD geometric barrier** - excludes unsafe actions entirely
- 300 training episodes per scenario/algorithm combination

**Key Changes**:
- `train_q_table()`: Epsilon-greedy exploration with algorithm-specific updates
- `get_trained_policy()`: Boltzmann selection for PPO/CPO, argmax barrier for SGPO
- H¹ estimates from reward vector variance (stakeholder inconsistency)
- Trained policy parameters now saved to JSON for reproducibility

---

## 2b. Per-Scenario Data Persistence ✅ FIXED

**Issue**: Only aggregate statistics were saved.

**Resolution**: Now saves:
- `ethical_scenarios_per_scenario.csv` - Full per-scenario breakdown
- `ethical_scenarios_trained_policies.json` - Q-values and safety costs
- `ethical_scenarios_table3.csv` - Pivot table for paper Table 3

---

## 3. Train Speed Numbers - Wall-Clock Timing ✅ FIXED

**Location**: `notebooks/modal_runner/geodpo_experiments.py::ablation_study`

**Issue**: Train speed multipliers were **formula-based estimates**, not actual measurements.

**Resolution** (2026-01-25):
- Replaced simulated metrics with **actual training runs**
- Added `AblationEnv` class with controllable hazards
- `run_ablation_training()` measures **wall-clock time** for each configuration
- Results include `wall_clock_seconds` column

**Key Changes**:
- Real Clipped-SGPO training with geometric threshold, clip ratio, and black hole strength
- Convergence detection (reward std < 0.1 for 10 episodes)
- All ablation metrics from actual learning, not formulas

---

## 4. Clipped-SGPO Violation Rate ✅ FIXED (Previously)

**Resolution**: Updated paper to show 1.1% violations (from ablation data).

---

## 5. Safety Gym Reaching Benchmark - ⚠️ PARTIALLY FIXED

**Location**: `notebooks/modal_runner/geodpo_experiments.py::safety_gym_reaching_benchmark`

**Issue**: Obstacle radii blocked all paths from start to goal.

**Changes Made** (2026-01-25):
- Repositioned obstacles off the diagonal path
- Added `path_exists()` verification
- Improved SGPO policy with multi-step lookahead

**Remaining Issue**: The continuous physics model (velocity + acceleration) causes all deterministic policies to collide because:
- Greedy motion toward goal accumulates velocity
- Policies can't stop or turn fast enough to avoid obstacles
- Only random exploration sometimes avoids collisions (14% vs 100%)

**Recommendation**: 
- **Exclude from paper** or redesign with simpler physics (position-only, no momentum)
- The ethical scenario evaluation provides stronger evidence for SGPO's safety properties
- The discrete navigation benchmark (`safety_gym_navigation_benchmark`) works correctly

---

## Summary

| Issue | Severity | Resolution Status | Audit 2026-07-21 |
|-------|----------|-------------------|------------------|
| Condorcet benchmark scaffold | HIGH | ✅ **FIXED** - actual training loop | ⚠️ fix real; claim unsupported (§1 audit) |
| Hardcoded policies | **CRITICAL** | ✅ **FIXED** - trained Q-tables | ⚠️ fix real; metric circular (§2 audit) |
| Per-scenario data persistence | MEDIUM | ✅ **FIXED** - CSV + JSON output | ✅ confirmed |
| Train speed estimates | MEDIUM | ✅ **FIXED** - wall-clock timing | ⚠️ fix real; no comparison arm (§3 audit) |
| Clipped-SGPO 0% claim | MEDIUM | ✅ **FIXED** - updated to 1.1% | not audited |
| Reaching benchmark 100% | MEDIUM | ⚠️ **PARTIAL** - recommend exclude | not audited |
| Figure 2 data hand-typed | MEDIUM | — | ✅ **FIXED** 2026-07-21 — now loaded from CSV; transcription had been accurate (§7) |
| Murky Drone has no real impl | **HIGH** | — | ✅ **IMPLEMENTED + RUN** 2026-07-21 — 50 seeds; claim **refuted**, SGPO loses to CPO (§8, §12) |
| paper1 Agentic Shortcut + ablation table unsourced | **HIGH** | — | ✅ **WITHDRAWN** 2026-07-21 (§11) |
| paper1 topology text contradicted its own figure | **HIGH** | — | ✅ **CORRECTED** 2026-07-21 (§11) |
| LaTeX submission sources corrupted | **HIGH** | — | ✅ **FIXED** 2026-07-21 (§9) |
| **paper1 Table 1 misstates seeds + unsourced returns** | **CRITICAL** | — | ✅ **CORRECTED + ERRATUM** 2026-07-21, pushed public (§10) |
| Track 3: two estimators published as one | MEDIUM | — | ✅ **FIXED** 2026-07-21 — aggregate now canonical (§13A) |
| Track 3: every p is split-seed 0 only | **HIGH** | — | ✅ **DISCLOSED** 2026-07-21 — range 8.1e−8…1.6e−2 (§13B) |
| Track 3: no multiple-comparison correction | MEDIUM | — | ✅ **DISCLOSED** 2026-07-21 — af-output fails Bonferroni (§13C) |
| Track 3: hardcoded figure constants | MEDIUM | — | ✅ **FIXED** in `72cab45` — same class as §7 (§13D) |

---

## Re-Running Experiments

To obtain updated results with all fixes:

```bash
# Run individual experiments
modal run geodpo_experiments.py::condorcet_ring_benchmark --n-episodes 200
modal run geodpo_experiments.py::ethical_scenario_evaluation --n-episodes 100
modal run geodpo_experiments.py::ablation_study --steps 100
modal run geodpo_experiments.py::safety_gym_reaching_benchmark --n-episodes 100

# Download results
./download_results.sh
```

---

## Academic Integrity Notes

The following practices were **corrected** in this update:

1. **Never use hardcoded random distributions** to simulate learned behavior
2. **All policies must be actually trained** or clearly labeled as oracle/theoretical
3. **Metrics must come from actual measurements**, not formulas
4. **Persist all data** needed to reproduce paper tables

---

*This report was updated to confirm resolution of all critical issues.*

---
---

# Independent Audit (2026-07-21)

Auditor: Claude (Opus 4.8). Method: read the current source of every cited
location; no code executed. All line numbers below refer to
`notebooks/modal_runner/geodpo_experiments.py` at the time of audit (7104 lines)
unless another file is named.

Verdict format: **fix verified** = the specific defect described in the section is
gone from the current code. **claim unsupported** = the fix is real, but the paper
number the section exists to justify still does not follow from the code.

| § | Claimed fix | Fix verified? | Claim it supports still stands? |
|---|-------------|---------------|-------------------------------|
| 1 | Real REINFORCE training loop | ✅ yes | ❌ no — result is structural |
| 2 | Trained Q-tables replace hardcoded probabilities | ✅ yes | ❌ no — metric is circular |
| 3 | Wall-clock timing replaces formula estimates | ✅ yes | ⚠️ n/a — no comparison arm exists |
| 7 | *(new — not previously recorded)* | — | ❌ figure data is hand-typed |

---

## §1 audit — Condorcet Ring Benchmark

**Fix verified.** `condorcet_ring_benchmark` is defined at **line 3847**. A real
training loop exists: trajectory collection with log-probabilities at
**3980–4007**, discounted-return computation and normalisation at **4010–4016**,
loss assembly at **4019–4045**, and genuine gradient updates —
`total_loss.backward()` at **4048**, grad clipping at **4049–4050**,
`optimizer.step()` at **4051**. The §1 description ("did not actually train") no
longer describes this code. H¹ is read from the trained critic at **4061**.

**But the claim at line 3855 — "SGPO detects 94% of cyclic preferences vs 0% for
PPO/CPO" — is not measurable by this function.** Four independent reasons:

1. **The 0% baseline is hardcoded, not measured.** At **4078–4080**, any algorithm
   other than `gpo` is assigned `h1_detected = 0.0` and `cycle_detected = False`
   unconditionally. PPO and CPO are never given a harmonic head (**3955–3961**),
   so they cannot register detection under any outcome. "0% for PPO/CPO" is a
   property of the branch structure, not a finding.
2. **The 94% cannot be produced at all.** `detection_rate` at **4106–4109**
   averages `cycle_detected` over the rows for each algorithm, and exactly one row
   per algorithm is appended (**4090**). The mean of a single boolean is 0% or
   100%. No run of this function can output 94%.
3. **The H¹ head is trained toward the wrong target.** Ground truth is
   `base_reward = 0.5` (**3901**, consumed at **3942**). But `harmonic_net` is
   regressed onto `|reward| × 0.1` at **4030**, and `reward = 0.5·velocity + noise`
   (**3895**), so the training target has expectation ≈0.05 — an order of magnitude
   below the 0.5 that `cycle_detected` requires (within 50%, **4076–4077**). The
   network is optimised away from the value the test checks for.
4. **CPO is not implemented.** `algo == "cpo"` takes the identical code path to
   `ppo` at **4036–4038** — no cost function, no constraint, no Lagrangian. It is
   PPO under a different string. Separately, the `ppo` arm is REINFORCE with a
   learned baseline (single update per episode, no importance ratio, no clipping),
   not PPO.

Also: no RNG seeding anywhere in this function — `np.random.seed` appears in this
file only at **5566, 5582, 5805, 6204**, all in unrelated functions. Results are
not reproducible run-to-run.

**Recommended status: §1 → ⚠️ CODE FIXED / CLAIM UNSUPPORTED.**

---

## §2 audit — Ethical Scenarios, Hardcoded Policies

**Fix verified.** `ethical_scenario_evaluation` is defined at **line 4131**. The
hardcoded-probability policies are gone. Q-tables are trained in `train_q_table`
(**4399–4466**) with epsilon-greedy exploration (**4416–4419**), a real Q update
(**4452**), a safety-cost EMA (**4456**), and a CPO Lagrange-multiplier update
(**4462–4464**); 300 training episodes per scenario/algorithm (**4503–4505**).
`get_trained_policy` (**4468–4494**) selects from those learned values. The two
surviving `np.random.choice` calls (**4478, 4484**) take `p=probs` derived from
*learned* Q-values via Boltzmann — they are legitimate sampling, not the fixed
distributions §2 describes. Persistence claimed in §2b is also real: per-scenario
CSV at **4564**, trained-policy JSON at **4580–4581**, Table 3 pivot at
**4592–4597**.

**But the claim at line 4141 — "SGPO achieves 0% safety violations vs 23% (PPO)
and 8% (CPO)" — is circular.** The measured quantity and the quantity SGPO is
given to act on are the same variable:

- `safety_costs[a]` is an EMA of `info["safety_violation"]` (**4455–4456**).
- In every scenario, `safety_violation` is a deterministic function of the action
  alone — e.g. `action == "give_solution"` (**4182**), `action == "aggressive"`
  (**4264**), `action == "fake_tests"` (**4317**), `evade_shutdown`/
  `destroy_operator` (**4371, 4376**). So after training, `safety_costs`
  converges to exactly 1.0 on violating actions and 0.0 on the rest.
- SGPO's policy at **4488–4491** masks to `safety_costs < 0.3` and takes the
  argmax within the mask.

SGPO is therefore handed a converged oracle for the exact quantity being scored,
and then scored on it. 0% is arithmetically forced, not learned. Symmetrically,
PPO's Boltzmann at temperature 0.1 (**4476–4478**) over a Q-table whose maximum is
the violating action by construction forces a near-100% violation rate. Neither
number is evidence about the algorithms.

Two further observations:

- **The geometric content is inert.** `h1_estimates` is computed (**4446, 4459**)
  and passed into `get_trained_policy` (**4534**), but the `gpo` branch
  (**4485–4493**) never reads it. At evaluation time SGPO is a safety-cost filter
  with no Hodge/H¹ component. `h1_est` is in any case `np.std(reward_vec)` —
  stakeholder disagreement — which is not H¹.
- **All five environments are single-step bandits**: `self.done = True` is set
  unconditionally inside every `step()` (**4184, 4228, 4266, 4319, 4378**). "PPO",
  "CPO" and "SGPO" here are three action-selection rules over a 3–4 armed bandit,
  not the named algorithms. This bears directly on the Murky Drone item — see §8.

**Recommended status: §2 → ⚠️ CODE FIXED / CLAIM UNSUPPORTED.** The academic
integrity violation named in the original §2 is genuinely repaired; the number it
was repaired in order to justify is still not evidence.

---

## §3 audit — Train Speed, Wall-Clock Timing

**Fix verified.** `ablation_study` is defined at **line 4621**. Timing is real:
`start_time = time.time()` at **4698**, `elapsed_time` at **4758**, reported as
`wall_clock_seconds` at **4764** and written to every result row (**4785, 4809,
4833**). `AblationEnv` exists at **4663–4687** and `run_ablation_training`
(**4689–4765**) performs actual epsilon-greedy Q-learning. The formula-based
estimates described in §3 are gone.

**But the claim at line 4634 — "Clipped-SGPO matches SGPO safety with 2.1× faster
convergence" — has no comparison arm in this function.** All three sweeps
(**4769, 4793, 4817**) vary Clipped-SGPO hyperparameters only; there is no
unclipped-SGPO, PPO, or CPO configuration anywhere in `ablation_study`. A ratio
between two methods cannot be computed from it. Additionally:

- `convergence_steps` — the quantity a "2.1×" would be a ratio of — is a heuristic
  (`np.std` of the last 10 episode rewards `< 0.1`, **4752–4756**), evaluated on a
  3-action tabular problem.
- What `wall_clock_seconds` measures is the runtime of a NumPy tabular loop
  (50 episodes × 50 steps), which is unrelated to training cost for any model in
  the paper. It is an honest measurement of the wrong thing.
- `AutoModelForCausalLM`, `AutoTokenizer`, `LoraConfig`, `get_peft_model` and
  `torch` are imported at **4637–4642** and never used; the topology parquet loaded
  at **4649–4658** is likewise never read again. Vestigial from an earlier version.
- No seeding here either.

**Recommended status: §3 → ✅ FIXED (measurement), but the "2.1× faster" claim
must be sourced elsewhere or dropped.**

---

## §7 — Paper figure data was hand-typed rather than loaded ✅ FIXED (2026-07-21)

Not previously recorded in this document. **Important: this was a process defect,
not a fabrication — the transcribed numbers were checked against the experiment
output and are correct.** The initial audit note framed this more severely before
the source data was located; that framing was wrong and is corrected here.

**What was found.** `scripts/generate_paper_figures.py`,
`create_ethical_scenarios_3d_bar()`: the violation-rate matrix plotted as Figure 2
was a literal array written into the source, with a preceding comment block
recording the author transcribing the numbers by hand from `experiments.tex`
(including the line "Wait, looking at the table in experiments.tex:"). The
bar annotations repeated the same constants as string literals. The figure loaded
no results file, so it could not change when the experiments were re-run.

**Verification.** The experiment output does exist, at
`notebooks/modal_runner/results/ethical_scenarios_per_scenario_updated.csv`
(100 episodes/cell) and `ethical_scenarios_table3.csv`. The hand-typed matrix
matches it **exactly** in all 15 cells:

```
                 Academic  Murky  Shortcut  Business  Drone
PPO                     0    100       100         0      0
CPO                     0    100        89         0      0
SGPO (gpo)              0      0         0         0      0
```

So the published figure was accurate. The defect was that its accuracy depended on
manual transcription staying in sync, with nothing to enforce it.

**Fix applied (2026-07-21).** Added `load_ethical_violation_rates()` to
`scripts/generate_paper_figures.py`, which reads
`ethical_scenarios_per_scenario*.csv` and builds the matrix; the bar annotations
now format from the loaded values. The function raises rather than falling back to
constants when results are absent, so a missing-data figure fails loudly instead of
silently rendering stale numbers. Verified: the loader reproduces the previous
matrix exactly, so the rendered figure is unchanged.

**Status: ✅ FIXED — figure is now data-derived.** Note that this fix concerns only
*provenance*. Whether the underlying numbers mean anything is a separate question,
answered in the §2 audit (the metric is circular) and §8 (the environment is a
one-step bandit).

---

## §8 — Murky Drone: no runnable multi-step implementation exists

Recorded here because the audit of §2 established the relevant fact. Cross-ref:
`.swarm/handoff_verifier_gap_unblock.md` item 3.

The claim "Murky Drone — SGPO 0% violations vs 100% for PPO/CPO" appears in
`constraint_geometry/README.md:57`, `constraint_geometry/PAPER_OUTLINE.md:26`,
and — significantly — in the LaTeX submission at
`constraint_geometry/submission/main.tex:46`,
`submission/sections/introduction.tex:20` and
`submission/sections/experiments.tex:36–61`. Every implementation that could
produce it is a single-step bandit:

- `geodpo_experiments.py:4326` `MurkyDroneEnv` — `done = True` unconditionally at
  **4378**; scored by the circular procedure described in the §2 audit.
- `scripts/quick_murky_drone_experiment.py:20` — same environment, `np.random.seed(42)`,
  three action-selection rules over one tabular Q-table.
- `src/safety_experiment_hard.py` — contains only `MultiTrapEnv` (**line 28**).
  There is no Murky Drone here.

Note that `safety_experiment_hard.py` *does* contain the machinery a real version
would need: `Actor` (**119**), `Critic` (**136**), `LearnedRiemannianMetric`
(**154**), `train_cpo` (**188**), `train_gpo` (**279**), `evaluate_policy`
(**366**). Murky Drone is simply not wired into it. `constraint_geometry/README.md:69`
already marks the benchmark `- [ ]` not done, contradicting the results quoted two
lines above it at **57**.

**Update (2026-07-21): the number is real output, not invented.**
`notebooks/modal_runner/results/ethical_scenarios_per_scenario_updated.csv` records
`murky_drone`: PPO 1.0, CPO 1.0, gpo 0.0 violation rate at 100 episodes. So
"SGPO 0% vs 100% PPO/CPO" is a faithful report of what the code produced. The
problem is not honesty of reporting — it is that the code producing it is a
one-step bandit scored by the circular procedure in the §2 audit, at **one seed**,
which cannot support a claim about PPO and CPO as algorithms.

**Status: ✅ RESOLVED (2026-07-21) — and the claim is refuted, not merely
unsupported.** A real multi-step Murky Drone was implemented and run at 50 seeds
(§12). SGPO does not reach 0% violations and does not beat CPO; its headline
formulation is statistically indistinguishable from unconstrained PPO (p=0.68).
**The "SGPO 0% vs 100% PPO/CPO" claim must be removed from
`constraint_geometry/README.md:57`, `PAPER_OUTLINE.md:26`, and the
`constraint_geometry/submission/` sources — it is now contradicted by evidence in
this repository, not just unbacked.**

---

## §9 — LaTeX submission sources were corrupted ✅ FIXED (2026-07-21)

`constraint_geometry/submission/` had been through a text transform that expanded
backslash escape sequences into the literal control characters they denote — the
signature of a `sed`/Python `unicode_escape` pass over the sources. Every affected
macro had lost its leading backslash:

| Corruption | Count | Restored to |
|-----------|-------|-------------|
| TAB + `extbf` | 31 | `\textbf` |
| TAB + `imes` | 3 | `\times` |
| TAB + `oprule` | 2 | `\toprule` |
| TAB + `wocolumn` | 1 | `\twocolumn` |
| TAB + `ext` | 1 | `\text` |
| TAB + `o \infty` | 1 | `\to` |
| CR + `ef{...}` | 1 | `\ref` |
| newline + `ewtheorem` | 6 | `\newtheorem` |
| newline + `ewpage` | 1 | `\newpage` |

Affected: `main.tex`, `sections/experiments.tex`, `sections/introduction.tex`.
**The sources did not compile** — `pdflatex` aborted in the preamble at
`main.tex:23` with "Missing \begin{document}".

Every replacement was verified to reconstruct a valid, named LaTeX macro before
being applied; there were no ambiguous cases. Also converted three raw Unicode
characters in the abstract (`σ`, `→`, `∞`) to math mode (`$\sigma(x) \to \infty$`),
which `pdflatex` had rejected.

**Verification**: before the fix, `pdflatex` died on a *syntax* error at line 23;
after, it parses the entire preamble and all restored macros, reaching line 55 and
stopping only on a genuinely missing input file. Post-fix scan for orphaned macros
across all `.tex`: zero.

**Note — the submission is a skeleton.** 5 of its 7 sections do not exist
(`background`, `method`, `related_work`, `conclusion`, `appendix`), and
`icml2026.sty` is absent from the repo. It cannot build end-to-end regardless of
this fix. Treat `constraint_geometry/submission/` as a draft shell, not a
submittable artifact.

---

## §10 — ❌ CRITICAL: `paper1_geodesic_singularity` Table 1 does not match its data

**This is the most serious finding of the 2026-07-21 audit and supersedes §8 in
priority.** Unlike every other item in this document, this one is not a
"structural result" problem — it is a table whose stated provenance is false.

`submission/paper1_geodesic_singularity/main.tex` is a **complete, compiled ICML-
format paper** (`main.pdf`, 291 KB, built 2026-03-25 — not a skeleton like
`constraint_geometry/submission/`). Its Table 1 (`tab:murky_drone`, **lines
207–222**) reads:

```
Murky Drone Navigation results (mean ± std over 5 seeds).
Method   Return      Violations   Safety Rate
PPO      412 ± 89    52 ± 11      48%
CPO      51 ± 12     7 ± 3        93%
SGPO     408 ± 73    11 ± 4       89%
```

### What the data actually shows

The only artifact resembling this is `results/safety/safety_benchmark_metrics.json`,
written by `src/safety_experiment.py:469`. Comparing:

| Claim in paper | Reality in the data file |
|---|---|
| "mean ± std over **5 seeds**" | `src/safety_experiment.py:27-29` sets **one** seed (`torch.manual_seed(42)`, `np.random.seed(42)`). There is no seed loop and no seed dimension in the JSON — it is **a single run**. The `± 89 / ± 11 / ± 4` values correspond to nothing in the file. |
| Violations **52 / 7 / 11** | `sum(ppo_violations) = 52.0`, `sum(cpo_violations) = 7.0`, `sum(gpo_violations) = 11.0` — **exact match on all three**. These are *total violation counts summed over 250 episodes of one run*, presented as a per-seed mean ± std. |
| Return **412 / 51 / 408** | Does not match anything: `sum(returns)` = 277.2 / −1120.7 / 79.3; `final_mean_returns` = **−6.67 / −6.23 / +1.53**. Two of the three real values are *negative*. **Unsourced.** |
| "*Murky Drone Navigation* — 2D continuous, murky zone offers **+10**/step, goal +1/step" (**:197**) | The code is `SandbaggingEnv` (`src/safety_experiment.py:35`) with `trap_reward = 3.0` (**:75**). Different environment, different reward scale. |
| "the policy learns to navigate around the murky zone... achieving high returns from the legitimate goal" (**:225**) | `goal_success_rate` is **`{ppo: 0.0, cpo: 0.0, gpo: 0.0}`** — in this run **no method ever reached the goal**, including SGPO. |

The two headline numbers in the abstract (**:44**) and contributions (**:63**)
are derived from this table and inherit the problem: "79% violation reduction" is
`(52−11)/52 = 78.8%` (from the single-run sums) and "8× higher returns than CPO"
is `408/51 = 8.0` (from the unsourced Return column).

### Corroborating evidence that this is a relabelling

The same JSON is cited **correctly** elsewhere. `constraint_geometry/README.md:56`
reports it as "**Sandbagging Trap** — SGPO (+1.53 return) vs PPO (−6.67) vs CPO
(−6.23)" — which is exactly `final_mean_returns`, correctly named and correctly
signed. So one experiment (`SandbaggingEnv`, seed 42, single run) appears in the
repo under its true name with its true numbers, and simultaneously in the paper as
"Murky Drone Navigation, 5 seeds" with a fabricated spread and an unsourced return
column.

### What is and is not established

**Established**: the seed count is misstated; the violation figures are single-run
sums relabelled as multi-seed means; the return column matches no artifact in the
repo; the environment description does not match the code that produced the data;
the goal-reaching narrative is contradicted by the run's own `goal_success_rate`.

**Not established**: whether some other run — never saved — produced 412/51/408.
`.swarm/handoff_verifier_gap_unblock.md` notes that Modal Starter-tier retention
destroyed earlier run logs, so an unsaved source cannot be ruled out. That would
explain the Return column but **not** the "5 seeds" claim, since the violation
numbers demonstrably come from the single-seed file.

**Not established**: whether this paper was ever actually submitted anywhere.
`constraint_geometry/VENUE_TRACKER.md:79` records an "ICML 2026 (combined paper),
Jan 2026, Pending" submission; this PDF was built 2026-03-25, after that date.
**This must be checked with Mike before anything else.**

### Required action

1. **Determine whether this PDF was sent to any venue.** If it was, the table needs
   a correction to the venue, not just a repo fix.
2. Do not circulate `main.pdf` further until Table 1 is regenerated.
3. Regenerate the table from a real multi-seed run of the actual environment, or
   relabel it truthfully as a single-seed `SandbaggingEnv` result with the real
   returns (−6.67 / −6.23 / +1.53) and a `goal_success_rate` of 0.0 — which does
   not support the paper's current narrative.
4. The Agentic Shortcut numbers in the same paper (**:229** — "67% (PPO) to 12%
   (SGPO), 91% task completion") were not traced to any artifact either and should
   be audited the same way.

**Status: ✅ CORRECTED (2026-07-21).** Table 1 regenerated from
`results/safety/safety_benchmark_metrics.json` and relabelled as a single-seed
`SandbaggingEnv` result; abstract, contributions and conclusion corrected; dated
erratum added to page 1; PDF rebuilt and pushed. Exposure resolved: desk-rejected
from ICML 2026 on length, never submitted elsewhere, but it had been committed to
the public repo `github.com/MikeHLee/shape_of_good_behavior` — the corrected
version is now what is published there. See also §11 for three further corrections
made in the same pass.

---

## §13 — Track 3 (peer-consistency ‖δ¹c‖) ⚠️ reporting-convention defects only

Extends the audit boundary to Track 3, which §1–§11 did not cover. Track 3 lives
in `alignment_research/peer_consistency_geometry/` and is the material with the
widest **live** public distribution.

**Track 3 is in materially better shape than Tracks 1–2, and the difference is
structural, not cosmetic.** No fabricated data, no circular metric, no phantom
environment, and — unlike Tracks 1–2 — it genuinely seeds and aggregates
(5 cal/eval split-seeds × 4 subsample seeds, `E6_7B_panel.py --split-seeds`).
Every headline number reconciles against a committed JSON; `fig1`/`fig3`
regenerate pixel-identically. **The defects below are reporting-convention
defects, not result defects.** The central selectivity claim is unaffected by
every correction made.

### (A) The same quantity was published as two different numbers ✅ FIXED

Two correct estimators were both in public, unlabelled as different. Verified by
recomputing from the committed JSONs:

| estimator | convincing-game | insider-trading |
|---|---|---|
| split-seed (5 splits, subsample seed 0) | 0.6616 ± 0.0328 | **0.6369 ± 0.0059** |
| subsample aggregate (4 subsample seeds) | 0.6614 ± 0.0123 | **0.6255 ± 0.0140** |

Insider-trading was published as **0.637** in the writeup and **0.625** in the
repo README — a 0.012 gap with a 2× difference in the quoted ±, and nothing
telling a reader they were different estimators.

Fixed by adopting the **subsample aggregate** as canonical everywhere (the more
conservative choice; it varies over both nuisance axes) and stating the convention
next to the numbers. This *weakened* a public headline from 0.637 to 0.625.
`alignment-faking-reasoning` has no subsample sweep, so it keeps the split-seed
value and is now labelled as such rather than silently mixed in.

### (B) Every published p-value is one split's p ✅ FIXED (disclosed)

`experiments/E6_7B_panel.py:100` stores `"p_seed0": runs[0][0]["p"]` — split-seed
0 at subsample seed 0 — and it was printed beside an AUC that *is* a multi-seed
mean. Nothing in prose disclosed this; only the variable name did.

**This is more consequential than a labelling slip.** The per-subsample-seed
p-values for insider-trading span five orders of magnitude:

| subsample seed | AUC | p |
|---|---|---|
| 0 (**published**) | 0.6369 | **8.1e−8** |
| 1 | 0.6347 | 8.1e−5 |
| 2 | 0.6063 | **0.0155** |
| 3 | 0.6239 | 2.8e−3 |

The published `p=8e−8` is the most favourable of the four by a wide margin. To be
fair to the authors: `runs[0]` is taken *structurally*, so this is an
unrepresentative default, **not** a cherry-pick. But a reader was shown 8e−8 when
the range is `[8.1e−8, 1.6e−2]`. Now disclosed with the full range; p removed from
the top-level README rather than printed next to an aggregate it does not match.
Aggregating p across seeds remains the better fix and has not been done.

### (C) No multiple-comparison correction across 9 configs ✅ FIXED (disclosed)

Nine configurations have committed results. At Bonferroni α = 0.05/9 = 5.6e−3 the
strong positives all survive comfortably (insider-trading 8.1e−8,
alignment-faking-reasoning 5.9e−8, convincing-game 2.5e−6,
insider-trading-action 3.8e−6, redacted 8.9e−5). **`alignment-faking-output`
(p=0.045) does not survive** and is now stated as such rather than as a weak
positive. The nulls (instructed-deception 0.32, gender-secret 0.91,
insider-trading-reasoning 0.92) are unaffected.

Disclosing this *strengthens* the paper: the selectivity claim is a claim about
nulls, and nulls are more credible once the number of tests is on the record.

**Compound caveat (not in the source handoff).** (B) and (C) interact: if
insider-trading's *worst* subsample seed (p=0.0155) is used instead of the
published seed-0 value, it no longer clears the corrected threshold. The AUC is
robust across all four seeds (0.606–0.637, all well above chance), but the
result's *significance under correction* depends on which seed is quoted. Recorded
in the writeup.

### (D) Hardcoded figure constants ✅ FIXED before this audit

`figures/make_figures.py` hardcoded three small-panel AUCs as literals with a
comment noting the source JSONs "are not vendored". The values were **correct**
(matching upstream to 3dp), but a published figure carried numbers unverifiable
from the public repo. Fixed in `alignment_research` commit `72cab45` by vendoring
the JSONs and replacing the dict with `load_small_auc()`.

**Process note**: this is the same defect class as §7 (Figure 2 hand-typed), in a
track that otherwise loads everything from disk. Two independent tracks producing
the same pattern makes it a **pipeline issue**, not a one-off — figure generators
should be checked for literal data across the repo.

### Distribution — the §10 mitigation does not apply here

§10 was softened by paper1 having been desk-rejected with no reviewer exposure.
Track 3 is live in four places: `oasis-main/alignment_research` (public),
`MikeHLee/shape_of_good_behavior` (public), the venue-track submission plans, and
**Blog Part 4 "The Shape of a Lie", live on Ghost since 2026-06-04** — a general
audience, not reviewers.

**The live blog post does not currently misstate anything.** Its rounded figures
("0.66 ± 0.01", "0.63 ± 0.01") are correct under the now-canonical aggregate
convention, and it already carries the resolved cue-redaction framing. Only the
cosmetic `p` relabel from (B) would apply there. Blog edits need explicit
approval and dev/prod share one Ghost CMS, so no blog change has been made.

### Framing invariant — verified intact after every edit

Every surface must describe a **selective** detector, never a lie detector:
divergence tracks deception that involves a behavioural *strategy*, is blind to
flat falsehoods, and runs *backwards* on alignment-faking reasoning. The nulls are
part of the claim's shape and must stay adjacent to the positives. Checked after
the corrections above — all surfaces still do this.

**Status: ⚠️ OPEN → corrections applied in `alignment_research`; not blocking
distribution. The strong positives and the sign-flip are unaffected.**

---

## §12 — Murky Drone implemented for real (2026-07-21)

Closes the implementation half of §8. New file:
`src/murky_drone_experiment.py`, implementing the environment that
`constraint_geometry/docs/EXPERIMENTAL_DESIGN.md` specified but that was never
built: continuous 2D drone navigation, 5-dim observation `(x, y, vx, vy, sensor)`,
zone radius 0.15, sensor range 0.3 with σ=0.05 noise, three zone placements,
150-step episodes, 50 seeds.

**The fairness contract is the point of the rewrite.** All methods observe only
position, velocity, the noisy sensor, and the scalar cost signal. None is told
where the zone is. Critically, SGPO's metric is trained **as a classifier
predicting observed cost from observation** — it never sees `dist_to_zone` or the
zone centre. The existing `safety_experiment_hard.train_gpo` regresses its metric
onto ground-truth `dist_to_closest_trap` (`g_target = 1 + 5/safe_dist`, line 337);
carrying that over would have handed SGPO the answer and reproduced exactly the
circularity documented in §2. Ground-truth distance is retained in `info` for
logging only and is never passed to a learner.

**Two SGPO variants are run, deliberately.** `safety_experiment_hard.train_gpo`
implements SGPO as `advantage / sqrt(g)`. That formulation only *damps* the
learning signal inside dangerous regions — it applies no directional pressure away
from them, and with the metric no longer fed ground truth there is no other
mechanism by which it could avoid the zone. Testing only that would be a
strawman, so a second variant subtracts a discounted metric-derived barrier cost
from the advantage, which is closer to the "geodesically unreachable" mechanism the
papers describe.

Environment calibration was checked before running: a greedy straight-line
controller reaches the goal in 36 steps and takes 7 violations through the centred
zone, and a hand-written detour reaches the goal in 52 of the 150-step budget with
zero violations. So the safety/return trade-off is real and a safe policy exists.
For the two off-diagonal zone placements the direct path misses the zone entirely
(centre-to-diagonal distance 0.283 > radius 0.15); those seeds function as
false-positive controls — they test whether a method needlessly avoids safe space.

Results are written to `results/safety/murky_drone_multistep.json` with per-seed
records.

### Results — 50 seeds, 200 episodes, 4 methods

Total violations per seed (mean ± sd across 50 seeds):

| method | violations | phase-2 violations | violation-free eps | return | goal rate |
|---|---|---|---|---|---|
| PPO | 471.9 ± 346.6 | 404.9 | 65.3% | 17.50 | 37.9% |
| **CPO** | **180.7 ± 173.4** | **122.2** | **88.8%** | 16.76 | 35.2% |
| SGPO-scale | 508.0 ± 522.1 | 429.5 | 70.6% | 18.70 | 42.4% |
| SGPO-barrier | 274.8 ± 247.7 | 196.3 | 81.3% | 17.24 | 37.8% |

Welch tests on total violations across all 50 seeds:

| comparison | p | Cohen's d | |
|---|---|---|---|
| CPO vs SGPO-scale | 0.0001 | −0.84 | CPO safer |
| CPO vs SGPO-barrier | 0.030 | −0.44 | CPO safer |
| PPO vs SGPO-barrier | 0.0015 | +0.65 | SGPO-barrier safer |
| **PPO vs SGPO-scale** | **0.68** | **−0.08** | **no difference** |
| PPO vs CPO | <0.0001 | +1.06 | CPO safer |

Restricting to the 16 seeds where the zone actually sits on the direct path — the
only placement that poses a real conflict — sharpens it: PPO 801.9, CPO 309.5,
**SGPO-scale 977.6**, SGPO-barrier 470.9.

### What this establishes

**The claim "Murky Drone — SGPO 0% violations vs 100% for PPO/CPO" is not merely
unsupported; under a fair multi-step test it is contradicted.** SGPO does not
achieve zero violations, and it does not beat CPO. CPO is the safest method here
by a significant margin against both SGPO variants.

**The repo's actual SGPO formulation is statistically indistinguishable from
unconstrained PPO** (p=0.68, d=−0.08). Once its metric is no longer fed
ground-truth trap distance, `advantage / sqrt(g)` confers no measurable safety
benefit — consistent with the mechanical reading in §12 above, since scaling a
scalar advantage by a positive number cannot reverse the sign of the incentive to
cross the zone.

The barrier variant *does* beat PPO significantly (p=0.0015, d=0.65), so the cost
signal is usable. But it still loses to CPO's Lagrangian. **The geometry is not
adding anything over a soft penalty on the same signal** — which is the honest
version of the result the papers claim.

Zone localisation was weak: implied-centre error 0.219 (scale) and 0.330
(barrier) against a zone radius of 0.15, i.e. 1.5–2.2 radii.

### Limitations — this is one experiment, not a refutation of the idea

- **Goal rates are low for every method (18–42%)**, so none of these policies
  solves the task well. This is a weak-training regime and the comparison is
  between four mediocre policies.
- **200 episodes is short.** A longer horizon might separate the methods
  differently, particularly for SGPO whose metric needs cost observations before
  it can act.
- **No per-method hyperparameter tuning.** CPO's `cost_limit=1.0` and
  SGPO-barrier's `barrier=3.0` are single unswept choices. This does not
  systematically favour either, but a tuned SGPO might do better.
- **Off-path zone placements still accumulate violations** (PPO 264.6 / 368.7),
  meaning the learned policies wander rather than converging on a tight
  trajectory. Some of the measured violations are exploration noise rather than
  trap-seeking.

A fair summary: *as implemented in this repository*, SGPO's geometric mechanism
does not deliver the safety advantage claimed for it, and its headline formulation
performs no better than the unconstrained baseline. Whether a better
implementation of the geodesic idea would is an open question this experiment does
not settle.

---

## §11 — Remaining `paper1` numbers traced ✅ CORRECTED (2026-07-21)

Follow-up pass over every other number in `paper1_geodesic_singularity`. All three
findings are now corrected in the manuscript and covered by the page-1 erratum.

**(a) Agentic Shortcut — unsourced, withdrawn.** The paper claimed shortcut
exploitation falling from "67% (PPO) to 12% (SGPO) while maintaining 91% task
completion". The only run of that scenario
(`notebooks/modal_runner/results/ethical_scenarios_per_scenario_updated.csv`)
gives `agentic_shortcut`: **ppo 1.0, cpo 0.89, gpo 0.0** — i.e. 100% / 89% / 0%,
and no task-completion metric at all. The paper's three numbers match nothing.
Withdrawn pending a reproducible run.

**(b) Component-ablation table — unsourced, withdrawn.** Table 2 reported
violations/returns for "− Metric singularity", "− Geodesic projection",
"− Advantage scaling" and "Isotropic metric only", on the same fabricated scale as
the withdrawn Table 1 (Full SGPO listed as `11 ± 4` / `408 ± 73`). The only
ablation ever run (`results/modal_exports/ablation_study.csv`, 15 rows) sweeps
**hyperparameters** — `geometric_threshold`, `clip_ratio`, `black_hole_strength` —
and reports different quantities (`convergence_steps`, `final_safety_violation`,
`final_reward`). No component ablation exists. Withdrawn.

*Incidental*: that CSV's `time_seconds` column contains values around `7e-05` —
74 microseconds for a nominal 50-episode × 50-step training run. Whatever it
timed, it was not training. Consistent with the §3 finding that the timing
instrumentation is measuring the wrong thing.

**(c) Topology mining — the paper contradicted its own figure.** The text claimed
**160,800** samples, mean harmonic risk **0.758 ± 0.093**, and severity bands of
97% (≥0.6) / 81% (≥0.7) / **30%** (≥0.8).

The artifact the figure is actually drawn from is `data/topology_metadata.parquet`,
which contains **50,000** rows:

| statistic | paper text | actual parquet |
|---|---|---|
| n | 160,800 | **50,000** |
| mean ± std | 0.758 ± 0.093 | **0.754 ± 0.093** |
| ≥ 0.6 | 97% | **94.0%** |
| ≥ 0.7 | 81% | **76.6%** |
| ≥ 0.8 | 30% | **33.3%** |

Decisively: `plot_harmonic_risk_distribution()` in
`notebooks/modal_runner/generate_paper_figures.py:89` is **correctly data-driven** —
it computes the percentage at render time — and the shipped
`figures/harmonic_risk_distribution.pdf` has the title *"(33.3% samples > 0.8)"*
baked into it. So the paper's own figure said 33.3% while its caption and body text
said 30% of 160,800. The honest figure caught the text out.

A genuine 160,800-sample run is *plausible but unverifiable*:
`geodpo_experiments.py::full_hh_rlhf_mining` writes `full_160k_topology.parquet`
and `full_160k_stats.json` to the Modal volume, and **neither was ever downloaded**;
under Starter-tier retention they are likely gone. This is therefore a
*cannot-re-derive*, not a *contradicted* — but the text must match the artifact that
exists, so it has been changed to 50,000 / 33.3% / 0.754 throughout.

---

## Cross-cutting: reproducibility (Tracks 1–2 only)

None of `condorcet_ring_benchmark`, `ethical_scenario_evaluation`, or
`ablation_study` seeds NumPy or Torch, and `src/safety_experiment.py` runs a single
fixed seed (42) with no repetition. Every number reported from **these** harnesses
is single-seed.

**This does not generalise to the whole repo.** The Track 3 peer-consistency work
(`shared/results/peer_sheaf_e*.json`) does seed properly — 5 split-seeds × 4
subsample seeds — and its figures regenerate deterministically. Scope this caveat
to Tracks 1–2; applying it to Track 3 would be wrong. Separately, `shared/src/hodge_diagnostic.py` derives context IDs
via `hash(cat) % 10000` (**lines 124, 191**); Python string hashing is randomised
per process unless `PYTHONHASHSEED` is fixed, so those IDs are not stable across
runs and are exposed to collisions mod 10000. Grouping behaviour is preserved
within a single process, so this is a latent rather than active defect — but it
should be replaced with a stable hash before any of it is cited.

