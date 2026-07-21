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
> **§10 is the exception and is CRITICAL.** Table 1 of the compiled paper
> `submission/paper1_geodesic_singularity/main.pdf` states "mean ± std over 5
> seeds" for a **single-seed run**, reports violation counts that are that run's
> 250-episode *sums*, carries a Return column matching no artifact in the repo, and
> describes an environment (+10 murky zone) that is not the one in the code
> (`SandbaggingEnv`, trap reward 3.0). Its own data shows `goal_success_rate = 0.0`
> for every method, contradicting the paper's narrative. **Read §10 first, and
> establish whether that PDF was sent anywhere before doing anything else.**

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
| Murky Drone has no real impl | **HIGH** | — | ❌ **OPEN** (§8 audit) |
| LaTeX submission sources corrupted | **HIGH** | — | ✅ **FIXED** 2026-07-21 (§9) |
| **paper1 Table 1 misstates seeds + unsourced returns** | **CRITICAL** | — | ❌ **OPEN — blocking distribution** (§10) |

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

**Status: ❌ OPEN — blocking.** Decision (2026-07-21, Mike): implement for real
rather than strike. Target: wire a multi-step Murky Drone into the existing
`safety_experiment_hard.py` trainers (`train_cpo` **188**, `train_gpo` **279**,
`Actor` **119**, `Critic` **136**, `LearnedRiemannianMetric` **154**) and run 50+
seeds. Until that lands, the number must not be cited as evidence about PPO/CPO.

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

**Status: ❌ OPEN — CRITICAL, blocking any further distribution of this paper.**

---

## Cross-cutting: reproducibility

None of `condorcet_ring_benchmark`, `ethical_scenario_evaluation`, or
`ablation_study` seeds NumPy or Torch. Every reported number is single-seed and
non-reproducible. Separately, `shared/src/hodge_diagnostic.py` derives context IDs
via `hash(cat) % 10000` (**lines 124, 191**); Python string hashing is randomised
per process unless `PYTHONHASHSEED` is fixed, so those IDs are not stable across
runs and are exposed to collisions mod 10000. Grouping behaviour is preserved
within a single process, so this is a latent rather than active defect — but it
should be replaced with a stable hash before any of it is cited.

