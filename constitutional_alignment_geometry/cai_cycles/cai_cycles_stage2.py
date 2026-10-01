"""cai_cycles_stage2.py — SGB-045 Stage 2 + constitution ablation on Modal.

What this collects
------------------
For each prompt: 5 candidate responses (K5), all 10 pairs, both presentation
orders, R paraphrases of the judge instruction, and one scored log-odds per
grader. The stored quantity per call is

    margin = logprob("A") - logprob("B")        (single forward pass, no generation)

and the replicate margin for a pair is the order-averaged value
    m_r = ( margin(a first) - margin(b first) ) / 2 .

Why this shape (logprob_certified.py, 2026-09-30, B=200 x 100 replicates):
  * Judge log-probs are deterministic, so sampled votes add nothing. The
    replicate unit is a PARAPHRASE, drawn iid from a pre-registered pool.
  * Per-pair test = exact sign test over paraphrases. 0% false positives in
    8 transitive conditions (near-ties, Cauchy tails, heteroscedastic edges,
    shared scale, saturating map + position bias, skewed noise, summing grader).
  * Adaptive 2-stage design: Rs paraphrases on all 10 pairs -> pick the most
    cycle-like oriented triangle -> Rt FRESH paraphrases on its 3 pairs.
    100% power at 25% cyclic prompts for 128-256 calls/prompt/grader.
  * Curl is NOT the test (cycle_harness.py: every curl null rejects transitive
    truths at up to 100%). Curl is reported descriptively only.

Grader families (principles.py):
  * 6 principle graders. Scored separately, so every mixture — single
    principle, leave-one-out, arbitrary weights, and the sampled-principle
    grader — is computed OFFLINE from these log-probs at no extra GPU cost.
  * constitution:full and 6 x constitution:mask_<k>. These read the
    constitution text, so they need their own calls. This is the causal test
    of Mike's ablation idea: if cycles come from mixed criteria, deleting the
    conflicting clause should remove them.

This file WRITES RAW MARGINS ONLY. No p-values, no decisions. Analysis is a
separate offline step so the pre-registered test cannot be tuned after seeing
the data.

Cost control: run `--pilot` first (20 prompts, principle graders, Rs=4/Rt=8)
to measure real throughput before the full run is priced.

Run (needs Modal auth + GPU):
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 constitutional_alignment_geometry/cai_cycles/cai_cycles_stage2.py --pilot

CPU smokes (no GPU):
    ... --smoke             # local, framework + manifest only
    ... --remote-smoke-deps # real image on CPU: imports, tokenizer, dataset, prompt build
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_AI_RESEARCH_ROOT = _HERE.parents[4]
for _p in (str(_AI_RESEARCH_ROOT), str(_HERE.parent), "/app", "/app/cai_cycles"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_modal import GPU, ModalExperiment, SourceDir, images, volumes  # noqa: E402
from shared_modal import secrets as sm_secrets  # noqa: E402

JUDGE = "Qwen/Qwen2.5-32B-Instruct"
DATASET = "berkeley-nest/Nectar"      # 7 ungated responses per prompt; we take 5
OUT_DIR = "/results/cai_cycles_stage2"

# A/B token ids are resolved at runtime from the tokenizer (judge-specific).
PROMPT_HEADER = (
    "You are comparing two candidate responses to the same question.\n\n"
    "Question:\n{prompt}\n\n"
    "Response A:\n{a}\n\nResponse B:\n{b}\n\n"
)


class CaiCyclesStage2(ModalExperiment):
    """Collect order-balanced, paraphrase-replicated judge log-odds for K5 prompts."""

    division = "SGB"
    inquiry_id = "SGB-045"
    # bf16 32B weights ~65 GB. Single forward pass per call, use_cache=False, so
    # no KV cache is retained; 80 GB holds it with a modest batch and a 2k cap.
    image = images.inference_base
    gpu = GPU.H100
    volumes = [volumes.sgb_panel_cache]
    secrets = [sm_secrets.huggingface_token]
    timeout_seconds = 60 * 60 * 6

    mount = "/results"
    entry_module = "cai_cycles_stage2"
    source_dirs = [SourceDir(str(_HERE.parent), "/app/cai_cycles")]

    def run(self, n_prompts: int = 200, k: int = 5, r_stage1: int = 8, r_stage2: int = 16,
            seed: int = 0, graders: str = "all", batch_size: int = 8,
            max_response_chars: int = 1200, max_length: int = 2048,
            judge: str = JUDGE, dataset: str = DATASET,
            resume: bool = True) -> dict:
        import json
        import os
        import time
        from itertools import combinations

        import numpy as np
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from principles import PARAPHRASE_POOL, grader_specs, paraphrase

        cache = "/results/hf_hub_cache_cai_cycles"
        os.environ["HF_HOME"] = cache
        os.environ["HF_DATASETS_CACHE"] = "/results/hf_datasets_cache"
        os.makedirs(OUT_DIR, exist_ok=True)

        # ---- grader selection ------------------------------------------------
        specs = grader_specs(include_constitution=(graders in ("all", "constitution")))
        if graders == "principles":
            specs = [s for s in specs if s["kind"] == "principle"]
        elif graders == "constitution":
            specs = [s for s in specs if s["kind"] == "constitution"]
        print(f"[cai] {len(specs)} graders: {[s['name'] for s in specs]}", flush=True)

        # ---- prompts: K responses each, deterministic in (n_prompts, k, seed) --
        ds = load_dataset(dataset, split="train")
        rng = np.random.default_rng(seed)
        blocks = []
        for i in rng.permutation(len(ds)):
            row = ds[int(i)]
            answers = [a["answer"].strip() for a in row["answers"][:k]]
            if len(answers) < k or any(not a for a in answers):
                continue
            blocks.append({"idx": int(i), "prompt": row["prompt"].strip()[:max_response_chars],
                           "answers": [a[:max_response_chars] for a in answers]})
            if len(blocks) >= n_prompts:
                break
        print(f"[cai] blocks={len(blocks)} k={k}", flush=True)
        EDGES = list(combinations(range(k), 2))
        TRIS = list(combinations(range(k), 3))

        # Paraphrase assignment: iid per block, DISJOINT between the two stages
        # (stage-2 validity needs fresh draws).
        P = len(PARAPHRASE_POOL)
        for b in blocks:
            br = np.random.default_rng(seed * 1_000_003 + b["idx"])
            perm = br.permutation(P)
            b["par1"] = [int(x) for x in np.resize(perm[: max(1, P // 2)], r_stage1)]
            b["par2"] = [int(x) for x in np.resize(perm[max(1, P // 2):], r_stage2)]

        # ---- judge ------------------------------------------------------------
        tok = AutoTokenizer.from_pretrained(judge)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"            # last position must be the real final token
        ids = {}
        for letter in ("A", "B"):
            cand = [tok.encode(letter, add_special_tokens=False),
                    tok.encode(" " + letter, add_special_tokens=False)]
            cand = [c for c in cand if len(c) == 1]
            if not cand:
                raise RuntimeError(f"judge tokenizer has no single-token '{letter}'")
            ids[letter] = [c[0] for c in cand]
        print(f"[cai] letter token ids: {ids}", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            judge, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
        model.eval()
        model.config.use_cache = False
        dev = next(model.parameters()).device

        def build(block, i, j, par_i, spec):
            """One judge prompt: response i shown as A, response j as B."""
            body = PROMPT_HEADER.format(prompt=block["prompt"],
                                        a=block["answers"][i], b=block["answers"][j])
            instr = paraphrase(par_i, spec["criterion"])
            if spec["kind"] == "constitution":
                instr = ("Judge according to this constitution:\n" + spec["criterion"]
                         + "\n\n" + paraphrase(par_i, "the response that better follows "
                                                      "the constitution above"))
            msg = [{"role": "user", "content": body + instr}]
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

        @torch.no_grad()
        def score(texts):
            """Return margin = logprob(A) - logprob(B) at the next position."""
            out = []
            for s in range(0, len(texts), batch_size):
                enc = tok(texts[s: s + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(dev)
                logits = model(**enc).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                la = torch.logsumexp(lp[:, ids["A"]], dim=-1)
                lb = torch.logsumexp(lp[:, ids["B"]], dim=-1)
                out.extend((la - lb).tolist())
            return out

        def margins(block, pairs, pars, spec):
            """Order-averaged margins: dict (i,j) -> list over paraphrases."""
            texts, key = [], []
            for (i, j) in pairs:
                for p in pars:
                    texts.append(build(block, i, j, p, spec)); key.append((i, j, p, +1))
                    texts.append(build(block, j, i, p, spec)); key.append((i, j, p, -1))
            vals = score(texts)
            acc: dict = {pr: {} for pr in pairs}
            for (i, j, p, sgn), v in zip(key, vals):
                acc[(i, j)].setdefault(p, []).append(sgn * v)
            return {pr: [sum(acc[pr][p]) / len(acc[pr][p]) for p in pars] for pr in pairs}

        # ---- collect ----------------------------------------------------------
        results, t0 = [], time.time()
        n_calls = 0
        for spec in specs:
            path = os.path.join(OUT_DIR, f"margins_{spec['name'].replace(':', '_')}"
                                         f"_n{len(blocks)}_k{k}_seed{seed}.jsonl")
            if resume and os.path.exists(path):
                print(f"[cai] {spec['name']}: cached", flush=True)
                self.checkpoint(spec["name"], path, volume=self.volumes[0].name)
                results.append({"grader": spec["name"], "path": path, "status": "cached"})
                continue
            tg = time.time()
            with open(path, "w") as fh:
                for bi, b in enumerate(blocks):
                    m1 = margins(b, EDGES, b["par1"], spec)
                    n_calls += 2 * len(EDGES) * len(b["par1"])
                    # stage-1 selection: most cycle-like oriented triangle, by the
                    # weakest of its three majority fractions (selection only —
                    # the test uses stage 2).
                    best, pick = -1.0, None
                    for (x, y, z) in TRIS:
                        for tri in ((x, y, z), (x, z, y)):
                            fr = []
                            for (a, c) in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                                v = m1[(min(a, c), max(a, c))]
                                s = 1 if a < c else -1
                                fr.append(sum(1 for t in v if s * t > 0) / len(v))
                            if min(fr) > best:
                                best, pick = min(fr), tri
                    tri_pairs = sorted({(min(a, c), max(a, c)) for a, c in
                                        ((pick[0], pick[1]), (pick[1], pick[2]), (pick[2], pick[0]))})
                    m2 = margins(b, tri_pairs, b["par2"], spec)
                    n_calls += 2 * len(tri_pairs) * len(b["par2"])
                    fh.write(json.dumps({
                        "grader": spec["name"], "mask": spec["mask"], "dataset_idx": b["idx"],
                        "k": k, "picked_triangle": list(pick),
                        "paraphrases_stage1": b["par1"], "paraphrases_stage2": b["par2"],
                        "stage1": {f"{i}-{j}": v for (i, j), v in m1.items()},
                        "stage2": {f"{i}-{j}": v for (i, j), v in m2.items()},
                    }) + "\n")
                    if bi % 20 == 0:
                        el = time.time() - tg
                        print(f"[cai] {spec['name']} block {bi}/{len(blocks)} "
                              f"{el:.0f}s calls={n_calls}", flush=True)
            self._commit_volume()
            self.checkpoint(spec["name"], path, volume=self.volumes[0].name)
            results.append({"grader": spec["name"], "path": path, "status": "fresh",
                            "seconds": time.time() - tg})
            print(f"[cai] {spec['name']} done in {time.time() - tg:.0f}s", flush=True)

        return {"judge": judge, "dataset": dataset, "n_prompts": len(blocks), "k": k,
                "r_stage1": r_stage1, "r_stage2": r_stage2, "seed": seed,
                "graders": [s["name"] for s in specs], "per_grader": results,
                "forward_passes": n_calls, "total_seconds": time.time() - t0,
                "out_dir": OUT_DIR,
                "note": "raw order-averaged margins only; sign test applied offline"}

    def _commit_volume(self) -> None:
        try:
            import modal
            modal.Volume.from_name(self.volumes[0].name).commit()
        except Exception:  # noqa: BLE001 — best-effort
            pass


class _SmokeOK(CaiCyclesStage2):
    """Local/CPU framework smoke: manifest wiring only, no model."""
    image = images.analysis_base
    gpu = GPU.CPU
    secrets: list = []
    timeout_seconds = 600

    def run(self, **kw) -> dict:          # noqa: D102
        from principles import PRINCIPLE_KEYS, grader_specs
        return {"graders": len(grader_specs()), "principles": PRINCIPLE_KEYS}


class _SmokeDeps(CaiCyclesStage2):
    """CPU container on the REAL image: imports, tokenizer, dataset, prompt build.

    Catches the ORG-002 class of bug (missing dep / tokenizer needing
    sentencepiece / dataset not loadable) without paying for an H100.
    """
    gpu = GPU.CPU
    timeout_seconds = 1800

    def run(self, judge: str = JUDGE, dataset: str = DATASET, **kw) -> dict:  # noqa: D102
        import os
        import datasets
        import torch  # noqa: F401
        import transformers
        from transformers import AutoTokenizer
        from principles import grader_specs, paraphrase

        os.environ["HF_HOME"] = "/results/hf_hub_cache_cai_cycles"
        os.environ["HF_DATASETS_CACHE"] = "/results/hf_datasets_cache"
        tok = AutoTokenizer.from_pretrained(judge)
        single = {L: [t for t in (tok.encode(L, add_special_tokens=False),
                                  tok.encode(" " + L, add_special_tokens=False)) if len(t) == 1]
                  for L in ("A", "B")}
        ds = datasets.load_dataset(dataset, split="train[:4]")
        row = ds[0]
        specs = grader_specs()
        body = PROMPT_HEADER.format(prompt=row["prompt"].strip()[:400],
                                    a=row["answers"][0]["answer"][:400],
                                    b=row["answers"][1]["answer"][:400])
        lens = {}
        for s in specs:
            txt = tok.apply_chat_template(
                [{"role": "user", "content": body + paraphrase(0, s["criterion"])}],
                tokenize=False, add_generation_prompt=True)
            lens[s["name"]] = len(tok.encode(txt))
        return {"deps_ok": True, "transformers": transformers.__version__,
                "datasets": datasets.__version__, "single_letter_tokens": single,
                "responses_in_row": len(row["answers"]), "graders": len(specs),
                "prompt_tokens": lens}


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--r-stage1", type=int, default=8)
    ap.add_argument("--r-stage2", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graders", choices=("all", "principles", "constitution"), default="all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pilot", action="store_true",
                    help="20 prompts, principle graders, Rs=4/Rt=8 — measures real "
                         "throughput and cost before the full run is priced")
    ap.add_argument("--smoke", action="store_true", help="local CPU framework smoke")
    ap.add_argument("--remote-smoke-deps", action="store_true",
                    help="CPU container on the real image (imports + tokenizer + dataset)")
    a = ap.parse_args()

    if a.smoke:
        m = _SmokeOK().launch()
        print(f"[smoke] status={m.status.value} results={m.results}")
        return 0 if m.status.value == "success" else 1
    if a.remote_smoke_deps:
        exp = _SmokeDeps()
        m = exp.launch_remote()
        print(f"[remote-smoke-deps] status={m.status.value}")
        print(f"[remote-smoke-deps] results={m.results}")
        print(f"[remote-smoke-deps] mirror: {exp.manifest_path}")
        return 0 if m.status.value == "success" else 1

    kw = dict(n_prompts=a.n_prompts, k=a.k, r_stage1=a.r_stage1, r_stage2=a.r_stage2,
              seed=a.seed, graders=a.graders, batch_size=a.batch_size)
    if a.pilot:
        kw.update(n_prompts=20, r_stage1=4, r_stage2=8, graders="principles")
    exp = CaiCyclesStage2()
    m = exp.launch_remote(**kw)
    print(f"[cai] status={m.status.value}")
    print(f"[cai] results={m.results}")
    print(f"[cai] mirror: {exp.manifest_path}")
    return 0 if m.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
