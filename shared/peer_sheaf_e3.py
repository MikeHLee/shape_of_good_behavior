"""peer_sheaf_e3.py — SGB-012 peer-consistency panel on shared_modal (ORG-002).

The ORG-002 rerouting of the 7–9B peer-consistency panel onto the
`shared_modal.ModalExperiment` abstraction. Where `shared/modal_peer_sheaf.py`
is a hand-written module of `@app.function`s that call `track()`, this expresses
the same experiment as a single declarative `ModalExperiment` subclass whose
`run()` body is shipped to the A100 by the framework, and whose manifest lands in
the workspace mirror (`ai_research/_runs/`) on both success and crash.

What this is the test of (see `ai_research/.swarm/queue.md` ORG-002): whether the
ResearchExperiment abstraction can cleanly express three sequential model loads on
one A100-40GB, cached `.npy` per model, and a deterministic eval seed. The verdict
is written up in `shared_modal/README.md`.

Panel (unchanged from modal_peer_sheaf.py; Llama-3.1-8B / Mistral-7B are gated on
HF so the original checklist panel was swapped for three ungated 7–9B instruct LMs
from three orgs — same cross-org design):
    Yi-1.5-9B-Chat  ·  Zephyr-7B-beta  ·  Qwen2.5-7B-Instruct

Run (needs Modal auth + A100 — costs GPU time):
    cd topics/shape_of_good_behavior
    ./venv/bin/python3 shared/peer_sheaf_e3.py --n 2000 --seed 0

Framework-only smoke test (no GPU, no Modal — validates the manifest + workspace
mirror guarantee locally, including the crash path):
    ./venv/bin/python3 shared/peer_sheaf_e3.py --smoke
    ./venv/bin/python3 shared/peer_sheaf_e3.py --smoke-crash
"""

from __future__ import annotations

import sys
from pathlib import Path

# shared_modal lives at the ai_research root. Locally this file is at
# <root>/topics/shape_of_good_behavior/shared/, so the root is parents[3].
_HERE = Path(__file__).resolve()
_AI_RESEARCH_ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent
for _p in (str(_AI_RESEARCH_ROOT), str(_HERE.parent), "/app"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_modal import GPU, ModalExperiment, SourceDir, images, volumes  # noqa: E402
from shared_modal import secrets as sm_secrets  # noqa: E402

PANEL = [
    ("Yi-1.5-9B", "01-ai/Yi-1.5-9B-Chat"),
    ("Zephyr-7B", "HuggingFaceH4/zephyr-7b-beta"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
]

OUT_DIR = "/results/peer_sheaf_e3_modal"


def _extract_final_response(dialogue: str) -> str:
    marker = "\n\nA:"
    idx = dialogue.rfind(marker)
    if idx < 0:
        return dialogue.strip()
    tail = dialogue[idx + len(marker):]
    cut = tail.find("\n\nH:")
    if cut >= 0:
        tail = tail[:cut]
    return tail.strip()


class PeerSheafE3(ModalExperiment):
    """SGB-012 7–9B panel embedder as a shared_modal experiment.

    NOTE on image (ORG-002 deviation from the queue checklist): the checklist said
    `image=images.finetune_base`, but this is a *forward-only* embedding job — no
    TRL, no PEFT, no training. `inference_base` is the correct lab and skips the
    heavy finetune deps, so it is used here and the discrepancy is recorded in the
    ORG-002 write-up. `accelerate` is already in inference_base for device_map.
    """

    division = "SGB"
    inquiry_id = "SGB-012"
    image = images.inference_base
    gpu = GPU.A100_40GB
    volumes = [volumes.sgb_panel_cache]
    secrets = [sm_secrets.huggingface_token]
    timeout_seconds = 60 * 60 * 3  # 3h — a tight cap hard-kills mid-numpy (see manifest note)

    mount = "/results"
    entry_module = "peer_sheaf_e3"
    source_dirs = [
        SourceDir(str(_HERE.parent), "/app/shared", sys_path=False),
        SourceDir(str(_HERE.parent / "src"), "/app/shared/src"),
    ]

    def run(self, n: int = 2000, seed: int = 0, batch_size: int = 4,
            max_length: int = 512, force_reembed: bool = False) -> dict:
        """Embed HH-RLHF chosen+rejected with each panel model, one at a time.

        Runs inside the Modal container. One model resident at a time: load →
        embed → save `.npy` → commit volume → free CUDA → next. Each `.npy` is
        recorded as a checkpoint in the run manifest as it lands, so a crash after
        model 2 leaves an auditable trail of what completed.
        """
        import gc
        import hashlib
        import os
        import time

        import numpy as np
        import torch
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer

        cache = "/results/hf_hub_cache_peer_sheaf"
        os.environ["HF_HOME"] = cache
        os.environ["TRANSFORMERS_CACHE"] = cache
        os.environ["HF_DATASETS_CACHE"] = "/results/hf_datasets_cache"
        os.makedirs(OUT_DIR, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[e3] device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
              flush=True)

        # Deterministic subsample: same (n, seed) → same pair indices.
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
        chosen, rejected = [], []
        for i in idx:
            c = _extract_final_response(ds[int(i)]["chosen"])
            r = _extract_final_response(ds[int(i)]["rejected"])
            if c and r:
                chosen.append(c)
                rejected.append(r)
        n_pairs = len(chosen)
        all_texts = chosen + rejected
        blob_hash = hashlib.sha1("\n----\n".join(all_texts).encode()).hexdigest()[:16]
        print(f"[e3] usable pairs={n_pairs} texts={len(all_texts)} blob={blob_hash}", flush=True)

        per_model: dict = {}
        for name, hf_id in PANEL:
            out_path = os.path.join(OUT_DIR, f"{name}_n{n_pairs}_seed{seed}.npy")
            if os.path.exists(out_path) and not force_reembed:
                arr = np.load(out_path)
                per_model[name] = {"path": out_path, "shape": list(arr.shape), "status": "cached"}
                self.checkpoint(name, out_path, volume=self.volumes[0].name)
                print(f"[e3] {name}: cached {arr.shape}", flush=True)
                continue

            print(f"[e3] === {name} ({hf_id}) ===", flush=True)
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(hf_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModel.from_pretrained(
                hf_id, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
            )
            model.eval()
            load_s = time.time() - t0

            chunks = []
            t1 = time.time()
            with torch.no_grad():
                for start in range(0, len(all_texts), batch_size):
                    batch = all_texts[start:start + batch_size]
                    enc = tok(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_length).to(device)
                    out = model(**enc, output_hidden_states=False)
                    last = out.last_hidden_state
                    lengths = enc.attention_mask.sum(dim=1) - 1
                    gi = lengths.view(-1, 1, 1).expand(-1, 1, last.size(-1))
                    pooled = last.gather(1, gi).squeeze(1)
                    chunks.append(pooled.to(torch.float32).cpu().numpy())
            arr = np.concatenate(chunks, axis=0).astype(np.float64)
            np.save(out_path, arr)
            self._commit_volume()
            embed_s = time.time() - t1
            per_model[name] = {"path": out_path, "shape": list(arr.shape),
                               "load_seconds": load_s, "embed_seconds": embed_s, "status": "fresh"}
            # Record in the manifest immediately — survives a later-model crash.
            self.checkpoint(name, out_path, volume=self.volumes[0].name)
            print(f"[e3] {name}: {arr.shape} saved load={load_s:.1f}s embed={embed_s:.1f}s", flush=True)

            del model, tok
            gc.collect()
            torch.cuda.empty_cache()

        return {
            "n_pairs": n_pairs,
            "seed": seed,
            "text_blob_sha1_16": blob_hash,
            "panel": [{"name": nm, "hf_id": h} for nm, h in PANEL],
            "per_model": per_model,
            "out_dir": OUT_DIR,
        }

    def _commit_volume(self) -> None:
        """Persist volume writes mid-run so a crash keeps the completed `.npy`s."""
        try:
            import modal
            modal.Volume.from_name(self.volumes[0].name).commit()
        except Exception:  # noqa: BLE001 — best-effort
            pass


# ---- framework smoke experiments (no GPU, no Modal) ------------------------


class _SmokeOK(ModalExperiment):
    """Trivial local experiment to validate the manifest + workspace mirror."""
    division = "SGB"
    inquiry_id = "SGB-012"
    image = images.analysis_base
    gpu = GPU.CPU
    volumes = [volumes.sgb_panel_cache]
    entry_module = "peer_sheaf_e3"

    def run(self, n: int = 8, seed: int = 0) -> dict:
        import numpy as np
        rng = np.random.default_rng(seed)
        return {"checksum": float(rng.random(n).sum()), "n": n, "seed": seed}


class _SmokeCrash(_SmokeOK):
    """Same, but raises partway — validates the FAILED-mirror crash path."""
    def run(self, n: int = 8, seed: int = 0) -> dict:
        raise RuntimeError("intentional smoke-test crash after partial work")


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="run the CPU smoke experiment locally (no Modal)")
    ap.add_argument("--smoke-crash", action="store_true",
                    help="run the crashing smoke experiment locally (no Modal)")
    a = ap.parse_args()

    if a.smoke or a.smoke_crash:
        # Local execution path (base ResearchExperiment.launch): run() in-process,
        # manifest mirrored to ai_research/_runs on RUNNING and terminal state.
        exp = (_SmokeCrash if a.smoke_crash else _SmokeOK)()
        try:
            m = exp.launch(n=a.n, seed=a.seed)
        except Exception as e:  # noqa: BLE001 — the crash path is under test
            print(f"[smoke] run raised as expected: {e}")
            print(f"[smoke] manifest mirrored to: {exp.manifest_path}")
            return 0
        print(f"[smoke] status={m.status.value} results={m.results}")
        print(f"[smoke] manifest mirrored to: {exp.manifest_path}")
        return 0

    m = PeerSheafE3().launch_remote(n=a.n, seed=a.seed)
    print(f"[e3] status={m.status.value}")
    print(f"[e3] manifest: {PeerSheafE3().manifest_path}")
    return 0 if m.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
