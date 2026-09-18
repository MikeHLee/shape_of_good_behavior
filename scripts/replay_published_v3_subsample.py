"""Part A: replay run_optimizer_comparison's subsample on the volume mapping.pkl.

Reproduces shared/modal_runner.py:97-121 exactly, but with an explicit RandomState per
draw (the original used the unseeded global RNG, so the exact draw is unrecoverable).

Input: pipeline/mapping.pkl from the Modal volume, downloaded read-only, e.g.
    modal volume get reward-hacking-results pipeline/mapping.pkl <DIR>/volume/mapping.pkl
Set REPLAY_DIR=<DIR> (outputs are written there as volume_<mode>.json).
"""
import os
import json, logging, pickle, sys, time
from multiprocessing import Pool
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.environ.get("REPLAY_DIR") or sys.exit("set REPLAY_DIR (see docstring)")
sys.path.insert(0, ROOT)
logging.disable(logging.CRITICAL)

MAP = pickle.load(open(f"{SP}/volume/mapping.pkl", "rb"))


def subsample(draw):
    from shared.src.preference_mapper import MappingResult
    m = MAP
    rng = np.random.RandomState(draw)
    idx = rng.choice(len(m.embedding_pairs), 500, replace=False)
    retained = set()
    for i in idx:
        e = m.preference_edges[i]
        retained.add(e[0]); retained.add(e[1])
    kept = [e for e in m.preference_edges if e[0] in retained and e[1] in retained]
    sub = MappingResult(
        preference_edges=kept, n_items=m.n_items,
        embedding_pairs=[m.embedding_pairs[i] for i in idx],
        danger_regions=m.danger_regions, constitutional_gradients=m.constitutional_gradients,
        exploit_embeddings_reduced=m.exploit_embeddings_reduced[idx],
        ideal_embeddings_reduced=m.ideal_embeddings_reduced[idx],
        exploit_embeddings=m.exploit_embeddings[idx], ideal_embeddings=m.ideal_embeddings[idx],
    )
    own = [tuple(m.preference_edges[i][:2]) for i in idx]
    aligned = sum(1 for k in range(500) if tuple(kept[k][:2]) == own[k])
    # how many of the first 500 kept edges are direct edges at all
    n_direct_total = len(m.embedding_pairs)
    direct_set = set(tuple(e[:2]) for e in m.preference_edges[:n_direct_total])
    first500_direct = sum(1 for e in kept[:500] if tuple(e[:2]) in direct_set)
    n_trace = sum(1 for i in idx if i >= 2000)
    return idx, sub, dict(draw=draw, n_edges=len(kept), aligned=aligned,
                          first500_are_direct=first500_direct, n_trace_pairs=n_trace)


def structural(draw):
    import torch
    torch.set_num_threads(1)
    from shared.src.config import PipelineConfig
    from shared.src.hodge_diagnostic import HodgeDiagnosticCritic
    idx, sub, info = subsample(draw)
    t0 = time.time()
    d = HodgeDiagnosticCritic(PipelineConfig()).diagnose_for_samples(
        sub.preference_edges, sub.n_items, embedding_pairs=sub.embedding_pairs)
    info["exploit_fraction"] = float(d.exploit_fraction)
    info["dphi_pos"] = int((d.sample_potential_diffs > 0).sum())
    info["diag_s"] = time.time() - t0
    return info


def baselines(args):
    draw, methods, nseeds = args
    import torch
    torch.set_num_threads(1)
    from shared.src.config import PipelineConfig
    from shared.src.optimizer_comparison import OptimizerBenchmark
    cfg = PipelineConfig(); cfg.rm_epochs = 50
    idx, sub, info = subsample(draw)
    tab = OptimizerBenchmark(cfg, sub).run(num_seeds=nseeds, methods=methods)
    info["exploit_fraction"] = float(tab.config["diagnosis_exploit_fraction"])
    info["means"] = {k: v["mean"] for k, v in tab.method_stats.items()}
    info["stds"] = {k: v["std"] for k, v in tab.method_stats.items()}
    return info


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    with Pool(12) as pool:
        if mode == "structural":
            out = pool.map(structural, range(int(sys.argv[2])))
        else:
            ndraws = int(sys.argv[2])
            methods = sys.argv[3].split(",")
            out = pool.map(baselines, [(d, methods, 30) for d in range(ndraws)])
    json.dump(out, open(f"{SP}/volume_{mode}.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")
