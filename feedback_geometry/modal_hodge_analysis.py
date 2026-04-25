"""
Modal runner for full HH-RLHF Hodge Analysis.
Proves computational tractability at scale.
"""

import modal

app = modal.App("hodge-analysis")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "scipy",
        "datasets",
        "huggingface_hub",
    )
)


@app.function(
    image=image,
    timeout=3600,  # 1 hour
    memory=65536,  # 64GB RAM for large sparse graph
)
def run_full_hodge_analysis(n_samples: int = None):
    """
    Run Hodge decomposition on full HH-RLHF dataset.
    
    Args:
        n_samples: If specified, use only first N samples. None = full dataset.
    """
    import time
    import json
    import numpy as np
    from datetime import datetime
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import cg
    from datasets import load_dataset
    
    print("=" * 60)
    print("FULL HH-RLHF HODGE ANALYSIS (Modal)")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Load dataset
    print("Loading HH-RLHF dataset...")
    t0 = time.time()
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    load_time = time.time() - t0
    print(f"  Loaded {len(dataset)} examples in {load_time:.1f}s")
    
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
        print(f"  Using first {len(dataset)} samples")
    
    # Extract preferences
    print("\nExtracting preferences...")
    preferences = []
    for item in dataset:
        preferences.append({
            'chosen': item['chosen'][:500],
            'rejected': item['rejected'][:500],
        })
    print(f"  Total preferences: {len(preferences)}")
    
    # Build preference graph
    print("\nBuilding preference graph...")
    t0 = time.time()
    
    responses = set()
    for pref in preferences:
        responses.add(pref['chosen'])
        responses.add(pref['rejected'])
    
    response_list = list(responses)
    response_to_idx = {r: i for i, r in enumerate(response_list)}
    n = len(response_list)
    
    adjacency = np.zeros((n, n))
    for pref in preferences:
        i = response_to_idx[pref['chosen']]
        j = response_to_idx[pref['rejected']]
        adjacency[i, j] += 1
    
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")
    print(f"  Unique responses (nodes): {n}")
    
    # Hodge decomposition
    print("\nComputing Hodge decomposition (sparse)...")
    t0 = time.time()
    
    adj_sparse = csr_matrix(adjacency)
    nnz = adj_sparse.nnz
    print(f"  Non-zero edges: {nnz}")
    
    out_degree = np.asarray(adj_sparse.sum(axis=1)).flatten()
    in_degree = np.asarray(adj_sparse.sum(axis=0)).flatten()
    degree = out_degree + in_degree
    sym_adj = adj_sparse + adj_sparse.T
    L = diags(degree, format='csr') - sym_adj
    divergence = out_degree - in_degree
    
    t_laplacian = time.time() - t0
    print(f"  Laplacian construction: {t_laplacian:.2f}s")
    
    t1 = time.time()
    L_reg = L + 1e-6 * diags(np.ones(n))
    phi, info = cg(L_reg, divergence, maxiter=2000, rtol=1e-6)
    t_solve = time.time() - t1
    print(f"  CG solve: {t_solve:.2f}s (converged={info==0})")
    
    t2 = time.time()
    rows, cols = adj_sparse.nonzero()
    grad_data = phi[rows] - phi[cols]
    gradient = csr_matrix((grad_data, (rows, cols)), shape=(n, n)).toarray()
    
    antisym = (adjacency - adjacency.T) / 2
    curl = antisym - gradient
    harmonic = adjacency - gradient - curl
    t_components = time.time() - t2
    print(f"  Component computation: {t_components:.2f}s")
    
    total_time = time.time() - t0
    
    # Norms
    total_norm = np.linalg.norm(adjacency, 'fro') ** 2
    grad_norm = np.linalg.norm(gradient, 'fro') ** 2
    curl_norm = np.linalg.norm(curl, 'fro') ** 2
    harm_norm = np.linalg.norm(harmonic, 'fro') ** 2
    reliability = grad_norm / total_norm
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes (unique responses): {n}")
    print(f"  Edges (preferences):      {nnz}")
    print(f"  Density:                  {nnz / (n*n) * 100:.4f}%")
    
    print(f"\nEnergy Distribution:")
    print(f"  ||Adjacency||² = {total_norm:.2f}")
    print(f"  ||Gradient||²  = {grad_norm:.2f} ({grad_norm/total_norm*100:.1f}%)")
    print(f"  ||Curl||²      = {curl_norm:.2f} ({curl_norm/total_norm*100:.1f}%)")
    print(f"  ||Harmonic||²  = {harm_norm:.2f} ({harm_norm/total_norm*100:.1f}%)")
    print(f"\n  GLOBAL RELIABILITY SCORE: {reliability:.4f}")
    
    print(f"\nTiming Breakdown:")
    print(f"  Graph construction:       {build_time:.2f}s")
    print(f"  Laplacian construction:   {t_laplacian:.2f}s")
    print(f"  CG solve:                 {t_solve:.2f}s")
    print(f"  Component computation:    {t_components:.2f}s")
    print(f"  TOTAL DECOMPOSITION:      {total_time:.2f}s")
    
    # Cycle detection (sampled)
    print("\nCycle Detection (sampled 10K triplets)...")
    np.random.seed(42)
    cycle_count = 0
    for _ in range(10000):
        i, j, k = np.random.choice(min(n, 10000), 3, replace=False)
        if adjacency[i, j] > 0 and adjacency[j, k] > 0 and adjacency[k, i] > 0:
            cycle_count += 1
    print(f"  3-cycles found: {cycle_count} / 10000 samples ({cycle_count/100:.2f}%)")
    
    # Per-edge analysis
    print("\nPer-Edge Component Analysis...")
    edge_stats = {'grad': [], 'curl': [], 'harm': []}
    for pref in preferences[:10000]:  # Sample for speed
        ci = response_to_idx.get(pref['chosen'])
        ri = response_to_idx.get(pref['rejected'])
        if ci is not None and ri is not None:
            g = abs(gradient[ci, ri])
            c = abs(curl[ci, ri])
            h = abs(harmonic[ci, ri])
            t = g + c + h + 1e-8
            edge_stats['grad'].append(g / t)
            edge_stats['curl'].append(c / t)
            edge_stats['harm'].append(h / t)
    
    print(f"  Gradient ratio: mean={np.mean(edge_stats['grad']):.4f}, std={np.std(edge_stats['grad']):.4f}")
    print(f"  Curl ratio:     mean={np.mean(edge_stats['curl']):.4f}, std={np.std(edge_stats['curl']):.4f}")
    print(f"  Harmonic ratio: mean={np.mean(edge_stats['harm']):.4f}, std={np.std(edge_stats['harm']):.4f}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(preferences),
        'graph': {'n_nodes': n, 'n_edges': nnz, 'density': nnz / (n*n)},
        'energy': {
            'total': float(total_norm),
            'gradient': float(grad_norm),
            'gradient_pct': float(grad_norm/total_norm*100),
            'curl': float(curl_norm),
            'curl_pct': float(curl_norm/total_norm*100),
            'harmonic': float(harm_norm),
            'harmonic_pct': float(harm_norm/total_norm*100),
            'reliability': float(reliability)
        },
        'timings': {
            'graph_build': float(build_time),
            'laplacian': float(t_laplacian),
            'cg_solve': float(t_solve),
            'components': float(t_components),
            'total': float(total_time)
        },
        'cycles': {'found': cycle_count, 'sampled': 10000, 'rate': cycle_count/10000},
        'per_edge': {
            'grad_mean': float(np.mean(edge_stats['grad'])),
            'curl_mean': float(np.mean(edge_stats['curl'])),
            'harm_mean': float(np.mean(edge_stats['harm']))
        }
    }
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


@app.local_entrypoint()
def main(samples: int = None):
    """Run Hodge analysis."""
    print("Launching Hodge Analysis on Modal...")
    
    if samples:
        print(f"Using {samples} samples")
        results = run_full_hodge_analysis.remote(n_samples=samples)
    else:
        print("Using FULL dataset")
        results = run_full_hodge_analysis.remote()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Dataset: {results['dataset_size']} preferences")
    print(f"Graph: {results['graph']['n_nodes']} nodes, {results['graph']['n_edges']} edges")
    print(f"Reliability Score: {results['energy']['reliability']:.4f}")
    print(f"Total Time: {results['timings']['total']:.2f}s")
    print(f"\nEnergy Distribution:")
    print(f"  Gradient: {results['energy']['gradient_pct']:.1f}%")
    print(f"  Curl:     {results['energy']['curl_pct']:.1f}%")
    print(f"  Harmonic: {results['energy']['harmonic_pct']:.1f}%")
