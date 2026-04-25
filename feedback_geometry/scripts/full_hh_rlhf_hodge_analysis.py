#!/usr/bin/env python3
"""
Full HH-RLHF Hodge Analysis Script

Demonstrates computational tractability of HodgeRank on the full Anthropic HH-RLHF dataset.
This is a methodology validation experiment - not downstream fine-tuning.

Key metrics:
1. Graph size (nodes, edges)
2. Hodge decomposition timings
3. Component energy distribution (gradient, curl, harmonic)
4. Filtering retention rates at various thresholds
5. Cycle detection statistics
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg


def build_preference_graph(preferences):
    """Build adjacency matrix from preferences."""
    # Collect unique responses
    responses = set()
    for pref in preferences:
        responses.add(pref['chosen'])
        responses.add(pref['rejected'])
    
    response_list = list(responses)
    response_to_idx = {r: i for i, r in enumerate(response_list)}
    n = len(response_list)
    
    # Build adjacency
    adjacency = np.zeros((n, n))
    for pref in preferences:
        i = response_to_idx[pref['chosen']]
        j = response_to_idx[pref['rejected']]
        adjacency[i, j] += 1  # chosen > rejected
    
    return adjacency, {'response_to_idx': response_to_idx, 'idx_to_response': response_list}


def sparse_hodge_decomposition(adjacency, verbose=True):
    """
    Compute Hodge decomposition using sparse methods.
    
    Returns:
        gradient, curl, harmonic matrices and timing info
    """
    n = adjacency.shape[0]
    
    if verbose:
        print(f"  Graph size: {n} nodes")
    
    t0 = time.time()
    
    # Convert to sparse
    adj_sparse = csr_matrix(adjacency)
    nnz = adj_sparse.nnz
    
    if verbose:
        print(f"  Non-zero edges: {nnz}")
    
    # Build graph Laplacian
    out_degree = np.asarray(adj_sparse.sum(axis=1)).flatten()
    in_degree = np.asarray(adj_sparse.sum(axis=0)).flatten()
    degree = out_degree + in_degree
    sym_adj = adj_sparse + adj_sparse.T
    L = diags(degree, format='csr') - sym_adj
    
    # Divergence
    divergence = out_degree - in_degree
    
    t_laplacian = time.time() - t0
    if verbose:
        print(f"  Laplacian construction: {t_laplacian:.2f}s")
    
    # Solve L @ phi = divergence using CG
    t1 = time.time()
    L_reg = L + 1e-6 * diags(np.ones(n))
    phi, info = cg(L_reg, divergence, maxiter=2000, tol=1e-6)
    t_solve = time.time() - t1
    
    if verbose:
        print(f"  CG solve: {t_solve:.2f}s (converged={info==0})")
    
    # Compute gradient
    t2 = time.time()
    rows, cols = adj_sparse.nonzero()
    grad_data = phi[rows] - phi[cols]
    gradient = csr_matrix((grad_data, (rows, cols)), shape=(n, n)).toarray()
    
    # Curl and harmonic
    antisym = (adjacency - adjacency.T) / 2
    curl = antisym - gradient
    harmonic = adjacency - gradient - curl
    t_components = time.time() - t2
    
    if verbose:
        print(f"  Component computation: {t_components:.2f}s")
    
    total_time = time.time() - t0
    
    # Compute norms
    total_norm = np.linalg.norm(adjacency, 'fro') ** 2
    grad_norm = np.linalg.norm(gradient, 'fro') ** 2
    curl_norm = np.linalg.norm(curl, 'fro') ** 2
    harm_norm = np.linalg.norm(harmonic, 'fro') ** 2
    
    return {
        'gradient': gradient,
        'curl': curl,
        'harmonic': harmonic,
        'timings': {
            'laplacian': t_laplacian,
            'cg_solve': t_solve,
            'components': t_components,
            'total': total_time
        },
        'norms': {
            'total': total_norm,
            'gradient': grad_norm,
            'curl': curl_norm,
            'harmonic': harm_norm
        },
        'graph_info': {
            'n_nodes': n,
            'n_edges': nnz
        }
    }


def analyze_filtering(preferences, decomp, node_info):
    """Analyze filtering at various retention levels."""
    response_to_idx = node_info['response_to_idx']
    gradient = decomp['gradient']
    curl = decomp['curl']
    harmonic = decomp['harmonic']
    
    # Compute per-edge scores
    scores = []
    for pref in preferences:
        chosen_idx = response_to_idx.get(pref['chosen'])
        rejected_idx = response_to_idx.get(pref['rejected'])
        
        if chosen_idx is None or rejected_idx is None:
            continue
        
        grad = abs(gradient[chosen_idx, rejected_idx])
        cur = abs(curl[chosen_idx, rejected_idx])
        har = abs(harmonic[chosen_idx, rejected_idx])
        total = grad + cur + har + 1e-8
        
        scores.append({
            'grad_ratio': grad / total,
            'curl_ratio': cur / total,
            'harm_ratio': har / total
        })
    
    scores = np.array([(s['grad_ratio'], s['curl_ratio'], s['harm_ratio']) for s in scores])
    
    # Analyze at different retention levels
    retention_levels = [0.9, 0.8, 0.7, 0.6, 0.5]
    results = {}
    
    for keep_frac in retention_levels:
        n_keep = int(len(scores) * keep_frac)
        
        # Gradient-based (reliability)
        grad_sorted = np.argsort(-scores[:, 0])
        grad_kept = scores[grad_sorted[:n_keep]]
        
        # Curl-based (remove high curl)
        curl_sorted = np.argsort(scores[:, 1])  # Ascending
        curl_kept = scores[curl_sorted[:n_keep]]
        
        # Harmonic-based
        harm_sorted = np.argsort(-scores[:, 2])
        harm_kept = scores[harm_sorted[:n_keep]]
        
        results[f'{int(keep_frac*100)}%'] = {
            'reliability': {
                'grad_ratio_mean': float(grad_kept[:, 0].mean()),
                'curl_ratio_mean': float(grad_kept[:, 1].mean()),
                'harm_ratio_mean': float(grad_kept[:, 2].mean())
            },
            'curl_filtered': {
                'grad_ratio_mean': float(curl_kept[:, 0].mean()),
                'curl_ratio_mean': float(curl_kept[:, 1].mean()),
                'harm_ratio_mean': float(curl_kept[:, 2].mean())
            },
            'harmonic': {
                'grad_ratio_mean': float(harm_kept[:, 0].mean()),
                'curl_ratio_mean': float(harm_kept[:, 1].mean()),
                'harm_ratio_mean': float(harm_kept[:, 2].mean())
            }
        }
    
    return results


def detect_cycles(adjacency, sample_size=1000):
    """Detect 3-cycles in the preference graph."""
    n = adjacency.shape[0]
    
    # Sample random triplets
    cycle_count = 0
    samples_checked = 0
    
    np.random.seed(42)
    for _ in range(sample_size):
        i, j, k = np.random.choice(n, 3, replace=False)
        samples_checked += 1
        
        # Check for cycle: i > j > k > i
        if adjacency[i, j] > 0 and adjacency[j, k] > 0 and adjacency[k, i] > 0:
            cycle_count += 1
    
    return {
        'cycles_found': cycle_count,
        'samples_checked': samples_checked,
        'cycle_rate': cycle_count / samples_checked if samples_checked > 0 else 0
    }


def main():
    print("=" * 60)
    print("FULL HH-RLHF HODGE ANALYSIS")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Load full dataset
    print("Loading HH-RLHF dataset...")
    t0 = time.time()
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    load_time = time.time() - t0
    print(f"  Loaded {len(dataset)} examples in {load_time:.1f}s")
    
    # Extract preferences
    print("\nExtracting preferences...")
    preferences = []
    for item in dataset:
        preferences.append({
            'chosen': item['chosen'][:500],  # Truncate for memory
            'rejected': item['rejected'][:500],
            'prompt': item['chosen'][:100]  # First part as prompt
        })
    print(f"  Total preferences: {len(preferences)}")
    
    # Build preference graph
    print("\nBuilding preference graph...")
    t0 = time.time()
    adjacency, node_info = build_preference_graph(preferences)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")
    print(f"  Unique responses: {adjacency.shape[0]}")
    
    # Compute Hodge decomposition
    print("\nComputing Hodge decomposition (sparse)...")
    decomp = sparse_hodge_decomposition(adjacency, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Energy distribution
    norms = decomp['norms']
    total = norms['total']
    print(f"\nEnergy Distribution:")
    print(f"  ||Adjacency||² = {total:.2f}")
    print(f"  ||Gradient||²  = {norms['gradient']:.2f} ({norms['gradient']/total*100:.1f}%)")
    print(f"  ||Curl||²      = {norms['curl']:.2f} ({norms['curl']/total*100:.1f}%)")
    print(f"  ||Harmonic||²  = {norms['harmonic']:.2f} ({norms['harmonic']/total*100:.1f}%)")
    
    reliability = norms['gradient'] / total
    print(f"\n  Global Reliability Score: {reliability:.4f}")
    
    # Timings
    timings = decomp['timings']
    print(f"\nTiming Breakdown:")
    print(f"  Laplacian construction: {timings['laplacian']:.2f}s")
    print(f"  CG solve:               {timings['cg_solve']:.2f}s")
    print(f"  Component computation:  {timings['components']:.2f}s")
    print(f"  TOTAL:                  {timings['total']:.2f}s")
    
    # Cycle detection
    print("\nCycle Detection (sampled)...")
    cycles = detect_cycles(adjacency, sample_size=10000)
    print(f"  3-cycles found: {cycles['cycles_found']} / {cycles['samples_checked']} samples")
    print(f"  Cycle rate: {cycles['cycle_rate']*100:.4f}%")
    
    # Filtering analysis
    print("\nFiltering Analysis at Various Retention Levels...")
    filter_results = analyze_filtering(preferences, decomp, node_info)
    
    for level, methods in filter_results.items():
        print(f"\n  Retention = {level}:")
        for method, stats in methods.items():
            print(f"    {method}: grad={stats['grad_ratio_mean']:.3f}, curl={stats['curl_ratio_mean']:.3f}, harm={stats['harm_ratio_mean']:.3f}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(preferences),
        'graph': {
            'n_nodes': decomp['graph_info']['n_nodes'],
            'n_edges': decomp['graph_info']['n_edges']
        },
        'energy_distribution': {
            'total': float(norms['total']),
            'gradient': float(norms['gradient']),
            'gradient_pct': float(norms['gradient'] / total * 100),
            'curl': float(norms['curl']),
            'curl_pct': float(norms['curl'] / total * 100),
            'harmonic': float(norms['harmonic']),
            'harmonic_pct': float(norms['harmonic'] / total * 100),
            'reliability_score': float(reliability)
        },
        'timings': {k: float(v) for k, v in timings.items()},
        'cycles': cycles,
        'filtering_analysis': filter_results
    }
    
    output_path = Path(__file__).parent.parent / "results" / "hodge_analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"full_hh_rlhf_hodge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
