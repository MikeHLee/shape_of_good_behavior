import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

def load_json(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def summarize_condorcet(data):
    if not data: return
    print("\n" + "="*50)
    print("CONDORCET CYCLE EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Ground Truth H¹: {data['ground_truth_h1']:.4f}")
    print(f"Learned H¹ (SGPO): {data['learned_h1']:.4f}")
    print(f"H¹ Error: {abs(data['ground_truth_h1'] - data['learned_h1']):.4f}")
    print("-" * 30)
    print(f"Empirical H¹ (PPO): {data['empirical_h1_ppo']:.4f}")
    print(f"Empirical H¹ (SGPO): {data['empirical_h1_gpo']:.4f}")

def summarize_safety(data):
    if not data: return
    print("\n" + "="*50)
    print("SAFETY BENCHMARK SUMMARY")
    print("="*50)
    final_stats = data.get('final_mean_violations', {})
    print("Mean Trap Violations (Last 50 Episodes):")
    print(f"  PPO: {final_stats.get('ppo', 'N/A')}")
    print(f"  CPO: {final_stats.get('cpo', 'N/A')}")
    print(f"  SGPO: {final_stats.get('gpo', 'N/A')}")
    
    # Calculate improvement
    if 'ppo' in final_stats and 'gpo' in final_stats:
        ppo_v = final_stats['ppo']
        gpo_v = final_stats['gpo']
        if ppo_v > 0:
            reduction = (ppo_v - gpo_v) / ppo_v * 100
            print(f"Safety Improvement (SGPO vs PPO): {reduction:.1f}%")

def summarize_style(data):
    if not data: return
    print("\n" + "="*50)
    print("LLM STYLE CYCLE SUMMARY")
    print("="*50)
    print(f"Ground Truth H¹: {data.get('ground_truth_h1', 'N/A')}")
    print(f"Learned Curl:    {data.get('learned_curl', 'N/A')}")
    print("-" * 30)
    print("Cycle Following Accuracy:")
    print(f"  PPO: {data.get('ppo_cycle_accuracy', 0):.1%}")
    print(f"  SGPO: {data.get('gpo_cycle_accuracy', 0):.1%}")
    print("-" * 30)
    print("Total Transitions:")
    print(f"  PPO: {data.get('ppo_transitions', 0)}")
    print(f"  SGPO: {data.get('gpo_transitions', 0)}")

def main():
    print("Aggregating Metrics from Experiment Logs...")
    
    condorcet_data = load_json('condorcet_metrics.json')
    safety_data = load_json('safety_benchmark_metrics.json')
    style_data = load_json('style_cycle_metrics.json')
    
    summarize_condorcet(condorcet_data)
    summarize_safety(safety_data)
    summarize_style(style_data)
    
    print("\n" + "="*50)
    print("Aggregation Complete.")

if __name__ == "__main__":
    main()
