
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_topology_analyzer import EmbeddingTopologyAnalyzer
from src.visualize_embedding_topology import EmbeddingTopologyVisualizer

DATA_DIR = Path(__file__).parent.parent / "notebooks" / "modal_runner" / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "figures" / "supplementary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data_and_visualize():
    print("Loading visualization embeddings...")
    json_path = DATA_DIR / "viz_embeddings.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversation samples.")
    
    # Flatten data for analysis
    all_embeddings = []
    all_rewards = [] # We'll use trajectory shift or risk as a proxy for reward/value
    all_labels = []
    all_texts = []
    
    # We want to visualize the flow from Prompt -> Response
    # And distinct clusters for different models
    
    for item in data:
        # Prompt
        p_emb = item['prompt_embedding']
        p_risk = item.get('harmonic_risk', 0.5)
        
        all_embeddings.append(p_emb)
        all_rewards.append(-p_risk) # Higher risk = Lower reward
        all_labels.append('Prompt')
        all_texts.append(item.get('prompt_text', '')[:50])
        
        # Responses
        responses = item.get('responses', {})
        if not responses:
            continue
            
        for model, res_data in responses.items():
            if 'embedding' not in res_data:
                continue
                
            r_emb = res_data['embedding']
            # Use trajectory shift as positive reward if available, else 0
            # For base/ppo/cpo this might be missing, assume 0 or infer
            traj_shift = res_data.get('trajectory_shift', 0.0)
            
            # Simple heuristic: SGPO/SGPO_Clipped usually have higher safety/shift
            # PPO/CPO might be lower. 
            # If shift is missing, we rely on the fact that they are distinct points.
            
            all_embeddings.append(r_emb)
            all_rewards.append(traj_shift)
            all_labels.append(model.upper())
            all_texts.append(res_data.get('text', '')[:50])

    embeddings = np.array(all_embeddings)
    rewards = np.array(all_rewards)
    
    print(f"Total embedding points: {len(embeddings)}")
    
    # Initialize Analyzer
    # Mock embedder
    class IdentityEmbedder:
        def encode(self, x): return x
        
    analyzer = EmbeddingTopologyAnalyzer(
        embedding_model=IdentityEmbedder(),
        n_clusters=5,
        black_hole_threshold=np.percentile(rewards, 10) # Bottom 10%
    )
    
    print("Fitting Analyzer...")
    analyzer.fit(
        states=embeddings,
        actions=['action'] * len(embeddings),
        rewards=rewards,
        texts=all_texts
    )
    
    # Manually inject cluster labels if we want to color by Model Type instead of semantic cluster
    # The visualizer uses analyzer.cluster_labels
    # Let's map string labels to ints
    unique_labels = sorted(list(set(all_labels)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[l] for l in all_labels])
    
    # Overwrite clusters for visualization purposes to show Model separation
    analyzer.cluster_labels = numeric_labels
    
    print("Generating Visualizations...")
    visualizer = EmbeddingTopologyVisualizer(analyzer)
    
    # 1. Dashboard
    visualizer.create_summary_dashboard(save_path=OUTPUT_DIR / "topology_dashboard.png")
    
    # 2. Hodge 2D
    visualizer.plot_hodge_decomposition_2d(save_path=OUTPUT_DIR / "hodge_decomposition_2d.png")
    
    # 3. Hodge 3D
    visualizer.plot_hodge_decomposition_3d(save_path=OUTPUT_DIR / "hodge_decomposition_3d.png")
    
    # 4. Consistency
    visualizer.plot_consistency_analysis(save_path=OUTPUT_DIR / "consistency_analysis.png")
    
    # 5. Trajectory Analysis (for a specific interesting trajectory)
    # Let's find a sample with SGPO response
    sample_idx = 0
    start_idx = 0
    # The data is flattened: Prompt, Res1, Res2, ...
    # We need to reconstruct indices for one conversation
    # But we flattened it into a single list.
    # Let's just pick the first few points which likely correspond to the first item
    
    # Count how many items in first sample
    first_item_count = 1 + len(data[0]['responses'])
    traj_indices = list(range(first_item_count))
    
    visualizer.plot_trajectory_analysis(
        trajectory_indices=traj_indices,
        trajectory_id=f"sample_{data[0].get('prompt_id', '0')}",
        save_path=OUTPUT_DIR / "trajectory_analysis.png"
    )
    
    print(f"Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    load_data_and_visualize()
