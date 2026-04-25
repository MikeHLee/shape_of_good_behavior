import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

RESULTS_DIR = "results"
FIGURES_DIR = "../../submission/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if filename.endswith('.parquet'):
        return pd.read_parquet(path)
    elif filename.endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown format: {filename}")

def plot_ethical_scenarios():
    print("Generating ethical_scenarios_3d.pdf...")
    try:
        df = load_data("ethical_scenarios_per_scenario_updated.csv")
        
        # Filter for relevant scenarios if needed, or just plot all
        # The paper mentions Murky Drone and Agentic Shortcut specifically for failures
        
        # Create a grouped bar chart
        plt.figure(figsize=(12, 6))
        
        # We want to show Safety Violation Rate by Method and Scenario
        # Convert rate to percentage
        df['violation_pct'] = df['safety_violation_rate'] * 100
        
        # Order: Random, PPO, CPO, SGPO
        order = ['random', 'ppo', 'cpo', 'gpo']
        
        # Rename for nicer legend
        method_map = {
            'random': 'Random',
            'ppo': 'PPO',
            'cpo': 'CPO (Lagrangian)',
            'gpo': 'SGPO (Ours)'
        }
        df['Method'] = df['algorithm'].map(method_map)
        
        # Nicer scenario names
        scenario_map = {
            'academic_integrity': 'Academic\nIntegrity',
            'drone_decision': 'Drone\nDecision',
            'murky_drone': 'Murky\nDrone',
            'agentic_shortcut': 'Agentic\nShortcut',
            'business_ethics': 'Business\nEthics'
        }
        df['Scenario'] = df['scenario'].map(scenario_map)
        
        g = sns.barplot(
            data=df,
            x='Scenario',
            y='violation_pct',
            hue='Method',
            hue_order=['Random', 'PPO', 'CPO (Lagrangian)', 'SGPO (Ours)'],
            palette=['gray', '#e74c3c', '#e67e22', '#27ae60']
        )
        
        plt.title('Safety Violation Rate by Scenario and Method', fontweight='bold')
        plt.ylabel('Safety Violation Rate (%)')
        plt.xlabel('')
        plt.ylim(0, 105)
        plt.legend(title='Method', loc='upper right')
        
        # Add labels on top of bars
        for container in g.containers:
            g.bar_label(container, fmt='%.0f%%', padding=3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "ethical_scenarios_3d.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, "ethical_scenarios_3d.png"), bbox_inches='tight', dpi=300)
        print("Done.")
    except Exception as e:
        print(f"Failed to generate ethical_scenarios_3d: {e}")

def plot_harmonic_risk_distribution():
    print("Generating harmonic_risk_distribution.pdf...")
    try:
        # We need the harmonic risk scores. 
        # Ideally from comparative_analysis.parquet or topology_metadata.parquet
        # Let's try comparative_analysis.parquet first as it has response data
        # actually topology_metadata.parquet likely has the distribution for the dataset
        
        # Check if topology_metadata exists, if not try comparative_analysis
        if os.path.exists(os.path.join(RESULTS_DIR, "topology_metadata.parquet")):
            df = load_data("topology_metadata.parquet")
            risk_col = "harmonic_risk"
        elif os.path.exists(os.path.join(RESULTS_DIR, "comparative_analysis.parquet")):
            df = load_data("comparative_analysis.parquet")
            risk_col = "prompt_harmonic_risk"
        else:
            print("No data found for harmonic risk distribution.")
            return

        plt.figure(figsize=(10, 5))
        
        # Histogram
        sns.histplot(df[risk_col], bins=30, kde=True, color='#8e44ad', alpha=0.6)
        
        # Add vertical line for threshold
        plt.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='Severe Inconsistency Threshold (0.8)')
        
        # Calculate pct above 0.8
        pct_above = (df[risk_col] >= 0.8).mean() * 100
        
        plt.title(f'Distribution of Harmonic Risk in Anthropic HH-RLHF\n({pct_above:.1f}% samples > 0.8)', fontweight='bold')
        plt.xlabel('Harmonic Risk (H¹ Magnitude)')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "harmonic_risk_distribution.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, "harmonic_risk_distribution.png"), bbox_inches='tight', dpi=300)
        print("Done.")
    except Exception as e:
        print(f"Failed to generate harmonic_risk_distribution: {e}")

def plot_ablation_study():
    print("Generating ablation_study.pdf...")
    try:
        df = load_data("ablation_study.csv")
        
        # Filter 
        # We have ablation_type: geometric_threshold, clip_ratio, black_hole_strength
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Geometric Threshold (using Safety Violation and Convergence)
        # Actually paper shows:
        # Left: Effect of geometric threshold tau
        # Right: Effect of black hole strength alpha
        
        # Left Plot: Geometric Threshold
        df_tau = df[df['ablation_type'] == 'geometric_threshold']
        
        ax1 = axes[0]
        # Twin axis for convergence steps
        ax1_twin = ax1.twinx()
        
        sns.lineplot(data=df_tau, x='parameter_value', y='final_safety_violation', marker='o', color='#e74c3c', ax=ax1, label='Safety Violation')
        sns.lineplot(data=df_tau, x='parameter_value', y='convergence_steps', marker='s', color='#3498db', ax=ax1_twin, label='Convergence Steps')
        
        ax1.set_title('Effect of Geometric Threshold (τ)', fontweight='bold')
        ax1.set_xlabel('Geometric Threshold (τ)')
        ax1.set_ylabel('Safety Violation Rate', color='#e74c3c')
        ax1_twin.set_ylabel('Convergence Steps', color='#3498db')
        ax1.tick_params(axis='y', labelcolor='#e74c3c')
        ax1_twin.tick_params(axis='y', labelcolor='#3498db')
        
        # Right Plot: Black Hole Strength
        df_alpha = df[df['ablation_type'] == 'black_hole_strength']
        
        ax2 = axes[1]
        
        sns.lineplot(data=df_alpha, x='parameter_value', y='final_safety_violation', marker='o', color='#27ae60', ax=ax2, label='Safety Violation')
        
        ax2.set_title('Effect of Black Hole Strength (α)', fontweight='bold')
        ax2.set_xlabel('Black Hole Strength (α)')
        ax2.set_ylabel('Safety Violation Rate', color='#27ae60')
        ax2.tick_params(axis='y', labelcolor='#27ae60')
        
        # Add annotation for 0%
        zero_violation = df_alpha[df_alpha['final_safety_violation'] == 0]
        if not zero_violation.empty:
            x_val = zero_violation['parameter_value'].min()
            ax2.annotate('Unconditional Safety (0%)', 
                         xy=(x_val, 0), 
                         xytext=(x_val, 0.1),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "ablation_study.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, "ablation_study.png"), bbox_inches='tight', dpi=300)
        print("Done.")
    except Exception as e:
        print(f"Failed to generate ablation_study: {e}")

if __name__ == "__main__":
    plot_ethical_scenarios()
    plot_harmonic_risk_distribution()
    plot_ablation_study()
