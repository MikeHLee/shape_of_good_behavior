"""
Visualization of Embedding Topology for Semantic RL

Creates interpretable visualizations of:
1. Hodge decomposition (gradient, curl, harmonic)
2. Reward manifold with black holes and cliffs
3. Trajectory flow analysis
4. Condorcet cycle detection
5. Cluster semantic maps

Output: Publication-quality figures for the ICML submission.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple
import warnings

from src.hodge_critic import HodgeCritic, FeedbackItem, TopologicalGradient
from src.embedding_topology_analyzer import (
    EmbeddingTopologyAnalyzer,
    TopologicalFeatures,
    InterpretableRegion,
    TrajectoryAnalysis,
)


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class EmbeddingTopologyVisualizer:
    """
    Creates visualizations of embedding space topology for interpretability.
    """
    
    def __init__(
        self,
        analyzer: EmbeddingTopologyAnalyzer,
        hodge_critic: Optional[HodgeCritic] = None,
        figsize: Tuple[int, int] = (12, 10),
        style: str = "seaborn-v0_8-whitegrid",
    ):
        self.analyzer = analyzer
        self.hodge_critic = hodge_critic
        self.figsize = figsize
        
        # Try to set style, fall back gracefully
        try:
            plt.style.use(style)
        except Exception:
            pass
        
        # Color scheme
        self.colors = {
            "gradient": "#2ecc71",      # Green
            "curl": "#e74c3c",          # Red
            "harmonic": "#3498db",      # Blue
            "black_hole": "#1a1a2e",    # Dark
            "cliff": "#f39c12",         # Orange
            "safe": "#27ae60",          # Green
            "trajectory": "#9b59b6",    # Purple
            "cluster_cmap": "tab10",
        }
    
    def plot_hodge_decomposition_2d(
        self,
        save_path: Optional[str] = None,
        show_vectors: bool = True,
    ) -> plt.Figure:
        """
        Plot 2D projection of Hodge decomposition.
        
        Shows gradient (green), curl (red), and harmonic (blue) components.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Project embeddings to 2D
        if self.analyzer.embeddings is None or len(self.analyzer.embeddings) < 3:
            return fig
        
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(self.analyzer.embeddings)
        
        # 1. Full manifold with rewards
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=self.analyzer.rewards,
            cmap="RdYlGn",
            s=60,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax1, label="Reward")
        ax1.set_title("Reward Manifold (PCA Projection)")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        
        # Mark black holes
        for idx in self.analyzer.black_hole_indices:
            ax1.scatter(
                coords_2d[idx, 0], coords_2d[idx, 1],
                c=self.colors["black_hole"],
                s=200,
                marker='X',
                zorder=10,
            )
        
        # Mark cliffs
        for idx in self.analyzer.cliff_indices:
            ax1.scatter(
                coords_2d[idx, 0], coords_2d[idx, 1],
                c=self.colors["cliff"],
                s=100,
                marker='^',
                zorder=10,
            )
        
        # 2. Gradient field
        ax2 = axes[0, 1]
        ax2.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=self.colors["gradient"],
            s=30,
            alpha=0.5,
        )
        
        if show_vectors and self.analyzer.hodge_gradient is not None:
            # Project gradient to 2D
            grad_2d = pca.transform(self.analyzer.hodge_gradient.reshape(1, -1))[0]
            center = coords_2d.mean(axis=0)
            scale = np.std(coords_2d) * 1.5
            
            ax2.arrow(
                center[0], center[1],
                scale * grad_2d[0], scale * grad_2d[1],
                head_width=0.1, head_length=0.05,
                fc=self.colors["gradient"], ec=self.colors["gradient"],
                linewidth=3,
            )
        
        grad_mag = np.linalg.norm(self.analyzer.hodge_gradient) if self.analyzer.hodge_gradient is not None else 0
        ax2.set_title(f"Gradient Component (||∇φ|| = {grad_mag:.3f})")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        
        # 3. Curl + Harmonic field
        ax3 = axes[1, 0]
        ax3.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=self.colors["curl"],
            s=30,
            alpha=0.5,
        )
        
        if show_vectors:
            center = coords_2d.mean(axis=0)
            scale = np.std(coords_2d) * 1.5
            
            # Curl vector
            if self.analyzer.hodge_curl is not None:
                curl_2d = pca.transform(self.analyzer.hodge_curl.reshape(1, -1))[0]
                ax3.arrow(
                    center[0], center[1],
                    scale * curl_2d[0], scale * curl_2d[1],
                    head_width=0.1, head_length=0.05,
                    fc=self.colors["curl"], ec=self.colors["curl"],
                    linewidth=3, label='Curl'
                )
            
            # Harmonic vector
            if self.analyzer.hodge_harmonic is not None:
                harm_2d = pca.transform(self.analyzer.hodge_harmonic.reshape(1, -1))[0]
                ax3.arrow(
                    center[0], center[1],
                    scale * harm_2d[0], scale * harm_2d[1],
                    head_width=0.1, head_length=0.05,
                    fc=self.colors["harmonic"], ec=self.colors["harmonic"],
                    linewidth=3, linestyle='--', label='Harmonic'
                )
        
        curl_mag = np.linalg.norm(self.analyzer.hodge_curl) if self.analyzer.hodge_curl is not None else 0
        harm_mag = np.linalg.norm(self.analyzer.hodge_harmonic) if self.analyzer.hodge_harmonic is not None else 0
        
        ax3.set_title(f"Non-Gradient Components\n(Curl={curl_mag:.2f}, Harm={harm_mag:.2f})")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        
        # 4. Combined flow field
        ax4 = axes[1, 1]
        
        # Color by cluster
        if self.analyzer.cluster_labels is not None:
            scatter = ax4.scatter(
                coords_2d[:, 0], coords_2d[:, 1],
                c=self.analyzer.cluster_labels,
                cmap=self.colors["cluster_cmap"],
                s=60,
                alpha=0.7,
            )
        else:
            ax4.scatter(
                coords_2d[:, 0], coords_2d[:, 1],
                c="gray",
                s=60,
                alpha=0.7,
            )
        
        # Draw trajectory lines
        for i in range(len(coords_2d) - 1):
            ax4.annotate(
                "",
                xy=(coords_2d[i+1, 0], coords_2d[i+1, 1]),
                xytext=(coords_2d[i, 0], coords_2d[i, 1]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=self.colors["trajectory"],
                    alpha=0.4,
                    lw=1,
                ),
            )
        
        # Add region labels
        for region in self.analyzer.regions:
            region_2d = pca.transform(region.centroid.reshape(1, -1))[0]
            ax4.annotate(
                region.label[:15],
                xy=(region_2d[0], region_2d[1]),
                fontsize=8,
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            )
        
        ax4.set_title("Semantic Clusters & Trajectory Flow")
        ax4.set_xlabel("PC1")
        ax4.set_ylabel("PC2")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_hodge_decomposition_3d(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 3D projection of Hodge decomposition with gradient vectors.
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if self.analyzer.embeddings is None or len(self.analyzer.embeddings) < 4:
            return fig
        
        # Project to 3D
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(self.analyzer.embeddings)
        
        # Color by reward
        scatter = ax.scatter(
            coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
            c=self.analyzer.rewards,
            cmap="RdYlGn",
            s=60,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="Reward", shrink=0.6)
        
        # Draw trajectory
        ax.plot(
            coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
            color=self.colors["trajectory"],
            alpha=0.5,
            linewidth=1,
        )
        
        # Draw Hodge gradient arrow
        if self.analyzer.hodge_gradient is not None:
            grad_3d = pca.transform(self.analyzer.hodge_gradient.reshape(1, -1))[0]
            center = coords_3d.mean(axis=0)
            scale = np.std(coords_3d) * 2
            
            arrow = Arrow3D(
                [center[0], center[0] + scale * grad_3d[0]],
                [center[1], center[1] + scale * grad_3d[1]],
                [center[2], center[2] + scale * grad_3d[2]],
                mutation_scale=15,
                lw=3,
                arrowstyle="-|>",
                color=self.colors["gradient"],
            )
            ax.add_artist(arrow)
        
        # Draw curl arrow
        if self.analyzer.hodge_curl is not None:
            curl_3d = pca.transform(self.analyzer.hodge_curl.reshape(1, -1))[0]
            center = coords_3d.mean(axis=0)
            scale = np.std(coords_3d) * 2
            
            arrow = Arrow3D(
                [center[0], center[0] + scale * curl_3d[0]],
                [center[1], center[1] + scale * curl_3d[1]],
                [center[2], center[2] + scale * curl_3d[2]],
                mutation_scale=15,
                lw=3,
                arrowstyle="-|>",
                color=self.colors["curl"],
            )
            ax.add_artist(arrow)
        
        # Mark black holes
        for idx in self.analyzer.black_hole_indices:
            ax.scatter(
                [coords_3d[idx, 0]], [coords_3d[idx, 1]], [coords_3d[idx, 2]],
                c=self.colors["black_hole"],
                s=200,
                marker='X',
            )
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("3D Reward Manifold with Hodge Decomposition")
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["gradient"],
                   markersize=10, label='Gradient (∇φ)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["curl"],
                   markersize=10, label='Curl (H¹)'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor=self.colors["black_hole"],
                   markersize=10, label='Black Hole'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_consistency_analysis(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot consistency analysis including H¹ cohomology and Condorcet cycles.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        features = self.analyzer.extract_features()
        
        # 1. Hodge decomposition bar chart
        ax1 = axes[0]
        components = ['Gradient\n(Learnable)', 'Curl\n(Inconsistency)', 'Harmonic\n(Global)']
        magnitudes = [
            features.gradient_magnitude,
            features.curl_magnitude,
            features.harmonic_magnitude,
        ]
        colors = [self.colors["gradient"], self.colors["curl"], self.colors["harmonic"]]
        
        bars = ax1.bar(components, magnitudes, color=colors, edgecolor='white', linewidth=2)
        ax1.set_ylabel("Magnitude")
        ax1.set_title("Hodge Decomposition Components")
        
        # Add value labels
        for bar, val in zip(bars, magnitudes):
            ax1.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}",
                ha='center', va='bottom', fontsize=10,
            )
        
        # 2. Consistency gauge
        ax2 = axes[1]
        h1 = features.h1_cohomology
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax2.plot(x, y, 'k-', linewidth=2)
        ax2.plot([-1, 1], [0, 0], 'k-', linewidth=2)
        
        # Zones
        ax2.fill_between(np.cos(np.linspace(0, np.pi/3, 50)),
                         0, np.sin(np.linspace(0, np.pi/3, 50)),
                         alpha=0.3, color='green', label='Consistent')
        ax2.fill_between(np.cos(np.linspace(np.pi/3, 2*np.pi/3, 50)),
                         0, np.sin(np.linspace(np.pi/3, 2*np.pi/3, 50)),
                         alpha=0.3, color='orange', label='Warning')
        ax2.fill_between(np.cos(np.linspace(2*np.pi/3, np.pi, 50)),
                         0, np.sin(np.linspace(2*np.pi/3, np.pi, 50)),
                         alpha=0.3, color='red', label='Inconsistent')
        
        # Needle
        needle_angle = np.pi - min(h1, 1.0) * np.pi
        ax2.arrow(
            0, 0,
            0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
            head_width=0.1, head_length=0.1, fc='black', ec='black',
        )
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f"H¹ Consistency Gauge: {h1:.3f}")
        ax2.legend(loc='lower center', ncol=3)
        
        # 3. Safety region pie chart
        ax3 = axes[2]
        n_total = features.n_points
        n_black_holes = features.n_black_holes
        n_cliffs = features.n_cliffs
        n_safe = n_total - n_black_holes - n_cliffs
        
        sizes = [n_safe, n_cliffs, n_black_holes]
        labels = [f'Safe\n({n_safe})', f'Cliffs\n({n_cliffs})', f'Black Holes\n({n_black_holes})']
        colors_pie = [self.colors["safe"], self.colors["cliff"], self.colors["black_hole"]]
        explode = (0, 0.05, 0.1)
        
        ax3.pie(
            sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.0f%%', shadow=True, startangle=90,
        )
        ax3.set_title("Safety Region Distribution")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_trajectory_analysis(
        self,
        trajectory_indices: List[int],
        trajectory_id: str = "trajectory",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot detailed trajectory analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        analysis = self.analyzer.analyze_trajectory(trajectory_indices, trajectory_id)
        
        # 1. Reward progression
        ax1 = axes[0, 0]
        traj_rewards = [self.analyzer.rewards[i] for i in trajectory_indices]
        steps = range(len(traj_rewards))
        
        ax1.plot(steps, traj_rewards, 'o-', color=self.colors["trajectory"], linewidth=2, markersize=8)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(steps, traj_rewards, alpha=0.3, color=self.colors["trajectory"])
        
        # Mark black holes and cliffs
        for i, idx in enumerate(trajectory_indices):
            if idx in self.analyzer.black_hole_indices:
                ax1.scatter([i], [traj_rewards[i]], c=self.colors["black_hole"],
                           s=150, marker='X', zorder=10, label='Black Hole' if i == 0 else '')
            if idx in self.analyzer.cliff_indices:
                ax1.scatter([i], [traj_rewards[i]], c=self.colors["cliff"],
                           s=100, marker='^', zorder=10, label='Cliff' if i == 0 else '')
        
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward")
        ax1.set_title(f"Reward Progression ({analysis.reward_trend})")
        ax1.legend()
        
        # 2. Cumulative reward
        ax2 = axes[0, 1]
        cumulative = np.cumsum(traj_rewards)
        ax2.plot(steps, cumulative, 'o-', color=self.colors["gradient"], linewidth=2, markersize=6)
        ax2.fill_between(steps, cumulative, alpha=0.3, color=self.colors["gradient"])
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Cumulative Reward")
        ax2.set_title(f"Cumulative Reward: {analysis.cumulative_reward:.2f}")
        
        # 3. Metrics summary
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        metrics_text = f"""
TRAJECTORY METRICS
{'='*40}
ID: {analysis.trajectory_id}
Steps: {analysis.n_steps}

PATH GEOMETRY
  Total Length:      {analysis.total_length:.2f}
  Euclidean Length:  {analysis.euclidean_length:.2f}
  Tortuosity:        {analysis.tortuosity:.2f}

REWARD STATISTICS
  Cumulative:        {analysis.cumulative_reward:.2f}
  Trend:             {analysis.reward_trend}

HODGE ALIGNMENT
  Gradient Alignment: {analysis.mean_gradient_alignment:.3f}
  Curl Exposure:      {analysis.curl_exposure:.3f}

SAFETY
  Min Black Hole Dist: {analysis.min_black_hole_distance:.2f}
  Cliff Crossings:     {analysis.n_cliff_crossings}
  Safety Score:        {analysis.safety_score:.2f}
"""
        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Region flow diagram
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if analysis.regions_visited:
            # Simple flow visualization
            regions = analysis.regions_visited[:8]  # Limit for readability
            n_regions = len(regions)
            
            for i, region in enumerate(regions):
                y = 0.9 - i * 0.1
                ax4.text(0.5, y, region[:20], ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
                
                if i < n_regions - 1:
                    ax4.annotate('', xy=(0.5, y - 0.03), xytext=(0.5, y - 0.07),
                               arrowprops=dict(arrowstyle='->', color='gray'))
            
            ax4.set_title("Region Progression")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def create_summary_dashboard(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive summary dashboard.
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Project embeddings
        if self.analyzer.embeddings is None or len(self.analyzer.embeddings) < 3:
            return fig
        
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(self.analyzer.embeddings)
        features = self.analyzer.extract_features()
        
        # 1. Main manifold plot (large)
        ax1 = fig.add_subplot(2, 2, 1)
        scatter = ax1.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=self.analyzer.rewards,
            cmap="RdYlGn",
            s=80,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax1, label="Reward")
        
        # Trajectory line
        ax1.plot(coords_2d[:, 0], coords_2d[:, 1],
                color=self.colors["trajectory"], alpha=0.4, linewidth=1)
        
        # Black holes
        for idx in self.analyzer.black_hole_indices:
            ax1.scatter(coords_2d[idx, 0], coords_2d[idx, 1],
                       c=self.colors["black_hole"], s=200, marker='X', zorder=10)
            circle = Circle((coords_2d[idx, 0], coords_2d[idx, 1]), 0.3,
                           fill=False, color=self.colors["black_hole"], linewidth=2, linestyle='--')
            ax1.add_patch(circle)
        
        # Hodge gradient
        if self.analyzer.hodge_gradient is not None:
            grad_2d = pca.transform(self.analyzer.hodge_gradient.reshape(1, -1))[0]
            center = coords_2d.mean(axis=0)
            scale = np.std(coords_2d) * 1.5
            ax1.arrow(center[0], center[1], scale * grad_2d[0], scale * grad_2d[1],
                     head_width=0.15, head_length=0.08,
                     fc=self.colors["gradient"], ec=self.colors["gradient"], linewidth=3)
        
        ax1.set_title("Semantic Reward Manifold", fontsize=12, fontweight='bold')
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        
        # 2. Hodge decomposition bars
        ax2 = fig.add_subplot(2, 2, 2)
        components = ['∇φ\n(Gradient)', '∇×ψ\n(Curl)', 'h\n(Harmonic)']
        magnitudes = [features.gradient_magnitude, features.curl_magnitude, features.harmonic_magnitude]
        colors = [self.colors["gradient"], self.colors["curl"], self.colors["harmonic"]]
        
        bars = ax2.bar(components, magnitudes, color=colors, edgecolor='white', linewidth=2)
        for bar, val in zip(bars, magnitudes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=10)
        
        ax2.set_ylabel("Magnitude")
        ax2.set_title("Hodge Decomposition", fontsize=12, fontweight='bold')
        
        # 3. Feature summary table
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.axis('off')
        
        summary_text = f"""
{'═'*45}
         TOPOLOGICAL ANALYSIS SUMMARY
{'═'*45}

MANIFOLD STRUCTURE
  Points:          {features.n_points}
  Dimension:       {features.embedding_dim}
  Clusters:        {features.n_clusters}
  
HODGE COHOMOLOGY
  H¹ Magnitude:    {features.h1_cohomology:.4f}
  Status:          {'✓ Consistent' if features.h1_cohomology < 0.1 else '⚠ Inconsistent'}
  
GEOMETRY
  Mean Curvature:  {features.mean_curvature:.4f}
  Max Curvature:   {features.max_curvature:.4f}
  
SAFETY
  Black Holes:     {features.n_black_holes}
  Cliffs:          {features.n_cliffs}
  Safe Fraction:   {features.safe_region_fraction:.1%}

CONNECTIVITY
  Components:      {features.n_connected_components}
  Graph Density:   {features.graph_density:.4f}
{'═'*45}
"""
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
        
        # 4. Cluster legend
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        region_text = "SEMANTIC REGIONS\n" + "─" * 30 + "\n\n"
        for i, region in enumerate(self.analyzer.regions[:6]):  # Limit to 6
            safety_icon = "🚫" if region.is_black_hole else "⚠️" if region.is_cliff_region else "✓"
            region_text += f"{safety_icon} {region.label[:25]}\n"
            region_text += f"   Points: {region.n_points}, Reward: {region.mean_reward:.2f}\n\n"
        
        ax4.text(0.05, 0.95, region_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
        
        plt.tight_layout()
        fig.suptitle("Embedding Topology Analysis Dashboard", fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


def demo_visualization():
    """Run visualization demo with sample data."""
    print("=" * 60)
    print("EMBEDDING TOPOLOGY VISUALIZATION DEMO")
    print("=" * 60)
    
    # Mock embedding model
    class MockEmbedder:
        def encode(self, texts):
            np.random.seed(42)
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(64)
                embeddings.append(emb / np.linalg.norm(emb))
            return np.array(embeddings)
    
    # Sample trajectory with rich semantic content
    texts = [
        "Agent starts in a safe room with a closed door",
        "Agent examines the door and finds a key",
        "Agent uses key to unlock the door",
        "Agent enters a dark corridor",
        "Agent hears ominous sounds ahead",
        "Agent proceeds cautiously forward",
        "Agent discovers a treasure room",
        "Agent collects valuable artifacts",
        "Agent triggers a hidden trap",
        "Agent barely escapes the trap",
        "Agent finds an alternative exit",
        "Agent emerges in sunlight",
        "Agent reaches the goal safely",
    ]
    
    # Rewards with clear structure
    rewards = [0.1, 0.2, 0.3, 0.1, -0.1, 0.0, 0.6, 0.8, -0.4, -0.2, 0.3, 0.5, 1.0]
    actions = [f"action_{i}" for i in range(len(texts))]
    
    # Initialize analyzer
    embedder = MockEmbedder()
    analyzer = EmbeddingTopologyAnalyzer(
        embedding_model=embedder,
        n_clusters=4,
        black_hole_threshold=-0.2,
    )
    
    embeddings = embedder.encode(texts)
    analyzer.fit(
        states=list(embeddings),
        actions=actions,
        rewards=rewards,
        texts=texts,
    )
    
    # Create visualizer
    visualizer = EmbeddingTopologyVisualizer(analyzer)
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    visualizer.plot_hodge_decomposition_2d(save_path="hodge_decomposition_2d.png")
    visualizer.plot_hodge_decomposition_3d(save_path="hodge_decomposition_3d.png")
    visualizer.plot_consistency_analysis(save_path="consistency_analysis.png")
    visualizer.plot_trajectory_analysis(
        trajectory_indices=list(range(len(texts))),
        trajectory_id="demo_trajectory",
        save_path="trajectory_analysis.png",
    )
    visualizer.create_summary_dashboard(save_path="topology_dashboard.png")
    
    print("\nAll visualizations saved!")
    print("  - hodge_decomposition_2d.png")
    print("  - hodge_decomposition_3d.png")
    print("  - consistency_analysis.png")
    print("  - trajectory_analysis.png")
    print("  - topology_dashboard.png")
    
    # Print feature summary
    features = analyzer.extract_features()
    print("\n" + features.summary())
    
    plt.show()


if __name__ == "__main__":
    demo_visualization()
