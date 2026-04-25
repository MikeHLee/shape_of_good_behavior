"""
3D Reward Manifold Visualization

Creates beautiful 3D visualizations of high-dimensional reward spaces:
1. Surface plots of reward manifolds projected to 3D
2. Locally connected point networks showing neighborhood structure
3. Hodge decomposition flow fields on the manifold
4. Black hole event horizons as singularities
5. Geodesic paths and policy trajectories

Mathematical Intuition:
- High-dimensional reward embeddings are projected to 3D via PCA/UMAP
- The reward value becomes the "height" (z-axis) of the manifold surface
- Local connectivity is shown as edges between nearby points
- The surface is interpolated using RBF or triangulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.interpolate import griddata, Rbf
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional, Tuple, Union
import warnings

from src.embedding_topology_analyzer import EmbeddingTopologyAnalyzer


class RewardManifold3D:
    """
    3D visualization of high-dimensional reward manifolds.
    
    The core idea: embed rewards not as scalars, but as points in a
    high-dimensional space. Then project to 3D for visualization while
    preserving local structure.
    """
    
    def __init__(
        self,
        analyzer: EmbeddingTopologyAnalyzer,
        projection_method: str = "pca",
        surface_resolution: int = 50,
        connectivity_k: int = 6,
    ):
        """
        Args:
            analyzer: EmbeddingTopologyAnalyzer with fitted data
            projection_method: "pca" or "custom"
            surface_resolution: Grid resolution for surface interpolation
            connectivity_k: Number of neighbors for local connectivity
        """
        self.analyzer = analyzer
        self.projection_method = projection_method
        self.surface_resolution = surface_resolution
        self.connectivity_k = connectivity_k
        
        # Cached projections
        self._coords_3d: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None
        
        # Color schemes
        self.reward_cmap = self._create_reward_colormap()
        self.colors = {
            "gradient": "#2ecc71",
            "curl": "#e74c3c",
            "harmonic": "#3498db",
            "black_hole": "#1a1a2e",
            "safe": "#27ae60",
            "trajectory": "#9b59b6",
            "network_edge": "#34495e",
            "surface": "viridis",
        }
    
    def _create_reward_colormap(self) -> LinearSegmentedColormap:
        """Create a custom colormap: red (bad) -> yellow -> green (good)."""
        colors = [
            (0.0, "#8B0000"),   # Dark red (very bad)
            (0.25, "#FF4500"),  # Orange red
            (0.5, "#FFD700"),   # Gold (neutral)
            (0.75, "#32CD32"),  # Lime green
            (1.0, "#006400"),   # Dark green (very good)
        ]
        return LinearSegmentedColormap.from_list(
            "reward_manifold",
            [(pos, color) for pos, color in colors]
        )
    
    def _project_to_3d(self) -> np.ndarray:
        """Project high-dimensional embeddings to 3D."""
        if self._coords_3d is not None:
            return self._coords_3d
        
        if self.analyzer.embeddings is None or len(self.analyzer.embeddings) < 3:
            return np.zeros((0, 3))
        
        self._pca = PCA(n_components=3)
        self._coords_3d = self._pca.fit_transform(self.analyzer.embeddings)
        return self._coords_3d
    
    def _build_connectivity_graph(self) -> List[Tuple[int, int]]:
        """Build local connectivity edges using k-nearest neighbors."""
        coords = self._project_to_3d()
        if len(coords) < 2:
            return []
        
        k = min(self.connectivity_k, len(coords) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)  # +1 for self
        nn.fit(coords)
        
        edges = []
        _, indices = nn.kneighbors(coords)
        
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self (index 0)
                if i < j:  # Avoid duplicates
                    edges.append((i, j))
        
        return edges
    
    def plot_reward_surface(
        self,
        ax: Optional[plt.Axes] = None,
        method: str = "triangulation",
        alpha: float = 0.8,
        show_points: bool = True,
        show_colorbar: bool = True,
    ) -> plt.Axes:
        """
        Plot the reward manifold as a 3D surface.
        
        Args:
            ax: Matplotlib 3D axes (created if None)
            method: "triangulation", "rbf", or "griddata"
            alpha: Surface transparency
            show_points: Whether to show data points
            show_colorbar: Whether to show colorbar
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        coords = self._project_to_3d()
        if len(coords) < 4:
            return ax
        
        rewards = np.array(self.analyzer.rewards)
        
        # Normalize rewards for coloring
        norm = Normalize(vmin=rewards.min(), vmax=rewards.max())
        
        if method == "triangulation":
            self._plot_triangulated_surface(ax, coords, rewards, norm, alpha)
        elif method == "rbf":
            self._plot_rbf_surface(ax, coords, rewards, norm, alpha)
        else:
            self._plot_griddata_surface(ax, coords, rewards, norm, alpha)
        
        # Plot actual data points
        if show_points:
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=rewards,
                cmap=self.reward_cmap,
                s=60,
                alpha=0.9,
                edgecolors='white',
                linewidth=0.5,
                zorder=10,
            )
            
            if show_colorbar:
                plt.colorbar(scatter, ax=ax, label="Reward", shrink=0.6, pad=0.1)
        
        # Mark black holes
        for idx in self.analyzer.black_hole_indices:
            ax.scatter(
                [coords[idx, 0]], [coords[idx, 1]], [coords[idx, 2]],
                c=self.colors["black_hole"],
                s=200,
                marker='X',
                zorder=20,
            )
            # Draw event horizon sphere
            self._draw_sphere(
                ax, coords[idx], radius=0.3, 
                color=self.colors["black_hole"], alpha=0.2
            )
        
        ax.set_xlabel("PC1 (Primary variation)", fontsize=10)
        ax.set_ylabel("PC2 (Secondary variation)", fontsize=10)
        ax.set_zlabel("PC3 (Tertiary variation)", fontsize=10)
        ax.set_title("Reward Manifold Surface", fontsize=12, fontweight='bold')
        
        return ax
    
    def _plot_triangulated_surface(
        self,
        ax: plt.Axes,
        coords: np.ndarray,
        rewards: np.ndarray,
        norm: Normalize,
        alpha: float,
    ):
        """Create surface via Delaunay triangulation."""
        try:
            # Use 2D projection for triangulation
            tri = Delaunay(coords[:, :2])
            
            # Create triangular mesh
            for simplex in tri.simplices:
                triangle = coords[simplex]
                triangle_rewards = rewards[simplex]
                avg_reward = np.mean(triangle_rewards)
                
                color = self.reward_cmap(norm(avg_reward))
                
                poly = Poly3DCollection(
                    [triangle],
                    alpha=alpha * 0.6,
                    facecolor=color,
                    edgecolor='gray',
                    linewidth=0.3,
                )
                ax.add_collection3d(poly)
        except Exception as e:
            warnings.warn(f"Triangulation failed: {e}")
    
    def _plot_rbf_surface(
        self,
        ax: plt.Axes,
        coords: np.ndarray,
        rewards: np.ndarray,
        norm: Normalize,
        alpha: float,
    ):
        """Create smooth surface via RBF interpolation."""
        try:
            # Create grid
            x_range = np.linspace(coords[:, 0].min(), coords[:, 0].max(), self.surface_resolution)
            y_range = np.linspace(coords[:, 1].min(), coords[:, 1].max(), self.surface_resolution)
            X, Y = np.meshgrid(x_range, y_range)
            
            # RBF interpolation for z (PC3) and reward
            rbf_z = Rbf(coords[:, 0], coords[:, 1], coords[:, 2], function='thin_plate')
            rbf_r = Rbf(coords[:, 0], coords[:, 1], rewards, function='thin_plate')
            
            Z = rbf_z(X, Y)
            R = rbf_r(X, Y)
            
            # Plot surface colored by reward
            surf = ax.plot_surface(
                X, Y, Z,
                facecolors=self.reward_cmap(norm(R)),
                alpha=alpha,
                rstride=1, cstride=1,
                linewidth=0,
                antialiased=True,
            )
        except Exception as e:
            warnings.warn(f"RBF interpolation failed: {e}")
    
    def _plot_griddata_surface(
        self,
        ax: plt.Axes,
        coords: np.ndarray,
        rewards: np.ndarray,
        norm: Normalize,
        alpha: float,
    ):
        """Create surface via griddata interpolation."""
        try:
            # Create grid
            x_range = np.linspace(coords[:, 0].min(), coords[:, 0].max(), self.surface_resolution)
            y_range = np.linspace(coords[:, 1].min(), coords[:, 1].max(), self.surface_resolution)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Interpolate
            Z = griddata(coords[:, :2], coords[:, 2], (X, Y), method='cubic')
            R = griddata(coords[:, :2], rewards, (X, Y), method='cubic')
            
            # Plot surface
            ax.plot_surface(
                X, Y, Z,
                facecolors=self.reward_cmap(norm(np.nan_to_num(R, nan=0.5))),
                alpha=alpha,
                rstride=1, cstride=1,
                linewidth=0,
                antialiased=True,
            )
        except Exception as e:
            warnings.warn(f"Griddata interpolation failed: {e}")
    
    def _draw_sphere(
        self,
        ax: plt.Axes,
        center: np.ndarray,
        radius: float,
        color: str,
        alpha: float = 0.3,
    ):
        """Draw a wireframe sphere (event horizon)."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)
    
    def plot_connectivity_network(
        self,
        ax: Optional[plt.Axes] = None,
        show_surface: bool = True,
        edge_alpha: float = 0.4,
        highlight_trajectory: bool = True,
    ) -> plt.Axes:
        """
        Plot the local connectivity network over the manifold.
        
        Shows how points are locally connected, revealing the 
        neighborhood structure of the reward space.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        coords = self._project_to_3d()
        if len(coords) < 2:
            return ax
        
        rewards = np.array(self.analyzer.rewards)
        norm = Normalize(vmin=rewards.min(), vmax=rewards.max())
        
        # Plot semi-transparent surface
        if show_surface:
            self._plot_triangulated_surface(ax, coords, rewards, norm, alpha=0.3)
        
        # Build and plot connectivity edges
        edges = self._build_connectivity_graph()
        
        edge_colors = []
        edge_segments = []
        
        for i, j in edges:
            segment = [coords[i], coords[j]]
            edge_segments.append(segment)
            
            # Color edge by average reward
            avg_reward = (rewards[i] + rewards[j]) / 2
            edge_colors.append(self.reward_cmap(norm(avg_reward)))
        
        edge_collection = Line3DCollection(
            edge_segments,
            colors=edge_colors,
            alpha=edge_alpha,
            linewidths=1.0,
        )
        ax.add_collection3d(edge_collection)
        
        # Plot points
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=rewards,
            cmap=self.reward_cmap,
            s=80,
            alpha=0.9,
            edgecolors='white',
            linewidth=1,
            zorder=10,
        )
        
        # Highlight trajectory path
        if highlight_trajectory:
            traj_segments = []
            for i in range(len(coords) - 1):
                traj_segments.append([coords[i], coords[i + 1]])
            
            traj_collection = Line3DCollection(
                traj_segments,
                colors=self.colors["trajectory"],
                alpha=0.8,
                linewidths=2.5,
                zorder=15,
            )
            ax.add_collection3d(traj_collection)
        
        plt.colorbar(scatter, ax=ax, label="Reward", shrink=0.6, pad=0.1)
        
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_zlabel("PC3", fontsize=10)
        ax.set_title("Local Connectivity Network", fontsize=12, fontweight='bold')
        
        return ax
    
    def plot_hodge_flow_field(
        self,
        ax: Optional[plt.Axes] = None,
        show_surface: bool = True,
        arrow_scale: float = 0.5,
    ) -> plt.Axes:
        """
        Plot Hodge decomposition as a flow field on the manifold.
        
        Shows:
        - Green arrows: Gradient direction (follow for improvement)
        - Red arrows: Curl direction (inconsistency)
        - Blue arrows: Harmonic direction (global structure)
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        coords = self._project_to_3d()
        if len(coords) < 2:
            return ax
        
        rewards = np.array(self.analyzer.rewards)
        norm = Normalize(vmin=rewards.min(), vmax=rewards.max())
        
        # Plot surface
        if show_surface:
            self._plot_triangulated_surface(ax, coords, rewards, norm, alpha=0.4)
        
        # Plot points
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=rewards,
            cmap=self.reward_cmap,
            s=50,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
        )
        
        center = coords.mean(axis=0)
        scale = np.std(coords) * arrow_scale
        
        # Project Hodge components to 3D
        if self._pca is not None:
            if self.analyzer.hodge_gradient is not None:
                grad_3d = self._pca.transform(
                    self.analyzer.hodge_gradient.reshape(1, -1)
                )[0]
                grad_3d = grad_3d / (np.linalg.norm(grad_3d) + 1e-8) * scale
                
                ax.quiver(
                    center[0], center[1], center[2],
                    grad_3d[0], grad_3d[1], grad_3d[2],
                    color=self.colors["gradient"],
                    linewidth=3,
                    arrow_length_ratio=0.2,
                    label="Gradient (∇φ)",
                )
            
            if self.analyzer.hodge_curl is not None:
                curl_3d = self._pca.transform(
                    self.analyzer.hodge_curl.reshape(1, -1)
                )[0]
                curl_3d = curl_3d / (np.linalg.norm(curl_3d) + 1e-8) * scale
                
                ax.quiver(
                    center[0], center[1], center[2],
                    curl_3d[0], curl_3d[1], curl_3d[2],
                    color=self.colors["curl"],
                    linewidth=3,
                    arrow_length_ratio=0.2,
                    label="Curl (H¹)",
                )
            
            if self.analyzer.hodge_harmonic is not None:
                harm_3d = self._pca.transform(
                    self.analyzer.hodge_harmonic.reshape(1, -1)
                )[0]
                harm_3d = harm_3d / (np.linalg.norm(harm_3d) + 1e-8) * scale * 0.7
                
                ax.quiver(
                    center[0], center[1], center[2],
                    harm_3d[0], harm_3d[1], harm_3d[2],
                    color=self.colors["harmonic"],
                    linewidth=3,
                    arrow_length_ratio=0.2,
                    label="Harmonic (h)",
                )
        
        ax.legend(loc='upper left')
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_zlabel("PC3", fontsize=10)
        ax.set_title("Hodge Flow Field on Reward Manifold", fontsize=12, fontweight='bold')
        
        return ax
    
    def plot_black_hole_geometry(
        self,
        ax: Optional[plt.Axes] = None,
        event_horizon_scale: float = 0.5,
    ) -> plt.Axes:
        """
        Visualize black hole regions as geometric singularities.
        
        Shows:
        - Black hole centers with event horizons
        - Warped geometry near singularities
        - Safe paths avoiding forbidden regions
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        coords = self._project_to_3d()
        if len(coords) < 2:
            return ax
        
        rewards = np.array(self.analyzer.rewards)
        norm = Normalize(vmin=rewards.min(), vmax=rewards.max())
        
        # Plot surface with warping near black holes
        try:
            tri = Delaunay(coords[:, :2])
            
            for simplex in tri.simplices:
                triangle = coords[simplex].copy()
                triangle_rewards = rewards[simplex]
                avg_reward = np.mean(triangle_rewards)
                
                # Warp geometry near black holes
                for idx in self.analyzer.black_hole_indices:
                    bh_center = coords[idx]
                    for i in range(3):
                        dist = np.linalg.norm(triangle[i] - bh_center)
                        if dist < 1.0 and dist > 0.1:
                            # Pull toward black hole
                            direction = bh_center - triangle[i]
                            warp_factor = 0.3 * (1.0 - dist) ** 2
                            triangle[i] += direction * warp_factor
                
                color = self.reward_cmap(norm(avg_reward))
                alpha = 0.5 if avg_reward > self.analyzer.black_hole_threshold else 0.3
                
                poly = Poly3DCollection(
                    [triangle],
                    alpha=alpha,
                    facecolor=color,
                    edgecolor='gray',
                    linewidth=0.2,
                )
                ax.add_collection3d(poly)
        except Exception:
            pass
        
        # Plot data points
        safe_mask = ~np.isin(np.arange(len(coords)), self.analyzer.black_hole_indices)
        
        ax.scatter(
            coords[safe_mask, 0], coords[safe_mask, 1], coords[safe_mask, 2],
            c=rewards[safe_mask],
            cmap=self.reward_cmap,
            s=50,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            label="Safe states",
        )
        
        # Black holes with event horizons
        for idx in self.analyzer.black_hole_indices:
            ax.scatter(
                [coords[idx, 0]], [coords[idx, 1]], [coords[idx, 2]],
                c=self.colors["black_hole"],
                s=300,
                marker='X',
                zorder=20,
                label="Black hole" if idx == self.analyzer.black_hole_indices[0] else "",
            )
            
            # Draw multiple event horizon rings
            for r in [0.2, 0.4, 0.6]:
                radius = r * event_horizon_scale
                self._draw_sphere(
                    ax, coords[idx], radius,
                    self.colors["black_hole"],
                    alpha=0.15 * (1 - r)
                )
        
        # Draw trajectory avoiding black holes
        traj_segments = []
        traj_colors = []
        for i in range(len(coords) - 1):
            traj_segments.append([coords[i], coords[i + 1]])
            
            # Color by safety
            is_near_bh = any(
                np.linalg.norm(coords[i] - coords[bh]) < event_horizon_scale
                for bh in self.analyzer.black_hole_indices
            )
            traj_colors.append('#e74c3c' if is_near_bh else '#2ecc71')
        
        traj_collection = Line3DCollection(
            traj_segments,
            colors=traj_colors,
            linewidths=2,
            alpha=0.8,
            zorder=15,
        )
        ax.add_collection3d(traj_collection)
        
        ax.legend(loc='upper left')
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_zlabel("PC3", fontsize=10)
        ax.set_title("Black Hole Geometry & Event Horizons", fontsize=12, fontweight='bold')
        
        return ax
    
    def create_manifold_gallery(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a gallery of manifold visualizations.
        
        Four views:
        1. Reward surface (triangulated)
        2. Connectivity network
        3. Hodge flow field
        4. Black hole geometry
        """
        fig = plt.figure(figsize=(16, 14))
        
        # 1. Reward surface
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self.plot_reward_surface(ax1, method="triangulation", show_colorbar=False)
        ax1.set_title("A. Reward Manifold Surface", fontsize=11, fontweight='bold')
        
        # 2. Connectivity network
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        self.plot_connectivity_network(ax2, show_surface=False)
        ax2.set_title("B. Local Connectivity Network", fontsize=11, fontweight='bold')
        
        # 3. Hodge flow field
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        self.plot_hodge_flow_field(ax3)
        ax3.set_title("C. Hodge Decomposition Flow", fontsize=11, fontweight='bold')
        
        # 4. Black hole geometry
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        self.plot_black_hole_geometry(ax4)
        ax4.set_title("D. Black Hole Event Horizons", fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        fig.suptitle(
            "3D Reward Manifold Visualization Gallery",
            fontsize=14, fontweight='bold', y=1.02
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_reward_height_surface(
        self,
        ax: Optional[plt.Axes] = None,
        use_reward_as_height: bool = True,
    ) -> plt.Axes:
        """
        Plot reward as the height dimension directly.
        
        Instead of using PC3 for height, use the actual reward value.
        This creates a "landscape" where valleys are bad and peaks are good.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        coords = self._project_to_3d()
        if len(coords) < 3:
            return ax
        
        rewards = np.array(self.analyzer.rewards)
        
        # Use PC1, PC2 for x, y and reward for z
        x = coords[:, 0]
        y = coords[:, 1]
        z = rewards if use_reward_as_height else coords[:, 2]
        
        # Create surface via triangulation
        try:
            tri = Delaunay(coords[:, :2])
            norm = Normalize(vmin=rewards.min(), vmax=rewards.max())
            
            for simplex in tri.simplices:
                triangle_xy = np.column_stack([x[simplex], y[simplex]])
                triangle_z = z[simplex]
                triangle_rewards = rewards[simplex]
                
                triangle_3d = np.column_stack([
                    triangle_xy[:, 0],
                    triangle_xy[:, 1],
                    triangle_z
                ])
                
                avg_reward = np.mean(triangle_rewards)
                color = self.reward_cmap(norm(avg_reward))
                
                poly = Poly3DCollection(
                    [triangle_3d],
                    alpha=0.7,
                    facecolor=color,
                    edgecolor='gray',
                    linewidth=0.3,
                )
                ax.add_collection3d(poly)
        except Exception as e:
            warnings.warn(f"Triangulation failed: {e}")
        
        # Plot points
        scatter = ax.scatter(
            x, y, z,
            c=rewards,
            cmap=self.reward_cmap,
            s=80,
            alpha=0.9,
            edgecolors='white',
            linewidth=1,
            zorder=10,
        )
        
        # Mark black holes as valleys
        for idx in self.analyzer.black_hole_indices:
            ax.scatter(
                [x[idx]], [y[idx]], [z[idx]],
                c=self.colors["black_hole"],
                s=250,
                marker='v',  # Downward triangle for "valley"
                zorder=20,
            )
        
        plt.colorbar(scatter, ax=ax, label="Reward (Height)", shrink=0.6, pad=0.1)
        
        ax.set_xlabel("PC1 (Semantic axis 1)", fontsize=10)
        ax.set_ylabel("PC2 (Semantic axis 2)", fontsize=10)
        z_label = "Reward Value" if use_reward_as_height else "PC3"
        ax.set_zlabel(z_label, fontsize=10)
        ax.set_title(
            "Reward Landscape (Height = Reward)",
            fontsize=12, fontweight='bold'
        )
        
        return ax
    
    def plot_geodesic_paths(
        self,
        ax: Optional[plt.Axes] = None,
        n_geodesics: int = 5,
    ) -> plt.Axes:
        """
        Visualize geodesic paths (shortest paths) on the manifold.
        
        These represent optimal policy trajectories that respect
        the geometry of the reward space.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        coords = self._project_to_3d()
        if len(coords) < 3:
            return ax
        
        rewards = np.array(self.analyzer.rewards)
        norm = Normalize(vmin=rewards.min(), vmax=rewards.max())
        
        # Plot faded surface
        self._plot_triangulated_surface(ax, coords, rewards, norm, alpha=0.3)
        
        # Plot points
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=rewards,
            cmap=self.reward_cmap,
            s=50,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
        )
        
        # Find high-reward "goal" states
        goal_indices = np.argsort(rewards)[-3:]  # Top 3 rewards
        start_indices = np.argsort(rewards)[:3]   # Bottom 3 rewards (start from bad)
        
        # Draw geodesic-like paths (straight lines in projection as approximation)
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, min(n_geodesics, len(start_indices))))
        
        for i, (start, goal) in enumerate(zip(start_indices, goal_indices)):
            if i >= n_geodesics:
                break
            
            # Simple linear interpolation as geodesic approximation
            t = np.linspace(0, 1, 20)
            path = np.outer(1 - t, coords[start]) + np.outer(t, coords[goal])
            
            ax.plot(
                path[:, 0], path[:, 1], path[:, 2],
                color=colors[i],
                linewidth=2.5,
                alpha=0.8,
                label=f"Path {i+1}" if i < 3 else None,
            )
            
            # Mark start and end
            ax.scatter(
                [coords[start, 0]], [coords[start, 1]], [coords[start, 2]],
                c='red', s=100, marker='o', zorder=15,
            )
            ax.scatter(
                [coords[goal, 0]], [coords[goal, 1]], [coords[goal, 2]],
                c='green', s=100, marker='*', zorder=15,
            )
        
        ax.legend(loc='upper left')
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_zlabel("PC3", fontsize=10)
        ax.set_title("Geodesic Paths on Reward Manifold", fontsize=12, fontweight='bold')
        
        return ax


def demo_3d_visualization():
    """Demonstrate 3D reward manifold visualization."""
    print("=" * 60)
    print("3D REWARD MANIFOLD VISUALIZATION DEMO")
    print("=" * 60)
    
    # Mock embedding model
    class MockEmbedder:
        def __init__(self, dim=64):
            self.dim = dim
        
        def encode(self, texts):
            embeddings = []
            for text in texts:
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(self.dim)
                
                # Add semantic structure
                for keyword, idx, val in [
                    ("safe", 0, 1), ("danger", 0, -1),
                    ("goal", 1, 1), ("trap", 1, -1),
                    ("reward", 2, 1), ("penalty", 2, -1),
                ]:
                    if keyword in text.lower():
                        emb[idx] += val * 0.5
                
                embeddings.append(emb / np.linalg.norm(emb))
            return np.array(embeddings)
    
    # Create rich trajectory
    texts = [
        "Agent starts in a safe room with a closed door",
        "Agent examines the door and finds a key",
        "Agent uses key to unlock the door safely",
        "Agent enters a dark corridor with danger ahead",
        "Agent hears ominous sounds, senses trap nearby",
        "Agent proceeds cautiously, avoiding the trap",
        "Agent discovers a treasure room with goal reward",
        "Agent collects valuable artifacts successfully",
        "Agent triggers a hidden trap - penalty incurred",
        "Agent barely escapes the trap with minor damage",
        "Agent finds an alternative safe exit path",
        "Agent emerges in sunlight, reaches safety",
        "Agent achieves the final goal with full reward",
        "User asks for harmful content - black hole state",
        "Agent refuses harmful request - maintains safety",
    ]
    
    rewards = [0.1, 0.2, 0.4, 0.2, -0.1, 0.3, 0.7, 0.9, -0.5, -0.2, 0.4, 0.6, 1.0, -0.9, 0.8]
    actions = [f"action_{i}" for i in range(len(texts))]
    
    # Initialize analyzer
    embedder = MockEmbedder()
    
    from src.embedding_topology_analyzer import EmbeddingTopologyAnalyzer
    
    analyzer = EmbeddingTopologyAnalyzer(
        embedding_model=embedder,
        n_clusters=4,
        black_hole_threshold=-0.4,
    )
    
    embeddings = embedder.encode(texts)
    analyzer.fit(
        states=list(embeddings),
        actions=actions,
        rewards=rewards,
        texts=texts,
    )
    
    # Create 3D visualizer
    visualizer = RewardManifold3D(analyzer, connectivity_k=4)
    
    print("\nGenerating 3D visualizations...")
    
    # Generate individual plots
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    visualizer.plot_reward_surface(ax1, method="triangulation")
    plt.savefig("reward_surface_3d.png", dpi=150, bbox_inches='tight')
    print("  - reward_surface_3d.png")
    
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    visualizer.plot_connectivity_network(ax2)
    plt.savefig("connectivity_network_3d.png", dpi=150, bbox_inches='tight')
    print("  - connectivity_network_3d.png")
    
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    visualizer.plot_reward_height_surface(ax3)
    plt.savefig("reward_landscape_3d.png", dpi=150, bbox_inches='tight')
    print("  - reward_landscape_3d.png")
    
    fig4 = plt.figure(figsize=(12, 10))
    ax4 = fig4.add_subplot(111, projection='3d')
    visualizer.plot_black_hole_geometry(ax4)
    plt.savefig("black_hole_geometry_3d.png", dpi=150, bbox_inches='tight')
    print("  - black_hole_geometry_3d.png")
    
    fig5 = plt.figure(figsize=(12, 10))
    ax5 = fig5.add_subplot(111, projection='3d')
    visualizer.plot_geodesic_paths(ax5)
    plt.savefig("geodesic_paths_3d.png", dpi=150, bbox_inches='tight')
    print("  - geodesic_paths_3d.png")
    
    # Create gallery
    visualizer.create_manifold_gallery(save_path="manifold_gallery_3d.png")
    print("  - manifold_gallery_3d.png")
    
    print("\nAll 3D visualizations saved!")
    plt.show()


if __name__ == "__main__":
    demo_3d_visualization()
