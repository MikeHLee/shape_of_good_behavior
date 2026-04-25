"""Unified 3D Plotly visualizations across all three tracks.

Generates 5 figures:
1. Constitutional Gradient Field (Track 3)
2. Alignment Boundary Topology (Track 3)
3. SGPO Trajectory Avoidance (Track 2)
4. Hodge Decomposition of Exploit Preferences (Track 1)
5. Hodge Filtering Comparison (Track 1)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from .config import PipelineConfig, CONSTRAINT_GEOMETRY_SRC
from .preference_mapper import MappingResult, DangerRegionSpec, EmbeddingPair
from .hodge_analysis import H1AnalysisResult, MultiSeedResult

logger = logging.getLogger(__name__)


class AlignmentVisualizer:
    """3D Plotly visualizations for cross-track reward hacking analysis."""

    def __init__(
        self,
        config: PipelineConfig,
        mapping: MappingResult,
        h1_result: Optional[H1AnalysisResult] = None,
        multi_seed: Optional[MultiSeedResult] = None,
    ):
        self.config = config
        self.mapping = mapping
        self.h1_result = h1_result
        self.multi_seed = multi_seed
        self.figures_dir = Path(config.figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Precompute 3D PCA projections
        self._pca3d = None
        self._exploit_3d = None
        self._ideal_3d = None

    def _compute_3d_projections(self):
        """PCA-reduce embeddings to 3D for visualization."""
        if self._exploit_3d is not None:
            return

        combined = np.vstack([
            self.mapping.exploit_embeddings,
            self.mapping.ideal_embeddings,
        ])
        self._pca3d = PCA(n_components=3, random_state=self.config.seed)
        combined_3d = self._pca3d.fit_transform(combined)
        n = len(self.mapping.exploit_embeddings)
        self._exploit_3d = combined_3d[:n]
        self._ideal_3d = combined_3d[n:]

    def _get_category_colors(self) -> Dict[str, str]:
        """Assign consistent colors to categories."""
        categories = sorted(set(ep.category for ep in self.mapping.embedding_pairs))
        palette = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
            "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
            "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
        ]
        return {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    def fig1_constitutional_gradient_field(self) -> str:
        """Track 3: Exploit points (red) + ideal points (blue) with gradient arrows."""
        import plotly.graph_objects as go

        self._compute_3d_projections()
        cat_colors = self._get_category_colors()

        fig = go.Figure()

        # Group by category
        categories: Dict[str, List[int]] = {}
        for i, ep in enumerate(self.mapping.embedding_pairs):
            categories.setdefault(ep.category, []).append(i)

        for cat, indices in categories.items():
            idx = np.array(indices)
            color = cat_colors.get(cat, "#999999")

            # Exploit points
            fig.add_trace(go.Scatter3d(
                x=self._exploit_3d[idx, 0],
                y=self._exploit_3d[idx, 1],
                z=self._exploit_3d[idx, 2],
                mode="markers",
                marker=dict(size=3, color="red", opacity=0.4),
                name=f"{cat} (exploit)",
                legendgroup=cat,
                showlegend=True,
            ))

            # Ideal points
            fig.add_trace(go.Scatter3d(
                x=self._ideal_3d[idx, 0],
                y=self._ideal_3d[idx, 1],
                z=self._ideal_3d[idx, 2],
                mode="markers",
                marker=dict(size=3, color="blue", opacity=0.4),
                name=f"{cat} (ideal)",
                legendgroup=cat,
                showlegend=False,
            ))

            # Gradient arrows (subsample for clarity)
            n_arrows = min(20, len(idx))
            arrow_idx = idx[np.linspace(0, len(idx) - 1, n_arrows, dtype=int)]
            for ai in arrow_idx:
                fig.add_trace(go.Scatter3d(
                    x=[self._exploit_3d[ai, 0], self._ideal_3d[ai, 0]],
                    y=[self._exploit_3d[ai, 1], self._ideal_3d[ai, 1]],
                    z=[self._exploit_3d[ai, 2], self._ideal_3d[ai, 2]],
                    mode="lines",
                    line=dict(color=color, width=1),
                    showlegend=False,
                ))

        # Per-principle mean gradient vectors as thick arrows
        for principle, grad in self.mapping.constitutional_gradients.items():
            if not principle.startswith("category:"):
                continue
            grad_3d = self._pca3d.transform(grad.reshape(1, -1))[0]
            origin = np.mean(self._exploit_3d, axis=0)
            fig.add_trace(go.Scatter3d(
                x=[origin[0], origin[0] + grad_3d[0] * 3],
                y=[origin[1], origin[1] + grad_3d[1] * 3],
                z=[origin[2], origin[2] + grad_3d[2] * 3],
                mode="lines+text",
                line=dict(color="black", width=4),
                text=["", principle.replace("category:", "")],
                textposition="top center",
                name=f"Gradient: {principle.replace('category:', '')}",
                showlegend=True,
            ))

        fig.update_layout(
            title="Constitutional Gradient Field (Track 3)",
            width=self.config.figure_width,
            height=self.config.figure_height,
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
        )

        path = str(self.figures_dir / "fig1_constitutional_gradient_field.html")
        fig.write_html(path)
        logger.info(f"Saved Fig 1: {path}")
        return path

    def fig2_alignment_boundary_topology(self) -> str:
        """Track 3: Decision boundary + black holes as dark spheres."""
        import plotly.graph_objects as go

        self._compute_3d_projections()

        fig = go.Figure()

        # Exploit and ideal points
        fig.add_trace(go.Scatter3d(
            x=self._exploit_3d[:, 0],
            y=self._exploit_3d[:, 1],
            z=self._exploit_3d[:, 2],
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.5),
            name="Exploits",
        ))
        fig.add_trace(go.Scatter3d(
            x=self._ideal_3d[:, 0],
            y=self._ideal_3d[:, 1],
            z=self._ideal_3d[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.5),
            name="Ideals",
        ))

        # Danger regions as dark spheres (black holes)
        for dr in self.mapping.danger_regions:
            # Project center to 3D
            center_full = np.zeros(self.mapping.exploit_embeddings.shape[1])
            center_full[: len(dr.center)] = dr.center
            center_3d = self._pca3d.transform(center_full.reshape(1, -1))[0]

            # Create sphere mesh
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            r = dr.radius * 0.5  # Scale for visibility
            x = center_3d[0] + r * np.outer(np.cos(u), np.sin(v))
            y = center_3d[1] + r * np.outer(np.sin(u), np.sin(v))
            z = center_3d[2] + r * np.outer(np.ones_like(u), np.cos(v))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, "rgb(20,20,20)"], [1, "rgb(60,0,0)"]],
                opacity=0.3,
                showscale=False,
                name=f"Danger: {dr.category}",
            ))

        # Decision boundary via midpoints
        midpoints = (self._exploit_3d + self._ideal_3d) / 2
        fig.add_trace(go.Scatter3d(
            x=midpoints[:, 0],
            y=midpoints[:, 1],
            z=midpoints[:, 2],
            mode="markers",
            marker=dict(size=2, color="green", opacity=0.3),
            name="Boundary (midpoints)",
        ))

        fig.update_layout(
            title="Alignment Boundary Topology (Track 3)",
            width=self.config.figure_width,
            height=self.config.figure_height,
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
        )

        path = str(self.figures_dir / "fig2_alignment_boundary_topology.html")
        fig.write_html(path)
        logger.info(f"Saved Fig 2: {path}")
        return path

    def fig3_sgpo_trajectory_avoidance(self) -> str:
        """Track 2: Danger regions as conformal metric heatmap + geodesic paths.

        Uses DangerRegion from conformal_safety.py and renders avoidance paths.
        """
        import plotly.graph_objects as go

        self._compute_3d_projections()

        # Set up conformal safety metric from danger regions
        src_str = str(CONSTRAINT_GEOMETRY_SRC)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

        try:
            from conformal_safety import ConformalSafetyMetric
        except ImportError:
            logger.warning("conformal_safety not importable, using basic visualization")
            return self._fig3_basic()

        safety_metric = ConformalSafetyMetric(base_sigma=0.0)
        for dr in self.mapping.danger_regions:
            safety_metric.add_danger_region(
                center=dr.center,
                radius=dr.radius,
                sharpness=2.0,
                name=dr.category,
            )

        fig = go.Figure()

        # Render danger regions
        for dr in self.mapping.danger_regions:
            center_full = np.zeros(self.mapping.exploit_embeddings.shape[1])
            center_full[: len(dr.center)] = dr.center
            center_3d = self._pca3d.transform(center_full.reshape(1, -1))[0]

            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            r = dr.radius * 0.5
            x = center_3d[0] + r * np.outer(np.cos(u), np.sin(v))
            y = center_3d[1] + r * np.outer(np.sin(u), np.sin(v))
            z = center_3d[2] + r * np.outer(np.ones_like(u), np.cos(v))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, "rgb(139,0,0)"], [1, "rgb(255,69,0)"]],
                opacity=0.25,
                showscale=False,
                name=f"Danger: {dr.category}",
            ))

        # Generate sample geodesic-like trajectories (straight-line + avoidance)
        np.random.seed(self.config.seed)
        n_trajectories = 5
        for t in range(n_trajectories):
            # Pick random start (safe region) and goal (safe region)
            start_idx = np.random.randint(len(self._ideal_3d))
            goal_idx = np.random.randint(len(self._ideal_3d))
            start = self._ideal_3d[start_idx]
            goal = self._ideal_3d[goal_idx]

            # Simulate trajectory with simple avoidance
            trajectory = self._simulate_avoidance_trajectory(start, goal, safety_metric)

            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode="lines+markers",
                marker=dict(size=2),
                line=dict(width=3),
                name=f"SGPO trajectory {t + 1}",
            ))

        fig.update_layout(
            title="SGPO Trajectory Avoidance (Track 2)",
            width=self.config.figure_width,
            height=self.config.figure_height,
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
        )

        path = str(self.figures_dir / "fig3_sgpo_trajectory_avoidance.html")
        fig.write_html(path)
        logger.info(f"Saved Fig 3: {path}")
        return path

    def _fig3_basic(self) -> str:
        """Fallback Fig 3 without conformal_safety imports."""
        import plotly.graph_objects as go

        self._compute_3d_projections()
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=self._ideal_3d[:, 0],
            y=self._ideal_3d[:, 1],
            z=self._ideal_3d[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.5),
            name="Safe states",
        ))
        fig.add_trace(go.Scatter3d(
            x=self._exploit_3d[:, 0],
            y=self._exploit_3d[:, 1],
            z=self._exploit_3d[:, 2],
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.5),
            name="Dangerous states",
        ))

        fig.update_layout(
            title="SGPO Trajectory Avoidance (Track 2) — Basic",
            width=self.config.figure_width,
            height=self.config.figure_height,
        )

        path = str(self.figures_dir / "fig3_sgpo_trajectory_avoidance.html")
        fig.write_html(path)
        return path

    def _simulate_avoidance_trajectory(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        safety_metric,
        n_steps: int = 50,
    ) -> np.ndarray:
        """Simulate a geodesic-like trajectory that avoids danger regions.

        Uses simple gradient-based avoidance in 3D projection space.
        """
        trajectory = [start.copy()]
        current = start.copy()

        for step in range(n_steps):
            # Direction toward goal
            direction = goal - current
            dist = np.linalg.norm(direction)
            if dist < 0.05:
                break
            direction = direction / dist

            # Check if next step would be in danger region
            step_size = dist / (n_steps - step)
            next_pos = current + direction * step_size

            # Simple repulsion from danger region centers
            for dr in self.mapping.danger_regions:
                center_full = np.zeros(self.mapping.exploit_embeddings.shape[1])
                center_full[: len(dr.center)] = dr.center
                center_3d = self._pca3d.transform(center_full.reshape(1, -1))[0]

                to_center = next_pos - center_3d
                dist_to_center = np.linalg.norm(to_center)
                if dist_to_center < dr.radius * 0.8 and dist_to_center > 1e-6:
                    # Push away from danger
                    repulsion = to_center / dist_to_center
                    next_pos += repulsion * step_size * 0.5

            current = next_pos
            trajectory.append(current.copy())

        trajectory.append(goal.copy())
        return np.array(trajectory)

    def fig4_hodge_decomposition(self) -> str:
        """Track 1: Hodge decomposition components on the preference graph."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self.h1_result is None:
            logger.warning("No H1 result available, skipping Fig 4")
            return ""

        decomp = self.h1_result.decomposition

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Gradient Component", "Harmonic (H1) Component", "Per-Category H1"],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        )

        # Component magnitudes
        gradient_mag = float(np.linalg.norm(decomp.gradient))
        harmonic_mag = float(np.linalg.norm(decomp.harmonic))
        total_mag = float(np.linalg.norm(decomp.original))

        fig.add_trace(
            go.Bar(
                x=["Gradient", "Harmonic", "Total"],
                y=[gradient_mag, harmonic_mag, total_mag],
                marker_color=["#377eb8", "#e41a1c", "#4daf4a"],
                name="Component Magnitudes",
            ),
            row=1,
            col=1,
        )

        # Harmonic component edge weights (cyclic inconsistencies)
        n_show = min(50, len(decomp.harmonic))
        fig.add_trace(
            go.Bar(
                x=list(range(n_show)),
                y=np.abs(decomp.harmonic[:n_show]),
                marker_color="#e41a1c",
                name="Harmonic edge weights",
            ),
            row=1,
            col=2,
        )

        # Per-category H1
        cats = list(self.h1_result.h1_per_category.keys())
        h1_vals = [self.h1_result.h1_per_category[c] for c in cats]
        fig.add_trace(
            go.Bar(
                x=cats,
                y=h1_vals,
                marker_color="#984ea3",
                name="Per-category H1",
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            title=f"Hodge Decomposition of Exploit Preferences (Track 1) — "
            f"Overall H1: {self.h1_result.h1_overall:.4f}",
            width=self.config.figure_width * 1.5,
            height=self.config.figure_height,
            showlegend=False,
        )

        path = str(self.figures_dir / "fig4_hodge_decomposition.html")
        fig.write_html(path)
        logger.info(f"Saved Fig 4: {path}")
        return path

    def fig5_hodge_filtering_comparison(self) -> str:
        """Track 1: Standard vs Hodge-filtered reward model comparison."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self.multi_seed is None:
            logger.warning("No multi-seed result, skipping Fig 5")
            return ""

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Exploit Resistance by Seed",
                "Distribution Comparison",
            ],
        )

        seeds = list(range(len(self.multi_seed.seed_results)))
        standard_vals = [r.standard_resistance for r in self.multi_seed.seed_results]
        hodge_vals = [r.hodge_resistance for r in self.multi_seed.seed_results]

        # Seed-by-seed comparison
        fig.add_trace(
            go.Scatter(
                x=seeds,
                y=standard_vals,
                mode="lines+markers",
                name="Standard RM",
                line=dict(color="#e41a1c"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=seeds,
                y=hodge_vals,
                mode="lines+markers",
                name="Hodge-filtered RM",
                line=dict(color="#377eb8"),
            ),
            row=1,
            col=1,
        )

        # Distribution comparison (box plots)
        fig.add_trace(
            go.Box(y=standard_vals, name="Standard RM", marker_color="#e41a1c"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Box(y=hodge_vals, name="Hodge-filtered RM", marker_color="#377eb8"),
            row=1,
            col=2,
        )

        comparison = self.multi_seed.comparison
        fig.update_layout(
            title=(
                f"Hodge Filtering Comparison — "
                f"Cohen's d: {comparison.get('cohens_d', 0):.3f} "
                f"({comparison.get('effect_size', 'N/A')}), "
                f"p={comparison.get('welch_t', {}).get('p_value', 1.0):.4f}"
            ),
            width=self.config.figure_width * 1.2,
            height=self.config.figure_height,
        )

        path = str(self.figures_dir / "fig5_hodge_filtering_comparison.html")
        fig.write_html(path)
        logger.info(f"Saved Fig 5: {path}")
        return path

    def generate_all(self) -> Dict[str, str]:
        """Generate all 5 figures and return paths."""
        paths = {}

        paths["fig1"] = self.fig1_constitutional_gradient_field()
        paths["fig2"] = self.fig2_alignment_boundary_topology()
        paths["fig3"] = self.fig3_sgpo_trajectory_avoidance()
        paths["fig4"] = self.fig4_hodge_decomposition()
        paths["fig5"] = self.fig5_hodge_filtering_comparison()

        logger.info(f"Generated {len([p for p in paths.values() if p])} figures")
        return paths
