#!/usr/bin/env python3
"""
Reward Manifold Explorer

Interactive UI for:
1. Episode browsing and replay
2. 3D manifold visualization (PCA projection)
3. Verbal feedback for reward curvature modification
4. Alternative action generation and ranking
5. Black hole creation from feedback

Usage:
    streamlit run apps/reward_manifold_explorer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

# Visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.error("Plotly required: pip install plotly")

# Embedding
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Our modules
from src.hodge_critic import HodgeCritic, FeedbackItem, TopologicalGradient
from src.embedding_topology_analyzer import EmbeddingTopologyAnalyzer
from src.sheaf_resolver import SheafResolver, Perspective


@dataclass
class EpisodeStep:
    """A single step in an episode."""
    step_idx: int
    state: str
    action: str
    next_state: str
    reward: float
    cost: float
    embedding: Optional[np.ndarray] = None
    alternative_actions: List[Dict] = None
    
    def __post_init__(self):
        if self.alternative_actions is None:
            self.alternative_actions = []


@dataclass
class Episode:
    """A complete episode with steps."""
    episode_id: str
    source: str  # alignment, chess, coding, etc.
    steps: List[EpisodeStep]
    total_reward: float
    total_cost: float
    metadata: Dict = None


class MockEmbeddingModel:
    """Mock embedding model for when sentence-transformers unavailable."""
    def __init__(self, dim=384):
        self.dim = dim
        self._cache = {}
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            if text not in self._cache:
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(self.dim)
                self._cache[text] = emb / np.linalg.norm(emb)
            embeddings.append(self._cache[text])
        return np.array(embeddings)


class SemanticClusterAnnotator:
    """
    Annotates clusters of states in the embedding space with semantic labels.
    Uses KMeans for clustering and simple TF-IDF-like keyword extraction.
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        
    def annotate(self, embeddings: np.ndarray, texts: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
        if not HAS_SKLEARN:
            return np.zeros(len(embeddings)), {}
            
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Cluster embeddings
        n_clusters = min(self.n_clusters, len(embeddings))
        if n_clusters < 2:
             return np.zeros(len(embeddings)), {0: "All"}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Extract keywords for each cluster
        cluster_labels = {}
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            for i in range(n_clusters):
                # Get texts in this cluster
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) == 0:
                    continue
                
                # Average TF-IDF vector for cluster
                cluster_center_tfidf = np.mean(tfidf_matrix[cluster_indices], axis=0).A1
                
                # Top 3 keywords
                top_indices = cluster_center_tfidf.argsort()[-3:][::-1]
                keywords = feature_names[top_indices]
                cluster_labels[i] = ", ".join(keywords).upper()
                
        except Exception as e:
            # Fallback if TF-IDF fails (e.g. empty vocab)
            print(f"Annotation failed: {e}")
            for i in range(n_clusters):
                cluster_labels[i] = f"Cluster {i}"
                
        return labels, cluster_labels


class RewardManifoldExplorer:
    """
    Main application class for the Reward Manifold Explorer.
    """
    
    def __init__(self):
        # Initialize session state
        if 'episodes' not in st.session_state:
            st.session_state.episodes = []
        if 'hodge_critic' not in st.session_state:
            st.session_state.hodge_critic = None
        if 'embeddings' not in st.session_state:
            st.session_state.embeddings = None
        if 'black_holes' not in st.session_state:
            st.session_state.black_holes = []
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
        if 'embedding_model' not in st.session_state:
            if HAS_TRANSFORMERS:
                st.session_state.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                st.session_state.embedding_model = MockEmbeddingModel()
        if 'cluster_annotator' not in st.session_state:
            st.session_state.cluster_annotator = SemanticClusterAnnotator(n_clusters=6)
        if 'topology_analyzer' not in st.session_state:
            st.session_state.topology_analyzer = None
    
    def load_episodes_from_scenarios(self):
        """Load episodes from scenario generators."""
        from src.scenarios.alignment import AlignmentScenarioGenerator
        from src.scenarios.strategic_games import ChessScenarioGenerator
        from src.scenarios.coding import CodingScenarioGenerator
        
        episodes = []
        
        # Load TextWorld Games if available
        tw_path = Path("tw_games")
        if tw_path.exists():
            import glob
            for file_path in glob.glob(str(tw_path / "*.json")):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    game_id = Path(file_path).stem
                    theme = data.get("grammar", {}).get("theme", "textworld")
                    
                    # Each quest can be an episode trajectory (solution path)
                    for q_idx, quest in enumerate(data.get("quests", [])):
                        commands = quest.get("commands", [])
                        desc = quest.get("desc", "Unknown Quest")
                        
                        steps = []
                        current_context = f"Quest: {desc}"
                        
                        for i, cmd in enumerate(commands):
                            # Simulate state progression via context accumulation
                            # (Since we can't run the game engine to get real observations)
                            step_state = current_context
                            
                            # Update context for next step
                            current_context += f" | > {cmd}"
                            
                            steps.append(EpisodeStep(
                                step_idx=i,
                                state=step_state,
                                action=cmd,
                                next_state=current_context if i < len(commands)-1 else "Quest Completed",
                                reward=0.1,  # Small progress reward
                                cost=0.0
                            ))
                        
                        # Final reward
                        if steps:
                            steps[-1].reward = 1.0
                        
                        episodes.append(Episode(
                            episode_id=f"TW-{game_id[:8]}-Q{q_idx}",
                            source="textworld",
                            steps=steps,
                            total_reward=1.0,
                            total_cost=0.0,
                            metadata={"full_desc": desc, "theme": theme}
                        ))
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        # Alignment scenarios
        align_gen = AlignmentScenarioGenerator()
        for i in range(5):
            transitions = align_gen.generate_episode(num_turns=5)
            steps = [
                EpisodeStep(
                    step_idx=j,
                    state=t.state,
                    action=t.action,
                    next_state=t.result,
                    reward=t.reward,
                    cost=t.cost,
                )
                for j, t in enumerate(transitions)
            ]
            episodes.append(Episode(
                episode_id=f"alignment_{i}",
                source="alignment",
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                total_cost=sum(s.cost for s in steps),
            ))
        
        # Chess scenarios
        chess_gen = ChessScenarioGenerator()
        for i in range(3):
            transitions = chess_gen.generate_episode(num_turns=5)
            steps = [
                EpisodeStep(
                    step_idx=j,
                    state=t.state,
                    action=f"{t.chosen_move.notation}: {t.chosen_move.description}",
                    next_state=t.resulting_state,
                    reward=t.reward,
                    cost=t.cost,
                    alternative_actions=[
                        {"action": f"{m.notation}: {m.description}", "score": m.strategic_value}
                        for m in t.available_moves if m != t.chosen_move
                    ],
                )
                for j, t in enumerate(transitions)
            ]
            episodes.append(Episode(
                episode_id=f"chess_{i}",
                source="chess",
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                total_cost=sum(s.cost for s in steps),
            ))
        
        # Coding scenarios
        code_gen = CodingScenarioGenerator()
        for i in range(3):
            transitions = code_gen.generate_episode(num_problems=4)
            steps = [
                EpisodeStep(
                    step_idx=j,
                    state=t.state,
                    action=t.action,
                    next_state=t.result,
                    reward=t.reward,
                    cost=t.cost,
                )
                for j, t in enumerate(transitions)
            ]
            episodes.append(Episode(
                episode_id=f"coding_{i}",
                source="coding",
                steps=steps,
                total_reward=sum(s.reward for s in steps),
                total_cost=sum(s.cost for s in steps),
            ))
        
        return episodes
    
    def compute_embeddings(self, episodes: List[Episode]):
        """Compute embeddings for all episode steps."""
        model = st.session_state.embedding_model
        
        all_texts = []
        for ep in episodes:
            for step in ep.steps:
                text = f"State: {step.state[:500]} Action: {step.action[:200]}"
                all_texts.append(text)
        
        if all_texts:
            embeddings = model.encode(all_texts)
            
            idx = 0
            for ep in episodes:
                for step in ep.steps:
                    step.embedding = embeddings[idx]
                    idx += 1
            
            st.session_state.embeddings = embeddings
        
        return episodes
    
    def initialize_hodge_critic(self, episodes: List[Episode]):
        """Initialize Hodge critic with episode feedback."""
        model = st.session_state.embedding_model
        critic = HodgeCritic(model)
        
        for ep in episodes:
            for step in ep.steps:
                critic.add_feedback(FeedbackItem(
                    state_text=step.state[:500],
                    action_text=step.action[:200],
                    next_state_text=step.next_state[:200] if step.next_state else None,
                    rank=step.reward,
                    critique=None,
                    evaluator_id=ep.source,
                ))
        
        st.session_state.hodge_critic = critic
        return critic
    
    def render_episode_browser(self):
        """Render the episode browsing interface."""
        st.header("Episode Browser")
        
        all_episodes = st.session_state.episodes
        if not all_episodes:
            st.info("No episodes loaded. Click 'Load Sample Episodes' to begin.")
            return
            
        # 1. Filters
        col1, col2 = st.columns([1, 2])
        with col1:
            # Source Filter
            sources = list(set(ep.source for ep in all_episodes))
            selected_sources = st.multiselect("Filter by Source", sources, default=sources)
        
        with col2:
            # TextWorld Theme Filter (if applicable)
            tw_episodes = [ep for ep in all_episodes if ep.source == 'textworld']
            if tw_episodes and 'textworld' in selected_sources:
                themes = list(set(ep.metadata.get("theme", "Unknown") for ep in tw_episodes))
                selected_themes = st.multiselect("Filter by TextWorld Theme", themes, default=themes)
            else:
                selected_themes = []
        
        # Apply Filters
        filtered_episodes = []
        for ep in all_episodes:
            if ep.source not in selected_sources:
                continue
            if ep.source == 'textworld' and selected_themes:
                if ep.metadata.get("theme", "Unknown") not in selected_themes:
                    continue
            filtered_episodes.append(ep)
            
        if not filtered_episodes:
            st.warning("No episodes match filters.")
            return

        # 2. Episode Selector
        # Create descriptive labels
        episode_options = []
        for ep in filtered_episodes:
            if ep.source == "textworld":
                label = f"🎮 {ep.metadata.get('theme', 'TW').title()} - {ep.metadata.get('full_desc', '')[:40]}..."
            else:
                label = f"📋 {ep.episode_id} ({ep.source})"
            episode_options.append(label)
            
        selected_idx = st.selectbox(
            "Select Episode", 
            range(len(filtered_episodes)), 
            format_func=lambda i: episode_options[i]
        )
        
        episode = filtered_episodes[selected_idx]
        
        # Episode summary
        st.subheader(f"Episode: {episode.episode_id}")
        if episode.source == "textworld":
            st.caption(f"**Quest:** {episode.metadata.get('full_desc', 'N/A')}")
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Steps", len(episode.steps))
        col2.metric("Total Reward", f"{episode.total_reward:.2f}")
        col3.metric("Total Cost", f"{episode.total_cost:.2f}")
        col4.metric("Source", episode.source)
        
        # Step-by-step replay
        st.subheader("Step-by-Step Replay")
        
        if not episode.steps:
            st.warning("Episode has no steps.")
            return
            
        step_idx = st.slider("Step", 0, len(episode.steps) - 1, 0)
        step = episode.steps[step_idx]
        
        # Display current step
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**State:**")
            st.text_area("", step.state, height=200, key=f"state_{episode.episode_id}_{step_idx}", disabled=True)
        
        with col2:
            st.markdown("**Action:**")
            st.text_area("", step.action, height=200, key=f"action_{episode.episode_id}_{step_idx}", disabled=True)
        
        # Reward/cost for this step
        col1, col2, col3 = st.columns(3)
        col1.metric("Step Reward", f"{step.reward:.2f}")
        col2.metric("Step Cost", f"{step.cost:.2f}")
        col3.metric("Step", f"{step_idx + 1}/{len(episode.steps)}")
        
        # Next state
        if step.next_state:
            with st.expander("Result/Next State"):
                st.write(step.next_state)
        
        # Alternative actions
        if step.alternative_actions:
            with st.expander(f"Alternative Actions ({len(step.alternative_actions)})"):
                for alt in step.alternative_actions:
                    st.markdown(f"- **{alt['action'][:100]}** (score: {alt.get('score', 'N/A')})")
        
        return episode, step_idx
    
    def render_manifold_visualization(self):
        """Render 3D manifold visualization."""
        st.header("Reward Manifold Visualization")
        
        # Lazy init for annotator if missing
        if 'cluster_annotator' not in st.session_state:
            st.session_state.cluster_annotator = SemanticClusterAnnotator(n_clusters=6)
        
        if st.session_state.embeddings is None or len(st.session_state.embeddings) < 3:
            st.warning("Need at least 3 embeddings for visualization. Load episodes first.")
            return
        
        embeddings = st.session_state.embeddings
        episodes = st.session_state.episodes
        
        # Collect data
        rewards = []
        labels = []
        sources = []
        texts = []
        costs = []
        
        for ep in episodes:
            for step in ep.steps:
                rewards.append(step.reward)
                labels.append(f"{ep.episode_id}:{step.step_idx}")
                sources.append(ep.source)
                costs.append(step.cost)
                # Text for annotation
                texts.append(f"{step.state} {step.action}")
        
        rewards = np.array(rewards)
        
        # Compute Clusters
        cluster_ids, cluster_names = st.session_state.cluster_annotator.annotate(embeddings, texts)
        
        # Dimensionality reduction options
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Projection Method", ["PCA", "t-SNE"])
        with col2:
            color_by = st.selectbox("Color By", ["Semantic Cluster", "Reward", "Source", "Cost"])
        
        # Compute projection
        if method == "PCA":
            reducer = PCA(n_components=3)
            coords_3d = reducer.fit_transform(embeddings)
        else:
            # t-SNE (slower)
            with st.spinner("Computing t-SNE..."):
                reducer = TSNE(n_components=3, perplexity=min(30, len(embeddings)-1), random_state=42)
                coords_3d = reducer.fit_transform(embeddings)
        
        # Color mapping
        if color_by == "Reward":
            colors = rewards
            colorscale = "RdYlGn"
            colorbar_title = "Reward"
        elif color_by == "Source":
            unique_sources = list(set(sources))
            source_map = {s: i for i, s in enumerate(unique_sources)}
            colors = [source_map[s] for s in sources]
            colorscale = "Viridis"
            colorbar_title = "Source ID"
        elif color_by == "Cost":
            colors = costs
            colorscale = "Reds"
            colorbar_title = "Cost"
        else: # Semantic Cluster
            colors = cluster_ids
            colorscale = "Rainbow"
            colorbar_title = "Cluster ID"
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Main scatter
        fig.add_trace(go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
                opacity=0.7,
            ),
            text=[f"{l}<br>Cluster: {cluster_names.get(c, 'Unknown')}" for l, c in zip(labels, cluster_ids)],
            hovertemplate="<b>%{text}</b><br>Reward: %{marker.color:.2f}<extra></extra>",
        ))
        
        # Add Cluster Annotations
        if color_by == "Semantic Cluster":
            unique_clusters = np.unique(cluster_ids)
            for cid in unique_clusters:
                # Find centroid of points in this cluster
                cluster_points = coords_3d[cluster_ids == cid]
                centroid = np.mean(cluster_points, axis=0)
                label = cluster_names.get(cid, f"Cluster {cid}")
                
                fig.add_trace(go.Scatter3d(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                    mode='text',
                    text=[label],
                    textposition="top center",
                    textfont=dict(size=12, color="white", family="Arial Black"),
                    showlegend=False
                ))

        # Add black holes if any
        if st.session_state.black_holes:
            for bh in st.session_state.black_holes:
                if 'coords_3d' in bh:
                    fig.add_trace(go.Scatter3d(
                        x=[bh['coords_3d'][0]],
                        y=[bh['coords_3d'][1]],
                        z=[bh['coords_3d'][2]],
                        mode='markers',
                        marker=dict(size=15, color='black', symbol='x'),
                        name=f"Black Hole: {bh.get('reason', 'User marked')[:20]}",
                    ))
        
        # Draw trajectory lines within episodes
        show_trajectories = st.checkbox("Show Episode Trajectories", value=True)
        if show_trajectories:
            idx = 0
            for ep in episodes:
                ep_coords = coords_3d[idx:idx + len(ep.steps)]
                fig.add_trace(go.Scatter3d(
                    x=ep_coords[:, 0],
                    y=ep_coords[:, 1],
                    z=ep_coords[:, 2],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip',
                ))
                idx += len(ep.steps)
        
        # Add Hodge gradient vector if available
        critic = st.session_state.hodge_critic
        if critic is not None:
            try:
                hodge_result = critic.compute_hodge_decomposition()
                
                # Project gradient to 3D
                if method == "PCA":
                    gradient_3d = reducer.transform(hodge_result.gradient_component.reshape(1, -1))[0]
                    
                    # Draw gradient arrow from center
                    center = coords_3d.mean(axis=0)
                    scale = np.std(coords_3d) * 2
                    
                    fig.add_trace(go.Scatter3d(
                        x=[center[0], center[0] + scale * gradient_3d[0]],
                        y=[center[1], center[1] + scale * gradient_3d[1]],
                        z=[center[2], center[2] + scale * gradient_3d[2]],
                        mode='lines+markers',
                        line=dict(color='green', width=8),
                        marker=dict(size=[0, 10], color='green'),
                        name=f"Hodge Gradient (H¹={hodge_result.h1_magnitude:.3f})",
                    ))
            except Exception as e:
                st.caption(f"Could not compute Hodge gradient: {e}")
        
        fig.update_layout(
            title="Reward Manifold (3D Projection)",
            scene=dict(
                xaxis_title="PC1" if method == "PCA" else "Dim 1",
                yaxis_title="PC2" if method == "PCA" else "Dim 2",
                zaxis_title="PC3" if method == "PCA" else "Dim 3",
            ),
            height=600,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store projection for feedback
        st.session_state.coords_3d = coords_3d
        st.session_state.projection_method = method
        if method == "PCA":
            st.session_state.pca_reducer = reducer
    
    def render_feedback_interface(self):
        """Render the feedback interface for modifying reward curvature."""
        st.header("Feedback Interface")
        
        st.markdown("""
        Provide feedback to modify the reward manifold:
        - **0.0** = Black hole (forbidden state)
        - **0.5** = Neutral
        - **1.0** = Global maximum (ideal outcome)
        
        Your feedback will:
        1. Add edges to the preference graph
        2. Update Hodge decomposition
        3. Create black holes for harmful states
        """)
        
        episodes = st.session_state.episodes
        if not episodes:
            st.info("Load episodes first.")
            return
        
        # Select state to provide feedback on
        st.subheader("Step Feedback")
        
        col1, col2 = st.columns(2)
        with col1:
            ep_idx = st.selectbox("Episode", range(len(episodes)), 
                                  format_func=lambda i: episodes[i].episode_id,
                                  key="feedback_ep")
        with col2:
            step_idx = st.selectbox("Step", range(len(episodes[ep_idx].steps)),
                                    key="feedback_step")
        
        episode = episodes[ep_idx]
        step = episode.steps[step_idx]
        
        # Show the state/action
        st.text_area("State", step.state[:500], height=100, disabled=True)
        st.text_area("Action", step.action[:300], height=80, disabled=True)
        
        # Feedback inputs
        col1, col2 = st.columns(2)
        with col1:
            feedback_score = st.slider(
                "Feedback Score",
                0.0, 1.0, float(step.reward),
                help="0=Black hole, 0.5=Neutral, 1=Optimal"
            )
        with col2:
            is_black_hole = st.checkbox("Mark as Black Hole (Forbidden)", value=feedback_score < 0.1)
        
        critique = st.text_input("Verbal Critique (optional)", placeholder="Explain why this is good/bad...")
        
        if st.button("Submit Feedback", type="primary"):
            # Update Hodge critic
            critic = st.session_state.hodge_critic
            if critic is not None:
                critic.add_feedback(FeedbackItem(
                    state_text=step.state[:500],
                    action_text=step.action[:200],
                    next_state_text=step.next_state[:200] if step.next_state else None,
                    rank=feedback_score,
                    critique=critique if critique else None,
                    evaluator_id="user_feedback",
                ))
            
            # Add black hole if marked
            if is_black_hole and step.embedding is not None:
                coords_3d = st.session_state.get('coords_3d')
                if coords_3d is not None:
                    # Find this step's 3D coordinates
                    flat_idx = 0
                    for ep in episodes:
                        for s in ep.steps:
                            if s is step:
                                break
                            flat_idx += 1
                        else:
                            continue
                        break
                    
                    if flat_idx < len(coords_3d):
                        st.session_state.black_holes.append({
                            'embedding': step.embedding,
                            'coords_3d': coords_3d[flat_idx],
                            'state': step.state[:100],
                            'reason': critique or "User marked as forbidden",
                            'score': feedback_score,
                        })
            
            # Record feedback
            st.session_state.feedback_history.append({
                'episode': episode.episode_id,
                'step': step_idx,
                'score': feedback_score,
                'critique': critique,
                'is_black_hole': is_black_hole,
            })
            
            st.success(f"Feedback recorded! Score: {feedback_score:.2f}")
            st.rerun()
        
        # Show feedback history
        with st.expander(f"Feedback History ({len(st.session_state.feedback_history)} items)"):
            for fb in reversed(st.session_state.feedback_history[-10:]):
                st.markdown(f"- **{fb['episode']}:{fb['step']}** → {fb['score']:.2f} {'🚫' if fb['is_black_hole'] else ''}")
    
    def render_action_ranking(self):
        """Render interface for ranking alternative actions."""
        st.header("Alternative Action Ranking")
        
        st.markdown("""
        Generate and rank alternative actions at each step.
        Rankings inform the Hodge critic to refine reward curvature.
        """)
        
        episodes = st.session_state.episodes
        if not episodes:
            st.info("Load episodes first.")
            return
        
        # Find steps with alternative actions
        steps_with_alts = []
        for ep in episodes:
            for step in ep.steps:
                if step.alternative_actions:
                    steps_with_alts.append((ep, step))
        
        if not steps_with_alts:
            st.info("No episodes have alternative actions yet. Chess scenarios include alternatives.")
            return
        
        # Select a step
        options = [f"{ep.episode_id}:{step.step_idx}" for ep, step in steps_with_alts]
        selected = st.selectbox("Select Step with Alternatives", range(len(options)), format_func=lambda i: options[i])
        
        ep, step = steps_with_alts[selected]
        
        # Show state
        st.text_area("State", step.state[:500], height=100, disabled=True)
        
        # Current action
        st.markdown("**Current Action:**")
        st.info(step.action[:300])
        
        # Alternative actions with ranking
        st.markdown("**Rank Alternative Actions:**")
        st.caption("Drag to reorder, or assign scores directly.")
        
        rankings = {}
        for i, alt in enumerate(step.alternative_actions):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Option {i+1}:** {alt['action'][:150]}")
            with col2:
                rankings[i] = st.number_input(f"Score", 0.0, 1.0, alt.get('score', 0.5), 
                                              key=f"rank_{selected}_{i}", label_visibility="collapsed")
        
        if st.button("Submit Rankings"):
            # Add preference pairs to Hodge critic
            critic = st.session_state.hodge_critic
            if critic is not None:
                sorted_alts = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
                
                # Create preference pairs
                for i in range(len(sorted_alts) - 1):
                    better_idx, better_score = sorted_alts[i]
                    worse_idx, worse_score = sorted_alts[i + 1]
                    
                    if better_score > worse_score:
                        # Add preference
                        critic.add_feedback(FeedbackItem(
                            state_text=step.state[:500],
                            action_text=step.alternative_actions[better_idx]['action'],
                            next_state_text=None,
                            rank=better_score,
                            critique=f"Preferred over option {worse_idx + 1}",
                        ))
                        critic.add_feedback(FeedbackItem(
                            state_text=step.state[:500],
                            action_text=step.alternative_actions[worse_idx]['action'],
                            next_state_text=None,
                            rank=worse_score,
                            critique=f"Less preferred than option {better_idx + 1}",
                        ))
            
            st.success("Rankings submitted to Hodge critic!")
    
    def render_hodge_analysis(self):
        """Render Hodge decomposition analysis."""
        st.header("Hodge Decomposition Analysis")
        
        critic = st.session_state.hodge_critic
        if critic is None or len(critic.feedback_items) < 2:
            st.info("Need more feedback for Hodge analysis. Provide feedback on episodes first.")
            return
        
        # Compute decomposition
        with st.spinner("Computing Hodge decomposition..."):
            try:
                hodge_result = critic.compute_hodge_decomposition()
            except Exception as e:
                st.error(f"Decomposition failed: {e}")
                return
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            h1_color = "green" if hodge_result.h1_magnitude < 0.1 else "orange" if hodge_result.h1_magnitude < 0.5 else "red"
            st.metric("H¹ Magnitude", f"{hodge_result.h1_magnitude:.4f}")
            st.caption(f":{h1_color}[{'Consistent' if hodge_result.h1_magnitude < 0.1 else 'Some inconsistency' if hodge_result.h1_magnitude < 0.5 else 'High inconsistency'}]")
        
        with col2:
            grad_norm = np.linalg.norm(hodge_result.gradient_component)
            st.metric("||∇φ|| (Gradient)", f"{grad_norm:.4f}")
            st.caption("Learnable reward direction")
        
        with col3:
            curl_norm = np.linalg.norm(hodge_result.curl_component)
            st.metric("||∇×ψ|| (Curl)", f"{curl_norm:.4f}")
            st.caption("Inconsistent/cyclic preferences")
        
        # Consistency report
        report = critic.get_consistency_report()
        
        st.subheader("Consistency Report")
        st.json(report)
        
        # Interpretation
        st.subheader("Interpretation")
        
        if hodge_result.h1_magnitude < 0.1:
            st.success("✅ **Feedback is highly consistent.** A global reward function exists.")
        elif hodge_result.h1_magnitude < 0.5:
            st.warning("⚠️ **Some inconsistency detected.** There may be cyclic preferences or evaluator disagreement.")
        else:
            st.error("❌ **High inconsistency.** Preferences contain significant cycles or contradictions.")
        
        st.markdown("""
        **What this means:**
        - **H¹ = 0**: Perfect consistency - all preferences can be explained by a single value function
        - **H¹ > 0**: There are "cycles" in preferences (A > B > C > A) that can't be resolved
        - **Gradient (∇φ)**: The "clean" reward direction after removing inconsistencies
        - **Curl (∇×ψ)**: The inconsistent part that forms cycles
        """)

    def render_topology_analysis(self):
        """Render topological analysis of the reward landscape."""
        st.header("Topological Analysis")
        
        if not st.session_state.episodes:
            st.info("Load episodes first.")
            return

        # Initialize analyzer if needed
        if st.session_state.topology_analyzer is None:
             st.session_state.topology_analyzer = EmbeddingTopologyAnalyzer(
                embedding_model=st.session_state.embedding_model,
                n_clusters=6,
                black_hole_threshold=0.1, # strict threshold for demo
                cliff_threshold=0.3
            )
        
        analyzer = st.session_state.topology_analyzer
        
        # Prepare data
        episodes = st.session_state.episodes
        states = []
        actions = []
        rewards = []
        texts = []
        
        for ep in episodes:
            for step in ep.steps:
                if step.embedding is not None:
                    states.append(step.embedding)
                    actions.append(step.action)
                    rewards.append(step.reward)
                    texts.append(f"{step.state} | {step.action}")
        
        if not states:
            st.warning("No embeddings found.")
            return
            
        # Fit analyzer
        with st.spinner("Analyzing topology..."):
            analyzer.fit(states, actions, rewards, texts)
            features = analyzer.extract_features()
        
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Safe Region", f"{features.safe_region_fraction:.1%}")
        col2.metric("Black Holes", features.n_black_holes)
        col3.metric("Clusters", features.n_clusters)
        col4.metric("Global H¹", f"{features.h1_cohomology:.3f}")
        
        # Interpretable Regions
        st.subheader("Semantic Regions")
        regions = analyzer.get_interpretable_regions()
        
        for region in regions:
            with st.expander(f"Region: {region['keywords']} (Reward: {region['mean_reward']:.2f})"):
                st.write(f"**Safety:** {'✅ Safe' if region['is_safe'] else '⚠️ Unsafe'}")
                st.write(f"**Flow Coherence:** {region['flow_coherence']:.2f}")
                st.write(f"**Sample Count:** {region['sample_count']}")
        
        # Trajectory Analysis
        st.subheader("Trajectory Safety Analysis")
        selected_ep_idx = st.selectbox("Select Trajectory", range(len(episodes)), 
                                       format_func=lambda i: episodes[i].episode_id,
                                       key="topo_traj_select")
        
        selected_ep = episodes[selected_ep_idx]
        
        # Map episode steps to global indices
        global_idx = 0
        traj_indices = []
        for i, ep in enumerate(episodes):
            if i == selected_ep_idx:
                traj_indices = list(range(global_idx, global_idx + len(ep.steps)))
                break
            global_idx += len(ep.steps)
            
        if traj_indices:
            traj_analysis = analyzer.analyze_trajectory(traj_indices, selected_ep.episode_id)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Safety Score", f"{traj_analysis.safety_score:.2f}")
            c2.metric("Reward Trend", traj_analysis.reward_trend)
            c3.metric("Gradient Alignment", f"{traj_analysis.mean_gradient_alignment:.3f}")
            
            st.markdown("**Detailed Report:**")
            st.text(traj_analysis.summary())

    def render_conflict_resolution(self):
        """Render Sheaf-Theoretic Conflict Resolution."""
        st.header("Sheaf Conflict Resolution")
        st.markdown("Simulate multi-perspective conflicts and find consensus.")
        
        episodes = st.session_state.episodes
        if not episodes:
            st.info("Load episodes first.")
            return

        # Select Step
        col1, col2 = st.columns(2)
        with col1:
            ep_idx = st.selectbox("Episode", range(len(episodes)), 
                                  format_func=lambda i: episodes[i].episode_id,
                                  key="conflict_ep")
        with col2:
            step_idx = st.selectbox("Step", range(len(episodes[ep_idx].steps)),
                                    key="conflict_step")
        
        step = episodes[ep_idx].steps[step_idx]
        st.text_area("Context", f"State: {step.state}\nAction: {step.action}", height=100, disabled=True)
        
        # Define Actions (use alternatives or defaults)
        actions = [step.action] + [alt['action'] for alt in step.alternative_actions]
        if len(actions) < 2:
            actions = [step.action, "Alternative A", "Alternative B", "Do Nothing"] # Defaults if no alts
        
        n_actions = len(actions)
        
        # Define Perspectives
        st.subheader("Perspectives")
        
        if 'perspectives' not in st.session_state:
            st.session_state.perspectives = [
                {'name': 'Safety', 'weight': 2.0},
                {'name': 'Efficiency', 'weight': 1.0},
                {'name': 'User Intent', 'weight': 1.0}
            ]
            
        perspective_objects = []
        
        cols = st.columns(len(st.session_state.perspectives))
        for idx, (col, p_data) in enumerate(zip(cols, st.session_state.perspectives)):
            with col:
                st.markdown(f"**{p_data['name']}** (w={p_data['weight']})")
                
                # Interactive sliders for probabilities
                preferred_idx = st.selectbox(f"Pref for {p_data['name']}", range(n_actions), 
                                             format_func=lambda i: actions[i][:30] + "..." if len(actions[i]) > 30 else actions[i],
                                             key=f"pref_{idx}")
                
                dist = np.ones(n_actions) * 0.1
                dist[preferred_idx] = 2.0 # Boost preferred
                dist = dist / dist.sum()
                
                perspective_objects.append(Perspective(
                    name=p_data['name'],
                    weight=p_data['weight'],
                    preference_distribution=dist
                ))
                
                st.bar_chart(dist)

        if st.button("Resolve Conflict", type="primary"):
            resolver = SheafResolver(perspective_objects, n_actions)
            analysis = resolver.compute_cohomology()
            
            st.divider()
            
            # Results
            c1, c2 = st.columns(2)
            c1.metric("Conflict Energy (H¹)", f"{analysis['obstruction_energy']:.4f}")
            c2.metric("Consistent?", "Yes" if analysis['is_consistent'] else "No")
            
            # Consensus
            consensus_dist = analysis['consensus_distribution']
            best_act_idx = np.argmax(consensus_dist)
            st.success(f"**Consensus Action:** {actions[best_act_idx]}")
            st.bar_chart(consensus_dist)
            
            # Suggestions
            st.subheader("Resolution Suggestions")
            suggestions = resolver.propose_resolution_path(analysis)
            for s in suggestions:
                st.info(s)
                
            # Pairwise
            if analysis['pairwise_conflicts']:
                with st.expander("Pairwise Conflicts"):
                    st.write(analysis['pairwise_conflicts'])

    def run(self):
        """Main application entry point."""
        st.set_page_config(
            page_title="Reward Manifold Explorer",
            page_icon="🌀",
            layout="wide",
        )
        
        st.title("🌀 Reward Manifold Explorer")
        st.caption("Interactive visualization and feedback for topological reward learning")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Controls")
            
            if st.button("Load Sample Episodes", type="primary"):
                with st.spinner("Generating episodes from scenarios..."):
                    episodes = self.load_episodes_from_scenarios()
                    episodes = self.compute_embeddings(episodes)
                    self.initialize_hodge_critic(episodes)
                    st.session_state.episodes = episodes
                st.success(f"Loaded {len(episodes)} episodes!")
            
            st.divider()
            
            st.metric("Episodes", len(st.session_state.episodes))
            st.metric("Feedback Items", len(st.session_state.feedback_history))
            st.metric("Black Holes", len(st.session_state.black_holes))
            
            if st.session_state.hodge_critic:
                st.metric("Hodge Feedback", len(st.session_state.hodge_critic.feedback_items))
            
            st.divider()
            
            if st.button("Clear All Data"):
                st.session_state.episodes = []
                st.session_state.hodge_critic = None
                st.session_state.embeddings = None
                st.session_state.black_holes = []
                st.session_state.feedback_history = []
                st.rerun()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📖 Episode Browser",
            "🌐 Manifold Visualization", 
            "💬 Feedback",
            "🔢 Action Ranking",
            "📊 Hodge Analysis",
            "🗺️ Topology Report",
            "🤝 Conflict Resolution",
        ])
        
        with tab1:
            self.render_episode_browser()
        
        with tab2:
            self.render_manifold_visualization()
        
        with tab3:
            self.render_feedback_interface()
        
        with tab4:
            self.render_action_ranking()
        
        with tab5:
            self.render_hodge_analysis()
            
        with tab6:
            self.render_topology_analysis()
            
        with tab7:
            self.render_conflict_resolution()


def main():
    app = RewardManifoldExplorer()
    app.run()


if __name__ == "__main__":
    main()
