#!/usr/bin/env python3
"""
Storytelling Machine Demo

Demonstrates the complete pipeline:
1. Generate episodes via small LLM playing a text adventure
2. Collect feedback (simulated or real)
3. Apply Hodge decomposition to learn reward manifold
4. Visualize the topological structure
5. Use GeoDPO to align policy with Hodge gradient

Requirements:
    pip install sentence-transformers openai numpy scipy plotly streamlit
"""

import argparse
import json
from dataclasses import asdict
from typing import List, Optional
import numpy as np

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using mock embeddings.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Visualization disabled.")

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.environments.storytelling_machine import (
    StorytellingMachine,
    MCPAction,
    ActionType,
    Page,
    Transition,
    WorldOracle,
)
from src.hodge_critic import (
    HodgeCritic,
    FeedbackItem,
    TopologicalGradient,
    GeoDPOLoss,
)


class MockEmbeddingModel:
    """Simple mock for when sentence-transformers isn't available."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._cache = {}
    
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            if text not in self._cache:
                # Deterministic pseudo-random based on text hash
                np.random.seed(hash(text) % (2**32))
                self._cache[text] = np.random.randn(self.dim)
                self._cache[text] /= np.linalg.norm(self._cache[text])
            embeddings.append(self._cache[text])
        return np.array(embeddings)


class RuleBasedOracle(WorldOracle):
    """
    Simple rule-based text adventure oracle for demo purposes.
    
    A tiny "escape the room" game.
    """
    
    def __init__(self):
        self.world_state = {
            "location": "start_room",
            "inventory": [],
            "door_locked": True,
            "key_found": False,
            "lever_pulled": False,
        }
        
        self.rooms = {
            "start_room": {
                "description": "A small stone room. A locked door is to the NORTH. A lever on the wall.",
                "exits": {"north": "corridor"},
                "items": [],
            },
            "corridor": {
                "description": "A dark corridor. The start room is SOUTH. A treasury is EAST.",
                "exits": {"south": "start_room", "east": "treasury"},
                "items": ["rusty_key"],
            },
            "treasury": {
                "description": "A room filled with gold. You've won!",
                "exits": {"west": "corridor"},
                "items": ["treasure"],
            },
        }
    
    def transition(self, page: Page, action: MCPAction):
        info = {"success": False, "message": ""}
        new_state = self.world_state.copy()
        
        if action.tool_name == "move":
            direction = action.parameters.get("direction", "").lower()
            current_room = self.rooms[self.world_state["location"]]
            
            if direction in current_room["exits"]:
                target = current_room["exits"][direction]
                
                # Check door lock
                if target == "corridor" and self.world_state["door_locked"]:
                    if not self.world_state["lever_pulled"]:
                        info["message"] = "The door is locked. A lever on the wall might help."
                    else:
                        new_state["location"] = target
                        info["success"] = True
                else:
                    new_state["location"] = target
                    info["success"] = True
            else:
                info["message"] = f"Cannot go {direction} from here."
        
        elif action.tool_name == "interact":
            target = action.parameters.get("target", "").lower()
            
            if target == "lever" and self.world_state["location"] == "start_room":
                new_state["lever_pulled"] = True
                new_state["door_locked"] = False
                info["success"] = True
                info["message"] = "You pull the lever. A click echoes—the door unlocks."
            
            elif target == "key" or target == "rusty_key":
                if self.world_state["location"] == "corridor" and "rusty_key" in self.rooms["corridor"]["items"]:
                    new_state["inventory"].append("rusty_key")
                    new_state["key_found"] = True
                    self.rooms["corridor"]["items"].remove("rusty_key")
                    info["success"] = True
                    info["message"] = "You pick up the rusty key."
            
            elif target == "treasure":
                if self.world_state["location"] == "treasury":
                    new_state["inventory"].append("treasure")
                    info["success"] = True
                    info["message"] = "You claim the treasure!"
        
        elif action.tool_name == "look":
            info["success"] = True
            info["message"] = "You examine the area carefully."
        
        self.world_state = new_state
        
        # Generate new page
        room = self.rooms[new_state["location"]]
        scene = room["description"]
        if info["message"]:
            scene = f"{info['message']} {scene}"
        if new_state["inventory"]:
            scene += f" (Inventory: {', '.join(new_state['inventory'])})"
        
        new_page = Page(
            scene_description=scene,
            hidden_state=new_state.copy(),
            step_number=page.step_number + 1,
            metadata={"success": info["success"]},
        )
        
        return new_page, info
    
    def render_observation(self, page: Page, agent_state):
        return page.scene_description
    
    def is_terminal(self, page: Page):
        if "treasure" in self.world_state.get("inventory", []):
            return True, "victory"
        if page.step_number > 20:
            return True, "timeout"
        return False, None


def generate_synthetic_trajectory(oracle: WorldOracle) -> List[Transition]:
    """Generate a trajectory by random exploration."""
    
    initial = Page(scene_description="A small stone room. A locked door is to the NORTH. A lever on the wall.")
    pages = [initial]
    transitions = []
    
    action_templates = [
        MCPAction(ActionType.OBSERVE, "look", {"target": "room"}),
        MCPAction(ActionType.ACT, "move", {"direction": "north"}),
        MCPAction(ActionType.ACT, "interact", {"target": "lever"}),
        MCPAction(ActionType.ACT, "interact", {"target": "key"}),
        MCPAction(ActionType.ACT, "move", {"direction": "east"}),
        MCPAction(ActionType.ACT, "interact", {"target": "treasure"}),
    ]
    
    for i, action in enumerate(action_templates):
        current_page = pages[-1]
        new_page, info = oracle.transition(current_page, action)
        
        transitions.append(Transition(
            from_page=current_page,
            action=action,
            to_page=new_page,
        ))
        pages.append(new_page)
        
        is_done, reason = oracle.is_terminal(new_page)
        if is_done:
            break
    
    return transitions


def assign_synthetic_feedback(transitions: List[Transition]) -> List[FeedbackItem]:
    """Assign synthetic feedback to transitions based on game progress."""
    
    feedback = []
    for i, t in enumerate(transitions):
        # Heuristic rewards based on game state
        hidden = t.to_page.hidden_state or {}
        
        rank = 0.5  # Neutral default
        critique = None
        
        if hidden.get("lever_pulled") and not transitions[max(0, i-1)].to_page.hidden_state.get("lever_pulled", False):
            rank = 0.8
            critique = "Good progress—unlocked the door."
        
        if hidden.get("key_found") and not transitions[max(0, i-1)].to_page.hidden_state.get("key_found", False):
            rank = 0.7
            critique = "Found a useful item."
        
        if "treasure" in hidden.get("inventory", []):
            rank = 1.0
            critique = "Victory! Optimal outcome."
        
        if "locked" in t.to_page.scene_description.lower() and t.action.tool_name == "move":
            rank = 0.3
            critique = "Wasted action—door was locked."
        
        feedback.append(FeedbackItem(
            state_text=t.from_page.scene_description,
            action_text=str(t.action),
            next_state_text=t.to_page.scene_description,
            rank=rank,
            critique=critique,
            evaluator_id="synthetic_oracle",
        ))
    
    return feedback


def visualize_manifold(
    critic: HodgeCritic,
    hodge_result: TopologicalGradient,
    title: str = "Reward Manifold",
):
    """Create 3D visualization of the reward manifold."""
    
    if not HAS_PLOTLY:
        print("Plotly not available. Skipping visualization.")
        return None
    
    if critic.embeddings is None or len(critic.embeddings) < 2:
        print("Not enough data points for visualization.")
        return None
    
    # Project to 3D using PCA
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(critic.embeddings)
    
    # Get ranks for coloring
    ranks = [item.rank for item in critic.feedback_items]
    
    # Project gradient to 3D
    gradient_3d = pca.transform(hodge_result.gradient_component.reshape(1, -1))[0]
    curl_3d = pca.transform(hodge_result.curl_component.reshape(1, -1))[0]
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Reward Manifold", "Hodge Decomposition"),
    )
    
    # Plot 1: Points colored by rank
    fig.add_trace(
        go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode="markers+text",
            marker=dict(
                size=8,
                color=ranks,
                colorscale="RdYlGn",
                colorbar=dict(title="Rank", x=0.45),
            ),
            text=[f"Step {i}" for i in range(len(ranks))],
            textposition="top center",
            name="States",
        ),
        row=1, col=1,
    )
    
    # Draw trajectory edges
    for i in range(len(coords_3d) - 1):
        fig.add_trace(
            go.Scatter3d(
                x=[coords_3d[i, 0], coords_3d[i+1, 0]],
                y=[coords_3d[i, 1], coords_3d[i+1, 1]],
                z=[coords_3d[i, 2], coords_3d[i+1, 2]],
                mode="lines",
                line=dict(color="gray", width=2),
                showlegend=False,
            ),
            row=1, col=1,
        )
    
    # Plot 2: Gradient and Curl vectors
    center = coords_3d.mean(axis=0)
    scale = 2.0
    
    # Gradient vector (green)
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], center[0] + scale * gradient_3d[0]],
            y=[center[1], center[1] + scale * gradient_3d[1]],
            z=[center[2], center[2] + scale * gradient_3d[2]],
            mode="lines+markers",
            marker=dict(size=5, color="green"),
            line=dict(color="green", width=5),
            name="∇φ (Gradient)",
        ),
        row=1, col=2,
    )
    
    # Curl vector (red)
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], center[0] + scale * curl_3d[0]],
            y=[center[1], center[1] + scale * curl_3d[1]],
            z=[center[2], center[2] + scale * curl_3d[2]],
            mode="lines+markers",
            marker=dict(size=5, color="red"),
            line=dict(color="red", width=5),
            name="∇×ψ (Curl)",
        ),
        row=1, col=2,
    )
    
    # Add points to second plot
    fig.add_trace(
        go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode="markers",
            marker=dict(size=5, color="lightblue"),
            name="States",
        ),
        row=1, col=2,
    )
    
    fig.update_layout(
        title=f"{title} | H¹ = {hodge_result.h1_magnitude:.4f}",
        height=600,
        width=1200,
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Storytelling Machine Demo")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to generate")
    parser.add_argument("--visualize", action="store_true", help="Show 3D visualization")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()
    
    print("=" * 60)
    print("STORYTELLING MACHINE DEMO")
    print("=" * 60)
    
    # Initialize embedding model
    if HAS_SENTENCE_TRANSFORMERS:
        print("\nLoading embedding model (all-MiniLM-L6-v2)...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        print("\nUsing mock embedding model...")
        embedding_model = MockEmbeddingModel()
    
    # Initialize Hodge Critic
    critic = HodgeCritic(embedding_model)
    
    # Generate episodes
    print(f"\nGenerating {args.episodes} episodes...")
    all_feedback = []
    
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1} ---")
        oracle = RuleBasedOracle()  # Fresh game state
        transitions = generate_synthetic_trajectory(oracle)
        
        print(f"  Steps: {len(transitions)}")
        for i, t in enumerate(transitions):
            print(f"    {i}: {t.action} -> {t.to_page.scene_description[:50]}...")
        
        # Assign synthetic feedback
        feedback = assign_synthetic_feedback(transitions)
        all_feedback.extend(feedback)
        
        # Add to critic
        for item in feedback:
            critic.add_feedback(item)
    
    # Compute Hodge decomposition
    print("\n" + "=" * 60)
    print("HODGE DECOMPOSITION")
    print("=" * 60)
    
    hodge_result = critic.compute_hodge_decomposition()
    
    print(f"\nH¹ Magnitude (Inconsistency): {hodge_result.h1_magnitude:.4f}")
    print(f"  - Low H¹ (<0.1): Consistent feedback")
    print(f"  - High H¹ (>0.5): Significant inconsistencies or cycles")
    
    print(f"\nGradient Component ||∇φ||: {np.linalg.norm(hodge_result.gradient_component):.4f}")
    print(f"Curl Component ||∇×ψ||: {np.linalg.norm(hodge_result.curl_component):.4f}")
    
    # Consistency report
    report = critic.get_consistency_report()
    print(f"\nConsistency Report:")
    print(f"  - Total feedback items: {report['total_feedback_items']}")
    print(f"  - Is consistent: {report['is_consistent']}")
    
    # Demo GeoDPO scoring
    print("\n" + "=" * 60)
    print("GEODPO ACTION RANKING")
    print("=" * 60)
    
    test_state = "A small stone room. A locked door is to the NORTH. A lever on the wall."
    test_actions = [
        "move(direction='north')",
        "interact(target='lever')",
        "look(target='room')",
    ]
    
    rankings = critic.rank_actions(test_state, test_actions)
    print(f"\nState: {test_state[:60]}...")
    print(f"Action rankings (by Hodge gradient alignment):")
    for action, score in rankings:
        print(f"  {score:+.4f}: {action}")
    
    # Visualize
    if args.visualize and HAS_PLOTLY:
        print("\nGenerating visualization...")
        fig = visualize_manifold(critic, hodge_result)
        if fig:
            fig.show()
    
    # Save results
    if args.output:
        results = {
            "episodes": args.episodes,
            "total_feedback": len(all_feedback),
            "h1_magnitude": hodge_result.h1_magnitude,
            "consistency_report": report,
            "action_rankings": rankings,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
