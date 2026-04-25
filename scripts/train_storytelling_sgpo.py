#!/usr/bin/env python3
"""
Storytelling Machine SGPO Training

Full integration of:
1. Storytelling Machine environment (semantic state space)
2. Hodge Critic (topological reward learning)
3. SGPO (Sheaf-Geodesic Policy Optimization with safety guarantees)

This script trains a policy to play text adventures while:
- Learning reward structure from feedback via Hodge decomposition
- Avoiding "black hole" regions (dangerous/forbidden states)
- Following geodesics on the semantic embedding manifold

Usage:
    python train_storytelling_gpo.py --episodes 100 --visualize
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.environments.storytelling_machine import (
    StorytellingMachine,
    MCPAction,
    ActionType,
    Page,
    Transition,
    WorldOracle,
)
from src.hodge_critic import HodgeCritic, FeedbackItem, TopologicalGradient
from src.semantic_mdp_rl import (
    SemanticSGPO,
    SemanticPPO,
    ManifoldAwarePolicyNetwork,
    SemanticPolicyNetwork,
    RolloutBuffer,
)

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class MockEmbeddingModel:
    """Simple mock for when sentence-transformers isn't available."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._cache = {}
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            if text not in self._cache:
                np.random.seed(hash(text) % (2**32))
                self._cache[text] = np.random.randn(self.dim)
                self._cache[text] /= np.linalg.norm(self._cache[text])
            embeddings.append(self._cache[text])
        return np.array(embeddings)


class RuleBasedOracle(WorldOracle):
    """
    Rule-based text adventure oracle for training.
    Implements a simple "escape the dungeon" game.
    """
    
    def __init__(self):
        self.reset_game()
    
    def reset_game(self):
        self.world_state = {
            "location": "cell",
            "inventory": [],
            "door_locked": True,
            "guard_asleep": False,
            "has_key": False,
            "alarm_triggered": False,
        }
        
        self.rooms = {
            "cell": {
                "description": "A dark prison cell. Iron bars block the way NORTH. A guard patrols outside.",
                "exits": {"north": "corridor"},
                "items": ["loose_brick"],
            },
            "corridor": {
                "description": "A dim corridor. The cell is SOUTH. A guard room is EAST. Exit is NORTH.",
                "exits": {"south": "cell", "east": "guard_room", "north": "exit"},
                "items": [],
            },
            "guard_room": {
                "description": "A guard room with a sleeping guard. A key ring hangs on the wall.",
                "exits": {"west": "corridor"},
                "items": ["key_ring"],
            },
            "exit": {
                "description": "Freedom! You've escaped the dungeon!",
                "exits": {},
                "items": [],
                "terminal": True,
            },
        }
    
    def transition(self, page: Page, action: MCPAction) -> Tuple[Page, Dict]:
        info = {"success": False, "message": "", "reward": 0.0, "cost": 0.0}
        new_state = self.world_state.copy()
        new_state["inventory"] = list(self.world_state["inventory"])
        
        if action.tool_name == "move":
            direction = action.parameters.get("direction", "").lower()
            current_room = self.rooms[self.world_state["location"]]
            
            if direction in current_room["exits"]:
                target = current_room["exits"][direction]
                
                # Check door lock for exit
                if target == "corridor" and self.world_state["door_locked"]:
                    info["message"] = "The cell door is locked."
                    info["cost"] = 0.1  # Small cost for wasted action
                elif target == "exit" and self.world_state["alarm_triggered"]:
                    info["message"] = "Guards block your escape! You're caught."
                    info["cost"] = 10.0  # High cost for getting caught
                    new_state["location"] = "cell"  # Reset to cell
                else:
                    new_state["location"] = target
                    info["success"] = True
                    if target == "exit":
                        info["reward"] = 10.0  # Big reward for escaping
            else:
                info["message"] = f"Cannot go {direction} from here."
        
        elif action.tool_name == "interact":
            target = action.parameters.get("target", "").lower()
            
            if "brick" in target and self.world_state["location"] == "cell":
                if "loose_brick" not in new_state["inventory"]:
                    new_state["inventory"].append("loose_brick")
                    info["success"] = True
                    info["message"] = "You pry loose a brick from the wall."
                    info["reward"] = 0.5
            
            elif "key" in target and self.world_state["location"] == "guard_room":
                if self.world_state["guard_asleep"]:
                    new_state["has_key"] = True
                    new_state["inventory"].append("key")
                    info["success"] = True
                    info["message"] = "You quietly take the key ring."
                    info["reward"] = 2.0
                else:
                    info["message"] = "The guard would notice!"
                    info["cost"] = 0.5
            
            elif "guard" in target:
                if "loose_brick" in self.world_state["inventory"]:
                    new_state["guard_asleep"] = True
                    info["success"] = True
                    info["message"] = "You knock out the guard with the brick."
                    info["reward"] = 1.0
                else:
                    info["message"] = "You have nothing to use against the guard."
                    new_state["alarm_triggered"] = True
                    info["cost"] = 5.0  # Triggering alarm is dangerous
            
            elif "door" in target or "bars" in target:
                if self.world_state["has_key"]:
                    new_state["door_locked"] = False
                    info["success"] = True
                    info["message"] = "You unlock the cell door."
                    info["reward"] = 1.0
                else:
                    info["message"] = "You need a key to unlock this."
        
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
        if new_state["alarm_triggered"]:
            scene += " [ALARM ACTIVE]"
        
        new_page = Page(
            scene_description=scene,
            hidden_state=new_state.copy(),
            step_number=page.step_number + 1,
            metadata={"success": info["success"], "reward": info["reward"], "cost": info["cost"]},
        )
        
        return new_page, info
    
    def render_observation(self, page: Page, agent_state: Dict) -> str:
        return page.scene_description
    
    def is_terminal(self, page: Page) -> Tuple[bool, Optional[str]]:
        if self.world_state["location"] == "exit":
            return True, "escaped"
        if page.step_number > 30:
            return True, "timeout"
        return False, None


@dataclass
class TrainingConfig:
    """Configuration for SGPO training."""
    embed_dim: int = 384
    num_actions: int = 10
    hidden_dims: List[int] = None
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    n_epochs: int = 5
    batch_size: int = 32
    hodge_reward_weight: float = 0.3
    black_hole_penalty: float = 50.0
    rollout_length: int = 100
    total_episodes: int = 100
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class StorytellingSGPOTrainer:
    """
    Trainer integrating Storytelling Machine + Hodge Critic + SGPO.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        embedding_model: Any = None,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Embedding model
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            if HAS_SENTENCE_TRANSFORMERS:
                print("Loading sentence-transformers model...")
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                print("Using mock embedding model...")
                self.embedding_model = MockEmbeddingModel(config.embed_dim)
        
        # Hodge Critic
        self.hodge_critic = HodgeCritic(self.embedding_model)
        
        # Policy network (manifold-aware for SGPO)
        self.policy = ManifoldAwarePolicyNetwork(
            embed_dim=config.embed_dim,
            num_actions=config.num_actions,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        # SGPO optimizer
        self.optimizer = SemanticSGPO(
            policy=self.policy,
            hodge_critic=self.hodge_critic,
            lr=config.lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            hodge_reward_weight=config.hodge_reward_weight,
            black_hole_penalty=config.black_hole_penalty,
            device=self.device,
        )
        
        # Action space
        self.action_space = self._build_action_space()
        
        # Training stats
        self.episode_rewards: List[float] = []
        self.episode_costs: List[float] = []
        self.h1_magnitudes: List[float] = []
    
    def _build_action_space(self) -> List[MCPAction]:
        """Define the discrete action space."""
        return [
            MCPAction(ActionType.OBSERVE, "look", {"target": "room"}),
            MCPAction(ActionType.ACT, "move", {"direction": "north"}),
            MCPAction(ActionType.ACT, "move", {"direction": "south"}),
            MCPAction(ActionType.ACT, "move", {"direction": "east"}),
            MCPAction(ActionType.ACT, "move", {"direction": "west"}),
            MCPAction(ActionType.ACT, "interact", {"target": "brick"}),
            MCPAction(ActionType.ACT, "interact", {"target": "key"}),
            MCPAction(ActionType.ACT, "interact", {"target": "guard"}),
            MCPAction(ActionType.ACT, "interact", {"target": "door"}),
            MCPAction(ActionType.COMPLETE, "complete", {"status": "done"}),
        ]
    
    def collect_episode(self, oracle: RuleBasedOracle) -> Tuple[RolloutBuffer, float, float, bool]:
        """
        Collect one episode of experience.
        
        Returns: (buffer, total_reward, total_cost, won)
        """
        buffer = RolloutBuffer()
        oracle.reset_game()
        
        initial_scene = oracle.rooms["cell"]["description"]
        current_page = Page(scene_description=initial_scene, step_number=0)
        
        total_reward = 0.0
        total_cost = 0.0
        done = False
        won = False
        
        while not done:
            # Embed current state
            state_embedding = self.embedding_model.encode([current_page.scene_description])[0]
            
            # Get action from policy
            action_idx, log_prob, value = self.optimizer.get_action(state_embedding)
            action = self.action_space[action_idx]
            
            # Execute action
            new_page, info = oracle.transition(current_page, action)
            
            # Get reward and cost
            reward = info.get("reward", 0.0)
            cost = info.get("cost", 0.0)
            
            # Check termination
            done, reason = oracle.is_terminal(new_page)
            won = reason == "escaped"
            
            # Store transition
            buffer.add(
                state=state_embedding,
                action=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                cost=cost,
                state_text=current_page.scene_description,
                action_text=str(action),
            )
            
            # Add to Hodge critic for learning
            self.hodge_critic.add_feedback(FeedbackItem(
                state_text=current_page.scene_description,
                action_text=str(action),
                next_state_text=new_page.scene_description,
                rank=reward / 10.0 + 0.5,  # Normalize to 0-1
                critique="Good" if reward > 0 else "Bad" if cost > 0 else "Neutral",
            ))
            
            total_reward += reward
            total_cost += cost
            current_page = new_page
        
        return buffer, total_reward, total_cost, won
    
    def learn_black_holes(self):
        """Identify dangerous regions from feedback and add as black holes."""
        # Find states with high costs (alarm triggered, caught, etc.)
        for item in self.hodge_critic.feedback_items:
            if item.rank < 0.45:  # Low rank = bad outcome (adjusted threshold)
                embedding = self.embedding_model.encode([item.state_text])[0]
                
                # Check if we already have a nearby black hole
                is_new = True
                for center in self.optimizer.black_hole_centers:
                    if np.linalg.norm(embedding - center) < 0.5:
                        is_new = False
                        break
                
                if is_new:
                    self.optimizer.add_black_hole(embedding, radius=0.3)
    
    def train(self, log_interval: int = 10):
        """
        Main training loop.
        """
        print("\n" + "="*60)
        print("STORYTELLING MACHINE SGPO TRAINING")
        print("="*60)
        print(f"Episodes: {self.config.total_episodes}")
        print(f"Rollout length: {self.config.rollout_length}")
        print(f"Hodge reward weight: {self.config.hodge_reward_weight}")
        print(f"Black hole penalty: {self.config.black_hole_penalty}")
        
        win_count = 0
        
        for ep in range(self.config.total_episodes):
            oracle = RuleBasedOracle()
            
            # Collect episode
            buffer, reward, cost, won = self.collect_episode(oracle)
            
            self.episode_rewards.append(reward)
            self.episode_costs.append(cost)
            if won:
                win_count += 1
            
            # Update policy
            if len(buffer.states) > 0:
                last_value = 0.0  # Terminal state value
                stats = self.optimizer.update(buffer, last_value)
                self.h1_magnitudes.append(stats.get("h1_magnitude", 0.0))
            
            # Learn black holes periodically
            if (ep + 1) % 20 == 0:
                self.learn_black_holes()
            
            # Log progress
            if (ep + 1) % log_interval == 0:
                recent_rewards = self.episode_rewards[-log_interval:]
                recent_costs = self.episode_costs[-log_interval:]
                win_rate = win_count / (ep + 1)
                
                print(f"\nEpisode {ep + 1}/{self.config.total_episodes}")
                print(f"  Avg reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg cost: {np.mean(recent_costs):.2f}")
                print(f"  Win rate: {win_rate:.1%}")
                print(f"  Black holes: {len(self.optimizer.black_hole_centers)}")
                if self.h1_magnitudes:
                    print(f"  H¹ magnitude: {self.h1_magnitudes[-1]:.4f}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Final win rate: {win_count / self.config.total_episodes:.1%}")
        print(f"Total black holes learned: {len(self.optimizer.black_hole_centers)}")
        
        return {
            "episode_rewards": self.episode_rewards,
            "episode_costs": self.episode_costs,
            "win_rate": win_count / self.config.total_episodes,
            "n_black_holes": len(self.optimizer.black_hole_centers),
        }
    
    def visualize_training(self):
        """Generate training visualization."""
        if not HAS_PLOTLY:
            print("Plotly not available for visualization.")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Episode Rewards",
                "Episode Costs",
                "H¹ Magnitude (Inconsistency)",
                "Cumulative Win Rate",
            ),
        )
        
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        # Rewards
        fig.add_trace(
            go.Scatter(x=episodes, y=self.episode_rewards, mode="lines", name="Reward"),
            row=1, col=1,
        )
        
        # Costs
        fig.add_trace(
            go.Scatter(x=episodes, y=self.episode_costs, mode="lines", name="Cost", line=dict(color="red")),
            row=1, col=2,
        )
        
        # H¹ magnitude
        if self.h1_magnitudes:
            fig.add_trace(
                go.Scatter(x=episodes[:len(self.h1_magnitudes)], y=self.h1_magnitudes, mode="lines", name="H¹"),
                row=2, col=1,
            )
        
        # Cumulative win rate
        wins = [1 if r > 5 else 0 for r in self.episode_rewards]
        cumulative_wins = np.cumsum(wins) / np.arange(1, len(wins) + 1)
        fig.add_trace(
            go.Scatter(x=episodes, y=cumulative_wins, mode="lines", name="Win Rate", line=dict(color="green")),
            row=2, col=2,
        )
        
        fig.update_layout(
            title="Storytelling Machine SGPO Training",
            height=600,
            width=1000,
            showlegend=False,
        )
        
        return fig


def main():
    parser = argparse.ArgumentParser(description="Train SGPO on Storytelling Machine")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hodge_weight", type=float, default=0.3, help="Hodge reward weight")
    parser.add_argument("--visualize", action="store_true", help="Show training visualization")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()
    
    config = TrainingConfig(
        total_episodes=args.episodes,
        lr=args.lr,
        hodge_reward_weight=args.hodge_weight,
    )
    
    trainer = StorytellingSGPOTrainer(config)
    results = trainer.train()
    
    if args.visualize:
        fig = trainer.visualize_training()
        if fig:
            fig.show()
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
