#!/usr/bin/env python3
"""
TextWorld Dataset Generator

Generates text adventure trajectories for training the Semantic State Machine.
Uses Microsoft TextWorld to create diverse goal-oriented games.

Outputs:
    - trajectories.jsonl: (state, action, next_state, reward) tuples
    - metadata.json: Dataset statistics and game configurations

Usage:
    python generate_textworld_data.py --num_games 100 --episodes_per_game 5 --output_dir data/textworld
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import textworld
    import textworld.gym
    from textworld import EnvInfos
    HAS_TEXTWORLD = True
except ImportError:
    HAS_TEXTWORLD = False
    print("Warning: TextWorld not installed. Run: pip install textworld")

try:
    import jericho
    HAS_JERICHO = True
except ImportError:
    HAS_JERICHO = False


@dataclass
class Transition:
    """Single state-action-state transition."""
    state: str
    action: str
    next_state: str
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class Episode:
    """Full episode trajectory."""
    game_id: str
    episode_id: int
    transitions: List[Dict]
    total_reward: float
    steps: int
    won: bool


def generate_textworld_games(
    output_dir: Path,
    num_games: int = 100,
    quest_length: int = 5,
    seed: int = 42,
) -> List[str]:
    """
    Generate TextWorld games of varying complexity.
    
    Returns list of paths to generated game files.
    """
    if not HAS_TEXTWORLD:
        raise RuntimeError("TextWorld not installed")
    
    game_files = []
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    
    for i in range(num_games):
        # Vary game parameters for diversity
        options = textworld.GameOptions()
        options.seeds = seed + i
        
        # Randomize complexity (use safer ranges)
        options.nb_rooms = random.randint(2, 5)
        options.nb_objects = random.randint(2, 5)
        options.quest_length = random.randint(1, min(3, quest_length))
        options.quest_breadth = 1  # Keep breadth simple to avoid generation failures
        
        try:
            # textworld.make() returns (game_file_path, game_object)
            game_file, game = textworld.make(options)
            game_files.append(game_file)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_games} games")
        except Exception as e:
            print(f"  Warning: Failed to generate game {i}: {e}")
    
    return game_files


def play_episode(
    game_file: str,
    max_steps: int = 50,
    exploration_rate: float = 0.3,
) -> Episode:
    """
    Play one episode of a TextWorld game using random exploration.
    
    Returns episode with all transitions.
    """
    request_infos = EnvInfos(
        description=True,
        inventory=True,
        admissible_commands=True,
        won=True,
        lost=True,
    )
    
    # Use TextWorld's native environment (not gym wrapper)
    env = textworld.start(game_file, request_infos)
    
    transitions = []
    game_state = env.reset()
    obs = game_state.description or game_state.feedback
    total_reward = 0.0
    done = False
    step = 0
    won = False
    
    while not done and step < max_steps:
        # Get admissible commands
        admissible = game_state.admissible_commands or ["look"]
        
        if not admissible:
            admissible = ["look"]
        
        # Epsilon-greedy action selection
        if random.random() < exploration_rate:
            action = random.choice(admissible)
        else:
            # Prefer goal-relevant actions (heuristic)
            goal_actions = [a for a in admissible if any(
                kw in a.lower() for kw in ["take", "open", "go", "unlock", "insert"]
            )]
            action = random.choice(goal_actions) if goal_actions else random.choice(admissible)
        
        game_state, reward, done = env.step(action)
        next_obs = game_state.description or game_state.feedback
        
        transitions.append(asdict(Transition(
            state=obs,
            action=action,
            next_state=next_obs,
            reward=reward,
            done=done,
            info={
                "inventory": game_state.inventory or "",
                "admissible_commands": admissible,
            },
        )))
        
        obs = next_obs
        total_reward += reward
        step += 1
        won = game_state.won if hasattr(game_state, 'won') else False
    
    env.close()
    
    return Episode(
        game_id=os.path.basename(game_file),
        episode_id=0,
        transitions=transitions,
        total_reward=total_reward,
        steps=step,
        won=won,
    )


def generate_dataset(
    output_dir: Path,
    num_games: int = 100,
    episodes_per_game: int = 5,
    quest_length: int = 5,
    seed: int = 42,
):
    """
    Generate full dataset of TextWorld trajectories.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TEXTWORLD DATASET GENERATION")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Games: {num_games}")
    print(f"Episodes per game: {episodes_per_game}")
    print(f"Max quest length: {quest_length}")
    
    # Generate games
    print(f"\n[1/3] Generating {num_games} TextWorld games...")
    game_files = generate_textworld_games(
        output_dir,
        num_games=num_games,
        quest_length=quest_length,
        seed=seed,
    )
    print(f"  Generated {len(game_files)} games")
    
    # Play episodes
    print(f"\n[2/3] Playing {episodes_per_game} episodes per game...")
    all_episodes = []
    total_transitions = 0
    wins = 0
    
    for gi, game_file in enumerate(game_files):
        for ep in range(episodes_per_game):
            try:
                episode = play_episode(game_file)
                episode.episode_id = ep
                all_episodes.append(asdict(episode))
                total_transitions += episode.steps
                if episode.won:
                    wins += 1
            except Exception as e:
                print(f"  Warning: Episode failed for {game_file}: {e}")
        
        if (gi + 1) % 20 == 0:
            print(f"  Processed {gi + 1}/{len(game_files)} games ({len(all_episodes)} episodes)")
    
    # Save dataset
    print(f"\n[3/3] Saving dataset...")
    
    # Save episodes as JSONL
    episodes_file = output_dir / "episodes.jsonl"
    with open(episodes_file, "w") as f:
        for episode in all_episodes:
            f.write(json.dumps(episode) + "\n")
    
    # Flatten to transitions
    transitions_file = output_dir / "transitions.jsonl"
    with open(transitions_file, "w") as f:
        for episode in all_episodes:
            for t in episode["transitions"]:
                t["game_id"] = episode["game_id"]
                t["episode_id"] = episode["episode_id"]
                f.write(json.dumps(t) + "\n")
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "num_games": len(game_files),
        "num_episodes": len(all_episodes),
        "num_transitions": total_transitions,
        "win_rate": wins / len(all_episodes) if all_episodes else 0,
        "config": {
            "quest_length": quest_length,
            "episodes_per_game": episodes_per_game,
            "seed": seed,
        },
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Episodes: {len(all_episodes)}")
    print(f"Transitions: {total_transitions}")
    print(f"Win rate: {metadata['win_rate']:.1%}")
    print(f"\nFiles:")
    print(f"  - {episodes_file}")
    print(f"  - {transitions_file}")
    print(f"  - {metadata_file}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate TextWorld dataset")
    parser.add_argument("--output_dir", type=str, default="data/textworld",
                        help="Output directory")
    parser.add_argument("--num_games", type=int, default=50,
                        help="Number of games to generate")
    parser.add_argument("--episodes_per_game", type=int, default=3,
                        help="Episodes per game")
    parser.add_argument("--quest_length", type=int, default=5,
                        help="Maximum quest length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    if not HAS_TEXTWORLD:
        print("ERROR: TextWorld not installed. Run:")
        print("  pip install textworld")
        return
    
    generate_dataset(
        output_dir=Path(args.output_dir),
        num_games=args.num_games,
        episodes_per_game=args.episodes_per_game,
        quest_length=args.quest_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
