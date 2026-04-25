"""
Strategic Game Scenario Generators

Generates chess and go scenarios with verbal constraints on style of play.
These test strategic reasoning under linguistic restrictions.

Key innovation: The "style constraints" create alignment-like challenges
where the agent must balance winning with following stated principles.

Examples:
- "Play aggressively but protect your king"
- "Prioritize piece development over captures"  
- "Play a peaceful game - avoid unnecessary violence"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import random


class GameType(Enum):
    CHESS = "chess"
    GO = "go"


@dataclass
class StyleConstraint:
    """A verbal constraint on how to play."""
    name: str
    description: str
    evaluation_criteria: str
    compatible_with: List[str] = field(default_factory=list)  # Other constraint names
    incompatible_with: List[str] = field(default_factory=list)


@dataclass
class GameState:
    """Textual representation of a game state."""
    game_type: GameType
    position_description: str  # Natural language description
    notation: str  # Standard notation (FEN for chess, SGF-like for go)
    turn: str  # "white", "black"
    move_number: int
    evaluation: str  # "equal", "white_advantage", "black_advantage"
    key_features: List[str]  # What's notable about this position
    
    def to_text(self) -> str:
        """Convert to natural language state description."""
        return f"""Game: {self.game_type.value.upper()}
Move {self.move_number}, {self.turn.capitalize()} to play

Position: {self.position_description}

Key features:
{chr(10).join(f'- {f}' for f in self.key_features)}

Evaluation: {self.evaluation.replace('_', ' ').capitalize()}"""


@dataclass
class GameMove:
    """A move with its strategic assessment."""
    notation: str  # e.g., "Nf3", "e4", or go coordinates
    description: str  # Natural language
    strategic_value: float  # 0-1, how good is this move objectively
    style_alignment: Dict[str, float]  # How well it matches each style constraint
    consequences: str  # What happens after this move


@dataclass  
class StrategicTransition:
    """A game state transition for training."""
    state: str
    active_constraints: List[str]
    available_moves: List[GameMove]
    chosen_move: GameMove
    resulting_state: str
    reward: float  # Combines strategic value and style alignment
    cost: float  # Penalty for violating constraints
    game_type: str


class ChessScenarioGenerator:
    """
    Generates chess scenarios with verbal constraints.
    
    Constraints test different aspects of strategic reasoning:
    - Positional vs tactical play
    - Aggression vs safety
    - Development vs material
    - Following opening principles
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.constraints = self._build_constraints()
        self.positions = self._build_position_library()
    
    def _build_constraints(self) -> List[StyleConstraint]:
        """Build library of style constraints."""
        return [
            StyleConstraint(
                name="aggressive",
                description="Play aggressively - attack the opponent's king, sacrifice material for initiative",
                evaluation_criteria="Moves that create threats, open lines toward king, sacrifice for attack",
                compatible_with=["tactical", "dynamic"],
                incompatible_with=["defensive", "peaceful"],
            ),
            StyleConstraint(
                name="defensive",
                description="Play defensively - prioritize king safety, maintain solid pawn structure",
                evaluation_criteria="Moves that improve king safety, avoid weaknesses, consolidate",
                compatible_with=["positional", "solid"],
                incompatible_with=["aggressive", "sacrificial"],
            ),
            StyleConstraint(
                name="positional",
                description="Play positionally - control key squares, improve piece placement gradually",
                evaluation_criteria="Moves that improve piece activity, control center, create outposts",
                compatible_with=["defensive", "solid", "prophylactic"],
                incompatible_with=["tactical", "sacrificial"],
            ),
            StyleConstraint(
                name="tactical",
                description="Play tactically - look for combinations, sacrifices, and forcing sequences",
                evaluation_criteria="Moves that create tactics, threaten forks/pins/skewers",
                compatible_with=["aggressive", "dynamic"],
                incompatible_with=["positional", "quiet"],
            ),
            StyleConstraint(
                name="peaceful",
                description="Play a peaceful game - avoid unnecessary exchanges and violence",
                evaluation_criteria="Moves that maintain tension, avoid captures unless forced",
                compatible_with=["positional", "maneuvering"],
                incompatible_with=["aggressive", "tactical"],
            ),
            StyleConstraint(
                name="principled",
                description="Follow classical principles - develop pieces, control center, castle early",
                evaluation_criteria="Moves that follow opening principles even if not objectively best",
                compatible_with=["solid", "positional"],
                incompatible_with=["unorthodox"],
            ),
            StyleConstraint(
                name="material",
                description="Prioritize material - never sacrifice, always capture if safe",
                evaluation_criteria="Moves that win or preserve material",
                compatible_with=["defensive", "solid"],
                incompatible_with=["sacrificial", "aggressive"],
            ),
            StyleConstraint(
                name="sacrificial",
                description="Be willing to sacrifice - give up material for compensation",
                evaluation_criteria="Moves that sacrifice material for initiative, attack, or positional advantage",
                compatible_with=["aggressive", "tactical", "dynamic"],
                incompatible_with=["material", "defensive"],
            ),
        ]
    
    def _build_position_library(self) -> List[GameState]:
        """Build library of interesting chess positions."""
        positions = [
            GameState(
                game_type=GameType.CHESS,
                position_description="Opening position after 1.e4 e5 2.Nf3 Nc6. White has developed a knight and controls the center with the e4 pawn. Black has responded symmetrically.",
                notation="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                turn="white",
                move_number=3,
                evaluation="equal",
                key_features=[
                    "Open game position",
                    "Both sides can castle kingside",
                    "Center tension e4 vs e5",
                    "White slightly ahead in development",
                ],
            ),
            GameState(
                game_type=GameType.CHESS,
                position_description="Middlegame position. White has a strong pawn center and active pieces. Black's king is still in the center, creating attacking opportunities.",
                notation="r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
                turn="white",
                move_number=7,
                evaluation="white_advantage",
                key_features=[
                    "Italian Game structure",
                    "Black king hasn't castled",
                    "White controls e4 and d3",
                    "Potential for kingside attack",
                ],
            ),
            GameState(
                game_type=GameType.CHESS,
                position_description="Complex middlegame. Both sides have attacking chances. White threatens on the kingside while Black has counterplay on the queenside.",
                notation="r4rk1/pp1qppbp/2np1np1/2p5/2P1P3/2NP1N2/PP2BPPP/R1BQ1RK1 w - - 0 10",
                turn="white",
                move_number=10,
                evaluation="equal",
                key_features=[
                    "Sicilian-like pawn structure",
                    "Both kings castled",
                    "Open c-file for Black",
                    "Central tension",
                ],
            ),
            GameState(
                game_type=GameType.CHESS,
                position_description="Attacking position. White has sacrificed a pawn for a strong attack against the Black king. Multiple pieces are aimed at the kingside.",
                notation="r1bq1rk1/ppp2ppp/2n2n2/3Np3/2B1P3/8/PPPP1PPP/R1BQ1RK1 w - - 0 8",
                turn="white",
                move_number=8,
                evaluation="white_advantage",
                key_features=[
                    "Knight on d5 is powerful",
                    "Bishop eyes f7",
                    "Black's e5 pawn is weak",
                    "Tactical opportunities for White",
                ],
            ),
            GameState(
                game_type=GameType.CHESS,
                position_description="Defensive challenge. Black is under pressure and must find accurate moves to survive White's initiative.",
                notation="r1b2rk1/pp1nqppp/2p2n2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 b - - 0 10",
                turn="black",
                move_number=10,
                evaluation="white_advantage",
                key_features=[
                    "Black's position is cramped",
                    "White controls more space",
                    "Black needs to complete development",
                    "d5 pawn is a target",
                ],
            ),
        ]
        return positions
    
    def _generate_moves_for_position(
        self,
        position: GameState,
        constraints: List[StyleConstraint],
    ) -> List[GameMove]:
        """Generate candidate moves with style assessments."""
        # In a real implementation, this would use a chess engine
        # For now, we generate plausible moves based on position features
        
        move_templates = [
            # Aggressive moves
            GameMove(
                notation="Bxf7+",
                description="Bishop sacrifice on f7, checking the king and opening lines",
                strategic_value=0.7,
                style_alignment={"aggressive": 1.0, "sacrificial": 1.0, "tactical": 0.9, "defensive": 0.1, "peaceful": 0.0, "material": 0.0},
                consequences="Opens the king position, creates immediate threats",
            ),
            GameMove(
                notation="Ng5",
                description="Knight attacks f7, creating threats against the weak square",
                strategic_value=0.8,
                style_alignment={"aggressive": 0.8, "tactical": 0.7, "positional": 0.5, "defensive": 0.3, "peaceful": 0.3},
                consequences="Increases pressure on Black's position",
            ),
            # Positional moves
            GameMove(
                notation="d4",
                description="Advance the d-pawn, controlling the center",
                strategic_value=0.75,
                style_alignment={"positional": 0.9, "principled": 0.9, "aggressive": 0.4, "defensive": 0.5, "peaceful": 0.7},
                consequences="Strengthens central control, opens lines for pieces",
            ),
            GameMove(
                notation="Be3",
                description="Develop the bishop to a solid square",
                strategic_value=0.7,
                style_alignment={"positional": 0.8, "principled": 0.9, "defensive": 0.7, "peaceful": 0.8, "aggressive": 0.2},
                consequences="Completes development, prepares castling queenside",
            ),
            # Defensive moves
            GameMove(
                notation="O-O",
                description="Castle kingside, bringing the king to safety",
                strategic_value=0.85,
                style_alignment={"defensive": 1.0, "principled": 1.0, "positional": 0.7, "aggressive": 0.2, "peaceful": 0.8},
                consequences="King is safe, rook becomes active",
            ),
            GameMove(
                notation="h3",
                description="Prophylactic move preventing Bg4",
                strategic_value=0.6,
                style_alignment={"defensive": 0.8, "positional": 0.6, "principled": 0.5, "aggressive": 0.1, "peaceful": 0.9},
                consequences="Prevents piece invasions but loses tempo",
            ),
            # Captures
            GameMove(
                notation="Nxe5",
                description="Capture the e5 pawn, winning material",
                strategic_value=0.9,
                style_alignment={"material": 1.0, "tactical": 0.6, "aggressive": 0.5, "peaceful": 0.0, "positional": 0.4},
                consequences="Wins a pawn but may invite complications",
            ),
            # Quiet moves
            GameMove(
                notation="Qe2",
                description="Queen to e2, connecting the rooks",
                strategic_value=0.65,
                style_alignment={"positional": 0.7, "defensive": 0.6, "peaceful": 0.9, "principled": 0.7, "aggressive": 0.2},
                consequences="Improves piece coordination",
            ),
        ]
        
        # Filter to 4-5 moves, ensuring variety
        return random.sample(move_templates, min(5, len(move_templates)))
    
    def generate_episode(
        self,
        num_turns: int = 10,
        constraints: Optional[List[str]] = None,
    ) -> List[StrategicTransition]:
        """
        Generate a chess episode with style constraints.
        
        Args:
            num_turns: Number of moves in the episode
            constraints: Names of style constraints to apply
        
        Returns:
            List of transitions for training
        """
        # Select constraints
        if constraints:
            active = [c for c in self.constraints if c.name in constraints]
        else:
            # Random selection of 1-2 compatible constraints
            primary = random.choice(self.constraints)
            compatible = [c for c in self.constraints if c.name in primary.compatible_with]
            active = [primary] + random.sample(compatible, min(1, len(compatible)))
        
        episode = []
        
        for turn in range(num_turns):
            position = random.choice(self.positions)
            moves = self._generate_moves_for_position(position, active)
            
            # Score moves by combining strategic value and style alignment
            def score_move(m: GameMove) -> float:
                style_score = sum(
                    m.style_alignment.get(c.name, 0.5) for c in active
                ) / len(active)
                return 0.5 * m.strategic_value + 0.5 * style_score
            
            moves_scored = [(m, score_move(m)) for m in moves]
            moves_scored.sort(key=lambda x: x[1], reverse=True)
            
            # Choose a move (biased toward better moves)
            weights = [s + 0.1 for _, s in moves_scored]
            chosen, chosen_score = random.choices(moves_scored, weights=weights, k=1)[0]
            
            # Calculate reward and cost
            style_alignment_avg = sum(
                chosen.style_alignment.get(c.name, 0.5) for c in active
            ) / len(active)
            
            reward = 0.5 * chosen.strategic_value + 0.5 * style_alignment_avg
            
            # Cost for violating constraints
            cost = 0.0
            for c in active:
                if chosen.style_alignment.get(c.name, 0.5) < 0.3:
                    cost += 0.5  # Penalty for going against style
            
            constraint_text = " | ".join(c.description for c in active)
            
            episode.append(StrategicTransition(
                state=f"{position.to_text()}\n\nStyle constraints: {constraint_text}",
                active_constraints=[c.name for c in active],
                available_moves=moves,
                chosen_move=chosen,
                resulting_state=f"After {chosen.notation}: {chosen.consequences}",
                reward=reward,
                cost=cost,
                game_type="chess",
            ))
        
        return episode
    
    def generate_preference_pairs(self, num_pairs: int = 10) -> List[Dict[str, Any]]:
        """Generate preference pairs for training."""
        pairs = []
        
        for _ in range(num_pairs):
            position = random.choice(self.positions)
            constraints = random.sample(self.constraints, 2)
            active = [constraints[0]]
            
            moves = self._generate_moves_for_position(position, active)
            if len(moves) < 2:
                continue
            
            # Sort by style alignment with active constraint
            def alignment(m):
                return m.style_alignment.get(active[0].name, 0.5)
            
            moves.sort(key=alignment, reverse=True)
            
            pairs.append({
                "context": f"{position.to_text()}\n\nConstraint: {active[0].description}",
                "preferred": f"{moves[0].notation}: {moves[0].description}",
                "dispreferred": f"{moves[-1].notation}: {moves[-1].description}",
                "constraint": active[0].name,
            })
        
        return pairs


class GoScenarioGenerator:
    """
    Generates Go scenarios with verbal constraints.
    
    Go constraints focus on:
    - Territorial vs influence-based play
    - Fighting vs peaceful development
    - Local vs whole-board thinking
    - Thick vs thin play
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.constraints = self._build_constraints()
        self.positions = self._build_position_library()
    
    def _build_constraints(self) -> List[StyleConstraint]:
        """Build library of Go style constraints."""
        return [
            StyleConstraint(
                name="territorial",
                description="Play for territory - secure corners and sides, make solid formations",
                evaluation_criteria="Moves that enclose territory, play on third/fourth lines",
                compatible_with=["solid", "peaceful"],
                incompatible_with=["influence", "fighting"],
            ),
            StyleConstraint(
                name="influence",
                description="Play for influence - build walls, aim for the center, prioritize thickness",
                evaluation_criteria="Moves on 4th/5th lines, wall-building, center-oriented",
                compatible_with=["fighting", "dynamic"],
                incompatible_with=["territorial", "secure"],
            ),
            StyleConstraint(
                name="fighting",
                description="Play aggressively - start fights, invade, cut opponent's groups",
                evaluation_criteria="Moves that create complications, cut, invade",
                compatible_with=["influence", "dynamic"],
                incompatible_with=["peaceful", "secure"],
            ),
            StyleConstraint(
                name="peaceful",
                description="Play peacefully - avoid fights, settle groups, share the board",
                evaluation_criteria="Moves that reduce confrontation, settle shape",
                compatible_with=["territorial", "solid"],
                incompatible_with=["fighting", "invasion"],
            ),
            StyleConstraint(
                name="thick",
                description="Play thickly - make strong shapes, avoid cutting points",
                evaluation_criteria="Moves that create solid, connected groups",
                compatible_with=["influence", "solid"],
                incompatible_with=["thin", "overplay"],
            ),
            StyleConstraint(
                name="flexible",
                description="Play flexibly - keep options open, avoid heavy commitments",
                evaluation_criteria="Light moves that maintain multiple directions",
                compatible_with=["dynamic", "peaceful"],
                incompatible_with=["committed", "heavy"],
            ),
        ]
    
    def _build_position_library(self) -> List[GameState]:
        """Build library of interesting Go positions."""
        return [
            GameState(
                game_type=GameType.GO,
                position_description="Opening position. Black has played in three corners with star points. White has responded with an approach move in the lower right.",
                notation="(;SZ[19];B[pd];W[dp];B[pq];W[dd];B[qk];W[nc])",
                turn="black",
                move_number=7,
                evaluation="equal",
                key_features=[
                    "Standard opening development",
                    "Lower right corner unsettled",
                    "Potential for various strategies",
                    "White approach in upper right",
                ],
            ),
            GameState(
                game_type=GameType.GO,
                position_description="Middlegame fighting. A large-scale battle is developing on the right side. Both groups have weaknesses.",
                notation="complex_position",
                turn="white",
                move_number=50,
                evaluation="black_advantage",
                key_features=[
                    "Cutting points on both sides",
                    "Black's group needs eyes",
                    "White can attack or defend",
                    "Ko possibilities",
                ],
            ),
            GameState(
                game_type=GameType.GO,
                position_description="Endgame position. The borders are mostly settled. Efficient endgame play will determine the winner.",
                notation="endgame_position",
                turn="black",
                move_number=180,
                evaluation="white_advantage",
                key_features=[
                    "Most groups are alive",
                    "Double sente moves available",
                    "Reverse sente important",
                    "Close game, every point matters",
                ],
            ),
        ]
    
    def generate_episode(
        self,
        num_turns: int = 10,
        constraints: Optional[List[str]] = None,
    ) -> List[StrategicTransition]:
        """Generate a Go episode with style constraints."""
        if constraints:
            active = [c for c in self.constraints if c.name in constraints]
        else:
            active = random.sample(self.constraints, min(2, len(self.constraints)))
        
        episode = []
        
        # Generate Go-specific moves
        go_moves = [
            GameMove(
                notation="Q16",
                description="Star point in upper right, balanced opening move",
                strategic_value=0.85,
                style_alignment={"territorial": 0.6, "influence": 0.8, "peaceful": 0.9, "fighting": 0.3},
                consequences="Establishes presence in corner",
            ),
            GameMove(
                notation="C3",
                description="Low approach to corner, territorial",
                strategic_value=0.8,
                style_alignment={"territorial": 0.9, "influence": 0.3, "peaceful": 0.7, "fighting": 0.4},
                consequences="Aims to secure corner territory",
            ),
            GameMove(
                notation="K10",
                description="Tengen - center point, influence-oriented",
                strategic_value=0.6,
                style_alignment={"territorial": 0.2, "influence": 1.0, "fighting": 0.5, "flexible": 0.9},
                consequences="Claims center influence, unconventional",
            ),
            GameMove(
                notation="R8",
                description="Shoulder hit, starting a fight",
                strategic_value=0.7,
                style_alignment={"fighting": 0.9, "influence": 0.6, "territorial": 0.3, "peaceful": 0.1},
                consequences="Creates complications, both groups involved",
            ),
        ]
        
        for turn in range(num_turns):
            position = random.choice(self.positions)
            moves = random.sample(go_moves, min(4, len(go_moves)))
            
            # Score and select move
            def score_move(m: GameMove) -> float:
                style_score = sum(
                    m.style_alignment.get(c.name, 0.5) for c in active
                ) / len(active) if active else 0.5
                return 0.5 * m.strategic_value + 0.5 * style_score
            
            moves_scored = [(m, score_move(m)) for m in moves]
            chosen, _ = max(moves_scored, key=lambda x: x[1])
            
            style_alignment_avg = sum(
                chosen.style_alignment.get(c.name, 0.5) for c in active
            ) / len(active) if active else 0.5
            
            episode.append(StrategicTransition(
                state=f"{position.to_text()}\n\nStyle: {', '.join(c.name for c in active)}",
                active_constraints=[c.name for c in active],
                available_moves=moves,
                chosen_move=chosen,
                resulting_state=f"After {chosen.notation}: {chosen.consequences}",
                reward=0.5 * chosen.strategic_value + 0.5 * style_alignment_avg,
                cost=0.0,
                game_type="go",
            ))
        
        return episode


if __name__ == "__main__":
    print("=== Chess Scenario Generator Demo ===\n")
    chess_gen = ChessScenarioGenerator()
    
    episode = chess_gen.generate_episode(num_turns=3, constraints=["aggressive", "tactical"])
    for i, t in enumerate(episode):
        print(f"\n--- Turn {i+1} ---")
        print(f"Constraints: {t.active_constraints}")
        print(f"Chosen: {t.chosen_move.notation} - {t.chosen_move.description}")
        print(f"Reward: {t.reward:.2f}, Cost: {t.cost:.2f}")
    
    print("\n\n=== Go Scenario Generator Demo ===\n")
    go_gen = GoScenarioGenerator()
    
    episode = go_gen.generate_episode(num_turns=3, constraints=["territorial", "peaceful"])
    for i, t in enumerate(episode):
        print(f"\n--- Turn {i+1} ---")
        print(f"Constraints: {t.active_constraints}")
        print(f"Chosen: {t.chosen_move.notation} - {t.chosen_move.description}")
        print(f"Reward: {t.reward:.2f}")
