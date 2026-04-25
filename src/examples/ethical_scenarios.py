"""
Ethical Scenario Simulations

Three realistic ethical scenarios demonstrating:
1. Academic Integrity (Condorcet cycles)
2. Military Drone Decision (Black holes)
3. Business Ethics (Stakeholder conflicts)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Action:
    """Represents an action in an ethical scenario."""
    name: str
    description: str


@dataclass
class State:
    """Represents a state in an ethical scenario."""
    name: str
    features: np.ndarray


class AcademicIntegrityEnv:
    """
    Simulated environment for academic integrity decisions.
    
    Student asks AI for homework help. AI must decide:
    - A: Provide full solution (cheating)
    - B: Give hints only (helpful)
    - C: Refuse entirely (unhelpful but safe)
    """
    
    def __init__(self):
        self.actions = {
            "give_solution": Action("give_solution", "Provide complete solution"),
            "give_hint": Action("give_hint", "Give helpful hints only"),
            "refuse": Action("refuse", "Refuse to help"),
            "ask_clarification": Action("ask_clarification", "Ask clarifying questions"),
        }
        
        self.stakeholder_preferences = {
            "student": {
                ("give_solution", "give_hint"): 0.8,
                ("give_hint", "refuse"): 0.6,
                ("give_solution", "refuse"): 0.9,
                ("give_solution", "ask_clarification"): 0.7,
            },
            "teacher": {
                ("refuse", "give_hint"): 0.3,
                ("give_hint", "give_solution"): 0.9,
                ("refuse", "give_solution"): 0.95,
                ("ask_clarification", "give_solution"): 0.8,
            },
            "alignment": {
                ("give_hint", "refuse"): 0.4,
                ("give_hint", "give_solution"): 0.85,
                ("ask_clarification", "give_solution"): 0.9,
                ("ask_clarification", "refuse"): 0.3,
            },
        }
        
        self.stakeholder_weights = {
            "student": 0.3,
            "teacher": 0.5,
            "alignment": 0.2,
        }
    
    def get_aggregated_preferences(self) -> Dict[Tuple[str, str], float]:
        """Aggregate preferences - may create cycles!"""
        aggregated = {}
        
        all_pairs = set()
        for prefs in self.stakeholder_preferences.values():
            all_pairs.update(prefs.keys())
        
        for pair in all_pairs:
            score = 0.0
            for stakeholder, prefs in self.stakeholder_preferences.items():
                weight = self.stakeholder_weights[stakeholder]
                if pair in prefs:
                    score += weight * prefs[pair]
                elif (pair[1], pair[0]) in prefs:
                    score -= weight * prefs[(pair[1], pair[0])]
            
            aggregated[pair] = score
        
        return aggregated
    
    def step(self, action: str, state_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Execute action, return (next_state, reward_vector, done).
        
        Reward is a VECTOR, not scalar.
        """
        reward_vector = np.array([
            self._student_utility(action),
            self._teacher_utility(action),
            self._learning_utility(action),
        ])
        
        next_state = self._transition(state_features, action)
        done = action in ["give_solution", "refuse"]
        
        return next_state, reward_vector, done
    
    def _student_utility(self, action: str) -> float:
        """Student wants quick answers."""
        utilities = {
            "give_solution": 1.0,
            "give_hint": 0.5,
            "refuse": 0.0,
            "ask_clarification": 0.3,
        }
        return utilities.get(action, 0.0)
    
    def _teacher_utility(self, action: str) -> float:
        """Teacher wants learning, not cheating."""
        utilities = {
            "give_solution": -1.0,
            "give_hint": 0.8,
            "refuse": 0.5,
            "ask_clarification": 0.6,
        }
        return utilities.get(action, 0.0)
    
    def _learning_utility(self, action: str) -> float:
        """Actual learning outcome."""
        utilities = {
            "give_solution": -0.5,
            "give_hint": 1.0,
            "refuse": 0.0,
            "ask_clarification": 0.7,
        }
        return utilities.get(action, 0.0)
    
    def _transition(self, state: np.ndarray, action: str) -> np.ndarray:
        """Simple state transition."""
        return state + np.random.randn(len(state)) * 0.1
    
    def detect_condorcet_cycle(self) -> Tuple[bool, float]:
        """Check for Condorcet cycles using simple cycle detection."""
        prefs = self.get_aggregated_preferences()
        
        actions = ["give_solution", "give_hint", "refuse"]
        
        for i, a1 in enumerate(actions):
            for j, a2 in enumerate(actions):
                if i >= j:
                    continue
                for k, a3 in enumerate(actions):
                    if k <= j:
                        continue
                    
                    score_12 = prefs.get((a1, a2), 0) - prefs.get((a2, a1), 0)
                    score_23 = prefs.get((a2, a3), 0) - prefs.get((a3, a2), 0)
                    score_31 = prefs.get((a3, a1), 0) - prefs.get((a1, a3), 0)
                    
                    if score_12 > 0 and score_23 > 0 and score_31 > 0:
                        cycle_strength = min(score_12, score_23, score_31)
                        return True, cycle_strength
        
        return False, 0.0


class DroneDecisionEnv:
    """
    Military drone ethical decision environment.
    
    Autonomous drone must decide on engagement:
    - A: Engage target (potential civilian risk)
    - B: Request human confirmation (delay)
    - C: Abort mission (safe but mission failure)
    
    Black Hole: Region where civilian casualties are certain.
    """
    
    def __init__(self):
        self.state_dim = 3
        
        self.black_holes = [
            {
                "center": np.array([0.9, 0.9, 0.1]),
                "radius": 0.2,
                "strength": 10.0,
                "description": "Certain civilian casualties",
            },
            {
                "center": np.array([0.5, 0.8, 0.05]),
                "radius": 0.15,
                "strength": 5.0,
                "description": "High collateral damage risk",
            },
        ]
        
        self.actions = ["engage", "request_confirmation", "abort"]
    
    def compute_metric(self, state: np.ndarray) -> float:
        """
        Riemannian metric with singularities at black holes.
        
        Metric diverges as g ~ 1/dist^2 near black holes.
        """
        g = 1.0
        
        for bh in self.black_holes:
            dist = np.linalg.norm(state - bh["center"])
            if dist < bh["radius"] * 3:
                g += bh["strength"] / (dist ** 2 + 1e-6)
        
        return g
    
    def is_in_black_hole(self, state: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Check if state is inside a black hole region."""
        for bh in self.black_holes:
            dist = np.linalg.norm(state - bh["center"])
            if dist < bh["radius"]:
                return True, bh["description"]
        
        return False, None
    
    def geodesic_reward(self, state: np.ndarray, action: str, 
                       next_state: np.ndarray) -> float:
        """
        Reward adjusted by metric - dangerous paths are expensive.
        
        The metric acts as a "cost" that makes paths near black holes
        have very low reward, effectively repelling the policy.
        """
        base_reward = self._mission_reward(action)
        
        mid_state = (state + next_state) / 2
        path_cost = self.compute_metric(mid_state)
        
        return base_reward / np.sqrt(path_cost)
    
    def _mission_reward(self, action: str) -> float:
        """Base mission reward (before geometric adjustment)."""
        rewards = {
            "engage": 1.0,
            "request_confirmation": 0.5,
            "abort": -0.5,
        }
        return rewards.get(action, 0.0)
    
    def step(self, state: np.ndarray, action: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)."""
        next_state = self._transition(state, action)
        
        in_bh, bh_desc = self.is_in_black_hole(next_state)
        
        if in_bh:
            reward = -100.0
            done = True
            info = {"black_hole": True, "description": bh_desc}
        else:
            reward = self.geodesic_reward(state, action, next_state)
            done = action in ["engage", "abort"]
            info = {"black_hole": False}
        
        return next_state, reward, done, info
    
    def _transition(self, state: np.ndarray, action: str) -> np.ndarray:
        """State transition based on action."""
        if action == "engage":
            return state + np.array([0.1, 0.1, 0.0])
        elif action == "request_confirmation":
            return state + np.array([0.0, 0.0, 0.1])
        else:
            return state + np.array([-0.1, -0.1, 0.0])
    
    def visualize_metric_landscape(self, resolution: int = 50, 
                                   save_path: str = None):
        """Visualize the metric landscape with black holes."""
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([X[i, j], Y[i, j], 0.5])
                Z[i, j] = np.log10(self.compute_metric(state) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd')
        plt.colorbar(contour, ax=ax, label='log10(Metric + 1)')
        
        for bh in self.black_holes:
            circle = plt.Circle(
                (bh["center"][0], bh["center"][1]),
                bh["radius"],
                color='black',
                alpha=0.7,
                label='Black Hole'
            )
            ax.add_patch(circle)
            
            ax.text(bh["center"][0], bh["center"][1], "⚠",
                   fontsize=20, ha='center', va='center', color='white')
        
        ax.set_xlabel("Target Confidence")
        ax.set_ylabel("Civilian Proximity")
        ax.set_title("Drone Decision Metric Landscape\n(Darker = Higher Cost)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class BusinessEthicsEnv:
    """
    Business decision environment with stakeholder conflicts.
    
    AI business advisor must recommend:
    - A: Aggressive tactics (high profit, ethical concerns)
    - B: Standard practices (moderate profit, ethical)
    - C: Conservative approach (low profit, very safe)
    """
    
    def __init__(self):
        self.actions = ["aggressive", "standard", "conservative"]
        
        self.stakeholders = {
            "shareholders": {
                "weight": 0.4,
                "preferences": {
                    ("aggressive", "standard"): 0.8,
                    ("standard", "conservative"): 0.7,
                    ("aggressive", "conservative"): 0.9,
                },
            },
            "employees": {
                "weight": 0.3,
                "preferences": {
                    ("standard", "aggressive"): 0.6,
                    ("conservative", "aggressive"): 0.9,
                    ("standard", "conservative"): 0.4,
                },
            },
            "regulators": {
                "weight": 0.3,
                "preferences": {
                    ("conservative", "standard"): 0.7,
                    ("conservative", "aggressive"): 0.95,
                    ("standard", "aggressive"): 0.8,
                },
            },
        }
    
    def build_preference_graph(self) -> Dict[Tuple[str, str], float]:
        """Build aggregated preference graph."""
        aggregated = {}
        
        for pair in [("aggressive", "standard"), ("standard", "conservative"), 
                     ("conservative", "aggressive")]:
            score = 0.0
            
            for stakeholder, data in self.stakeholders.items():
                weight = data["weight"]
                prefs = data["preferences"]
                
                if pair in prefs:
                    score += weight * prefs[pair]
                elif (pair[1], pair[0]) in prefs:
                    score -= weight * prefs[(pair[1], pair[0])]
            
            aggregated[pair] = score
        
        return aggregated
    
    def detect_condorcet_cycles(self) -> Tuple[bool, float]:
        """Use simple cycle detection to find preference cycles."""
        pref_graph = self.build_preference_graph()
        
        a_to_s = pref_graph.get(("aggressive", "standard"), 0)
        s_to_c = pref_graph.get(("standard", "conservative"), 0)
        c_to_a = pref_graph.get(("conservative", "aggressive"), 0)
        
        if a_to_s > 0 and s_to_c > 0 and c_to_a > 0:
            h1_estimate = min(a_to_s, s_to_c, c_to_a)
            return True, h1_estimate
        
        s_to_a = -a_to_s
        c_to_s = -s_to_c
        a_to_c = -c_to_a
        
        if s_to_a > 0 and a_to_c > 0 and c_to_s > 0:
            h1_estimate = min(s_to_a, a_to_c, c_to_s)
            return True, h1_estimate
        
        return False, 0.0
    
    def analyze_stakeholder_conflicts(self) -> Dict:
        """Analyze conflicts between stakeholders."""
        conflicts = []
        
        for action1 in self.actions:
            for action2 in self.actions:
                if action1 >= action2:
                    continue
                
                votes = {}
                for stakeholder, data in self.stakeholders.items():
                    prefs = data["preferences"]
                    if (action1, action2) in prefs:
                        votes[stakeholder] = action1
                    elif (action2, action1) in prefs:
                        votes[stakeholder] = action2
                    else:
                        votes[stakeholder] = "neutral"
                
                if len(set(votes.values())) > 1:
                    conflicts.append({
                        "pair": (action1, action2),
                        "votes": votes,
                    })
        
        return {
            "conflicts": conflicts,
            "num_conflicts": len(conflicts),
        }


def run_examples():
    """Run all ethical scenario examples."""
    print("=" * 80)
    print("ETHICAL SCENARIO SIMULATIONS")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("1. ACADEMIC INTEGRITY SCENARIO")
    print("=" * 80)
    
    academic_env = AcademicIntegrityEnv()
    
    print("\nStakeholder preferences:")
    for stakeholder, weight in academic_env.stakeholder_weights.items():
        print(f"  {stakeholder} (weight={weight})")
    
    prefs = academic_env.get_aggregated_preferences()
    print("\nAggregated preferences (top 5):")
    for pair, score in sorted(prefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {pair[0]} vs {pair[1]}: {score:+.3f}")
    
    has_cycle, strength = academic_env.detect_condorcet_cycle()
    if has_cycle:
        print(f"\n⚠ CONDORCET CYCLE DETECTED! Strength: {strength:.3f}")
        print("  No scalar reward can satisfy all stakeholders.")
    else:
        print("\n✓ No Condorcet cycle detected")
    
    print("\n" + "=" * 80)
    print("2. MILITARY DRONE SCENARIO")
    print("=" * 80)
    
    drone_env = DroneDecisionEnv()
    
    print(f"\nBlack hole regions: {len(drone_env.black_holes)}")
    for i, bh in enumerate(drone_env.black_holes):
        print(f"  {i+1}. {bh['description']}")
        print(f"     Center: {bh['center']}, Radius: {bh['radius']}")
    
    test_states = [
        np.array([0.3, 0.3, 0.5]),
        np.array([0.9, 0.9, 0.1]),
        np.array([0.5, 0.8, 0.05]),
    ]
    
    print("\nMetric values at test states:")
    for i, state in enumerate(test_states):
        metric = drone_env.compute_metric(state)
        in_bh, desc = drone_env.is_in_black_hole(state)
        print(f"  State {i+1}: metric={metric:.2f}, in_black_hole={in_bh}")
        if in_bh:
            print(f"    ⚠ {desc}")
    
    print("\nGenerating metric landscape visualization...")
    fig = drone_env.visualize_metric_landscape(
        save_path="../../figures/examples/drone_metric_landscape.png"
    )
    print("✓ Saved to figures/examples/drone_metric_landscape.png")
    
    print("\n" + "=" * 80)
    print("3. BUSINESS ETHICS SCENARIO")
    print("=" * 80)
    
    business_env = BusinessEthicsEnv()
    
    print("\nStakeholders:")
    for stakeholder, data in business_env.stakeholders.items():
        print(f"  {stakeholder} (weight={data['weight']})")
    
    pref_graph = business_env.build_preference_graph()
    print("\nAggregated preferences:")
    for pair, score in pref_graph.items():
        print(f"  {pair[0]} vs {pair[1]}: {score:+.3f}")
    
    has_cycle, h1 = business_env.detect_condorcet_cycles()
    if has_cycle:
        print(f"\n⚠ CONDORCET CYCLE DETECTED! H¹ estimate: {h1:.3f}")
        print("  No scalar reward can satisfy all stakeholders.")
    else:
        print("\n✓ No Condorcet cycle detected")
    
    conflicts = business_env.analyze_stakeholder_conflicts()
    print(f"\nStakeholder conflicts: {conflicts['num_conflicts']}")
    for conflict in conflicts['conflicts']:
        print(f"  {conflict['pair']}: {conflict['votes']}")
    
    plt.show()
    
    return academic_env, drone_env, business_env


if __name__ == "__main__":
    academic, drone, business = run_examples()
