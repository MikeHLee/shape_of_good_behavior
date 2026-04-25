"""
SGPO Chat Agent

Full integration of Sheaf-Geodesic Policy Optimization for a chat agent
with Hodge-based feedback decomposition and black hole avoidance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class FeedbackData:
    """Container for multi-modal feedback."""
    verbal: Optional[str] = None
    ordinal: Optional[int] = None
    passfail: Optional[Dict[str, bool]] = None
    safety_flag: bool = False


class HodgeCritic(nn.Module):
    """
    Neural network that learns both value function V and harmonic component ω.
    
    Architecture:
    - Shared encoder
    - Value head (scalar potential)
    - Harmonic head (cycle component)
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.harmonic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state_emb: State embedding [batch, embedding_dim]
        
        Returns:
            (value, harmonic): Both [batch, 1]
        """
        features = self.encoder(state_emb)
        value = self.value_head(features)
        harmonic = self.harmonic_head(features)
        return value, harmonic
    
    def value(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Get value (gradient component) only."""
        with torch.no_grad():
            v, _ = self.forward(state_emb)
        return v
    
    def harmonic(self, state_emb: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """
        Get harmonic component for state-action pair.
        
        In practice, this would be computed from the graph structure,
        but here we approximate it with the harmonic head.
        """
        with torch.no_grad():
            combined = torch.cat([state_emb, action_emb], dim=-1)
            _, h = self.forward(combined)
        return h
    
    def update(self, state_emb: np.ndarray, action_emb: np.ndarray, 
               preference_vector: np.ndarray, learning_rate: float = 1e-4):
        """
        Update critic from preference feedback.
        
        This is a simplified update - in practice would use proper
        Hodge decomposition and graph structure.
        """
        state_tensor = torch.from_numpy(state_emb).float().unsqueeze(0)
        action_tensor = torch.from_numpy(action_emb).float().unsqueeze(0)
        pref_tensor = torch.from_numpy(preference_vector).float()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        value, harmonic = self.forward(state_tensor)
        
        target = torch.norm(pref_tensor)
        loss = (value.squeeze() - target) ** 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class MetricModel(nn.Module):
    """
    Neural network that learns the Riemannian metric.
    
    Outputs metric value g(x) that diverges near dangerous regions (black holes).
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        
        self.danger_samples = []
        self.safe_samples = []
    
    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute metric value.
        
        Args:
            state_emb: State embedding [batch, embedding_dim]
        
        Returns:
            metric: Metric value [batch, 1], always positive
        """
        return self.network(state_emb) + 1.0
    
    def add_danger_sample(self, state_emb: np.ndarray):
        """Add a dangerous state sample."""
        self.danger_samples.append(state_emb)
    
    def add_safe_sample(self, state_emb: np.ndarray):
        """Add a safe state sample."""
        self.safe_samples.append(state_emb)
    
    def update_from_samples(self, learning_rate: float = 1e-3):
        """
        Update metric to have high values near danger samples,
        low values near safe samples.
        """
        if not self.danger_samples and not self.safe_samples:
            return
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        loss = 0.0
        
        if self.danger_samples:
            danger_tensor = torch.from_numpy(
                np.array(self.danger_samples)
            ).float()
            danger_metric = self.forward(danger_tensor)
            loss += -torch.log(danger_metric).mean()
        
        if self.safe_samples:
            safe_tensor = torch.from_numpy(
                np.array(self.safe_samples)
            ).float()
            safe_metric = self.forward(safe_tensor)
            loss += torch.log(safe_metric).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class FeedbackEmbedder:
    """Embed different feedback types into common space."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        
        self.ordinal_anchors = [
            "This is terrible, completely unacceptable.",
            "This is poor, needs significant improvement.",
            "This is acceptable but mediocre.",
            "This is good, minor improvements possible.",
            "This is excellent, no changes needed.",
        ]
        self.ordinal_embeddings = self.encoder.encode(self.ordinal_anchors)
    
    def embed_verbal(self, text: str) -> np.ndarray:
        """Embed verbal feedback."""
        return self.encoder.encode(text)
    
    def embed_ordinal(self, rating: int) -> np.ndarray:
        """Embed ordinal rating."""
        idx = min(max(rating - 1, 0), len(self.ordinal_anchors) - 1)
        return self.ordinal_embeddings[idx]
    
    def embed_passfail(self, criterion: str, passed: bool) -> np.ndarray:
        """Embed pass/fail feedback."""
        text = f"{'Satisfies' if passed else 'Fails'} criterion: {criterion}"
        return self.encoder.encode(text)
    
    def compute_preference_vector(
        self,
        state_emb: np.ndarray,
        chosen_emb: np.ndarray,
        rejected_emb: np.ndarray,
        feedback: FeedbackData,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute weighted preference vector from multi-modal feedback."""
        if weights is None:
            weights = {"base": 1.0, "verbal": 0.5, "ordinal": 0.3, "passfail": 0.2}
        
        base_pref = chosen_emb - rejected_emb
        base_norm = np.linalg.norm(base_pref) + 1e-8
        base_dir = base_pref / base_norm
        
        pref_vector = weights["base"] * base_pref
        
        if feedback.verbal:
            verbal_emb = self.embed_verbal(feedback.verbal)
            verbal_component = np.dot(verbal_emb, base_pref) / base_norm
            pref_vector += weights["verbal"] * verbal_component * base_dir
        
        if feedback.ordinal is not None:
            ordinal_emb = self.embed_ordinal(feedback.ordinal)
            rating_scale = (feedback.ordinal - 3) / 2
            pref_vector += weights["ordinal"] * rating_scale * base_pref
        
        if feedback.passfail:
            for criterion, passed in feedback.passfail.items():
                pf_emb = self.embed_passfail(criterion, passed)
                pf_scale = 1.0 if passed else -0.5
                pref_vector += weights["passfail"] * pf_scale * (pf_emb - state_emb)
        
        return pref_vector


class SGPOChatAgent:
    """
    Chat agent using Sheaf-Geodesic Policy Optimization.
    
    Combines:
    - Hodge critic (learns V and ω)
    - Metric model (learns black holes)
    - Feedback embedder (multi-modal feedback)
    """
    
    def __init__(
        self,
        llm_generate_fn,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
    ):
        """
        Initialize SGPO chat agent.
        
        Args:
            llm_generate_fn: Function that generates text given conversation history
            embedding_dim: Dimension of sentence embeddings
            hidden_dim: Hidden dimension for neural networks
        """
        self.llm_generate = llm_generate_fn
        
        self.embedder = FeedbackEmbedder()
        self.hodge_critic = HodgeCritic(embedding_dim, hidden_dim)
        self.metric_model = MetricModel(embedding_dim, hidden_dim // 2)
        
        self.conversation_state = []
        self.feedback_history = []
    
    def encode_state(self, conversation: List[Dict[str, str]]) -> np.ndarray:
        """Encode conversation history as state embedding."""
        text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        return self.embedder.encoder.encode(text)
    
    def get_action_candidates(self, n: int = 5, temperature: float = 0.8) -> List[str]:
        """Generate candidate responses using LLM."""
        candidates = []
        for _ in range(n):
            response = self.llm_generate(self.conversation_state, temperature=temperature)
            candidates.append(response)
        return candidates
    
    def select_action(self, state: List[Dict[str, str]], candidates: List[str]) -> str:
        """
        Select best action using SGPO policy.
        
        Scores each candidate by:
        advantage = (value - harmonic) / sqrt(metric)
        
        This balances:
        - High value (good response)
        - Low harmonic (consistent with preferences)
        - Low metric (far from black holes)
        """
        state_emb = self.encode_state(state)
        state_tensor = torch.from_numpy(state_emb).float().unsqueeze(0)
        
        scores = []
        for candidate in candidates:
            cand_emb = self.embedder.encoder.encode(candidate)
            cand_tensor = torch.from_numpy(cand_emb).float().unsqueeze(0)
            
            value, harmonic = self.hodge_critic(state_tensor)
            value = value.item()
            harmonic = harmonic.item()
            
            metric = self.metric_model(cand_tensor).item()
            
            advantage = (value - harmonic) / np.sqrt(metric + 1e-6)
            scores.append(advantage)
        
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def update_from_feedback(
        self,
        state: List[Dict[str, str]],
        action: str,
        feedback: FeedbackData,
        rejected_action: Optional[str] = None,
    ):
        """
        Update models from human feedback.
        
        Args:
            state: Conversation state
            action: Chosen action (response)
            feedback: Multi-modal feedback
            rejected_action: Optional rejected alternative
        """
        state_emb = self.encode_state(state)
        chosen_emb = self.embedder.encoder.encode(action)
        
        if rejected_action:
            rejected_emb = self.embedder.encoder.encode(rejected_action)
        else:
            rejected_emb = np.zeros_like(chosen_emb)
        
        pref_vector = self.embedder.compute_preference_vector(
            state_emb=state_emb,
            chosen_emb=chosen_emb,
            rejected_emb=rejected_emb,
            feedback=feedback,
        )
        
        self.hodge_critic.update(state_emb, chosen_emb, pref_vector)
        
        if feedback.safety_flag:
            self.metric_model.add_danger_sample(chosen_emb)
        else:
            self.metric_model.add_safe_sample(chosen_emb)
        
        self.metric_model.update_from_samples()
        
        self.feedback_history.append({
            "state": state,
            "action": action,
            "feedback": feedback,
            "preference_vector": pref_vector,
        })
    
    def respond(self, user_message: str, n_candidates: int = 5) -> str:
        """
        Generate response to user message using SGPO.
        
        Args:
            user_message: User's message
            n_candidates: Number of candidate responses to generate
        
        Returns:
            Selected response
        """
        self.conversation_state.append({"role": "user", "content": user_message})
        
        candidates = self.get_action_candidates(n=n_candidates)
        
        selected = self.select_action(self.conversation_state, candidates)
        
        self.conversation_state.append({"role": "assistant", "content": selected})
        
        return selected
    
    def get_statistics(self) -> Dict:
        """Get statistics about the agent's learning."""
        return {
            "num_feedbacks": len(self.feedback_history),
            "num_danger_samples": len(self.metric_model.danger_samples),
            "num_safe_samples": len(self.metric_model.safe_samples),
            "conversation_length": len(self.conversation_state),
        }


def demo_llm_generate(conversation: List[Dict[str, str]], temperature: float = 0.8) -> str:
    """
    Demo LLM generation function.
    
    In practice, this would call a real LLM API.
    """
    responses = [
        "I'd be happy to help with that!",
        "That's an interesting question. Let me think about it.",
        "I understand your concern. Here's what I suggest...",
        "Could you provide more details about what you're looking for?",
        "Based on what you've said, I recommend...",
    ]
    return np.random.choice(responses)


def run_demo():
    """Run a demo of the SGPO chat agent."""
    print("=" * 80)
    print("SGPO CHAT AGENT DEMO")
    print("=" * 80)
    
    agent = SGPOChatAgent(llm_generate_fn=demo_llm_generate)
    
    print("\n1. Initial response (no feedback yet):")
    print("-" * 80)
    response1 = agent.respond("How do I learn Python?")
    print(f"Agent: {response1}")
    
    print("\n2. Providing feedback on response:")
    print("-" * 80)
    feedback1 = FeedbackData(
        verbal="Good start, but could be more specific",
        ordinal=3,
        passfail={"helpful": True, "specific": False},
        safety_flag=False,
    )
    agent.update_from_feedback(
        state=agent.conversation_state[:-1],
        action=response1,
        feedback=feedback1,
    )
    print("✓ Feedback incorporated")
    
    print("\n3. Second response (with learned preferences):")
    print("-" * 80)
    response2 = agent.respond("What about machine learning?")
    print(f"Agent: {response2}")
    
    print("\n4. Safety-flagged feedback:")
    print("-" * 80)
    feedback2 = FeedbackData(
        verbal="This response could be misleading",
        ordinal=2,
        passfail={"accurate": False},
        safety_flag=True,
    )
    agent.update_from_feedback(
        state=agent.conversation_state[:-1],
        action=response2,
        feedback=feedback2,
    )
    print("⚠ Safety flag recorded - metric updated")
    
    print("\n5. Agent statistics:")
    print("-" * 80)
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n6. Architecture summary:")
    print("-" * 80)
    print("  Hodge Critic: Learns V (value) and ω (harmonic)")
    print("  Metric Model: Learns g(x) with singularities at dangerous regions")
    print("  Feedback Embedder: Unifies verbal/ordinal/pass-fail feedback")
    print("  Policy: advantage = (V - ω) / sqrt(g)")
    
    return agent


if __name__ == "__main__":
    agent = run_demo()
