"""
Topological Safety Gym: General-purpose safety library using sheaf theory.

Extends sheaf-theoretic safety beyond text embeddings to arbitrary decision spaces:
- Continuous control (MuJoCo, robotics)
- Discrete navigation (grid worlds)
- Image-based control (Atari, visual robotics)
- Hybrid spaces

Key Components:
- TopologicalSpace: Abstract base class for any decision space
- ContinuousControlSpace: For MuJoCo-style environments
- DiscreteNavigationSpace: For grid worlds and discrete tasks
- ImageStateSpace: For visual observations
- TopologicalSafetyWrapper: Gym wrapper adding safety metrics
"""

from .topological_space import TopologicalSpace
from .continuous_space import ContinuousControlSpace
from .discrete_space import DiscreteNavigationSpace
from .wrapper import TopologicalSafetyWrapper

__version__ = "0.1.0"

__all__ = [
    "TopologicalSpace",
    "ContinuousControlSpace",
    "DiscreteNavigationSpace",
    "TopologicalSafetyWrapper",
]
