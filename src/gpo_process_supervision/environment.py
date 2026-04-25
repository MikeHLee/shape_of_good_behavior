"""
Navigation environment with topological anomalies.
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional


class AnomalyType(Enum):
    NONE = "none"
    BLACK_HOLE = "black_hole"
    CLIFF = "cliff"
    WORMHOLE = "wormhole"
    PLATEAU = "plateau"


class AnomalyNavigationEnv:
    """2D navigation environment with topological anomalies.
    
    Anomalies:
    - Black Holes: Catastrophic failure regions (hazards)
    - Cliffs: Narrow passages where small errors cause failure
    - Wormholes: Shortcuts that bypass intended behavior
    - Plateaus: Regions with no progress signal
    """
    
    def __init__(
        self,
        goal: np.ndarray = None,
        max_steps: int = 200,
        dt: float = 0.1,
        enable_wormhole: bool = True,
        enable_plateau: bool = True,
    ):
        self.goal = goal if goal is not None else np.array([2.5, 2.5])
        self.max_steps = max_steps
        self.dt = dt
        self.obs_dim = 4  # [x, y, vx, vy]
        self.act_dim = 2  # [ax, ay]
        
        # Black holes (hazards)
        self.black_holes = [
            {'center': np.array([1.0, 0.5]), 'radius': 0.3, 'severity': 'fatal'},
            {'center': np.array([0.5, 1.8]), 'radius': 0.25, 'severity': 'fatal'},
        ]
        
        # Cliff region (narrow passage)
        self.cliff_passage = {
            'x_range': (1.4, 1.6),
            'y_range': (1.0, 2.0),
            'safe_width': 0.12,
        }
        
        # Wormhole (shortcut)
        self.enable_wormhole = enable_wormhole
        self.wormhole = {
            'entry': np.array([0.3, 0.3]),
            'exit': np.array([2.2, 2.2]),
            'radius': 0.2,
        }
        
        # Plateau region
        self.enable_plateau = enable_plateau
        self.plateau = {
            'center': np.array([1.8, 0.5]),
            'radius': 0.4,
        }
        
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.steps = 0
        self.trajectory = [self.pos.copy()]
        self.step_anomalies: List[AnomalyType] = []
        self.used_wormhole = False
        self.plateau_steps = 0
        return self._obs()
    
    def _obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel]).astype(np.float32)
    
    def _check_anomalies(self) -> Tuple[AnomalyType, Dict]:
        """Check which anomaly the agent is currently in."""
        info = {}
        
        # Check black holes
        for bh in self.black_holes:
            dist = np.linalg.norm(self.pos - bh['center'])
            if dist < bh['radius']:
                return AnomalyType.BLACK_HOLE, {
                    'severity': bh['severity'],
                    'center': bh['center'].tolist(),
                    'distance': dist,
                }
        
        # Check cliff passage
        cp = self.cliff_passage
        if cp['x_range'][0] <= self.pos[0] <= cp['x_range'][1]:
            if cp['y_range'][0] <= self.pos[1] <= cp['y_range'][1]:
                center_x = (cp['x_range'][0] + cp['x_range'][1]) / 2
                dist_from_center = abs(self.pos[0] - center_x)
                if dist_from_center > cp['safe_width']:
                    return AnomalyType.CLIFF, {
                        'distance_from_safe': dist_from_center,
                        'position': self.pos.tolist(),
                    }
        
        # Check wormhole entry
        if self.enable_wormhole:
            wh = self.wormhole
            if np.linalg.norm(self.pos - wh['entry']) < wh['radius']:
                self.pos = wh['exit'].copy()
                self.used_wormhole = True
                return AnomalyType.WORMHOLE, {
                    'teleported': True,
                    'from': wh['entry'].tolist(),
                    'to': wh['exit'].tolist(),
                }
        
        # Check plateau
        if self.enable_plateau:
            pl = self.plateau
            if np.linalg.norm(self.pos - pl['center']) < pl['radius']:
                self.plateau_steps += 1
                if self.plateau_steps > 10:
                    return AnomalyType.PLATEAU, {
                        'steps_in_plateau': self.plateau_steps,
                        'center': pl['center'].tolist(),
                    }
            else:
                self.plateau_steps = 0
        
        return AnomalyType.NONE, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        self.vel = np.clip(self.vel + action * self.dt, -2.0, 2.0)
        self.pos = self.pos + self.vel * self.dt
        self.steps += 1
        self.trajectory.append(self.pos.copy())
        
        # Check anomalies
        anomaly_type, anomaly_info = self._check_anomalies()
        self.step_anomalies.append(anomaly_type)
        
        # Distance to goal
        dist = np.linalg.norm(self.pos - self.goal)
        reached_goal = dist < 0.3
        
        # Terminal conditions
        done = (
            reached_goal or
            anomaly_type == AnomalyType.BLACK_HOLE or
            anomaly_type == AnomalyType.CLIFF or
            self.steps >= self.max_steps
        )
        
        info = {
            'anomaly_type': anomaly_type,
            'anomaly_info': anomaly_info,
            'reached_goal': reached_goal,
            'distance_to_goal': dist,
            'used_wormhole': self.used_wormhole,
            'step': self.steps,
            'position': self.pos.tolist(),
        }
        
        return self._obs(), 0.0, done, info
    
    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)
    
    def get_step_anomalies(self) -> List[AnomalyType]:
        return self.step_anomalies
    
    def get_env_config(self) -> Dict:
        """Return environment configuration for visualization."""
        return {
            'goal': self.goal.tolist(),
            'black_holes': [
                {'center': bh['center'].tolist(), 'radius': bh['radius']}
                for bh in self.black_holes
            ],
            'cliff_passage': {
                'x_range': self.cliff_passage['x_range'],
                'y_range': self.cliff_passage['y_range'],
                'safe_width': self.cliff_passage['safe_width'],
            },
            'wormhole': {
                'entry': self.wormhole['entry'].tolist(),
                'exit': self.wormhole['exit'].tolist(),
                'radius': self.wormhole['radius'],
                'enabled': self.enable_wormhole,
            },
            'plateau': {
                'center': self.plateau['center'].tolist(),
                'radius': self.plateau['radius'],
                'enabled': self.enable_plateau,
            },
        }
