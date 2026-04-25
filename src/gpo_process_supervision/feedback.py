"""
Process supervision feedback data structures and simulated supervisor.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal

from environment import AnomalyNavigationEnv, AnomalyType


@dataclass
class StepFeedback:
    """Per-step process supervision feedback."""
    step_idx: int
    progress: Literal[-1, 0, 1]  # -1=regress, 0=same, 1=progress
    quality: float  # 0-1 step quality score
    is_discontinuity: bool  # Did something suddenly change?
    anomaly_flag: Optional[AnomalyType] = None
    comment: Optional[str] = None


@dataclass
class TrajectoryFeedback:
    """Complete trajectory with multi-scale feedback."""
    id: str
    observations: np.ndarray
    actions: np.ndarray
    path: np.ndarray
    
    # Outcome-level feedback
    overall_quality: float  # 1-5 scale
    reached_goal: bool
    hit_catastrophe: bool
    total_steps: int
    
    # Step-level feedback (process supervision)
    step_feedback: List[StepFeedback] = field(default_factory=list)
    
    # Anomaly probes
    shortcut_detected: bool = False
    shortcut_assessment: Optional[Literal[
        "legitimate_efficiency",
        "gaming_the_system",
        "violates_spirit",
        "unsure"
    ]] = None
    
    critical_step: Optional[int] = None  # Cliff detection
    plateau_range: Optional[Tuple[int, int]] = None
    
    # Metadata
    used_wormhole: bool = False
    
    def __hash__(self):
        return hash(self.id)
    
    def get_progress_curve(self) -> List[int]:
        """Get progress values for all steps."""
        return [sf.progress for sf in self.step_feedback]
    
    def get_quality_curve(self) -> List[float]:
        """Get quality values for all steps."""
        return [sf.quality for sf in self.step_feedback]
    
    def get_discontinuities(self) -> List[int]:
        """Get indices of discontinuity steps."""
        return [sf.step_idx for sf in self.step_feedback if sf.is_discontinuity]


@dataclass
class AnomalyCandidate:
    """Detected anomaly candidate."""
    anomaly_type: AnomalyType
    trajectory_id: str
    location: Optional[np.ndarray] = None
    step_range: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    residual: float = 0.0
    description: str = ""


class ProcessSupervisor:
    """Simulates human process supervision feedback.
    
    Can operate in two modes:
    1. Automatic (simulated): Uses ground truth to generate feedback
    2. Interactive: Stores partial feedback for human input via UI
    """
    
    def __init__(
        self,
        env: AnomalyNavigationEnv,
        noise_level: float = 0.1,
        interactive: bool = False,
    ):
        self.env = env
        self.noise_level = noise_level
        self.interactive = interactive
        self._pending_feedback: Dict[str, TrajectoryFeedback] = {}
    
    def evaluate_trajectory_auto(self, traj_data: Dict) -> TrajectoryFeedback:
        """Automatically generate feedback (simulated human)."""
        path = traj_data['path']
        obs = traj_data['observations']
        actions = traj_data['actions']
        step_anomalies = traj_data.get('step_anomalies', [])
        
        goal = self.env.goal
        
        # Generate step-level feedback
        step_feedback = []
        prev_dist = np.linalg.norm(path[0] - goal)
        
        for i in range(len(actions)):
            curr_pos = path[i + 1] if i + 1 < len(path) else path[-1]
            curr_dist = np.linalg.norm(curr_pos - goal)
            
            # Progress indicator
            if curr_dist < prev_dist - 0.05:
                progress = 1
            elif curr_dist > prev_dist + 0.05:
                progress = -1
            else:
                progress = 0
            
            # Quality score
            base_quality = 0.5 + 0.3 * progress
            
            # Penalize being near hazards
            for bh in self.env.black_holes:
                dist_to_hazard = np.linalg.norm(curr_pos - bh['center'])
                if dist_to_hazard < bh['radius'] * 2:
                    base_quality -= 0.2 * (1 - dist_to_hazard / (bh['radius'] * 2))
            
            quality = np.clip(base_quality, 0.0, 1.0)
            
            # Discontinuity detection
            is_discontinuity = (
                i < len(step_anomalies) and
                step_anomalies[i] != AnomalyType.NONE
            )
            
            anomaly_flag = step_anomalies[i] if i < len(step_anomalies) else None
            
            # Add noise
            if random.random() < self.noise_level:
                progress = random.choice([-1, 0, 1])
            
            step_feedback.append(StepFeedback(
                step_idx=i,
                progress=progress,
                quality=quality,
                is_discontinuity=is_discontinuity,
                anomaly_flag=anomaly_flag,
            ))
            
            prev_dist = curr_dist
        
        # Outcome-level evaluation
        reached_goal = traj_data.get('reached_goal', False)
        hit_catastrophe = any(
            a in [AnomalyType.BLACK_HOLE, AnomalyType.CLIFF]
            for a in step_anomalies
        )
        used_wormhole = traj_data.get('used_wormhole', False)
        
        # Overall quality
        if hit_catastrophe:
            overall_quality = 1.0
        elif reached_goal:
            overall_quality = 5.0 if not used_wormhole else 3.0
        else:
            overall_quality = 2.0
        
        # Anomaly probes
        shortcut_detected = used_wormhole
        shortcut_assessment = "gaming_the_system" if used_wormhole else None
        
        # Find critical step (cliff)
        critical_step = None
        for sf in step_feedback:
            if sf.anomaly_flag == AnomalyType.CLIFF:
                critical_step = sf.step_idx
                break
        
        # Find plateau range
        plateau_range = None
        plateau_start = None
        for i, sf in enumerate(step_feedback):
            if sf.anomaly_flag == AnomalyType.PLATEAU:
                if plateau_start is None:
                    plateau_start = i
            elif plateau_start is not None:
                plateau_range = (plateau_start, i)
                break
        
        return TrajectoryFeedback(
            id=f"traj_{random.randint(0, 1000000)}",
            observations=obs,
            actions=actions,
            path=path,
            overall_quality=overall_quality,
            reached_goal=reached_goal,
            hit_catastrophe=hit_catastrophe,
            total_steps=len(actions),
            step_feedback=step_feedback,
            shortcut_detected=shortcut_detected,
            shortcut_assessment=shortcut_assessment,
            critical_step=critical_step,
            plateau_range=plateau_range,
            used_wormhole=used_wormhole,
        )
    
    def create_pending_feedback(self, traj_data: Dict) -> str:
        """Create pending feedback for interactive evaluation."""
        traj_id = f"traj_{random.randint(0, 1000000)}"
        
        path = traj_data['path']
        obs = traj_data['observations']
        actions = traj_data['actions']
        step_anomalies = traj_data.get('step_anomalies', [])
        
        # Create partial feedback (step-level to be filled by human)
        self._pending_feedback[traj_id] = TrajectoryFeedback(
            id=traj_id,
            observations=obs,
            actions=actions,
            path=path,
            overall_quality=0.0,  # To be filled
            reached_goal=traj_data.get('reached_goal', False),
            hit_catastrophe=any(
                a in [AnomalyType.BLACK_HOLE, AnomalyType.CLIFF]
                for a in step_anomalies
            ),
            total_steps=len(actions),
            step_feedback=[],  # To be filled
            used_wormhole=traj_data.get('used_wormhole', False),
        )
        
        return traj_id
    
    def update_feedback(
        self,
        traj_id: str,
        overall_quality: Optional[float] = None,
        step_feedback: Optional[List[StepFeedback]] = None,
        shortcut_detected: Optional[bool] = None,
        shortcut_assessment: Optional[str] = None,
        critical_step: Optional[int] = None,
        plateau_range: Optional[Tuple[int, int]] = None,
    ) -> TrajectoryFeedback:
        """Update pending feedback with human input."""
        if traj_id not in self._pending_feedback:
            raise ValueError(f"No pending feedback for {traj_id}")
        
        fb = self._pending_feedback[traj_id]
        
        if overall_quality is not None:
            fb.overall_quality = overall_quality
        if step_feedback is not None:
            fb.step_feedback = step_feedback
        if shortcut_detected is not None:
            fb.shortcut_detected = shortcut_detected
        if shortcut_assessment is not None:
            fb.shortcut_assessment = shortcut_assessment
        if critical_step is not None:
            fb.critical_step = critical_step
        if plateau_range is not None:
            fb.plateau_range = plateau_range
        
        return fb
    
    def finalize_feedback(self, traj_id: str) -> TrajectoryFeedback:
        """Finalize and return pending feedback."""
        if traj_id not in self._pending_feedback:
            raise ValueError(f"No pending feedback for {traj_id}")
        
        fb = self._pending_feedback.pop(traj_id)
        return fb
    
    def get_pending(self, traj_id: str) -> Optional[TrajectoryFeedback]:
        """Get pending feedback for review."""
        return self._pending_feedback.get(traj_id)
