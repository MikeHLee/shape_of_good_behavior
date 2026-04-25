"""
Streamlit app for SGPO with interactive process supervision.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.figure import Figure
import torch
from typing import Dict, List, Optional
import time

from environment import AnomalyNavigationEnv, AnomalyType
from feedback import StepFeedback, TrajectoryFeedback, ProcessSupervisor
from models import AnomalyAwareMetric
from trainer import SGPOTrainer, TrainingConfig

# Page config
st.set_page_config(
    page_title="SGPO Process Supervision Demo",
    page_icon="🎯",
    layout="wide",
)

# Initialize session state
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'current_traj' not in st.session_state:
    st.session_state.current_traj = None
if 'current_traj_id' not in st.session_state:
    st.session_state.current_traj_id = None
if 'training_mode' not in st.session_state:
    st.session_state.training_mode = 'interactive'
if 'episode_results' not in st.session_state:
    st.session_state.episode_results = []
if 'step_feedback' not in st.session_state:
    st.session_state.step_feedback = {}
if 'use_gpo' not in st.session_state:
    st.session_state.use_gpo = True
if 'experiment_log' not in st.session_state:
    st.session_state.experiment_log = []


def log_event(event: str, details: str = ""):
    """Add event to experiment log."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.experiment_log.append({
        'time': timestamp,
        'event': event,
        'details': details,
    })
    if len(st.session_state.experiment_log) > 50:
        st.session_state.experiment_log = st.session_state.experiment_log[-50:]


def create_env_plot(env: AnomalyNavigationEnv, trajectory: np.ndarray = None) -> Figure:
    """Create environment visualization."""
    fig, ax = plt.subplots(figsize=(8, 8))
    config = env.get_env_config()
    
    # Draw black holes (hazards)
    for i, bh in enumerate(config['black_holes']):
        circle = Circle(
            bh['center'], bh['radius'],
            fill=True, color='red', alpha=0.4,
            label='Black Hole' if i == 0 else None
        )
        ax.add_patch(circle)
        ax.plot(*bh['center'], 'rx', markersize=15, markeredgewidth=3)
    
    # Draw cliff passage
    cp = config['cliff_passage']
    rect = Rectangle(
        (cp['x_range'][0], cp['y_range'][0]),
        cp['x_range'][1] - cp['x_range'][0],
        cp['y_range'][1] - cp['y_range'][0],
        fill=True, color='orange', alpha=0.3, label='Cliff Zone'
    )
    ax.add_patch(rect)
    
    # Draw safe zone in cliff
    center_x = (cp['x_range'][0] + cp['x_range'][1]) / 2
    safe_rect = Rectangle(
        (center_x - cp['safe_width'], cp['y_range'][0]),
        cp['safe_width'] * 2,
        cp['y_range'][1] - cp['y_range'][0],
        fill=True, color='green', alpha=0.3, label='Safe Path'
    )
    ax.add_patch(safe_rect)
    
    # Draw wormhole
    if config['wormhole']['enabled']:
        wh = config['wormhole']
        entry_circle = Circle(
            wh['entry'], wh['radius'],
            fill=True, color='purple', alpha=0.4, label='Wormhole'
        )
        ax.add_patch(entry_circle)
        exit_circle = Circle(
            wh['exit'], wh['radius'],
            fill=False, color='purple', linestyle='--', linewidth=2
        )
        ax.add_patch(exit_circle)
        ax.annotate(
            '', xy=wh['exit'], xytext=wh['entry'],
            arrowprops=dict(arrowstyle='->', color='purple', lw=2, ls='--')
        )
    
    # Draw plateau
    if config['plateau']['enabled']:
        pl = config['plateau']
        plateau_circle = Circle(
            pl['center'], pl['radius'],
            fill=True, color='gray', alpha=0.3, label='Plateau'
        )
        ax.add_patch(plateau_circle)
    
    # Draw goal and start
    ax.plot(*config['goal'], 'g*', markersize=25, label='Goal')
    ax.plot(0, 0, 'ko', markersize=12, label='Start')
    
    # Draw trajectory
    if trajectory is not None and len(trajectory) > 1:
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
        for i in range(len(trajectory) - 1):
            ax.plot(
                [trajectory[i, 0], trajectory[i+1, 0]],
                [trajectory[i, 1], trajectory[i+1, 1]],
                color=colors[i], linewidth=2
            )
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'o', color=colors[-1], markersize=10)
    
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(-0.5, 3.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Navigation Environment')
    plt.tight_layout()
    return fig


def create_metric_plot(metric: AnomalyAwareMetric, env: AnomalyNavigationEnv) -> Figure:
    """Create Riemannian metric visualization."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    X, Y, Z = metric.get_metric_field(resolution=40)
    contour = ax.contourf(X, Y, Z, levels=20, cmap='hot')
    fig.colorbar(contour, ax=ax, label='log(g(x) + 1)')
    
    config = env.get_env_config()
    for bh in config['black_holes']:
        circle = Circle(
            bh['center'], bh['radius'],
            fill=False, color='white', linewidth=2
        )
        ax.add_patch(circle)
    
    ax.plot(*config['goal'], 'g*', markersize=20)
    ax.plot(0, 0, 'wo', markersize=10)
    
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(-0.5, 3.0)
    ax.set_aspect('equal')
    ax.set_title('Learned Riemannian Metric\n(Higher = More Dangerous)')
    plt.tight_layout()
    return fig


def create_progress_plot(traj_data: Dict, env: AnomalyNavigationEnv) -> Figure:
    """Create step-by-step progress visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    path = traj_data['path']
    goal = env.goal
    
    distances = [np.linalg.norm(p - goal) for p in path]
    ax1 = axes[0]
    ax1.plot(distances, 'b-', linewidth=2)
    ax1.set_ylabel('Distance to Goal')
    ax1.set_title('Progress Over Time')
    ax1.grid(True, alpha=0.3)
    
    anomalies = traj_data.get('step_anomalies', [])
    for i, anom in enumerate(anomalies):
        if anom != AnomalyType.NONE and i < len(distances):
            color = {
                AnomalyType.BLACK_HOLE: 'red',
                AnomalyType.CLIFF: 'orange',
                AnomalyType.WORMHOLE: 'purple',
                AnomalyType.PLATEAU: 'gray',
            }.get(anom, 'black')
            ax1.axvline(i, color=color, alpha=0.5, linestyle='--')
    
    if len(path) > 1:
        velocities = [np.linalg.norm(path[i+1] - path[i]) / 0.1 for i in range(len(path)-1)]
        ax2 = axes[1]
        ax2.plot(velocities, 'g-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Speed')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def render_algorithm_explanation():
    """Render explanation of PPO vs SGPO."""
    st.markdown("""
    ### 🔬 Algorithm Comparison
    
    | Aspect | PPO (Standard) | SGPO (Geodesic) |
    |--------|---------------|----------------|
    | **Advantage** | A(s,a) = Q(s,a) - V(s) | A_geo = A(s,a) / √g(s) |
    | **Near hazards** | Normal updates | Suppressed updates (high g) |
    | **Safety** | Soft penalties | Geometric barriers |
    | **Key idea** | Clip policy ratio | Warp reward space |
    """)


def render_step_feedback_ui(traj_data: Dict, traj_id: str) -> Dict:
    """Render UI for step-level feedback."""
    st.subheader("📝 Step-Level Feedback")
    
    path = traj_data['path']
    n_steps = len(traj_data['actions'])
    
    if n_steps == 0:
        st.warning("No actions in trajectory")
        return {'progress': [], 'quality': [], 'discontinuities': []}
    
    if traj_id not in st.session_state.step_feedback:
        st.session_state.step_feedback[traj_id] = {
            'progress': [0] * n_steps,
            'quality': [0.5] * n_steps,
            'discontinuities': [],
        }
    
    fb = st.session_state.step_feedback[traj_id]
    
    step_idx = st.slider("Select Step to Review", 0, max(0, n_steps - 1), 0, key=f"step_slider_{traj_id}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Step {step_idx}** of {n_steps}")
        if step_idx < len(path):
            st.write(f"Position: ({path[step_idx][0]:.2f}, {path[step_idx][1]:.2f})")
        
        progress_options = [("↑ Closer to goal", 1), ("→ Same distance", 0), ("↓ Further away", -1)]
        current_idx = 1 - fb['progress'][step_idx] if step_idx < len(fb['progress']) else 1
        progress = st.radio(
            "Progress toward goal:",
            options=progress_options,
            format_func=lambda x: x[0],
            key=f"progress_{traj_id}_{step_idx}",
            index=max(0, min(2, current_idx)),
        )
        if step_idx < len(fb['progress']):
            fb['progress'][step_idx] = progress[1]
    
    with col2:
        current_quality = fb['quality'][step_idx] if step_idx < len(fb['quality']) else 0.5
        quality = st.slider(
            "Step Quality (0=bad, 1=good)",
            0.0, 1.0, current_quality,
            key=f"quality_{traj_id}_{step_idx}"
        )
        if step_idx < len(fb['quality']):
            fb['quality'][step_idx] = quality
        
        is_discontinuity = st.checkbox(
            "⚠️ Something suddenly changed here (discontinuity)",
            value=step_idx in fb['discontinuities'],
            key=f"disc_{traj_id}_{step_idx}"
        )
        if is_discontinuity and step_idx not in fb['discontinuities']:
            fb['discontinuities'].append(step_idx)
        elif not is_discontinuity and step_idx in fb['discontinuities']:
            fb['discontinuities'].remove(step_idx)
    
    return fb


def render_anomaly_probes_ui(traj_data: Dict, traj_id: str) -> Dict:
    """Render UI for anomaly-specific probes."""
    st.subheader("🔍 Anomaly Detection Probes")
    
    n_steps = len(traj_data['actions'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🌀 Wormhole Detection**")
        st.caption("Did the agent take a shortcut that bypasses intended behavior?")
        shortcut_detected = st.checkbox(
            "Shortcut detected",
            value=traj_data.get('used_wormhole', False),
            key=f"shortcut_{traj_id}"
        )
        
        shortcut_assessment = None
        if shortcut_detected:
            shortcut_assessment = st.selectbox(
                "Assessment:",
                options=[
                    "legitimate_efficiency",
                    "gaming_the_system",
                    "violates_spirit",
                    "unsure"
                ],
                format_func=lambda x: {
                    "legitimate_efficiency": "✅ Legitimate efficiency",
                    "gaming_the_system": "⚠️ Gaming the system",
                    "violates_spirit": "❌ Violates spirit of task",
                    "unsure": "❓ Unsure",
                }.get(x, x),
                key=f"shortcut_assess_{traj_id}"
            )
    
    with col2:
        st.write("**🏔️ Cliff Detection**")
        st.caption("Was there a critical step where a small error caused failure?")
        critical_step = st.number_input(
            "Critical step (-1 if none):",
            min_value=-1,
            max_value=max(0, n_steps - 1),
            value=-1,
            key=f"critical_{traj_id}"
        )
        critical_step = None if critical_step < 0 else critical_step
        
        st.write("**📉 Plateau Detection**")
        st.caption("Did the agent spin its wheels without making progress?")
        plateau_start = st.number_input(
            "Plateau start (-1 if none):",
            min_value=-1,
            max_value=max(0, n_steps - 1),
            value=-1,
            key=f"plateau_start_{traj_id}"
        )
        plateau_end = st.number_input(
            "Plateau end:",
            min_value=-1,
            max_value=n_steps,
            value=-1,
            key=f"plateau_end_{traj_id}"
        )
        
        plateau_range = None
        if plateau_start >= 0 and plateau_end > plateau_start:
            plateau_range = (plateau_start, plateau_end)
    
    return {
        'shortcut_detected': shortcut_detected,
        'shortcut_assessment': shortcut_assessment,
        'critical_step': critical_step,
        'plateau_range': plateau_range,
    }


def main():
    st.title("🎯 SGPO Process Supervision Demo")
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Experiment Configuration")
    
    # Algorithm selection with explanation
    st.sidebar.subheader("Algorithm")
    use_gpo = st.sidebar.radio(
        "Select Algorithm",
        options=[True, False],
        format_func=lambda x: "🌐 SGPO (Geodesic)" if x else "📊 PPO (Standard)",
        index=0 if st.session_state.use_gpo else 1,
        help="SGPO uses Riemannian geometry to create safety barriers. PPO is the standard baseline."
    )
    st.session_state.use_gpo = use_gpo
    
    if use_gpo:
        st.sidebar.success("**SGPO Active**: Policy updates are scaled by 1/√g(s), suppressing learning near hazards.")
    else:
        st.sidebar.info("**PPO Active**: Standard clipped policy gradient without geometric safety.")
    
    # Training mode
    st.sidebar.subheader("Training Mode")
    training_mode = st.sidebar.radio(
        "Mode",
        options=['interactive', 'automatic'],
        format_func=lambda x: "🖱️ Interactive (manual feedback)" if x == 'interactive' else "🤖 Automatic (simulated)",
        index=0 if st.session_state.training_mode == 'interactive' else 1,
    )
    st.session_state.training_mode = training_mode
    
    # Environment options
    st.sidebar.subheader("Environment")
    enable_wormhole = st.sidebar.checkbox("Enable Wormhole (shortcut)", value=True)
    enable_plateau = st.sidebar.checkbox("Enable Plateau (low-signal zone)", value=True)
    
    # Initialize trainer
    if st.sidebar.button("🔄 Reset Experiment") or st.session_state.trainer is None:
        env = AnomalyNavigationEnv(
            enable_wormhole=enable_wormhole,
            enable_plateau=enable_plateau,
        )
        config = TrainingConfig()
        st.session_state.trainer = SGPOTrainer(
            env, config,
            interactive=(training_mode == 'interactive')
        )
        st.session_state.current_traj = None
        st.session_state.current_traj_id = None
        st.session_state.episode_results = []
        st.session_state.step_feedback = {}
        st.session_state.experiment_log = []
        log_event("Experiment initialized", f"Algorithm: {'SGPO' if use_gpo else 'PPO'}, Mode: {training_mode}")
        st.success("Experiment reset!")
    
    trainer = st.session_state.trainer
    
    # Main content
    st.markdown("---")
    
    # Top row: Environment + Experiment Log
    col_env, col_log = st.columns([2, 1])
    
    with col_env:
        st.subheader("🗺️ Environment")
        
        # Legend
        with st.expander("📖 Environment Legend", expanded=False):
            st.markdown("""
            - 🔴 **Black Holes**: Fatal hazards (agent dies)
            - 🟠 **Cliff Zone**: Narrow passage (small errors = failure)
            - 🟢 **Safe Path**: Navigate through cliff zone here
            - 🟣 **Wormhole**: Teleports agent (shortcut/cheat)
            - ⚫ **Plateau**: Low-signal zone (hard to learn)
            - ⭐ **Goal**: Target destination
            """)
        
        traj = None
        if st.session_state.current_traj is not None:
            traj = st.session_state.current_traj.get('path')
        
        env_fig = create_env_plot(trainer.env, traj)
        st.pyplot(env_fig)
        plt.close(env_fig)
    
    with col_log:
        st.subheader("📋 Experiment Log")
        
        # Current status
        algo_badge = "🌐 SGPO" if st.session_state.use_gpo else "📊 PPO"
        mode_badge = "🖱️ Interactive" if training_mode == 'interactive' else "🤖 Auto"
        st.markdown(f"**Status**: {algo_badge} | {mode_badge}")
        st.markdown(f"**Episodes**: {len(st.session_state.episode_results)}")
        
        if st.session_state.episode_results:
            recent = st.session_state.episode_results[-10:]
            goal_rate = sum(1 for r in recent if r.get('reached_goal', False)) / len(recent)
            st.markdown(f"**Recent Goal Rate**: {goal_rate:.0%}")
        
        # Log entries
        st.markdown("**Recent Events:**")
        log_container = st.container()
        with log_container:
            for entry in reversed(st.session_state.experiment_log[-10:]):
                st.text(f"[{entry['time']}] {entry['event']}")
                if entry['details']:
                    st.caption(f"  → {entry['details']}")
    
    st.markdown("---")
    
    # Training controls
    col_ctrl, col_stats = st.columns([1, 1])
    
    with col_ctrl:
        st.subheader("🎮 Training Controls")
        
        if training_mode == 'automatic':
            n_episodes = st.number_input("Episodes to run", 1, 500, 20)
            
            if st.button("▶️ Run Training Batch", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                log_event("Training started", f"{n_episodes} episodes with {'SGPO' if use_gpo else 'PPO'}")
                
                for ep in range(n_episodes):
                    result = trainer.run_episode(use_gpo=use_gpo)
                    st.session_state.episode_results.append(result)
                    
                    progress_bar.progress((ep + 1) / n_episodes)
                    
                    ret = result.get('return', 0)
                    goal = '✓' if result.get('reached_goal', False) else '✗'
                    status_text.text(f"Episode {ep+1}/{n_episodes}: Return={ret:.1f}, Goal={goal}")
                
                log_event("Training completed", f"{n_episodes} episodes")
                st.success(f"Completed {n_episodes} episodes!")
                st.rerun()
        
        else:  # Interactive mode
            st.markdown("**Interactive Mode**: Run one episode, then provide feedback.")
            
            if st.session_state.current_traj is None:
                if st.button("▶️ Run Single Episode", type="primary"):
                    log_event("Episode started", f"Using {'SGPO' if use_gpo else 'PPO'}")
                    result = trainer.run_episode(use_gpo=use_gpo)
                    
                    if result['status'] == 'awaiting_feedback':
                        st.session_state.current_traj = result['traj_data']
                        st.session_state.current_traj_id = result['trajectory_id']
                        
                        traj_data = result['traj_data']
                        n_steps = len(traj_data['actions'])
                        goal = '✓' if traj_data.get('reached_goal', False) else '✗'
                        wormhole = '🌀' if traj_data.get('used_wormhole', False) else ''
                        
                        log_event("Trajectory collected", f"{n_steps} steps, Goal={goal} {wormhole}")
                        st.info(f"Collected trajectory with {n_steps} steps. Please provide feedback below.")
                        st.rerun()
            else:
                st.warning("⏳ Feedback pending - complete the form below to continue")
    
    with col_stats:
        st.subheader("📊 Training Statistics")
        
        if st.session_state.episode_results:
            results = st.session_state.episode_results
            recent = results[-20:] if len(results) >= 20 else results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                returns = [r.get('return', 0) for r in recent]
                st.metric("Avg Return", f"{np.mean(returns):.1f}")
            with col2:
                goals = [r.get('reached_goal', False) for r in recent]
                st.metric("Goal Rate", f"{np.mean(goals):.0%}")
            with col3:
                wormholes = [r.get('used_wormhole', False) for r in recent]
                st.metric("Wormhole Use", f"{np.mean(wormholes):.0%}")
            
            # Anomalies detected
            anomalies = trainer.state.detected_anomalies
            if anomalies:
                st.markdown("**Detected Anomalies:**")
                for anom in anomalies[-5:]:
                    icon = {
                        AnomalyType.BLACK_HOLE: "🕳️",
                        AnomalyType.CLIFF: "🏔️",
                        AnomalyType.WORMHOLE: "🌀",
                        AnomalyType.PLATEAU: "📉",
                    }.get(anom.anomaly_type, "❓")
                    st.write(f"{icon} {anom.anomaly_type.value}: {anom.description}")
        else:
            st.info("No training data yet. Run some episodes to see statistics.")
    
    # Metric visualization
    with st.expander("🔥 View Riemannian Metric (SGPO Safety Field)", expanded=False):
        render_algorithm_explanation()
        metric_fig = create_metric_plot(trainer.metric, trainer.env)
        st.pyplot(metric_fig)
        plt.close(metric_fig)
    
    # Feedback section (interactive mode with pending trajectory)
    if training_mode == 'interactive' and st.session_state.current_traj is not None:
        st.markdown("---")
        st.header("📝 Provide Process Supervision Feedback")
        
        traj_data = st.session_state.current_traj
        traj_id = st.session_state.current_traj_id
        
        # Show trajectory summary
        n_steps = len(traj_data['actions'])
        reached = "✅ Reached goal!" if traj_data.get('reached_goal', False) else "❌ Did not reach goal"
        wormhole = "🌀 Used wormhole!" if traj_data.get('used_wormhole', False) else ""
        st.markdown(f"**Trajectory Summary**: {n_steps} steps | {reached} {wormhole}")
        
        # Progress visualization
        st.subheader("📈 Trajectory Progress")
        progress_fig = create_progress_plot(traj_data, trainer.env)
        st.pyplot(progress_fig)
        plt.close(progress_fig)
        
        # Overall quality
        st.subheader("⭐ Overall Quality")
        overall_quality = st.slider(
            "Rate the overall trajectory (1=terrible, 5=excellent)",
            1.0, 5.0, 3.0, 0.5,
            key="overall_quality"
        )
        
        # Step-level feedback
        step_fb = render_step_feedback_ui(traj_data, traj_id)
        
        # Anomaly probes
        anomaly_probes = render_anomaly_probes_ui(traj_data, traj_id)
        
        # Submit button
        st.markdown("---")
        if st.button("✅ Submit Feedback & Continue Training", type="primary"):
            # Build step feedback list
            step_feedback_list = []
            for i in range(len(traj_data['actions'])):
                progress_val = step_fb['progress'][i] if i < len(step_fb['progress']) else 0
                quality_val = step_fb['quality'][i] if i < len(step_fb['quality']) else 0.5
                sf = StepFeedback(
                    step_idx=i,
                    progress=progress_val,
                    quality=quality_val,
                    is_discontinuity=i in step_fb['discontinuities'],
                )
                step_feedback_list.append(sf)
            
            # Update supervisor
            trainer.supervisor.update_feedback(
                traj_id,
                overall_quality=overall_quality,
                step_feedback=step_feedback_list,
                shortcut_detected=anomaly_probes['shortcut_detected'],
                shortcut_assessment=anomaly_probes['shortcut_assessment'],
                critical_step=anomaly_probes['critical_step'],
                plateau_range=anomaly_probes['plateau_range'],
            )
            
            # Continue training
            result = trainer.continue_after_feedback(traj_id, traj_data, use_gpo=use_gpo)
            st.session_state.episode_results.append(result)
            
            # Log
            log_event(
                "Feedback submitted",
                f"Quality={overall_quality:.1f}, Return={result.get('return', 0):.1f}, Anomalies={result.get('anomalies_detected', 0)}"
            )
            
            # Clear
            st.session_state.current_traj = None
            st.session_state.current_traj_id = None
            
            st.success(f"Episode {result.get('episode', '?')}: Return={result.get('return', 0):.1f}")
            st.rerun()
    
    # Training history
    if len(st.session_state.episode_results) > 5:
        st.markdown("---")
        st.subheader("📈 Training History")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        results = st.session_state.episode_results
        returns = [r.get('return', 0) for r in results]
        window = min(10, len(returns))
        
        if len(returns) > window:
            smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
            axes[0].plot(smoothed, 'b-', linewidth=2)
        else:
            axes[0].plot(returns, 'b-', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Return')
        axes[0].set_title(f'Training Returns ({"SGPO" if use_gpo else "PPO"})')
        axes[0].grid(True, alpha=0.3)
        
        goals = [float(r.get('reached_goal', False)) for r in results]
        if len(goals) > window:
            goal_smooth = np.convolve(goals, np.ones(window)/window, mode='valid')
            axes[1].plot(goal_smooth, 'g-', linewidth=2, label='Goal Rate')
        else:
            axes[1].plot(goals, 'g-', linewidth=2, label='Goal Rate')
        
        wormholes = [float(r.get('used_wormhole', False)) for r in results]
        if len(wormholes) > window:
            wh_smooth = np.convolve(wormholes, np.ones(window)/window, mode='valid')
            axes[1].plot(wh_smooth, 'm--', linewidth=2, label='Wormhole Rate')
        else:
            axes[1].plot(wormholes, 'm--', linewidth=2, label='Wormhole Rate')
        
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Rate')
        axes[1].set_title('Success Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
