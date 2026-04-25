import { ProjectedPoint, TrajectoryFrame, EmbeddingQuintet, PointType } from '../types';
import { computePCA } from './pca';

/**
 * Generate animation frames showing trajectory evolution over time.
 * 
 * Modes:
 * - 'trajectory': Animate one point moving along its trajectory
 * - 'evolution': Show how the manifold evolves (simulated training steps)
 * - 'comparison': Sequentially reveal different model outputs
 */

export function generateTrajectoryFrames(
  data: EmbeddingQuintet[],
  mode: 'trajectory' | 'evolution' | 'comparison' = 'comparison'
): TrajectoryFrame[] {
  if (mode === 'comparison') {
    return generateComparisonFrames(data);
  } else if (mode === 'evolution') {
    return generateEvolutionFrames(data);
  }
  return generateSingleTrajectoryFrames(data);
}

/**
 * Comparison mode: Sequentially reveal different model outputs
 * Frame 0: Just prompts
 * Frame 1: Prompts + Base
 * Frame 2: Prompts + Base + PPO
 * ...etc
 */
function generateComparisonFrames(data: EmbeddingQuintet[]): TrajectoryFrame[] {
  const frames: TrajectoryFrame[] = [];
  const modelOrder: PointType[] = ['prompt', 'base', 'ppo', 'cpo', 'gpo', 'gpo_clipped'];
  const modelLabels = ['Prompts', 'Base Model', 'PPO', 'CPO', 'GPO', 'Clipped-GPO'];
  
  // Compute PCA on all points first for consistent projection
  const allEmbeddings: number[][] = [];
  const allMeta: { quintetIdx: number; type: PointType; text: string; risk: number; shift?: number }[] = [];
  
  data.forEach((quintet, qIdx) => {
    allEmbeddings.push(quintet.prompt_embedding);
    allMeta.push({
      quintetIdx: qIdx,
      type: 'prompt',
      text: quintet.prompt_text,
      risk: quintet.harmonic_risk,
    });
    
    const responseTypes = ['base', 'ppo', 'cpo', 'gpo', 'gpo_clipped'] as const;
    responseTypes.forEach(type => {
      const response = quintet.responses[type];
      if (response) {
        allEmbeddings.push(response.embedding);
        allMeta.push({
          quintetIdx: qIdx,
          type,
          text: response.text,
          risk: quintet.harmonic_risk,
          shift: response.trajectory_shift,
        });
      }
    });
  });
  
  const projected = computePCA(allEmbeddings, 2);
  
  const allProjectedPoints: ProjectedPoint[] = projected.map((coords, i) => ({
    x: coords[0],
    y: coords[1],
    type: allMeta[i].type,
    prompt_id: allMeta[i].quintetIdx,
    text: allMeta[i].text,
    harmonic_risk: allMeta[i].risk,
    trajectory_shift: allMeta[i].shift,
  }));
  
  // Generate frames progressively revealing models
  for (let i = 0; i < modelOrder.length; i++) {
    const visibleTypes = modelOrder.slice(0, i + 1);
    const framePoints = allProjectedPoints.filter(p => visibleTypes.includes(p.type));
    
    frames.push({
      frameIndex: i,
      timestamp: i * 1000,
      points: framePoints,
      metadata: {
        step: i,
        description: `Showing: ${modelLabels.slice(0, i + 1).join(' → ')}`,
      },
    });
  }
  
  return frames;
}

/**
 * Evolution mode: Simulate how manifold changes during training
 * Adds noise to positions that gradually converges to final state
 */
function generateEvolutionFrames(data: EmbeddingQuintet[], numFrames: number = 20): TrajectoryFrame[] {
  const frames: TrajectoryFrame[] = [];
  
  // Get final projected points
  const allEmbeddings: number[][] = [];
  const allMeta: { quintetIdx: number; type: PointType; text: string; risk: number }[] = [];
  
  data.forEach((quintet, qIdx) => {
    allEmbeddings.push(quintet.prompt_embedding);
    allMeta.push({ quintetIdx: qIdx, type: 'prompt', text: quintet.prompt_text, risk: quintet.harmonic_risk });
    
    const responseTypes = ['base', 'ppo', 'cpo', 'gpo', 'gpo_clipped'] as const;
    responseTypes.forEach(type => {
      const response = quintet.responses[type];
      if (response) {
        allEmbeddings.push(response.embedding);
        allMeta.push({ quintetIdx: qIdx, type, text: response.text, risk: quintet.harmonic_risk });
      }
    });
  });
  
  const finalProjected = computePCA(allEmbeddings, 2);
  
  // Generate frames with decreasing noise
  for (let frame = 0; frame < numFrames; frame++) {
    const progress = frame / (numFrames - 1);
    const noiseScale = (1 - progress) * 2; // Noise decreases from 2 to 0
    
    const framePoints: ProjectedPoint[] = finalProjected.map((coords, i) => {
      // Add noise that decreases over time
      const noise = noiseScale * (Math.random() - 0.5);
      return {
        x: coords[0] + noise,
        y: coords[1] + noise,
        type: allMeta[i].type,
        prompt_id: allMeta[i].quintetIdx,
        text: allMeta[i].text,
        harmonic_risk: allMeta[i].risk,
      };
    });
    
    frames.push({
      frameIndex: frame,
      timestamp: frame * 500,
      points: framePoints,
      metadata: {
        step: frame,
        reward: progress * 0.8, // Simulated improving reward
        risk: (1 - progress) * 0.5, // Simulated decreasing risk
        description: `Training epoch ${frame + 1}/${numFrames}`,
      },
    });
  }
  
  return frames;
}

/**
 * Single trajectory mode: Animate one point's journey through the manifold
 */
function generateSingleTrajectoryFrames(data: EmbeddingQuintet[]): TrajectoryFrame[] {
  const frames: TrajectoryFrame[] = [];
  
  if (data.length === 0) return frames;
  
  // Use first quintet as the trajectory to animate
  const quintet = data[0];
  const trajectory: { embedding: number[]; type: PointType; text: string }[] = [
    { embedding: quintet.prompt_embedding, type: 'prompt', text: quintet.prompt_text },
  ];
  
  const responseTypes = ['base', 'ppo', 'cpo', 'gpo', 'gpo_clipped'] as const;
  responseTypes.forEach(type => {
    const response = quintet.responses[type];
    if (response) {
      trajectory.push({ embedding: response.embedding, type, text: response.text });
    }
  });
  
  // Project all points including background
  const allEmbeddings = trajectory.map(t => t.embedding);
  const projected = computePCA(allEmbeddings, 2);
  
  // Create frames for each step in the trajectory
  for (let i = 0; i < trajectory.length; i++) {
    const visiblePoints: ProjectedPoint[] = projected.slice(0, i + 1).map((coords, j) => ({
      x: coords[0],
      y: coords[1],
      type: trajectory[j].type,
      prompt_id: 0,
      text: trajectory[j].text,
      harmonic_risk: quintet.harmonic_risk,
    }));
    
    frames.push({
      frameIndex: i,
      timestamp: i * 1000,
      points: visiblePoints,
      metadata: {
        step: i,
        description: `Step ${i + 1}: ${trajectory[i].type}`,
      },
    });
  }
  
  return frames;
}

/**
 * Generate frames from safety gym benchmark data
 * For discrete navigation or continuous reaching tasks
 */
export function generateSafetyGymFrames(
  trajectoryData: Array<{
    step: number;
    position: [number, number] | [number, number, number];
    reward: number;
    risk: number;
    collision: boolean;
  }>,
  hazards: Array<{ center: [number, number]; radius: number }> = []
): TrajectoryFrame[] {
  return trajectoryData.map((step, i) => ({
    frameIndex: i,
    timestamp: step.step * 100,
    points: [
      {
        x: step.position[0],
        y: step.position[1],
        z: step.position.length > 2 ? step.position[2] : undefined,
        type: 'gpo' as PointType,
        prompt_id: 0,
        text: `Step ${step.step}`,
        harmonic_risk: step.risk,
      },
      // Add hazard markers
      ...hazards.map((h, hIdx) => ({
        x: h.center[0],
        y: h.center[1],
        type: 'prompt' as PointType,
        prompt_id: -1 - hIdx,
        text: `Hazard ${hIdx + 1}`,
        harmonic_risk: 1.0,
      })),
    ],
    metadata: {
      step: step.step,
      reward: step.reward,
      risk: step.risk,
      description: step.collision ? '⚠️ Collision!' : `Position: (${step.position.join(', ')})`,
    },
  }));
}
