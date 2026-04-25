export interface ResponseData {
  text: string;
  embedding: number[];
  trajectory_shift: number;
}

export interface EmbeddingQuintet {
  prompt_id: number;
  prompt_text: string;
  harmonic_risk: number;
  prompt_embedding: number[];
  responses: {
    base?: ResponseData;
    ppo?: ResponseData;
    cpo?: ResponseData;
    gpo?: ResponseData;
    gpo_clipped?: ResponseData;
  };
}

export type PointType = 'prompt' | 'base' | 'ppo' | 'cpo' | 'gpo' | 'gpo_clipped';

export interface ProjectedPoint {
  x: number;
  y: number;
  z?: number;
  type: PointType;
  prompt_id: number;
  text: string;
  harmonic_risk: number;
  trajectory_shift?: number;
}

export interface PlotSettings {
  showPrompts: boolean;
  showBase: boolean;
  showPPO: boolean;
  showCPO: boolean;
  showGPO: boolean;
  showClippedGPO: boolean;
  connectLines: boolean;
  opacity: number;
  highlightedId: number | null;
}

export interface DataAvailability {
  available: boolean;
  files: string[];
  missing: string[];
}

// Animation types
export interface AnimationSettings {
  isPlaying: boolean;
  currentFrame: number;
  totalFrames: number;
  playbackSpeed: number;  // ms per frame
  mode: 'trajectory' | 'evolution' | 'comparison';
}

export interface TrajectoryFrame {
  frameIndex: number;
  timestamp: number;
  points: ProjectedPoint[];
  metadata?: {
    step: number;
    reward?: number;
    risk?: number;
    description?: string;
  };
}
