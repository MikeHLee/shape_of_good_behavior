import { PointType } from '../types';

export const COLORS: Record<PointType, string> = {
  prompt: '#4B5563',      // Gray-600
  base: '#EF4444',        // Red-500
  ppo: '#F59E0B',         // Amber-500
  cpo: '#3B82F6',         // Blue-500
  gpo: '#10B981',         // Emerald-500
  gpo_clipped: '#8B5CF6', // Violet-500
};

export const COLOR_LABELS: Record<PointType, string> = {
  prompt: 'Prompt',
  base: 'Base GPT-2',
  ppo: 'PPO',
  cpo: 'CPO',
  gpo: 'GPO',
  gpo_clipped: 'Clipped-GPO',
};
