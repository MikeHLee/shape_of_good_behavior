import React from 'react';
import { ProjectedPoint } from '../types';
import { COLORS, COLOR_LABELS } from '../utils/colors';

interface Props {
  point: ProjectedPoint | null;
}

export const HoverCard: React.FC<Props> = ({ point }) => {
  if (!point) {
    return (
      <div className="p-4 bg-white border rounded-lg shadow h-48 flex items-center justify-center text-gray-400">
        Hover over a point to see details
      </div>
    );
  }

  return (
    <div className="p-4 bg-white border rounded-lg shadow">
      <div className="flex items-center gap-2 mb-2">
        <span 
          className="w-4 h-4 rounded"
          style={{ backgroundColor: COLORS[point.type] }}
        />
        <span className="font-bold">{COLOR_LABELS[point.type]}</span>
        <span className="text-gray-500">#{point.prompt_id}</span>
      </div>
      
      <div className="mb-2">
        <span className="text-sm text-gray-600">Harmonic Risk: </span>
        <span className={point.harmonic_risk > 0.7 ? 'text-red-600 font-bold' : 'text-gray-800'}>
          {point.harmonic_risk.toFixed(3)}
        </span>
      </div>
      
      {point.trajectory_shift !== undefined && (
        <div className="mb-2">
          <span className="text-sm text-gray-600">Trajectory Shift: </span>
          <span className={point.trajectory_shift < 0 ? 'text-green-600' : 'text-red-600'}>
            {point.trajectory_shift > 0 ? '+' : ''}{(point.trajectory_shift * 100).toFixed(1)}%
          </span>
          <span className="text-xs text-gray-500 ml-1">
            ({point.trajectory_shift < 0 ? 'safer' : 'riskier'})
          </span>
        </div>
      )}
      
      <div className="mt-3 p-2 bg-gray-50 rounded text-sm max-h-32 overflow-y-auto">
        {point.text}
      </div>
    </div>
  );
};
