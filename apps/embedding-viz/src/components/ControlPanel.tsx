import React from 'react';
import { PlotSettings, PointType } from '../types';
import { COLORS, COLOR_LABELS } from '../utils/colors';

interface Props {
  settings: PlotSettings;
  onChange: (settings: PlotSettings) => void;
  promptCount: number;
  onClearHighlight: () => void;
}

export const ControlPanel: React.FC<Props> = ({ 
  settings, 
  onChange, 
  promptCount,
  onClearHighlight,
}) => {
  const toggles: { key: keyof PlotSettings; type: PointType }[] = [
    { key: 'showPrompts', type: 'prompt' },
    { key: 'showBase', type: 'base' },
    { key: 'showPPO', type: 'ppo' },
    { key: 'showCPO', type: 'cpo' },
    { key: 'showGPO', type: 'gpo' },
    { key: 'showClippedGPO', type: 'gpo_clipped' },
  ];

  return (
    <div className="p-4 bg-gray-100 rounded-lg">
      <h3 className="font-bold mb-3">Display Settings</h3>
      
      <div className="space-y-2 mb-4">
        {toggles.map(({ key, type }) => (
          <label key={key} className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings[key] as boolean}
              onChange={(e) => onChange({ ...settings, [key]: e.target.checked })}
              className="rounded"
            />
            <span 
              className="w-3 h-3 rounded" 
              style={{ backgroundColor: COLORS[type] }} 
            />
            <span className="text-sm">{COLOR_LABELS[type]}</span>
          </label>
        ))}
      </div>
      
      <hr className="my-3" />
      
      <label className="flex items-center gap-2 mb-3 cursor-pointer">
        <input
          type="checkbox"
          checked={settings.connectLines}
          onChange={(e) => onChange({ ...settings, connectLines: e.target.checked })}
          className="rounded"
        />
        <span className="text-sm">Show trajectory lines</span>
      </label>
      
      <label className="block mb-4">
        <span className="text-sm text-gray-600">Opacity: {settings.opacity.toFixed(1)}</span>
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.1"
          value={settings.opacity}
          onChange={(e) => onChange({ ...settings, opacity: parseFloat(e.target.value) })}
          className="w-full mt-1"
        />
      </label>
      
      <hr className="my-3" />
      
      <div className="text-sm text-gray-600 mb-2">
        {promptCount} prompts loaded
      </div>
      
      {settings.highlightedId !== null && (
        <button
          onClick={onClearHighlight}
          className="w-full px-3 py-2 bg-gray-200 hover:bg-gray-300 rounded text-sm"
        >
          Clear Selection (#{settings.highlightedId})
        </button>
      )}
    </div>
  );
};
