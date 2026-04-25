import React, { useMemo, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { ProjectedPoint, PlotSettings, PointType, AnimationSettings, TrajectoryFrame } from '../types';
import { COLORS, COLOR_LABELS } from '../utils/colors';
import { Layout } from 'plotly.js';

// Use looser typing for Plotly data to support customdata
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PlotData = any;

interface Props {
  frames: TrajectoryFrame[];
  settings: PlotSettings;
  animationSettings: AnimationSettings;
  onAnimationChange: (settings: Partial<AnimationSettings>) => void;
  onHover: (point: ProjectedPoint | null) => void;
  onSelect: (promptId: number) => void;
  is3D?: boolean;
}

export const AnimatedManifoldPlot: React.FC<Props> = ({
  frames,
  settings,
  animationSettings,
  onAnimationChange,
  onHover,
  onSelect,
  is3D = false,
}) => {
  const { isPlaying, currentFrame, playbackSpeed } = animationSettings;

  // Auto-advance frames when playing
  useEffect(() => {
    if (!isPlaying || frames.length === 0) return;
    
    const timer = setInterval(() => {
      onAnimationChange({
        currentFrame: (currentFrame + 1) % frames.length,
      });
    }, playbackSpeed);
    
    return () => clearInterval(timer);
  }, [isPlaying, currentFrame, playbackSpeed, frames.length, onAnimationChange]);

  const currentPoints = useMemo(() => {
    if (frames.length === 0) return [];
    return frames[Math.min(currentFrame, frames.length - 1)]?.points || [];
  }, [frames, currentFrame]);

  const currentMetadata = useMemo(() => {
    if (frames.length === 0) return null;
    return frames[Math.min(currentFrame, frames.length - 1)]?.metadata || null;
  }, [frames, currentFrame]);

  const buildTraces = useCallback((points: ProjectedPoint[]): PlotData[] => {
    const result: PlotData[] = [];
    const types: PointType[] = ['prompt', 'base', 'ppo', 'cpo', 'gpo', 'gpo_clipped'];
    
    const typeSettings: Record<PointType, boolean> = {
      prompt: settings.showPrompts,
      base: settings.showBase,
      ppo: settings.showPPO,
      cpo: settings.showCPO,
      gpo: settings.showGPO,
      gpo_clipped: settings.showClippedGPO,
    };

    // Add trajectory lines for highlighted quintet
    if (settings.connectLines && settings.highlightedId !== null) {
      const promptId = settings.highlightedId;
      const quintetPoints = points.filter(p => p.prompt_id === promptId);
      const prompt = quintetPoints.find(p => p.type === 'prompt');
      
      if (prompt) {
        types.slice(1).forEach(type => {
          if (!typeSettings[type]) return;
          const response = quintetPoints.find(p => p.type === type);
          if (response) {
            if (is3D) {
              result.push({
                type: 'scatter3d',
                mode: 'lines+markers',
                x: [prompt.x, response.x],
                y: [prompt.y, response.y],
                z: [prompt.z || 0, response.z || 0],
                line: { color: COLORS[type], width: 4 },
                marker: { size: 4, color: COLORS[type] },
                showlegend: false,
                hoverinfo: 'skip',
              } as any);
            } else {
              result.push({
                type: 'scatter',
                mode: 'lines+markers',
                x: [prompt.x, response.x],
                y: [prompt.y, response.y],
                line: { color: COLORS[type], width: 3 },
                marker: { size: 4, color: COLORS[type] },
                showlegend: false,
                hoverinfo: 'skip',
              } as any);
            }
          }
        });
      }
    }

    // Add scatter points
    const isHighlighted = settings.highlightedId !== null;
    
    types.forEach(type => {
      if (!typeSettings[type]) return;
      const typePoints = points.filter(p => p.type === type);
      
      if (is3D) {
        result.push({
          type: 'scatter3d',
          mode: 'markers',
          name: COLOR_LABELS[type],
          x: typePoints.map(p => p.x),
          y: typePoints.map(p => p.y),
          z: typePoints.map(p => p.z || 0),
          marker: {
            color: COLORS[type],
            size: type === 'prompt' ? 10 : 7,
            opacity: isHighlighted 
              ? typePoints.map(p => p.prompt_id === settings.highlightedId ? 1 : 0.15)
              : settings.opacity,
            symbol: type === 'prompt' ? 'circle' : 'square',
          },
          text: typePoints.map(p => `${p.text.slice(0, 80)}...`),
          customdata: typePoints,
          hovertemplate: '<b>%{text}</b><br>Risk: %{customdata.harmonic_risk:.3f}<extra></extra>',
        });
      } else {
        result.push({
          type: 'scatter',
          mode: 'markers',
          name: COLOR_LABELS[type],
          x: typePoints.map(p => p.x),
          y: typePoints.map(p => p.y),
          marker: {
            color: COLORS[type],
            size: type === 'prompt' ? 14 : 10,
            opacity: isHighlighted 
              ? typePoints.map(p => p.prompt_id === settings.highlightedId ? 1 : 0.15)
              : settings.opacity,
            symbol: type === 'prompt' ? 'circle' : 'square',
            line: {
              color: isHighlighted 
                ? typePoints.map(p => p.prompt_id === settings.highlightedId ? '#000' : 'white') 
                : 'white',
              width: type === 'prompt' ? 2 : 1,
            },
          },
          text: typePoints.map(p => `${p.text.slice(0, 80)}...`),
          customdata: typePoints,
          hovertemplate: '<b>%{text}</b><br>Risk: %{customdata.harmonic_risk:.3f}<extra></extra>',
        });
      }
    });

    return result;
  }, [settings, is3D]);

  const traces = useMemo(() => buildTraces(currentPoints), [buildTraces, currentPoints]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const layout: any = useMemo(() => {
    const baseLayout: Partial<Layout> = {
      title: {
        text: `Manifold Evolution${currentMetadata ? ` - Step ${currentMetadata.step}` : ''}`,
        font: { size: 18 },
      },
      hovermode: 'closest',
      showlegend: true,
      legend: { x: 1, y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
      paper_bgcolor: 'white',
      plot_bgcolor: '#fafafa',
      margin: { l: 60, r: 30, t: 80, b: 60 },
      annotations: currentMetadata ? [{
        x: 0.02,
        y: 0.98,
        xref: 'paper',
        yref: 'paper',
        text: `Frame: ${currentFrame + 1}/${frames.length}${currentMetadata.reward !== undefined ? `<br>Reward: ${currentMetadata.reward.toFixed(3)}` : ''}${currentMetadata.risk !== undefined ? `<br>Risk: ${currentMetadata.risk.toFixed(3)}` : ''}`,
        showarrow: false,
        font: { size: 12 },
        bgcolor: 'rgba(255,255,255,0.9)',
        bordercolor: '#ccc',
        borderwidth: 1,
        borderpad: 4,
      }] : [],
    };

    if (is3D) {
      return {
        ...baseLayout,
        scene: {
          xaxis: { title: 'PC1', gridcolor: '#e5e7eb' },
          yaxis: { title: 'PC2', gridcolor: '#e5e7eb' },
          zaxis: { title: 'PC3', gridcolor: '#e5e7eb' },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
        },
      };
    }

    return {
      ...baseLayout,
      xaxis: { title: 'PCA Component 1', zeroline: false, gridcolor: '#e5e7eb' },
      yaxis: { title: 'PCA Component 2', zeroline: false, gridcolor: '#e5e7eb' },
    };
  }, [currentMetadata, currentFrame, frames.length, is3D]);

  return (
    <div className="space-y-4">
      <Plot
        data={traces}
        layout={layout}
        config={{
          toImageButtonOptions: { format: 'png', filename: 'manifold_animation', scale: 2 },
          displayModeBar: true,
          scrollZoom: true,
        }}
        onHover={(event) => {
          const point = event.points[0];
          if (point?.customdata) onHover(point.customdata as unknown as ProjectedPoint);
        }}
        onUnhover={() => onHover(null)}
        onClick={(event) => {
          const point = event.points[0];
          if (point?.customdata) {
            const p = point.customdata as unknown as ProjectedPoint;
            onSelect(p.prompt_id);
          }
        }}
        style={{ width: '100%', height: is3D ? '550px' : '500px' }}
      />
      
      {/* Animation Controls */}
      <div className="flex items-center gap-4 px-4 py-3 bg-gray-100 rounded-lg">
        <button
          onClick={() => onAnimationChange({ isPlaying: !isPlaying })}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors font-medium"
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        
        <button
          onClick={() => onAnimationChange({ currentFrame: 0 })}
          className="px-3 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors"
          title="Reset to start"
        >
          ⏮
        </button>
        
        <button
          onClick={() => onAnimationChange({ 
            currentFrame: Math.max(0, currentFrame - 1) 
          })}
          className="px-3 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors"
          title="Previous frame"
        >
          ◀
        </button>
        
        <input
          type="range"
          min={0}
          max={Math.max(0, frames.length - 1)}
          value={currentFrame}
          onChange={(e) => onAnimationChange({ currentFrame: parseInt(e.target.value) })}
          className="flex-1"
        />
        
        <button
          onClick={() => onAnimationChange({ 
            currentFrame: Math.min(frames.length - 1, currentFrame + 1) 
          })}
          className="px-3 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors"
          title="Next frame"
        >
          ▶
        </button>
        
        <span className="text-sm text-gray-600 min-w-[80px]">
          {currentFrame + 1} / {frames.length}
        </span>
        
        <select
          value={playbackSpeed}
          onChange={(e) => onAnimationChange({ playbackSpeed: parseInt(e.target.value) })}
          className="px-2 py-1 border rounded text-sm"
        >
          <option value={100}>Fast (100ms)</option>
          <option value={250}>Normal (250ms)</option>
          <option value={500}>Slow (500ms)</option>
          <option value={1000}>Very Slow (1s)</option>
        </select>
      </div>
    </div>
  );
};
