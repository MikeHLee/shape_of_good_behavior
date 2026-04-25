import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ProjectedPoint, PlotSettings, PointType } from '../types';
import { COLORS, COLOR_LABELS } from '../utils/colors';
import { Data, PlotMouseEvent } from 'plotly.js';

interface Props {
  points: ProjectedPoint[];
  settings: PlotSettings;
  onHover: (point: ProjectedPoint | null) => void;
  onSelect: (promptId: number) => void;
}

export const ManifoldPlot: React.FC<Props> = ({ points, settings, onHover, onSelect }) => {
  const traces = useMemo(() => {
    const result: Data[] = [];
    
    const types: PointType[] = ['prompt', 'base', 'ppo', 'cpo', 'gpo', 'gpo_clipped'];
    const typeSettings: Record<PointType, boolean> = {
      prompt: settings.showPrompts,
      base: settings.showBase,
      ppo: settings.showPPO,
      cpo: settings.showCPO,
      gpo: settings.showGPO,
      gpo_clipped: settings.showClippedGPO,
    };
    
    const isHighlighted = settings.highlightedId !== null;
    
    // Add trajectory lines FIRST (so they render behind points)
    if (settings.connectLines && settings.highlightedId !== null) {
      const promptId = settings.highlightedId;
      const quintetPoints = points.filter(p => p.prompt_id === promptId);
      const prompt = quintetPoints.find(p => p.type === 'prompt');
      
      if (prompt) {
        types.slice(1).forEach(type => {
          if (!typeSettings[type]) return;
          const response = quintetPoints.find(p => p.type === type);
          if (response) {
            result.push({
              type: 'scatter',
              mode: 'lines+markers',
              x: [prompt.x, response.x],
              y: [prompt.y, response.y],
              line: { color: COLORS[type], width: 3 },
              marker: { size: 4, color: COLORS[type] },
              showlegend: false,
              hoverinfo: 'skip',
            } as Data);
          }
        });
      }
    }
    
    // Add scatter points (rendered on top of lines)
    types.forEach(type => {
      if (!typeSettings[type]) return;
      
      const typePoints = points.filter(p => p.type === type);
      
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
            color: isHighlighted ? typePoints.map(p => p.prompt_id === settings.highlightedId ? '#000' : 'white') : 'white',
            width: type === 'prompt' ? 2 : 1,
          },
        },
        text: typePoints.map(p => `${p.text.slice(0, 80)}...`),
        customdata: typePoints,
        hovertemplate: '<b>%{text}</b><br>Risk: %{customdata.harmonic_risk:.3f}<extra></extra>',
      } as Data);
    });
    
    return result;
  }, [points, settings]);

  const handleHover = (event: PlotMouseEvent) => {
    const point = event.points[0];
    if (point?.customdata) {
      onHover(point.customdata as ProjectedPoint);
    }
  };

  const handleClick = (event: PlotMouseEvent) => {
    const point = event.points[0];
    if (point?.customdata) {
      const p = point.customdata as ProjectedPoint;
      onSelect(p.prompt_id);
    }
  };

  return (
    <Plot
      data={traces}
      layout={{
        title: {
          text: 'Manifold Trajectories: Model Comparison',
          font: { size: 18 },
        },
        xaxis: { 
          title: 'PCA Component 1',
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        yaxis: { 
          title: 'PCA Component 2',
          zeroline: false,
          gridcolor: '#e5e7eb',
        },
        hovermode: 'closest',
        showlegend: true,
        legend: { 
          x: 1, 
          y: 1,
          bgcolor: 'rgba(255,255,255,0.8)',
        },
        paper_bgcolor: 'white',
        plot_bgcolor: '#fafafa',
        margin: { l: 60, r: 30, t: 50, b: 60 },
      }}
      config={{
        toImageButtonOptions: {
          format: 'png',
          filename: 'manifold_trajectories',
          scale: 2,
        },
        displayModeBar: true,
        scrollZoom: true,
        modeBarButtonsToAdd: ['downloadImage' as never],
      }}
      onHover={handleHover}
      onUnhover={() => onHover(null)}
      onClick={handleClick}
      style={{ width: '100%', height: '600px' }}
    />
  );
};
