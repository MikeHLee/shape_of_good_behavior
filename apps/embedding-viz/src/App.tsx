import React, { useState, useMemo, useEffect } from 'react';
import { ManifoldPlot } from './components/ManifoldPlot';
import { ControlPanel } from './components/ControlPanel';
import { HoverCard } from './components/HoverCard';
import { DataStatus } from './components/DataStatus';
import { loadVizEmbeddings, generateMockData } from './utils/dataLoader';
import { computePCA } from './utils/pca';
import { PlotSettings, ProjectedPoint, EmbeddingQuintet, PointType } from './types';

const defaultSettings: PlotSettings = {
  showPrompts: true,
  showBase: true,
  showPPO: true,
  showCPO: true,
  showGPO: true,
  showClippedGPO: true,
  connectLines: true,
  opacity: 0.7,
  highlightedId: null,
};

function transformToPoints(data: EmbeddingQuintet[]): ProjectedPoint[] {
  // Collect all embeddings for PCA
  const allEmbeddings: number[][] = [];
  const embeddingMeta: { promptId: number; type: PointType; text: string; risk: number; shift?: number }[] = [];
  
  data.forEach(quintet => {
    // Prompt embedding
    allEmbeddings.push(quintet.prompt_embedding);
    embeddingMeta.push({
      promptId: quintet.prompt_id,
      type: 'prompt',
      text: quintet.prompt_text,
      risk: quintet.harmonic_risk,
    });
    
    // Response embeddings
    const responseTypes = ['base', 'ppo', 'cpo', 'gpo', 'gpo_clipped'] as const;
    responseTypes.forEach(type => {
      const response = quintet.responses[type as keyof typeof quintet.responses];
      if (response) {
        allEmbeddings.push(response.embedding);
        embeddingMeta.push({
          promptId: quintet.prompt_id,
          type,
          text: response.text,
          risk: quintet.harmonic_risk,
          shift: response.trajectory_shift,
        });
      }
    });
  });
  
  // Compute PCA
  const projected = computePCA(allEmbeddings, 2);
  
  // Transform to ProjectedPoints
  return projected.map((coords, i) => ({
    x: coords[0],
    y: coords[1],
    type: embeddingMeta[i].type,
    prompt_id: embeddingMeta[i].promptId,
    text: embeddingMeta[i].text,
    harmonic_risk: embeddingMeta[i].risk,
    trajectory_shift: embeddingMeta[i].shift,
  }));
}

export default function App() {
  const [data, setData] = useState<EmbeddingQuintet[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [_error, _setError] = useState<string | null>(null);
  const [usingMockData, setUsingMockData] = useState(false);
  const [settings, setSettings] = useState<PlotSettings>(defaultSettings);
  const [hoveredPoint, setHoveredPoint] = useState<ProjectedPoint | null>(null);
  
  // Load data on mount
  useEffect(() => {
    loadVizEmbeddings()
      .then(loadedData => {
        setData(loadedData);
        setUsingMockData(false);
      })
      .catch(() => {
        // Fall back to mock data
        console.log('Using mock data - real data not available');
        setData(generateMockData(20));
        setUsingMockData(true);
      })
      .finally(() => setLoading(false));
  }, []);

  const points = useMemo(() => {
    if (!data) return [];
    return transformToPoints(data);
  }, [data]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading embeddings...</div>
      </div>
    );
  }
  
  if (_error && !data) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-xl text-red-600">Error: {_error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold mb-2">
          Manifold Trajectory Visualization
        </h1>
        <p className="text-gray-600 mb-6">
          Interactive exploration of reward space trajectories across different alignment methods
        </p>
        
        {usingMockData && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm">
            <span className="font-semibold text-yellow-800">⚠ Using mock data</span>
            <span className="text-gray-600 ml-2">
              Run Modal experiments and download results to see real data.
            </span>
          </div>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3 bg-white rounded-lg shadow p-4">
            <ManifoldPlot
              points={points}
              settings={settings}
              onHover={setHoveredPoint}
              onSelect={(id) => setSettings(prev => ({ 
                ...prev, 
                highlightedId: prev.highlightedId === id ? null : id 
              }))}
            />
          </div>
          
          <div className="space-y-4">
            <DataStatus />
            <HoverCard point={hoveredPoint} />
            <ControlPanel
              settings={settings}
              onChange={setSettings}
              promptCount={data?.length || 0}
              onClearHighlight={() => setSettings(prev => ({ ...prev, highlightedId: null }))}
            />
          </div>
        </div>
        
        <div className="mt-6 text-sm text-gray-500">
          <p><strong>Click</strong> on a point to highlight its quintet trajectory.</p>
          <p><strong>Hover</strong> to see full text and metrics.</p>
          <p>Use the <strong>camera icon</strong> in the plot toolbar to export as PNG.</p>
        </div>
      </div>
    </div>
  );
}
