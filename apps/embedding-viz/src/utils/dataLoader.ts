import { EmbeddingQuintet, DataAvailability } from '../types';

const DATA_BASE_PATH = '/data';

export async function loadVizEmbeddings(): Promise<EmbeddingQuintet[]> {
  const dataPath = `${DATA_BASE_PATH}/viz_embeddings.json`;
  
  try {
    const response = await fetch(dataPath);
    if (!response.ok) {
      throw new Error(`Failed to load data: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    if (!Array.isArray(data)) {
      throw new Error('Invalid data format: expected array');
    }
    
    // Validate each entry
    data.forEach((entry, idx) => {
      if (!entry.prompt_id || !entry.prompt_text || !entry.prompt_embedding) {
        throw new Error(`Invalid entry at index ${idx}: missing required fields`);
      }
      if (!entry.responses || typeof entry.responses !== 'object') {
        throw new Error(`Invalid entry at index ${idx}: missing responses`);
      }
    });
    
    return data as EmbeddingQuintet[];
  } catch (error) {
    console.error('Error loading visualization data:', error);
    throw error;
  }
}

export async function checkDataAvailability(): Promise<DataAvailability> {
  const requiredFiles = [
    'viz_embeddings.json',
  ];
  
  const available: string[] = [];
  const missing: string[] = [];
  
  for (const file of requiredFiles) {
    try {
      const response = await fetch(`${DATA_BASE_PATH}/${file}`, { method: 'HEAD' });
      if (response.ok) {
        available.push(file);
      } else {
        missing.push(file);
      }
    } catch {
      missing.push(file);
    }
  }
  
  return {
    available: missing.length === 0,
    files: available,
    missing,
  };
}

// Generate mock data for development/testing when real data is not available
export function generateMockData(count: number = 20): EmbeddingQuintet[] {
  const mockData: EmbeddingQuintet[] = [];
  
  const samplePrompts = [
    "How do I improve my coding skills?",
    "Explain quantum computing simply",
    "What are the best practices for ML?",
    "How do I hack into a system?",
    "Write a poem about nature",
    "How can I cheat on an exam?",
    "Explain climate change",
    "Create a malicious script",
    "What is reinforcement learning?",
    "How to bypass security systems?",
  ];
  
  for (let i = 0; i < count; i++) {
    const promptText = samplePrompts[i % samplePrompts.length];
    const isHighRisk = promptText.toLowerCase().includes('hack') || 
                       promptText.toLowerCase().includes('cheat') ||
                       promptText.toLowerCase().includes('malicious') ||
                       promptText.toLowerCase().includes('bypass');
    
    const baseEmbedding = Array.from({ length: 384 }, () => Math.random() * 2 - 1);
    
    const addNoise = (arr: number[], scale: number) => 
      arr.map(v => v + (Math.random() - 0.5) * scale);
    
    const promptEmb = baseEmbedding;
    const baseShift = isHighRisk ? 0.15 : 0.05;
    const ppoShift = isHighRisk ? 0.08 : 0.03;
    const cpoShift = isHighRisk ? -0.12 : -0.02;
    const gpoShift = isHighRisk ? -0.25 : -0.05;
    const clippedShift = isHighRisk ? -0.22 : -0.04;
    
    mockData.push({
      prompt_id: i + 1,
      prompt_text: promptText,
      harmonic_risk: isHighRisk ? 0.7 + Math.random() * 0.25 : Math.random() * 0.4,
      prompt_embedding: promptEmb,
      responses: {
        base: {
          text: `Base response to: ${promptText}`,
          embedding: addNoise(baseEmbedding, 0.3),
          trajectory_shift: baseShift,
        },
        ppo: {
          text: `PPO response to: ${promptText}`,
          embedding: addNoise(baseEmbedding, 0.25),
          trajectory_shift: ppoShift,
        },
        cpo: {
          text: `CPO response (safer) to: ${promptText}`,
          embedding: addNoise(baseEmbedding, 0.35),
          trajectory_shift: cpoShift,
        },
        gpo: {
          text: `GPO response (geometric safety) to: ${promptText}`,
          embedding: addNoise(baseEmbedding, 0.4),
          trajectory_shift: gpoShift,
        },
        gpo_clipped: {
          text: `Clipped-GPO response to: ${promptText}`,
          embedding: addNoise(baseEmbedding, 0.38),
          trajectory_shift: clippedShift,
        },
      },
    });
  }
  
  return mockData;
}
