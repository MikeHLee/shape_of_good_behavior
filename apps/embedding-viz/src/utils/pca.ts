export function computePCA(data: number[][], nComponents: number = 2): number[][] {
  const n = data.length;
  if (n === 0) return [];
  
  const d = data[0].length;
  
  // Center the data
  const mean = new Array(d).fill(0);
  data.forEach(row => row.forEach((v, i) => mean[i] += v / n));
  const centered = data.map(row => row.map((v, i) => v - mean[i]));
  
  // Compute covariance matrix
  const cov: number[][] = new Array(d).fill(null).map(() => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += centered[k][i] * centered[k][j];
      }
      cov[i][j] = sum / (n - 1);
    }
  }
  
  // Power iteration for top eigenvectors
  const components = powerIteration(cov, nComponents);
  
  // Project data
  return centered.map(row => {
    return components.map(comp => 
      row.reduce((sum, v, i) => sum + v * comp[i], 0)
    );
  });
}

function powerIteration(matrix: number[][], nComponents: number): number[][] {
  const d = matrix.length;
  const components: number[][] = [];
  
  for (let c = 0; c < nComponents; c++) {
    let v = new Array(d).fill(0).map(() => Math.random() - 0.5);
    
    for (let iter = 0; iter < 100; iter++) {
      // Multiply by matrix
      const newV = new Array(d).fill(0);
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          newV[i] += matrix[i][j] * v[j];
        }
      }
      
      // Orthogonalize against previous components
      components.forEach(comp => {
        const dot = newV.reduce((s, x, i) => s + x * comp[i], 0);
        newV.forEach((_, i) => newV[i] -= dot * comp[i]);
      });
      
      // Normalize
      const norm = Math.sqrt(newV.reduce((s, x) => s + x * x, 0));
      if (norm > 1e-10) {
        v = newV.map(x => x / norm);
      }
    }
    
    components.push(v);
  }
  
  return components;
}
