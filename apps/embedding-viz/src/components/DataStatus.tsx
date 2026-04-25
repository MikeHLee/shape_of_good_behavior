import React, { useEffect, useState } from 'react';
import { checkDataAvailability } from '../utils/dataLoader';
import { DataAvailability } from '../types';

export const DataStatus: React.FC = () => {
  const [status, setStatus] = useState<DataAvailability | null>(null);
  
  useEffect(() => {
    checkDataAvailability().then(setStatus);
  }, []);
  
  if (!status) return null;
  
  if (status.available) {
    return (
      <div className="p-3 bg-green-50 border border-green-200 rounded text-sm">
        <span className="text-green-700 font-semibold">✓ Data loaded</span>
        <span className="text-gray-600 ml-2">({status.files.length} files)</span>
      </div>
    );
  }
  
  return (
    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded">
      <h3 className="font-bold text-yellow-800 mb-2">⚠ Using Mock Data</h3>
      <p className="text-sm text-gray-700 mb-2">
        Real experiment data not found. To load real data:
      </p>
      <code className="block bg-gray-800 text-green-400 p-2 rounded text-xs mb-2">
        modal volume get geodpo-data /data ./data/
      </code>
      <p className="text-xs text-gray-600">Missing: {status.missing.join(', ')}</p>
    </div>
  );
};
