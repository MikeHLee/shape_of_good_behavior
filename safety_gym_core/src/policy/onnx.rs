//! ONNX Runtime inference for trained policies.

use crate::{Result, SafetyGymError};
use ndarray::Array1;
use std::path::Path;

/// ONNX-based policy for inference.
///
/// Loads a trained policy from ONNX format and provides
/// inference capabilities for deployment.
pub struct OnnxPolicy {
    session: ort::Session,
    input_dim: usize,
    output_dim: usize,
}

impl OnnxPolicy {
    /// Load an ONNX model from file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let session = ort::Session::builder()
            .map_err(|e| SafetyGymError::OnnxError(e.to_string()))?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)
            .map_err(|e| SafetyGymError::OnnxError(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| SafetyGymError::OnnxError(e.to_string()))?;
        
        // Get input/output dimensions from model metadata
        let inputs = session.inputs.clone();
        let outputs = session.outputs.clone();
        
        let input_dim = inputs.first()
            .and_then(|i| i.input_type.tensor_dimensions())
            .and_then(|dims| dims.last().copied())
            .unwrap_or(0) as usize;
        
        let output_dim = outputs.first()
            .and_then(|o| o.output_type.tensor_dimensions())
            .and_then(|dims| dims.last().copied())
            .unwrap_or(0) as usize;
        
        Ok(Self {
            session,
            input_dim,
            output_dim,
        })
    }
    
    /// Get the expected input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    
    /// Get the output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    
    /// Run inference on a single state.
    pub fn predict(&self, state: &[f32]) -> Result<Vec<f32>> {
        use ort::inputs;
        
        let input_array = ndarray::Array2::from_shape_vec(
            (1, state.len()),
            state.to_vec(),
        ).map_err(|e| SafetyGymError::OnnxError(e.to_string()))?;
        
        let outputs = self.session
            .run(inputs![input_array].map_err(|e| SafetyGymError::OnnxError(e.to_string()))?)
            .map_err(|e| SafetyGymError::OnnxError(e.to_string()))?;
        
        let output = outputs.first()
            .ok_or_else(|| SafetyGymError::OnnxError("No output from model".to_string()))?;
        
        let output_tensor = output.try_extract_tensor::<f32>()
            .map_err(|e| SafetyGymError::OnnxError(e.to_string()))?;
        
        Ok(output_tensor.view().iter().copied().collect())
    }
    
    /// Run batch inference.
    pub fn predict_batch(&self, states: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        use ort::inputs;
        
        if states.is_empty() {
            return Ok(Vec::new());
        }
        
        let batch_size = states.len();
        let state_dim = states[0].len();
        
        let flat: Vec<f32> = states.iter().flatten().copied().collect();
        let input_array = ndarray::Array2::from_shape_vec(
            (batch_size, state_dim),
            flat,
        ).map_err(|e| SafetyGymError::OnnxError(e.to_string()))?;
        
        let outputs = self.session
            .run(inputs![input_array].map_err(|e| SafetyGymError::OnnxError(e.to_string()))?)
            .map_err(|e| SafetyGymError::OnnxError(e.to_string()))?;
        
        let output = outputs.first()
            .ok_or_else(|| SafetyGymError::OnnxError("No output from model".to_string()))?;
        
        let output_tensor = output.try_extract_tensor::<f32>()
            .map_err(|e| SafetyGymError::OnnxError(e.to_string()))?;
        
        let view = output_tensor.view();
        let output_dim = self.output_dim;
        
        Ok((0..batch_size)
            .map(|i| {
                let start = i * output_dim;
                let end = start + output_dim;
                view.iter().skip(start).take(output_dim).copied().collect()
            })
            .collect())
    }
}

impl std::fmt::Debug for OnnxPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxPolicy")
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests require an actual ONNX model file
    // They are marked as ignored by default
    
    #[test]
    #[ignore]
    fn test_load_model() {
        // Would require a test model file
    }
}
