//! # Safety Gym Core
//!
//! Rust core library for sheaf-theoretic safety in reinforcement learning.
//!
//! This crate provides:
//! - `TopologicalSpace` trait for arbitrary decision spaces
//! - Discrete and continuous space implementations
//! - SGPO policy inference via ONNX
//! - Godot GDExtension bindings (optional)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Python (Training)                        │
//! │  Safety Gym Environment + SGPO Policy (PyTorch)              │
//! │                         │                                   │
//! │                         ▼                                   │
//! │  Export Layer (ONNX / TorchScript)                         │
//! └─────────────────────────────────────────────────────────────┘
//!                           │
//!                           ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Rust Core Library                        │
//! │  ┌──────────────────┐  ┌──────────────────┐                │
//! │  │  ort (ONNX RT)   │  │  TopologicalSpace│                │
//! │  └──────────────────┘  └──────────────────┘                │
//! └─────────────────────────────────────────────────────────────┘
//!                           │
//!           ┌───────────────┴───────────────┐
//!           ▼                               ▼
//! ┌─────────────────────┐     ┌─────────────────────────────────┐
//! │  C FFI Bindings     │     │  Godot GDExtension              │
//! └─────────────────────┘     └─────────────────────────────────┘
//! ```

pub mod topology;
pub mod policy;
pub mod bindings;

pub use topology::{TopologicalSpace, BlackHoleRegion, TopologyData};
pub use topology::discrete::DiscreteNavigationSpace;
pub use topology::continuous::ContinuousControlSpace;

#[cfg(feature = "onnx")]
pub use policy::onnx::OnnxPolicy;

pub use policy::gpo::SGPOPolicy;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error types for the library
#[derive(Debug, thiserror::Error)]
pub enum SafetyGymError {
    #[error("State out of bounds: {0}")]
    OutOfBounds(String),
    
    #[error("Invalid embedding dimension: expected {expected}, got {got}")]
    InvalidEmbeddingDim { expected: usize, got: usize },
    
    #[error("No topology data available")]
    NoTopologyData,
    
    #[error("ONNX inference error: {0}")]
    #[cfg(feature = "onnx")]
    OnnxError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SafetyGymError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
