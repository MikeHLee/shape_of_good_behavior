//! Policy modules for SGPO inference.

pub mod control;
pub mod gpo;

#[cfg(feature = "onnx")]
pub mod onnx;
