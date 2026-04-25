//! FFI bindings for external integrations.

pub mod c_api;

#[cfg(feature = "godot")]
pub mod godot;
