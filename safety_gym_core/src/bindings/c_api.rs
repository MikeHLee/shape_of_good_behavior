//! C FFI bindings for integration with C/C++/Unity.
//!
//! These functions provide a stable C ABI for using the safety gym
//! from languages that support C FFI.

use std::ffi::{c_char, c_float, c_int};
use std::slice;

use crate::topology::discrete::DiscreteNavigationSpace;
use crate::topology::continuous::ContinuousControlSpace;
use crate::topology::TopologicalSpace;
use crate::policy::gpo::SGPOPolicy;

/// Opaque handle to a discrete navigation space
pub struct DiscreteSpaceHandle(DiscreteNavigationSpace);

/// Opaque handle to a continuous control space
pub struct ContinuousSpaceHandle(ContinuousControlSpace);

/// Opaque handle to a SGPO policy
pub struct SGPOPolicyHandle<S: TopologicalSpace>(SGPOPolicy<S>);

// ============================================================================
// Discrete Navigation Space
// ============================================================================

/// Create a new discrete navigation space.
///
/// # Safety
/// Returns a raw pointer that must be freed with `discrete_space_free`.
#[no_mangle]
pub unsafe extern "C" fn discrete_space_new(
    width: c_int,
    height: c_int,
    embedding_dim: c_int,
    seed: u64,
) -> *mut DiscreteSpaceHandle {
    let space = DiscreteNavigationSpace::new(
        (width as usize, height as usize),
        vec![],
        embedding_dim as usize,
        seed,
    );
    Box::into_raw(Box::new(DiscreteSpaceHandle(space)))
}

/// Free a discrete navigation space.
///
/// # Safety
/// `handle` must be a valid pointer returned by `discrete_space_new`.
#[no_mangle]
pub unsafe extern "C" fn discrete_space_free(handle: *mut DiscreteSpaceHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Add a hazard position.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn discrete_space_add_hazard(
    handle: *mut DiscreteSpaceHandle,
    x: c_int,
    y: c_int,
) {
    if let Some(h) = handle.as_mut() {
        h.0.add_hazard((x, y));
    }
}

/// Check if a position is safe.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn discrete_space_is_safe(
    handle: *const DiscreteSpaceHandle,
    x: c_int,
    y: c_int,
) -> c_int {
    handle.as_ref()
        .map(|h| if h.0.is_safe(&(x, y)) { 1 } else { 0 })
        .unwrap_or(0)
}

/// Compute distance between two positions.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn discrete_space_distance(
    handle: *const DiscreteSpaceHandle,
    x1: c_int,
    y1: c_int,
    x2: c_int,
    y2: c_int,
) -> c_float {
    handle.as_ref()
        .map(|h| h.0.distance(&(x1, y1), &(x2, y2)))
        .unwrap_or(0.0)
}

/// Compute harmonic risk at a position.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn discrete_space_harmonic_risk(
    handle: *const DiscreteSpaceHandle,
    x: c_int,
    y: c_int,
    k: c_int,
) -> c_float {
    handle.as_ref()
        .map(|h| h.0.compute_harmonic_risk(&(x, y), k as usize))
        .unwrap_or(0.5)
}

/// Get embedding for a position.
///
/// # Safety
/// `handle` must be a valid pointer.
/// `out_embedding` must point to a buffer of at least `embedding_dim` floats.
#[no_mangle]
pub unsafe extern "C" fn discrete_space_embed(
    handle: *const DiscreteSpaceHandle,
    x: c_int,
    y: c_int,
    out_embedding: *mut c_float,
    embedding_dim: c_int,
) -> c_int {
    let Some(h) = handle.as_ref() else { return -1 };
    
    let embedding = h.0.embed(&(x, y));
    let dim = embedding.len().min(embedding_dim as usize);
    
    let out = slice::from_raw_parts_mut(out_embedding, dim);
    for (i, &val) in embedding.iter().take(dim).enumerate() {
        out[i] = val;
    }
    
    dim as c_int
}

// ============================================================================
// Continuous Control Space
// ============================================================================

/// Create a new continuous control space.
///
/// # Safety
/// Returns a raw pointer that must be freed with `continuous_space_free`.
#[no_mangle]
pub unsafe extern "C" fn continuous_space_new(
    x_min: c_float,
    x_max: c_float,
    y_min: c_float,
    y_max: c_float,
) -> *mut ContinuousSpaceHandle {
    let space = ContinuousControlSpace::new(
        [[x_min, x_max], [y_min, y_max]],
        vec![],
        None,
    );
    Box::into_raw(Box::new(ContinuousSpaceHandle(space)))
}

/// Free a continuous control space.
///
/// # Safety
/// `handle` must be a valid pointer returned by `continuous_space_new`.
#[no_mangle]
pub unsafe extern "C" fn continuous_space_free(handle: *mut ContinuousSpaceHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Add an obstacle to the continuous space.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn continuous_space_add_obstacle(
    handle: *mut ContinuousSpaceHandle,
    x: c_float,
    y: c_float,
    radius: c_float,
) {
    if let Some(h) = handle.as_mut() {
        h.0.add_obstacle([x, y], radius);
    }
}

/// Check if a position is safe.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn continuous_space_is_safe(
    handle: *const ContinuousSpaceHandle,
    x: c_float,
    y: c_float,
) -> c_int {
    handle.as_ref()
        .map(|h| if h.0.is_safe(&[x, y]) { 1 } else { 0 })
        .unwrap_or(0)
}

/// Compute distance between two positions.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn continuous_space_distance(
    handle: *const ContinuousSpaceHandle,
    x1: c_float,
    y1: c_float,
    x2: c_float,
    y2: c_float,
) -> c_float {
    handle.as_ref()
        .map(|h| h.0.distance(&[x1, y1], &[x2, y2]))
        .unwrap_or(0.0)
}

/// Simulate one physics step.
///
/// # Safety
/// `handle` must be a valid pointer.
/// `out_pos` must point to a buffer of at least 2 floats.
/// `out_vel` must point to a buffer of at least 2 floats.
#[no_mangle]
pub unsafe extern "C" fn continuous_space_step(
    handle: *const ContinuousSpaceHandle,
    pos_x: c_float,
    pos_y: c_float,
    vel_x: c_float,
    vel_y: c_float,
    action_x: c_float,
    action_y: c_float,
    out_pos: *mut c_float,
    out_vel: *mut c_float,
) {
    let Some(h) = handle.as_ref() else { return };
    
    let (new_pos, new_vel) = h.0.step(
        &[pos_x, pos_y],
        &[vel_x, vel_y],
        &[action_x, action_y],
    );
    
    let pos_out = slice::from_raw_parts_mut(out_pos, 2);
    let vel_out = slice::from_raw_parts_mut(out_vel, 2);
    
    pos_out[0] = new_pos[0];
    pos_out[1] = new_pos[1];
    vel_out[0] = new_vel[0];
    vel_out[1] = new_vel[1];
}

// ============================================================================
// Version Info
// ============================================================================

/// Get the library version string.
///
/// # Safety
/// Returns a pointer to a static string. Do not free.
#[no_mangle]
pub extern "C" fn safety_gym_version() -> *const c_char {
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_discrete_space_c_api() {
        unsafe {
            let handle = discrete_space_new(10, 10, 64, 42);
            assert!(!handle.is_null());
            
            discrete_space_add_hazard(handle, 5, 5);
            
            assert_eq!(discrete_space_is_safe(handle, 0, 0), 1);
            assert_eq!(discrete_space_is_safe(handle, 5, 5), 0);
            
            let dist = discrete_space_distance(handle, 0, 0, 3, 4);
            assert_eq!(dist, 7.0);
            
            discrete_space_free(handle);
        }
    }
    
    #[test]
    fn test_continuous_space_c_api() {
        unsafe {
            let handle = continuous_space_new(-10.0, 10.0, -10.0, 10.0);
            assert!(!handle.is_null());
            
            continuous_space_add_obstacle(handle, 0.0, 0.0, 1.0);
            
            assert_eq!(continuous_space_is_safe(handle, 5.0, 5.0), 1);
            assert_eq!(continuous_space_is_safe(handle, 0.0, 0.0), 0);
            
            continuous_space_free(handle);
        }
    }
}
