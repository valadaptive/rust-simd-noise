//! AVX2 and FMA3 Accelerated noise functions.
//! CPUs since ~2013 (Intel) and ~2015 (AMD) support this.
//! It is about twice as fast as the SSE2 version.
//!
//! Use `is_x86_feature_detected!("avx2")` provided
//! by the Rust stanard library to detect at runtime.
//!
//! When using the `get_` functions, you will get a performance boost when width
//! is evenly divisble by 8, and when it is not small relative height and depth.

use crate::noise::cell_32;
use crate::noise::cell_64;
use crate::noise::fbm_32;
use crate::noise::fbm_64;
use crate::noise::ridge_32;
use crate::noise::ridge_64;
use crate::noise::simplex_32;
use crate::noise::simplex_64;
use crate::noise::turbulence_32;
use crate::noise::turbulence_64;
use crate::shared::scale_noise;
use crate::{CellDistanceFunction, CellReturnType, DimensionalBeing, NoiseType};

use simdeez::avx2::{Avx2, F32x8, F64x4};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::f32;

cellular!("2d", cellular_2d, __m256, F32x8, i32, cell_32, Avx2);
cellular!("3d", cellular_3d, __m256, F32x8, i32, cell_32, Avx2);
cellular!("2d", cellular_2d_f64, __m256d, F64x4, i64, cell_64, Avx2);
cellular!("3d", cellular_3d_f64, __m256d, F64x4, i64, cell_64, Avx2);

simplex!("1d", simplex_1d, __m256, F32x8, i32, simplex_32, Avx2);
simplex!("2d", simplex_2d, __m256, F32x8, i32, simplex_32, Avx2);
simplex!("3d", simplex_3d, __m256, F32x8, i32, simplex_32, Avx2);
simplex!("4d", simplex_4d, __m256, F32x8, i32, simplex_32, Avx2);
simplex!("1d", simplex_1d_f64, __m256d, F64x4, i64, simplex_64, Avx2);
simplex!("2d", simplex_2d_f64, __m256d, F64x4, i64, simplex_64, Avx2);
simplex!("3d", simplex_3d_f64, __m256d, F64x4, i64, simplex_64, Avx2);
simplex!("4d", simplex_4d_f64, __m256d, F64x4, i64, simplex_64, Avx2);

fbm!("1d", fbm_1d, __m256, F32x8, i32, fbm_32, Avx2);
fbm!("2d", fbm_2d, __m256, F32x8, i32, fbm_32, Avx2);
fbm!("3d", fbm_3d, __m256, F32x8, i32, fbm_32, Avx2);
fbm!("4d", fbm_4d, __m256, F32x8, i32, fbm_32, Avx2);
fbm!("1d", fbm_1d_f64, __m256d, F64x4, i64, fbm_64, Avx2);
fbm!("2d", fbm_2d_f64, __m256d, F64x4, i64, fbm_64, Avx2);
fbm!("3d", fbm_3d_f64, __m256d, F64x4, i64, fbm_64, Avx2);
fbm!("4d", fbm_4d_f64, __m256d, F64x4, i64, fbm_64, Avx2);

ridge!("1d", ridge_1d, __m256, F32x8, i32, ridge_32, Avx2);
ridge!("2d", ridge_2d, __m256, F32x8, i32, ridge_32, Avx2);
ridge!("3d", ridge_3d, __m256, F32x8, i32, ridge_32, Avx2);
ridge!("4d", ridge_4d, __m256, F32x8, i32, ridge_32, Avx2);
ridge!("1d", ridge_1d_f64, __m256d, F64x4, i64, ridge_64, Avx2);
ridge!("2d", ridge_2d_f64, __m256d, F64x4, i64, ridge_64, Avx2);
ridge!("3d", ridge_3d_f64, __m256d, F64x4, i64, ridge_64, Avx2);
ridge!("4d", ridge_4d_f64, __m256d, F64x4, i64, ridge_64, Avx2);

/// Get a single value of 2d turbulence.
pub unsafe fn turbulence_1d(
    x: __m256,
    lacunarity: __m256,
    gain: __m256,
    octaves: u8,
    seed: i32,
) -> __m256 {
    turbulence_32::turbulence_1d::<Avx2>(F32x8(x), F32x8(lacunarity), F32x8(gain), octaves, seed).0
}

/// Get a single value of 2d turbulence.
pub unsafe fn turbulence_1d_f64(
    x: __m256d,
    lacunarity: __m256d,
    gain: __m256d,
    octaves: u8,
    seed: i64,
) -> __m256d {
    turbulence_64::turbulence_1d::<Avx2>(F64x4(x), F64x4(lacunarity), F64x4(gain), octaves, seed).0
}

/// Gets a width sized block of 1d noise, unscaled.
/// `start_x` can be used to provide an offset in the
/// coordinates. Results are unscaled, 'min' and 'max' noise values
/// are returned so you can scale and transform the noise as you see fit
/// in a single pass.
pub unsafe fn get_1d_noise(noise_type: &NoiseType) -> (Vec<f32>, f32, f32) {
    crate::noise_helpers_32::get_1d_noise::<Avx2>(noise_type)
}

pub unsafe fn get_1d_noise_64(noise_type: &NoiseType) -> (Vec<f64>, f64, f64) {
    crate::noise_helpers_64::get_1d_noise_f64::<Avx2>(noise_type)
}

/// Gets a width sized block of scaled 2d noise
/// `start_x` can be used to provide an offset in the
/// coordinates.
/// `scaled_min` and `scaled_max` specify the range you want the noise scaled to.
pub unsafe fn get_1d_scaled_noise(noise_type: &NoiseType) -> Vec<f32> {
    let (mut noise, min, max) = get_1d_noise(noise_type);
    let dim = noise_type.get_dimensions();
    scale_noise::<Avx2>(dim.min, dim.max, min, max, &mut noise);
    noise
}

/// Get a single value of 2d turbulence.
pub unsafe fn turbulence_2d(
    x: __m256,
    y: __m256,
    lac: __m256,
    gain: __m256,
    octaves: u8,
    seed: i32,
) -> __m256 {
    turbulence_32::turbulence_2d::<Avx2>(F32x8(x), F32x8(y), F32x8(lac), F32x8(gain), octaves, seed)
        .0
}

/// Get a single value of 2d turbulence.
pub unsafe fn turbulence_2d_f64(
    x: __m256d,
    y: __m256d,
    lac: __m256d,
    gain: __m256d,
    octaves: u8,
    seed: i64,
) -> __m256d {
    turbulence_64::turbulence_2d::<Avx2>(F64x4(x), F64x4(y), F64x4(lac), F64x4(gain), octaves, seed)
        .0
}
/// Gets a width X height sized block of 2d noise, unscaled.
/// `start_x` and `start_y` can be used to provide an offset in the
/// coordinates. Results are unscaled, 'min' and 'max' noise values
/// are returned so you can scale and transform the noise as you see fit
/// in a single pass.
pub unsafe fn get_2d_noise(noise_type: &NoiseType) -> (Vec<f32>, f32, f32) {
    crate::noise_helpers_32::get_2d_noise::<Avx2>(noise_type)
}

pub unsafe fn get_2d_noise_64(noise_type: &NoiseType) -> (Vec<f64>, f64, f64) {
    crate::noise_helpers_64::get_2d_noise_f64::<Avx2>(noise_type)
}

/// Gets a width X height sized block of scaled 2d noise
/// `start_x` and `start_y` can be used to provide an offset in the
/// coordinates.
/// `scaled_min` and `scaled_max` specify the range you want the noise scaled to.
pub unsafe fn get_2d_scaled_noise(noise_type: &NoiseType) -> Vec<f32> {
    let (mut noise, min, max) = get_2d_noise(noise_type);
    let dim = noise_type.get_dimensions();
    scale_noise::<Avx2>(dim.min, dim.max, min, max, &mut noise);
    noise
}

/// Get a single value of 3d turbulence.
pub unsafe fn turbulence_3d(
    x: __m256,
    y: __m256,
    z: __m256,
    lac: __m256,
    gain: __m256,
    octaves: u8,
    seed: i32,
) -> __m256 {
    turbulence_32::turbulence_3d::<Avx2>(
        F32x8(x),
        F32x8(y),
        F32x8(z),
        F32x8(lac),
        F32x8(gain),
        octaves,
        seed,
    )
    .0
}

/// Get a single value of 3d turbulence.
pub unsafe fn turbulence_3d_f64(
    x: __m256d,
    y: __m256d,
    z: __m256d,
    lac: __m256d,
    gain: __m256d,
    octaves: u8,
    seed: i64,
) -> __m256d {
    turbulence_64::turbulence_3d::<Avx2>(
        F64x4(x),
        F64x4(y),
        F64x4(z),
        F64x4(lac),
        F64x4(gain),
        octaves,
        seed,
    )
    .0
}

/// Gets a width X height X depth sized block of 3d noise, unscaled,
/// `start_x`,`start_y` and `start_z` can be used to provide an offset in the
/// coordinates. Results are unscaled, 'min' and 'max' noise values
/// are returned so you can scale and transform the noise as you see fit
/// in a single pass.
pub unsafe fn get_3d_noise(noise_type: &NoiseType) -> (Vec<f32>, f32, f32) {
    crate::noise_helpers_32::get_3d_noise::<Avx2>(noise_type)
}

pub unsafe fn get_3d_noise_64(noise_type: &NoiseType) -> (Vec<f64>, f64, f64) {
    crate::noise_helpers_64::get_3d_noise_f64::<Avx2>(noise_type)
}

/// Gets a width X height X depth sized block of scaled 3d noise
/// `start_x`, `start_y` and `start_z` can be used to provide an offset in the
/// coordinates.
/// `scaled_min` and `scaled_max` specify the range you want the noise scaled to.
pub unsafe fn get_3d_scaled_noise(noise_type: &NoiseType) -> Vec<f32> {
    let (mut noise, min, max) = get_3d_noise(noise_type);
    let dim = noise_type.get_dimensions();
    scale_noise::<Avx2>(dim.min, dim.max, min, max, &mut noise);
    noise
}

/// Get a single value of 4d turbulence.
pub unsafe fn turbulence_4d(
    x: __m256,
    y: __m256,
    z: __m256,
    w: __m256,
    lac: __m256,
    gain: __m256,
    octaves: u8,
    seed: i32,
) -> __m256 {
    turbulence_32::turbulence_4d::<Avx2>(
        F32x8(x),
        F32x8(y),
        F32x8(z),
        F32x8(w),
        F32x8(lac),
        F32x8(gain),
        octaves,
        seed,
    )
    .0
}

/// Get a single value of 4d turbulence.
pub unsafe fn turbulence_4d_f64(
    x: __m256d,
    y: __m256d,
    z: __m256d,
    w: __m256d,
    lac: __m256d,
    gain: __m256d,
    octaves: u8,
    seed: i64,
) -> __m256d {
    turbulence_64::turbulence_4d::<Avx2>(
        F64x4(x),
        F64x4(y),
        F64x4(z),
        F64x4(w),
        F64x4(lac),
        F64x4(gain),
        octaves,
        seed,
    )
    .0
}

/// Gets a width X height X depth x time sized block of 4d noise, unscaled,
/// `start_*` can be used to provide an offset in the
/// coordinates. Results are unscaled, 'min' and 'max' noise values
/// are returned so you can scale and transform the noise as you see fit
/// in a single pass.
pub unsafe fn get_4d_noise(noise_type: &NoiseType) -> (Vec<f32>, f32, f32) {
    crate::noise_helpers_32::get_4d_noise::<Avx2>(noise_type)
}

pub unsafe fn get_4d_noise_64(noise_type: &NoiseType) -> (Vec<f64>, f64, f64) {
    crate::noise_helpers_64::get_4d_noise_f64::<Avx2>(noise_type)
}

/// Gets a width X height X depth X time sized block of scaled 4d noise
/// `start_*` can be used to provide an offset in the
/// coordinates.
/// `scaled_min` and `scaled_max` specify the range you want the noise scaled to.
pub unsafe fn get_4d_scaled_noise(noise_type: &NoiseType) -> Vec<f32> {
    let (mut noise, min, max) = get_4d_noise(noise_type);
    let dim = noise_type.get_dimensions();
    scale_noise::<Avx2>(dim.min, dim.max, min, max, &mut noise);
    noise
}
