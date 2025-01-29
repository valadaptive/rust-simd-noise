use std::mem::MaybeUninit;

use crate::dimensional_being::DimensionalBeing;
pub use crate::noise_dimensions::NoiseDimensions;
pub use crate::noise_type::NoiseType;
use crate::shared::{get_scaled_noise, slice_to_maybe_uninit_mut};

pub trait Settings: DimensionalBeing + Sized {
    fn default(dim: NoiseDimensions) -> Self;
    fn with_seed(&mut self, seed: i32) -> &mut Self;
    fn with_freq(&mut self, freq: f32) -> &mut Self;
    fn with_freq_2d(&mut self, freq_x: f32, freq_y: f32) -> &mut Self;
    fn with_freq_3d(&mut self, freq_x: f32, freq_y: f32, freq_z: f32) -> &mut Self;
    fn with_freq_4d(&mut self, freq_x: f32, freq_y: f32, freq_z: f32, freq_w: f32) -> &mut Self;

    fn get_freq_x(&self) -> f32;
    fn get_freq_y(&self) -> f32;
    fn get_freq_z(&self) -> f32;
    fn get_freq_w(&self) -> f32;

    /// If you want to call noise functions by hand, call wrap on the settings
    /// to get back a NoiseType to call the noise functions with
    fn wrap(self) -> NoiseType;


    fn generate_into_maybe_uninit(self, result: &mut [MaybeUninit<f32>]);

    fn generate_into(self, result: &mut [f32]) {
        self.generate_into_maybe_uninit(slice_to_maybe_uninit_mut(result))
    }

    /// Generate a chunk of noise based on your settings, and the min and max value
    /// generated, so you can scale it as you wish
    fn generate(self) -> Vec<f32> {
        let size = self.get_dimensions().len();
        let mut result = Vec::<f32>::with_capacity(size);
        // Safety: we initialized the Vec with a capacity of `size` so its spare capacity must be at least that much.
        // Unfortunately, there *is* a bounds check here otherwise, which regresses performance a lot.
        self.generate_into_maybe_uninit(
            unsafe { result.spare_capacity_mut().get_unchecked_mut(..size) }
        );
        unsafe { result.set_len(size); }
        result
    }

    fn validate(&self);

    /// Generate a chunk of noise with values scaled from min to max
    fn generate_scaled(self) -> Vec<f32> {
        get_scaled_noise(self.get_dimensions(), self.generate())
    }
}

pub trait SimplexSettings {
    fn with_lacunarity(&mut self, lacunarity: f32) -> &mut Self;
    fn with_gain(&mut self, gain: f32) -> &mut Self;
    fn with_octaves(&mut self, octaves: u8) -> &mut Self;
}

mod cellular2_settings;
mod cellular_settings;
mod fbm_settings;
mod gradient_settings;
mod ridge_settings;
mod turbulence_settings;

pub use cellular2_settings::Cellular2Settings;
pub use cellular_settings::CellularSettings;
pub use fbm_settings::FbmSettings;
pub use gradient_settings::GradientSettings;
pub use ridge_settings::RidgeSettings;
pub use turbulence_settings::TurbulenceSettings;
