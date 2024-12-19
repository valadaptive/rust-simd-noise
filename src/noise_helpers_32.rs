use crate::dimensional_being::DimensionalBeing;
use crate::Settings;

use simdeez::prelude::*;

use std::f32;
use std::mem::MaybeUninit;

pub trait Sample32<S: Simd>: DimensionalBeing + Settings {
    fn sample_1d(&self, x: S::Vf32) -> S::Vf32;
    fn sample_2d(&self, x: S::Vf32, y: S::Vf32) -> S::Vf32;
    fn sample_3d(&self, x: S::Vf32, y: S::Vf32, z: S::Vf32) -> S::Vf32;
    fn sample_4d(&self, x: S::Vf32, y: S::Vf32, z: S::Vf32, w: S::Vf32) -> S::Vf32;
}

#[inline(always)]
pub unsafe fn get_1d_noise_helper_f32<S: Simd, Settings: Sample32<S>>(
    settings: Settings,
    result: &mut [MaybeUninit<f32>],
) -> (f32, f32) {
    let dim = settings.get_dimensions();
    let freq_x = S::Vf32::set1(settings.get_freq_x());
    let start_x = dim.x;
    let width = dim.width;
    assert_eq!(result.len(), width);
    let mut min_s = S::Vf32::set1(f32::MAX);
    let mut max_s = S::Vf32::set1(f32::MIN);

    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let result_ptr = result.as_mut_ptr() as *mut f32;
    let mut i = 0;
    let vector_width = S::Vf32::WIDTH;
    let remainder = width % vector_width;
    let mut x_arr = Vec::<f32>::with_capacity(vector_width);
    let x_ptr = x_arr.as_mut_ptr();
    for i in (0..vector_width).rev() {
        x_ptr.add(i).write(start_x + i as f32);
    }
    x_arr.set_len(vector_width);
    let mut x = S::Vf32::load_from_ptr_unaligned(x_ptr);
    for _ in 0..width / vector_width {
        let f = settings.sample_1d(x * freq_x);
        max_s = max_s.max(f);
        min_s = min_s.min(f);
        f.copy_to_ptr_unaligned(result_ptr.add(i));
        i += vector_width;
        x = x + S::Vf32::set1(vector_width as f32);
    }
    if remainder != 0 {
        let f = settings.sample_1d(x * freq_x);
        for j in 0..remainder {
            let n = f[j];
            result_ptr.add(i).write(n);
            // Note: This is unecessary for large images
            if n < min {
                min = n;
            }
            if n > max {
                max = n;
            }
            i += 1;
        }
    }
    for i in 0..vector_width {
        if min_s[i] < min {
            min = min_s[i];
        }
        if max_s[i] > max {
            max = max_s[i];
        }
    }
    (min, max)
}

#[inline(always)]
pub unsafe fn get_2d_noise_helper_f32<S: Simd, Settings: Sample32<S>>(
    settings: Settings,
    result: &mut [MaybeUninit<f32>],
) -> (f32, f32) {
    let dim = settings.get_dimensions();
    let freq_x = S::Vf32::set1(settings.get_freq_x());
    let freq_y = S::Vf32::set1(settings.get_freq_y());
    let start_x = dim.x;
    let width = dim.width;
    let start_y = dim.y;
    let height = dim.height;
    assert_eq!(result.len(), width * height);

    let mut min_s = S::Vf32::set1(f32::MAX);
    let mut max_s = S::Vf32::set1(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let result_ptr = result.as_mut_ptr() as *mut f32;
    let mut y = S::Vf32::set1(start_y);
    let mut i = 0;
    let vector_width = S::Vf32::WIDTH;
    let remainder = width % vector_width;
    let mut x_arr = Vec::<f32>::with_capacity(vector_width);
    let x_ptr = x_arr.as_mut_ptr();
    for i in (0..vector_width).rev() {
        x_ptr.add(i).write(start_x + i as f32);
    }
    x_arr.set_len(vector_width);
    for _ in 0..height {
        let mut x = S::Vf32::load_from_ptr_unaligned(x_ptr);
        for _ in 0..width / vector_width {
            let f = settings.sample_2d(x * freq_x, y * freq_y);
            max_s = max_s.max(f);
            min_s = min_s.min(f);
            f.copy_to_ptr_unaligned(result_ptr.add(i));
            i += vector_width;
            x = x + S::Vf32::set1(vector_width as f32);
        }
        if remainder != 0 {
            let f = settings.sample_2d(x * freq_x, y * freq_y);
            for j in 0..remainder {
                let n = f[j];
                result_ptr.add(i).write(n);
                if n < min {
                    min = n;
                }
                if n > max {
                    max = n;
                }
                i += 1;
            }
        }
        y = y + S::Vf32::set1(1.0);
    }
    for i in 0..vector_width {
        if min_s[i] < min {
            min = min_s[i];
        }
        if max_s[i] > max {
            max = max_s[i];
        }
    }
    (min, max)
}

#[inline(always)]
unsafe fn get_3d_noise_helper_f32<S: Simd, Settings: Sample32<S>>(
    settings: Settings,
    result: &mut [MaybeUninit<f32>],
) -> (f32, f32) {
    let dim = settings.get_dimensions();
    let freq_x = S::Vf32::set1(settings.get_freq_x());
    let freq_y = S::Vf32::set1(settings.get_freq_y());
    let freq_z = S::Vf32::set1(settings.get_freq_z());
    let start_x = dim.x;
    let width = dim.width;
    let start_y = dim.y;
    let height = dim.height;
    let start_z = dim.z;
    let depth = dim.depth;
    assert_eq!(result.len(), width * height * depth);

    let mut min_s = S::Vf32::set1(f32::MAX);
    let mut max_s = S::Vf32::set1(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let result_ptr = result.as_mut_ptr() as *mut f32;
    let mut i = 0;
    let vector_width = S::Vf32::WIDTH;
    let remainder = width % vector_width;
    let mut x_arr = Vec::<f32>::with_capacity(vector_width);
    let x_ptr = x_arr.as_mut_ptr();
    for i in (0..vector_width).rev() {
        x_ptr.add(i).write(start_x + i as f32);
    }
    x_arr.set_len(vector_width);

    let mut z = S::Vf32::set1(start_z);
    for _ in 0..depth {
        let mut y = S::Vf32::set1(start_y);
        for _ in 0..height {
            let mut x = S::Vf32::load_from_ptr_unaligned(&x_arr[0]);
            for _ in 0..width / vector_width {
                let f = settings.sample_3d(x * freq_x, y * freq_y, z * freq_z);
                max_s = max_s.max(f);
                min_s = min_s.min(f);
                f.copy_to_ptr_unaligned(result_ptr.add(i));
                i += vector_width;
                x = x + S::Vf32::set1(vector_width as f32);
            }
            if remainder != 0 {
                let f = settings.sample_3d(x * freq_x, y * freq_y, z * freq_z);
                for j in 0..remainder {
                    let n = f[j];
                    result_ptr.add(i).write(n);
                    if n < min {
                        min = n;
                    }
                    if n > max {
                        max = n;
                    }
                    i += 1;
                }
            }
            y = y + S::Vf32::set1(1.0);
        }
        z = z + S::Vf32::set1(1.0);
    }
    for i in 0..vector_width {
        if min_s[i] < min {
            min = min_s[i];
        }
        if max_s[i] > max {
            max = max_s[i];
        }
    }
    (min, max)
}

#[inline(always)]
unsafe fn get_4d_noise_helper_f32<S: Simd, Settings: Sample32<S>>(
    settings: Settings,
    result: &mut [MaybeUninit<f32>],
) -> (f32, f32) {
    let dim = settings.get_dimensions();
    let freq_x = S::Vf32::set1(settings.get_freq_x());
    let freq_y = S::Vf32::set1(settings.get_freq_y());
    let freq_z = S::Vf32::set1(settings.get_freq_z());
    let freq_w = S::Vf32::set1(settings.get_freq_w());
    let start_x = dim.x;
    let width = dim.width;
    let start_y = dim.y;
    let height = dim.height;
    let start_z = dim.z;
    let depth = dim.depth;
    let start_w = dim.w;
    let time = dim.time;
    assert_eq!(result.len(), width * height * depth * time);

    let mut min_s = S::Vf32::set1(f32::MAX);
    let mut max_s = S::Vf32::set1(f32::MIN);
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    let result_ptr = result.as_mut_ptr() as *mut f32;
    let mut i = 0;
    let vector_width = S::Vf32::WIDTH;
    let remainder = width % vector_width;
    let mut x_arr = Vec::<f32>::with_capacity(vector_width);
    let x_ptr = x_arr.as_mut_ptr();
    for i in (0..vector_width).rev() {
        x_ptr.add(i).write(start_x + i as f32);
    }
    x_arr.set_len(vector_width);
    let mut w = S::Vf32::set1(start_w);
    for _ in 0..time {
        let mut z = S::Vf32::set1(start_z);
        for _ in 0..depth {
            let mut y = S::Vf32::set1(start_y);
            for _ in 0..height {
                let mut x = S::Vf32::load_from_ptr_unaligned(&x_arr[0]);
                for _ in 0..width / vector_width {
                    let f = settings.sample_4d(x * freq_x, y * freq_y, z * freq_z, w * freq_w);
                    max_s = max_s.max(f);
                    min_s = min_s.min(f);
                    f.copy_to_ptr_unaligned(result_ptr.add(i));
                    i += vector_width;
                    x = x + S::Vf32::set1(vector_width as f32);
                }
                if remainder != 0 {
                    let f = settings.sample_4d(x * freq_x, y * freq_y, z * freq_z, w * freq_w);
                    for j in 0..remainder {
                        let n = f[j];
                        result_ptr.add(i).write(n);
                        // Note: This is unecessary for large images
                        if n < min {
                            min = n;
                        }
                        if n > max {
                            max = n;
                        }
                        i += 1;
                    }
                }
                y = y + S::Vf32::set1(1.0);
            }
            z = z + S::Vf32::set1(1.0);
        }
        w = w + S::Vf32::set1(1.0);
    }
    for i in 0..vector_width {
        if min_s[i] < min {
            min = min_s[i];
        }
        if max_s[i] > max {
            max = max_s[i];
        }
    }
    (min, max)
}

macro_rules! generate_noise_helper_dispatch {
    ($helper_name:ident, $settings_ty:ty, $settings_name:ident, $result:ident) => {{
        #[allow(non_camel_case_types)]
        {
            use simdeez::prelude::*;
            simd_runtime_generate!(
                fn gen_noise($settings_name: &$settings_ty, result: &mut [MaybeUninit<f32>]) -> (f32, f32) {
                    $helper_name::<S, $settings_ty>(*$settings_name, result)
                }
            );
            gen_noise($settings_name, $result)
        }
    }};
}

pub mod get_1d_noise {
    use std::mem::MaybeUninit;

    use crate::{noise_helpers_32::get_1d_noise_helper_f32, NoiseType};
    use crate::{FbmSettings, GradientSettings, RidgeSettings, TurbulenceSettings};
    #[allow(dead_code)]
    pub fn get_1d_noise(noise_type: &NoiseType, result: &mut [MaybeUninit<f32>]) -> (f32, f32) {
        match noise_type {
            NoiseType::Fbm(s) => {
                generate_noise_helper_dispatch!(get_1d_noise_helper_f32, FbmSettings, s, result)
            }
            NoiseType::Ridge(s) => {
                generate_noise_helper_dispatch!(get_1d_noise_helper_f32, RidgeSettings, s, result)
            }
            NoiseType::Turbulence(s) => {
                generate_noise_helper_dispatch!(get_1d_noise_helper_f32, TurbulenceSettings, s, result)
            }
            NoiseType::Gradient(s) => {
                generate_noise_helper_dispatch!(get_1d_noise_helper_f32, GradientSettings, s, result)
            }
            NoiseType::Cellular(_) => {
                panic!("not implemented");
            }
            NoiseType::Cellular2(_) => {
                panic!("not implemented");
            }
        }
    }
}

pub mod get_2d_noise {
    use std::mem::MaybeUninit;

    use crate::{noise_helpers_32::get_2d_noise_helper_f32, NoiseType};
    use crate::{
        Cellular2Settings, CellularSettings, FbmSettings, GradientSettings, RidgeSettings,
        TurbulenceSettings,
    };
    /// Gets a width X height sized block of 2d noise, unscaled.
    /// `start_x` and `start_y` can be used to provide an offset in the
    /// coordinates. Results are unscaled, 'min' and 'max' noise values
    /// are returned so you can scale and transform the noise as you see fit
    /// in a single pass.
    #[allow(dead_code)]
    pub fn get_2d_noise(noise_type: &NoiseType, result: &mut [MaybeUninit<f32>]) -> (f32, f32) {
        match noise_type {
            NoiseType::Fbm(s) => {
                generate_noise_helper_dispatch!(get_2d_noise_helper_f32, FbmSettings, s, result)
            }
            NoiseType::Ridge(s) => {
                generate_noise_helper_dispatch!(get_2d_noise_helper_f32, RidgeSettings, s, result)
            }
            NoiseType::Turbulence(s) => {
                generate_noise_helper_dispatch!(get_2d_noise_helper_f32, TurbulenceSettings, s, result)
            }
            NoiseType::Gradient(s) => {
                generate_noise_helper_dispatch!(get_2d_noise_helper_f32, GradientSettings, s, result)
            }
            NoiseType::Cellular(s) => {
                generate_noise_helper_dispatch!(get_2d_noise_helper_f32, CellularSettings, s, result)
            }
            NoiseType::Cellular2(s) => {
                generate_noise_helper_dispatch!(get_2d_noise_helper_f32, Cellular2Settings, s, result)
            }
        }
    }
}

pub mod get_3d_noise {
    use std::mem::MaybeUninit;

    use crate::{noise_helpers_32::get_3d_noise_helper_f32, NoiseType};
    use crate::{
        Cellular2Settings, CellularSettings, FbmSettings, GradientSettings, RidgeSettings,
        TurbulenceSettings,
    };
    /// Gets a width X height X depth sized block of 3d noise, unscaled,
    /// `start_x`,`start_y` and `start_z` can be used to provide an offset in the
    /// coordinates. Results are unscaled, 'min' and 'max' noise values
    /// are returned so you can scale and transform the noise as you see fit
    /// in a single pass.
    #[allow(dead_code)]
    pub fn get_3d_noise(noise_type: &NoiseType, result: &mut [MaybeUninit<f32>]) -> (f32, f32) {
        match noise_type {
            NoiseType::Fbm(s) => {
                generate_noise_helper_dispatch!(get_3d_noise_helper_f32, FbmSettings, s, result)
            }
            NoiseType::Ridge(s) => {
                generate_noise_helper_dispatch!(get_3d_noise_helper_f32, RidgeSettings, s, result)
            }
            NoiseType::Turbulence(s) => {
                generate_noise_helper_dispatch!(get_3d_noise_helper_f32, TurbulenceSettings, s, result)
            }
            NoiseType::Gradient(s) => {
                generate_noise_helper_dispatch!(get_3d_noise_helper_f32, GradientSettings, s, result)
            }
            NoiseType::Cellular(s) => {
                generate_noise_helper_dispatch!(get_3d_noise_helper_f32, CellularSettings, s, result)
            }
            NoiseType::Cellular2(s) => {
                generate_noise_helper_dispatch!(get_3d_noise_helper_f32, Cellular2Settings, s, result)
            }
        }
    }
}

pub mod get_4d_noise {
    use std::mem::MaybeUninit;

    use crate::{noise_helpers_32::get_4d_noise_helper_f32, NoiseType};
    use crate::{FbmSettings, GradientSettings, RidgeSettings, TurbulenceSettings};
    #[allow(dead_code)]
    pub fn get_4d_noise(noise_type: &NoiseType, result: &mut [MaybeUninit<f32>]) -> (f32, f32) {
        match noise_type {
            NoiseType::Fbm(s) => {
                generate_noise_helper_dispatch!(get_4d_noise_helper_f32, FbmSettings, s, result)
            }
            NoiseType::Ridge(s) => {
                generate_noise_helper_dispatch!(get_4d_noise_helper_f32, RidgeSettings, s, result)
            }
            NoiseType::Turbulence(s) => {
                generate_noise_helper_dispatch!(get_4d_noise_helper_f32, TurbulenceSettings, s, result)
            }
            NoiseType::Gradient(s) => {
                generate_noise_helper_dispatch!(get_4d_noise_helper_f32, GradientSettings, s, result)
            }
            NoiseType::Cellular(_) => {
                panic!("not implemented");
            }
            NoiseType::Cellular2(_) => {
                panic!("not implemented");
            }
        }
    }
}
