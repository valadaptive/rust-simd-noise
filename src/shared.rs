use std::mem::MaybeUninit;

use simdeez::prelude::*;

use crate::NoiseDimensions;

#[inline(always)]
pub unsafe fn scale_noise<S: Simd>(
    scale_min: f32,
    scale_max: f32,
    min: f32,
    max: f32,
    data: &mut Vec<f32>,
) {
    let scale_range = scale_max - scale_min;
    let range = max - min;
    let multiplier = scale_range / range;
    let offset = scale_min - min * multiplier;
    let vector_width = S::Vf32::WIDTH;
    let mut i = 0;
    if data.len() >= vector_width {
        while i <= data.len() - vector_width {
            let value = (S::Vf32::set1(multiplier) * S::Vf32::load_from_ptr_unaligned(&data[i]))
                + S::Vf32::set1(offset);
            value.copy_to_ptr_unaligned(data.get_unchecked_mut(i));
            i += vector_width;
        }
    }
    i = data.len() - (data.len() % vector_width);
    while i < data.len() {
        *data.get_unchecked_mut(i) = data.get_unchecked(i) * multiplier + offset;
        i += 1;
    }
}

simd_runtime_generate!(
    fn get_min_max(noise: &[f32]) -> (f32, f32) {
        let mut min_s = S::Vf32::set1(f32::MAX);
        let mut max_s = S::Vf32::set1(f32::MIN);

        let mut min = f32::MAX;
        let mut max = f32::MIN;

        let chunks = noise.chunks_exact(S::Vf32::WIDTH);
        for sample in chunks.remainder() {
            min = min.min(*sample);
            max = max.max(*sample);
        }
        for chunk in chunks {
            min_s = min_s.min(unsafe { S::Vf32::load_from_ptr_unaligned(chunk.as_ptr()) });
            max_s = max_s.max(unsafe { S::Vf32::load_from_ptr_unaligned(chunk.as_ptr()) });
        }
        for (min_s, max_s) in min_s.iter().zip(max_s.iter()) {
            min = min.min(min_s);
            max = max.max(max_s);
        }

        (min, max)
    }
);

simd_runtime_generate!(
    fn get_scaled_noise_inner(dim: NoiseDimensions, noise: &mut Vec<f32>, bounds: (f32, f32)) {
        let (min, max) = bounds;
        scale_noise::<S>(dim.min, dim.max, min, max, noise);
    }
);

pub(crate) fn get_scaled_noise(
    dimensions: NoiseDimensions,
    mut noise: Vec<f32>,
) -> Vec<f32> {
    let bounds = get_min_max(&noise);
    get_scaled_noise_inner(dimensions, &mut noise, bounds);
    noise
}

pub(crate) fn slice_to_maybe_uninit_mut<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
    // Safety: we know these are all initialized, so it's fine to transmute into a type that makes fewer assumptions
    unsafe { std::mem::transmute(slice) }
}
