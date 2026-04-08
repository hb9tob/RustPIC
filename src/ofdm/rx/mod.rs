pub mod sync;
pub mod mode_detect;
pub mod equalizer;
pub mod demapper;
pub mod frame;

use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::ofdm::params::*;

/// FFT-demodulates one CP-stripped OFDM symbol window.
///
/// Takes [`FFT_SIZE`] time-domain samples (cyclic prefix **already removed**)
/// and returns the [`NUM_CARRIERS`] scaled complex values for the active
/// subcarriers (bins [`FIRST_BIN`] … [`LAST_BIN`]).
///
/// Scaling: `1 / √FFT_SIZE` so that a transmitted symbol with unit amplitude
/// is recovered with unit amplitude on a flat unit-gain channel.
pub fn ofdm_demodulate(fft_window: &[Complex32]) -> Vec<Complex32> {
    debug_assert_eq!(fft_window.len(), FFT_SIZE);
    let mut buf = fft_window.to_vec();
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_forward(FFT_SIZE).process(&mut buf);
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();
    (0..NUM_CARRIERS)
        .map(|k| buf[carrier_to_bin(k)] * scale)
        .collect()
}
