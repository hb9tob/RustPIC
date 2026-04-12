//! Pilot-based frame synchronisation (replaces ZC correlator).
//!
//! Uses two mechanisms:
//! 1. **CP correlation** for coarse symbol timing: the cyclic prefix is a copy
//!    of the last CP_LEN samples of the FFT window → correlating with the tail
//!    gives a peak at the correct symbol boundary.
//! 2. **Frequency pilot detection** for frame start: once symbol timing is
//!    found, FFT each symbol and check if the scattered pilots match the DRM
//!    Mode B pattern. A match confirms this is a RustPIC frame.

use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::ofdm::params::*;
use crate::ofdm::drm_pilots::{drm_pilot_value, drm_pilot_indices};

/// Result of pilot-based sync.
#[derive(Debug, Clone)]
pub struct PilotSyncResult {
    /// Sample index of the first data symbol in the input buffer.
    pub data_start: usize,
    /// Per-carrier channel estimate from the detected symbol's pilots.
    pub channel_est: Vec<Complex32>,
    /// Pilot correlation metric (0..1).
    pub metric: f32,
}

/// Scans `samples` for a valid RustPIC frame using CP correlation + pilot
/// detection.
///
/// Returns the position of the first data symbol and a channel estimate
/// derived from the detected pilots.
pub fn find_frame_by_pilots(
    samples: &[Complex32],
    threshold: f32,
) -> Option<PilotSyncResult> {
    if samples.len() < 3 * SYMBOL_LEN {
        return None;
    }

    // ── Stage 1: CP correlation for coarse symbol timing ─────────────────
    // For each candidate position d, compute:
    //   C(d) = |Σ_{n=0}^{CP_LEN-1} r[d+n] · conj(r[d+n+FFT_SIZE])| / energy
    // Peak at d = symbol start (where CP matches FFT tail).
    let search_end = samples.len() - SYMBOL_LEN;
    let step = SYMBOL_LEN / 4; // coarse search step

    let mut best_cp_pos = 0;
    let mut best_cp_metric = 0.0_f32;

    let mut d = 0;
    while d < search_end {
        let m = cp_correlation(samples, d);
        if m > best_cp_metric {
            best_cp_metric = m;
            best_cp_pos = d;
        }
        d += step;
    }

    // Refine to single-sample around the coarse peak
    let refine_lo = best_cp_pos.saturating_sub(step);
    let refine_hi = (best_cp_pos + step).min(search_end);
    for d in refine_lo..refine_hi {
        let m = cp_correlation(samples, d);
        if m > best_cp_metric {
            best_cp_metric = m;
            best_cp_pos = d;
        }
    }

    if best_cp_metric < 0.3 {
        return None;
    }

    // ── Stage 2: Pilot detection — find which sym_idx matches ────────────
    // Try a few symbol positions around the CP peak and check the scattered
    // pilot pattern for each candidate sym_idx.
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();

    let mut best_result: Option<PilotSyncResult> = None;
    let mut best_pilot_metric = 0.0_f32;

    // Search a few symbols before and after the CP peak
    for sym_offset in 0..10_usize {
        let pos = if best_cp_pos >= sym_offset * SYMBOL_LEN {
            best_cp_pos - sym_offset * SYMBOL_LEN
        } else {
            continue;
        };

        if pos + SYMBOL_LEN > samples.len() { continue; }

        // FFT the candidate symbol (strip CP)
        let fft_in = &samples[pos + CP_LEN..pos + CP_LEN + FFT_SIZE];
        let mut buf = fft_in.to_vec();
        fft.process(&mut buf);

        // Try each possible sym_idx (0..SYMBOLS_PER_FRAME)
        for try_sym in 0..SYMBOLS_PER_FRAME {
            let pilot_pos = drm_pilot_indices(try_sym);
            if pilot_pos.is_empty() { continue; }

            // Compute pilot correlation: how well do the received pilots
            // match the expected DRM pilot values?
            let mut sum = Complex32::new(0.0, 0.0);
            let mut count = 0;
            for &k in &pilot_pos {
                let y = buf[carrier_to_bin(k)] * scale;
                let expected = drm_pilot_value(k, try_sym);
                // Normalized correlation
                if y.norm() > 1e-6 && expected.norm() > 1e-6 {
                    sum += (y / y.norm()) * (expected / expected.norm()).conj();
                    count += 1;
                }
            }
            let metric = if count > 0 { sum.norm() / count as f32 } else { 0.0 };

            if metric > best_pilot_metric && metric >= threshold {
                best_pilot_metric = metric;

                // Channel estimate from this symbol's pilots
                let channel_est: Vec<Complex32> = (0..NUM_CARRIERS)
                    .map(|k| {
                        let y = buf[carrier_to_bin(k)] * scale;
                        let expected = drm_pilot_value(k, try_sym);
                        if expected.norm_sqr() > 1e-10 {
                            y * expected.conj() / expected.norm_sqr()
                        } else {
                            y
                        }
                    })
                    .collect();

                best_result = Some(PilotSyncResult {
                    data_start: pos,
                    channel_est,
                    metric,
                });
            }
        }
    }

    best_result
}

/// CP correlation metric at position `d`.
fn cp_correlation(samples: &[Complex32], d: usize) -> f32 {
    if d + SYMBOL_LEN > samples.len() { return 0.0; }

    let mut corr = Complex32::new(0.0, 0.0);
    let mut energy = 0.0_f32;

    for n in 0..CP_LEN {
        let a = samples[d + n];
        let b = samples[d + n + FFT_SIZE];
        corr += a * b.conj();
        energy += a.norm_sqr() + b.norm_sqr();
    }

    if energy > 1e-10 {
        2.0 * corr.norm() / energy
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::tx::ofdm_modulate_scattered;

    #[test]
    fn detect_scattered_symbol() {
        // Build a scattered-pilot OFDM symbol
        let n_data = crate::ofdm::drm_pilots::drm_num_data(0);
        let data = vec![Complex32::new(1.0, 0.0); n_data];
        let sym = ofdm_modulate_scattered(&data, 0);

        // Convert to real (like TX pipeline)
        let samples: Vec<Complex32> = sym.iter()
            .map(|s| Complex32::new(s.re, 0.0))
            .collect();

        // CP correlation should be high at position 0
        let m = cp_correlation(&samples, 0);
        assert!(m > 0.5, "CP correlation should be high, got {m:.3}");
    }
}
