//! Pilot-based frame synchronisation (replaces ZC correlator).
//!
//! Inspired by QSSTV's DRM demodulator — uses two mechanisms:
//!
//! 1. **CP correlation** for coarse symbol timing: the cyclic prefix is a copy
//!    of the last CP_LEN samples of the FFT window → correlating with the tail
//!    gives a peak at the correct symbol boundary.
//!
//! 2. **Scattered pilot detection** for frame start: once symbol timing is
//!    found, FFT each candidate and check if the DRM Mode B scattered pilot
//!    pattern matches.  A match confirms this is a RustPIC frame and tells us
//!    which sym_idx we're at (0..SYMBOLS_PER_FRAME−1).
//!
//! With the sym_idx known, we navigate backward to find the mode header and
//! decode the transmission parameters.
//!
//! Works with real (Im=0) or analytic (Hilbert-filtered) signals.

use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::ofdm::params::*;
use crate::ofdm::drm_pilots::{drm_pilot_value, drm_pilot_indices};

/// Result of pilot-based sync.
#[derive(Debug, Clone)]
pub struct PilotSyncResult {
    /// Sample index (absolute, within the search buffer) of the detected
    /// OFDM symbol start (including CP).
    pub symbol_pos: usize,
    /// The DRM frame symbol index that was detected (0..SYMBOLS_PER_FRAME−1).
    pub sym_idx: usize,
    /// Per-carrier channel estimate H[k] derived from the detected symbol's
    /// scattered pilots (interpolated across all carriers).
    pub channel_est: Vec<Complex32>,
    /// Pilot correlation metric (0..1, higher = better match).
    pub metric: f32,
}

/// Scans `samples` for OFDM symbols carrying valid DRM scattered pilots.
///
/// Searches from `start` with a sliding CP correlation, then verifies
/// pilot patterns.  Returns the position and sym_idx of the first match.
///
/// # Arguments
/// * `samples` — full audio buffer (Complex32, from Hilbert filter or Im=0)
/// * `start`   — sample offset to begin scanning
/// * `cp_threshold` — minimum CP correlation metric (0.3 is a good default)
/// * `pilot_threshold` — minimum pilot correlation metric (0.5 is a good default)
pub fn scan_for_pilots(
    samples: &[Complex32],
    start: usize,
    cp_threshold: f32,
    pilot_threshold: f32,
) -> Option<PilotSyncResult> {
    let min_len = SYMBOL_LEN + FFT_SIZE; // need room for CP + FFT window
    if start + min_len > samples.len() {
        return None;
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();

    // ── Stage 1: find CP correlation peaks ──────────────────────────────
    // Coarse scan: step by SYMBOL_LEN/4 to find approximate symbol boundaries.
    let search_end = samples.len().saturating_sub(SYMBOL_LEN);
    let coarse_step = SYMBOL_LEN / 4;

    let mut d = start;
    while d < search_end {
        let m = cp_correlation(samples, d);
        if m >= cp_threshold {
            // Refine to single-sample precision around this peak
            let refine_lo = d.saturating_sub(coarse_step);
            let refine_hi = (d + coarse_step).min(search_end);

            let mut best_pos = d;
            let mut best_metric = m;
            for r in refine_lo..refine_hi {
                let rm = cp_correlation(samples, r);
                if rm > best_metric {
                    best_metric = rm;
                    best_pos = r;
                }
            }

            // ── Stage 2: pilot detection at this symbol ─────────────────
            if best_pos + SYMBOL_LEN <= samples.len() {
                if let Some(result) = try_pilot_match(
                    samples, best_pos, &fft, scale, pilot_threshold,
                ) {
                    return Some(result);
                }
            }
        }
        d += coarse_step;
    }

    None
}

/// Tries to match scattered pilots at a candidate symbol position.
///
/// Tests all `sym_idx ∈ 0..SYMBOLS_PER_FRAME` and returns the best match
/// if above threshold.
fn try_pilot_match(
    samples: &[Complex32],
    pos: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    scale: f32,
    threshold: f32,
) -> Option<PilotSyncResult> {
    if pos + SYMBOL_LEN > samples.len() {
        return None;
    }

    // FFT the candidate symbol (strip CP)
    let fft_in = &samples[pos + CP_LEN..pos + CP_LEN + FFT_SIZE];
    let mut buf = fft_in.to_vec();
    fft.process(&mut buf);

    let mut best_metric = 0.0_f32;
    let mut best_sym_idx = 0usize;

    for try_sym in 0..SYMBOLS_PER_FRAME {
        let pilot_pos = drm_pilot_indices(try_sym);
        if pilot_pos.is_empty() {
            continue;
        }

        // Normalised phase correlation between received and expected pilots
        let mut sum = Complex32::new(0.0, 0.0);
        let mut count = 0;
        for &k in &pilot_pos {
            let y = buf[carrier_to_bin(k)] * scale;
            let expected = drm_pilot_value(k, try_sym);
            if y.norm() > 1e-6 && expected.norm() > 1e-6 {
                sum += (y / y.norm()) * (expected / expected.norm()).conj();
                count += 1;
            }
        }
        let metric = if count > 0 { sum.norm() / count as f32 } else { 0.0 };

        if metric > best_metric {
            best_metric = metric;
            best_sym_idx = try_sym;
        }
    }

    if best_metric < threshold {
        return None;
    }

    // ── Channel estimate from the detected pilots ───────────────────────
    let pilot_pos = drm_pilot_indices(best_sym_idx);
    let mut h_pilots: Vec<(usize, Complex32)> = Vec::new();

    for &k in &pilot_pos {
        let y = buf[carrier_to_bin(k)] * scale;
        let expected = drm_pilot_value(k, best_sym_idx);
        if expected.norm_sqr() > 1e-10 {
            h_pilots.push((k, y / expected));
        }
    }

    // Linear interpolation across all carriers
    let channel_est = interpolate_channel(&h_pilots, NUM_CARRIERS);

    Some(PilotSyncResult {
        symbol_pos: pos,
        sym_idx: best_sym_idx,
        channel_est,
        metric: best_metric,
    })
}

/// Linearly interpolates a sparse set of pilot channel estimates across
/// all `n_carriers` active carriers.
fn interpolate_channel(pilots: &[(usize, Complex32)], n_carriers: usize) -> Vec<Complex32> {
    if pilots.is_empty() {
        return vec![Complex32::new(1.0, 0.0); n_carriers];
    }

    let mut h = vec![Complex32::new(0.0, 0.0); n_carriers];

    for k in 0..n_carriers {
        // Find the two nearest pilots bracketing carrier k
        let mut lo: Option<(usize, Complex32)> = None;
        let mut hi: Option<(usize, Complex32)> = None;
        for &(pk, ph) in pilots {
            if pk <= k {
                lo = Some((pk, ph));
            }
            if pk >= k && hi.is_none() {
                hi = Some((pk, ph));
            }
        }

        h[k] = match (lo, hi) {
            (Some((lk, lh)), Some((hk, hh))) if lk != hk => {
                // Linear interpolation
                let t = (k - lk) as f32 / (hk - lk) as f32;
                lh * (1.0 - t) + hh * t
            }
            (Some((_, ph)), _) => ph,
            (_, Some((_, ph))) => ph,
            (None, None) => Complex32::new(1.0, 0.0),
        };
    }

    h
}

/// CP correlation metric at position `d`.
///
/// Exploits the cyclic prefix: the first CP_LEN samples of an OFDM symbol
/// are a copy of the last CP_LEN samples of the FFT window.
///
/// Returns a normalised metric in [0, 1].  Values > 0.5 strongly indicate
/// a valid OFDM symbol boundary.
pub fn cp_correlation(samples: &[Complex32], d: usize) -> f32 {
    if d + SYMBOL_LEN > samples.len() {
        return 0.0;
    }

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

/// Given a detected sym_idx and the transmission preamble structure,
/// compute the absolute sample position of the mode header start.
///
/// `preamble_syms` = number of non-data symbols before the first data symbol
/// (currently 2 ZC + MODE_HEADER_REPEAT = 5).
///
/// Returns None if the detected symbol is too close to the start of the
/// buffer (mode header would be before the audio).
pub fn mode_header_pos_from_pilot(
    detected_pos: usize,
    detected_sym_idx: usize,
    preamble_syms: usize,
    header_offset_syms: usize, // how many symbols before mode header (2 for ZC#1+ZC#2)
) -> Option<usize> {
    // The detected symbol's absolute position in the frame is detected_sym_idx.
    // Data symbols start at frame position preamble_syms.
    // Mode header starts at frame position header_offset_syms.
    if detected_sym_idx < preamble_syms {
        // Detected a non-data symbol — shouldn't happen with pilot matching
        return None;
    }

    let syms_back = detected_sym_idx - header_offset_syms;
    let bytes_back = syms_back * SYMBOL_LEN;
    detected_pos.checked_sub(bytes_back)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::tx::ofdm_modulate_scattered;
    use crate::ofdm::rx::hilbert::HilbertFilter;

    #[test]
    fn cp_correlation_on_real_signal() {
        let n_data = crate::ofdm::drm_pilots::drm_num_data(0);
        let data = vec![Complex32::new(1.0, 0.0); n_data];
        let sym = ofdm_modulate_scattered(&data, 0);

        // Convert to real (like TX pipeline) then back to complex
        let samples: Vec<Complex32> = sym.iter()
            .map(|s| Complex32::new(s.re, 0.0))
            .collect();

        let m = cp_correlation(&samples, 0);
        assert!(m > 0.5, "CP correlation should be high on real signal, got {m:.3}");
    }

    #[test]
    fn cp_correlation_on_analytic_signal() {
        let n_data = crate::ofdm::drm_pilots::drm_num_data(0);
        let data = vec![Complex32::new(1.0, 0.0); n_data];
        let sym = ofdm_modulate_scattered(&data, 0);

        // Convert to real then through Hilbert filter
        let real: Vec<f32> = sym.iter().map(|s| s.re).collect();
        let mut hf = HilbertFilter::new();
        // Pad with silence for filter warm-up
        let mut padded = vec![0.0_f32; 200];
        padded.extend_from_slice(&real);
        padded.extend(vec![0.0_f32; 200]);
        let analytic = hf.process(&padded);

        // CP correlation should work on analytic signal too
        // Skip the initial transient (200 + ~76 delay samples)
        let offset = 200 + hf.delay();
        let m = cp_correlation(&analytic, offset);
        assert!(m > 0.3, "CP correlation should work on analytic signal, got {m:.3}");
    }

    #[test]
    fn pilot_detection_on_analytic_signal() {
        // Build 3 consecutive scattered-pilot symbols (sym_idx 5, 6, 7)
        let preamble_syms = 5;
        let mut real_samples = vec![0.0_f32; 300]; // silence padding

        for sym_idx in 0..3 {
            let abs_idx = preamble_syms + sym_idx;
            let n_data = crate::ofdm::drm_pilots::drm_num_data(abs_idx);
            let data = vec![Complex32::new(1.0, 0.0); n_data];
            let sym = ofdm_modulate_scattered(&data, abs_idx);
            real_samples.extend(sym.iter().map(|s| s.re));
        }
        real_samples.extend(vec![0.0_f32; 300]);

        // Hilbert filter
        let mut hf = HilbertFilter::new();
        let analytic = hf.process(&real_samples);

        // Scan for pilots
        let result = scan_for_pilots(&analytic, 0, 0.3, 0.4);
        assert!(result.is_some(), "should detect pilot-bearing symbols");

        let r = result.unwrap();
        assert!(r.metric > 0.4, "pilot metric should be reasonable: {:.3}", r.metric);
        // The detected sym_idx should be one of [5, 6, 7] (mod SYMBOLS_PER_FRAME)
        let expected: Vec<usize> = (0..3).map(|i| (preamble_syms + i) % SYMBOLS_PER_FRAME).collect();
        assert!(
            expected.contains(&r.sym_idx),
            "detected sym_idx {} should be in {:?}", r.sym_idx, expected
        );
    }
}
