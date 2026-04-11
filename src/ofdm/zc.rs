//! Zadoff–Chu (ZC) sequence generation.
//!
//! ZC sequences are CAZAC (Constant Amplitude Zero Auto-Correlation) sequences.
//! They are used as the frame preamble and for periodic re-synchronisation.
//!
//! ## Frequency-domain placement
//!
//! The ZC sequence of length [`ZC_LEN`] = 72 is placed on the 72 active
//! subcarriers (FFT bins [`FIRST_BIN`] … [`LAST_BIN`]).  All other bins are
//! zero.  After an IFFT and prepending a cyclic prefix the result is one
//! complete OFDM symbol of length [`SYMBOL_LEN`] = 288 samples.
//!
//! ## Formula
//!
//! ```text
//! x_u(n) = exp(−j π u n (n + cf) / N_zc)
//! ```
//! with cf = N_zc mod 2 (0 for even length, 1 for odd).

use num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

use crate::ofdm::params::*;

// ── ZC sequence generation ────────────────────────────────────────────────────

/// Generates a Zadoff–Chu sequence of `length` elements with root `u`.
///
/// The sequence has unit amplitude on every element and ideal periodic
/// auto-correlation (zero sidelobes) for prime `length`.
pub fn generate_zc(root: u32, length: usize) -> Vec<Complex32> {
    let n_f = length as f32;
    let u_f = root as f32;
    let cf  = (length % 2) as f32; // 0 for even, 1 for odd
    (0..length)
        .map(|n| {
            let nf = n as f32;
            let phase = -PI * u_f * nf * (nf + cf) / n_f;
            Complex32::new(phase.cos(), phase.sin())
        })
        .collect()
}

// ── Preamble construction ─────────────────────────────────────────────────────

/// Builds the time-domain ZC preamble OFDM symbol **including** the cyclic
/// prefix (total length [`SYMBOL_LEN`]).
///
/// Steps:
/// 1. Generate ZC of length [`ZC_LEN`] in the frequency domain.
/// 2. Zero-pad to [`FFT_SIZE`] and place on active subcarriers.
/// 3. IFFT → time domain.
/// 4. Prepend CP (last [`CP_LEN`] samples of the FFT window).
/// 5. Normalise to unit average power.
pub fn build_preamble() -> Vec<Complex32> {
    let zc = generate_zc(ZC_ROOT, ZC_LEN);
    zc_freq_to_time_domain(&zc)
}

/// Converts a frequency-domain ZC vector (length = [`NUM_CARRIERS`]) into a
/// time-domain OFDM symbol with cyclic prefix.
///
/// This is the low-level building block used by both the TX modulator and the
/// RX channel estimator reference.
pub(crate) fn zc_freq_to_time_domain(zc: &[Complex32]) -> Vec<Complex32> {
    assert_eq!(
        zc.len(), NUM_CARRIERS,
        "ZC freq vector must have length NUM_CARRIERS={NUM_CARRIERS}"
    );

    // Build FFT input: place ZC on active subcarriers.
    let mut freq = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    for (k, &s) in zc.iter().enumerate() {
        freq[carrier_to_bin(k)] = s;
    }

    // Inverse FFT (rustfft computes unnormalised IDFT: divide by √N to keep
    // unit average power across the active subcarriers).
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(FFT_SIZE);
    ifft.process(&mut freq); // freq is now the time-domain signal
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();
    let time: Vec<Complex32> = freq.iter().map(|&s| s * scale).collect();

    // Prepend cyclic prefix: the last CP_LEN samples of the FFT window.
    let mut sym = Vec::with_capacity(SYMBOL_LEN);
    sym.extend_from_slice(&time[FFT_SIZE - CP_LEN..]);
    sym.extend_from_slice(&time);
    sym
}

// ── Frequency-domain reference ────────────────────────────────────────────────

/// Returns the known transmitted ZC values in the **frequency domain**,
/// one `Complex32` per active subcarrier (length = [`NUM_CARRIERS`]).
///
/// Used by the channel estimator:
/// ```text
/// H_hat[k] = Y_rx[k] / X_zc[k]
/// ```
pub fn zc_freq_reference() -> Vec<Complex32> {
    generate_zc(ZC_ROOT, ZC_LEN)
}

// ── Conjugate root (for cross-correlation kernel) ─────────────────────────────

/// Returns the time-domain ZC preamble with **conjugated** samples — the
/// matched-filter kernel used by the [`ZcCorrelator`].
///
/// [`ZcCorrelator`]: crate::ofdm::rx::sync::ZcCorrelator
#[allow(dead_code)]
pub(crate) fn build_preamble_conj() -> Vec<Complex32> {
    build_preamble().into_iter().map(|s| s.conj()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn zc_unit_amplitude() {
        let zc = generate_zc(ZC_ROOT, ZC_LEN);
        for (n, s) in zc.iter().enumerate() {
            assert!(
                (s.norm() - 1.0).abs() < 1e-6,
                "ZC sample {n} has amplitude {}", s.norm()
            );
        }
    }

    #[test]
    fn preamble_length() {
        let p = build_preamble();
        assert_eq!(p.len(), SYMBOL_LEN);
    }

    #[test]
    fn preamble_autocorr_peak_at_zero() {
        // |corr(p, p)[0]|² must be the maximum among all lags.
        let p = build_preamble();
        let n = p.len();

        // Zero-lag
        let peak: f32 = p.iter().map(|s| s.norm_sqr()).sum();

        // A few non-zero lags should be strictly smaller
        for lag in 1..=5 {
            let side: Complex32 = p[lag..]
                .iter()
                .zip(p[..n - lag].iter())
                .map(|(&a, &b)| a * b.conj())
                .sum();
            assert!(
                side.norm() < peak,
                "auto-correlation sidelobe at lag {lag} ({:.3}) ≥ peak ({peak:.3})",
                side.norm()
            );
        }
    }
}
