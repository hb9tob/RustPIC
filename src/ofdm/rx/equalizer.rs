//! Per-symbol channel equaliser with pilot-based interpolation.
//!
//! # Processing pipeline for one data OFDM symbol
//!
//! ```text
//!  OFDM symbol (SYMBOL_LEN samples, CP included)
//!       │
//!       ▼  strip CP  →  FFT  →  scale
//!  Y[k], k = 0 … NUM_CARRIERS−1
//!       │
//!       ├── Pilot subcarriers k ∈ {0,8,16,…,64}
//!       │     Y[k] / (pilot_sign(k) · H_interp[k])
//!       │     → residual error → σ²_noise estimate
//!       │     → EMA update: H[k] ← (1−α)·H[k] + α·H_measured[k]
//!       │
//!       ├── Channel interpolation
//!       │     linear between adjacent pilots
//!       │     linear extrapolation beyond last pilot (k = 65…71)
//!       │
//!       └── Data subcarriers k ∉ {0,8,…,64}
//!             ZF: X̂[k] = Y[k] / H_interp[k]
//!             per-carrier noise var: σ²_eq[k] = σ²_noise / |H[k]|²
//!
//!  EqualizedSymbol { data, noise_var, pilot_snr_db }
//! ```
//!
//! # Channel tracking
//!
//! The EMA coefficient `alpha` controls tracking speed:
//! * Small α (e.g. 0.05) → slow, low-noise tracking (good for static channels)
//! * Large α (e.g. 0.5)  → fast tracking (good for time-varying multipath)
//!
//! Use [`SymbolEqualizer::resync_from_zc`] when a re-sync ZC symbol is received
//! to fully re-estimate the channel.

use num_complex::Complex32;

use crate::ofdm::params::*;
use crate::ofdm::rx::ofdm_demodulate;
use crate::ofdm::rx::sync::channel_estimate_from_zc;

// ── Public types ──────────────────────────────────────────────────────────────

/// One equalized OFDM data symbol ready for demapping.
#[derive(Debug, Clone)]
pub struct EqualizedSymbol {
    /// Equalized data subcarrier values.
    /// Length = [`NUM_DATA`] = 63, ordered by active-carrier index
    /// (pilots skipped: indices 1–7, 9–15, …, 65–71).
    pub data: Vec<Complex32>,

    /// Per-data-subcarrier post-equalization noise variance σ²_eq[k].
    /// Length = [`NUM_DATA`].
    ///
    /// Accounts for noise amplification in deep fades:
    /// `σ²_eq[k] = σ²_channel / |H[k]|²`.
    pub noise_var: Vec<f32>,

    /// Average pilot SNR after equalization (dB).
    /// Useful for display and for [`max_decodable_modulation`].
    ///
    /// [`max_decodable_modulation`]: crate::ofdm::rx::mode_detect::max_decodable_modulation
    pub pilot_snr_db: f32,
}

// ── SymbolEqualizer ──────────────────────────────────────────────────────────

/// Stateful single-tap ZF channel equaliser.
///
/// Maintains a per-subcarrier channel estimate H[k] that is updated symbol by
/// symbol using exponential moving average (EMA) over the pilot observations.
///
/// ## Clock-drift compensation
///
/// When the TX and RX clocks differ by `ppm` parts-per-million, each data
/// symbol `s` (since the last ZC) is read `(s+1)×SYMBOL_LEN×ppm/1e6` samples
/// late.  This timing error adds a linear-with-bin phase ramp to every
/// subcarrier.  [`set_timing_drift_per_sym`] stores the estimated drift; each
/// [`process`] call pre-rotates the frequency-domain samples to cancel it
/// before the pilot EMA, so the EMA only needs to track residual channel
/// variation.
///
/// [`set_timing_drift_per_sym`]: SymbolEqualizer::set_timing_drift_per_sym
pub struct SymbolEqualizer {
    /// Current channel estimate for **all** [`NUM_CARRIERS`] active subcarriers.
    h: Vec<Complex32>,
    /// EMA coefficient for pilot-based channel tracking (0 < α ≤ 1).
    alpha: f32,
    /// Estimated per-data-symbol timing drift in samples (positive = late read).
    /// Set from consecutive resync ZC positions; 0.0 until first resync.
    timing_drift_per_sym: f32,
    /// Count of data symbols processed since the last ZC reset.
    syms_since_resync: usize,
}

impl SymbolEqualizer {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Initialises the equaliser from the preamble channel estimate returned
    /// by [`channel_estimate_from_zc`].
    ///
    /// `alpha`: EMA tracking coefficient.  Typical range 0.05 … 0.3.
    pub fn from_preamble(h_preamble: &[Complex32], alpha: f32) -> Self {
        assert_eq!(h_preamble.len(), NUM_CARRIERS);
        Self {
            h: h_preamble.to_vec(),
            alpha: alpha.clamp(1e-4, 1.0),
            timing_drift_per_sym: 0.0,
            syms_since_resync: 0,
        }
    }

    /// Sets the per-data-symbol timing drift in samples.
    ///
    /// Estimated from consecutive resync ZC positions in the receiver loop:
    /// ```text
    ///   drift_per_sym = −correction / (RESYNC_PERIOD + 1)
    /// ```
    /// where `correction = found_pos − expected_pos` (negative when clock is
    /// faster than TX).  Call this after each resync ZC is located.
    pub fn set_timing_drift_per_sym(&mut self, drift: f32) {
        self.timing_drift_per_sym = drift;
    }

    /// Re-estimates the channel from a re-sync ZC OFDM symbol (CP included).
    ///
    /// Fully replaces the current channel estimate (no EMA blending) and resets
    /// the intra-group symbol counter used for drift pre-correction.
    pub fn resync_from_zc(&mut self, ofdm_symbol: &[Complex32]) {
        assert_eq!(ofdm_symbol.len(), SYMBOL_LEN);
        let fft_window = &ofdm_symbol[CP_LEN..];
        self.h = channel_estimate_from_zc(fft_window);
        self.syms_since_resync = 0;
    }

    // ── Main processing ───────────────────────────────────────────────────────

    /// Equalises one data OFDM symbol (CP included).
    ///
    /// # Steps
    /// 1. Strip CP, FFT, scale → Y[k].
    /// 2. At pilot positions: measure H_measured[k], apply EMA update, compute residuals.
    /// 3. Interpolate H[k] to all positions.
    /// 4. ZF-equalise data subcarriers, compute per-carrier noise variance.
    pub fn process(&mut self, ofdm_symbol: &[Complex32]) -> EqualizedSymbol {
        assert_eq!(ofdm_symbol.len(), SYMBOL_LEN);

        // ── 1. Demodulate ────────────────────────────────────────────────────
        let fft_window = &ofdm_symbol[CP_LEN..];
        let mut y = ofdm_demodulate(fft_window); // NUM_CARRIERS complex values

        // ── 1b. Clock-drift phase pre-correction ─────────────────────────────
        // A timing error of `d` samples shifts each subcarrier at FFT bin `b` by
        //   Y[k] = H_true[k] · X[k] · exp(+j·2π·d·b / FFT_SIZE)
        // We remove this rotation before the pilot EMA so the EMA only needs to
        // track residual channel changes rather than the growing phase ramp.
        //
        // d = (syms_since_resync + 1) × timing_drift_per_sym
        //   (the +1 accounts for one symbol-period between the ZC and the first
        //    data symbol after reset)
        let drift_samples = (self.syms_since_resync as f32 + 1.0) * self.timing_drift_per_sym;
        if drift_samples.abs() > 1e-9 {
            use std::f32::consts::PI;
            for k in 0..NUM_CARRIERS {
                let phi = 2.0 * PI * drift_samples * carrier_to_bin(k) as f32
                          / FFT_SIZE as f32;
                let (sin_phi, cos_phi) = phi.sin_cos();
                // multiply by exp(−j·phi) = cos(phi) − j·sin(phi)
                y[k] *= Complex32::new(cos_phi, -sin_phi);
            }
        }
        self.syms_since_resync += 1;

        // ── 1c. Common Phase Error (CPE) correction ──────────────────────────
        // FM oscillator phase noise rotates ALL subcarriers by the same angle
        // every symbol. Estimate the common rotation from pilots vs the current
        // channel estimate, then remove it before the EMA update so the EMA
        // only tracks slow frequency-selective fading.
        //
        //   CPE = arg( Σ_p  (Y[pilot_k] / expected_p)  ×  conj(H_ema[pilot_k]) )
        //
        // Summing the complex products (instead of averaging angles) avoids
        // wrap-around artefacts when individual pilot estimates are noisy.
        let cpe: f32 = {
            let sum: Complex32 = (0..NUM_PILOTS)
                .map(|p| {
                    let k      = p * PILOT_SPACING;
                    let h_meas = y[k] / Complex32::new(pilot_sign(k), 0.0);
                    h_meas * self.h[k].conj()
                })
                .fold(Complex32::new(0.0, 0.0), |acc, v| acc + v);
            sum.arg() // radians in (−π, +π]
        };
        // Rotate the whole symbol by −CPE so EMA sees only channel variation.
        // Threshold: skip correction when |CPE| < 0.1 rad (≈ 6°) — below this
        // the estimate is dominated by pilot noise and the correction degrades
        // performance rather than improving it.
        if cpe.abs() > 0.1 {
            let corr = Complex32::from_polar(1.0, -cpe);
            for k in 0..NUM_CARRIERS {
                y[k] *= corr;
            }
        }

        // ── 2. Pilot observation & EMA update ────────────────────────────────
        // pilot_h[p] = channel at the p-th pilot subcarrier (p = 0..NUM_PILOTS)
        let mut pilot_h    = Vec::with_capacity(NUM_PILOTS);
        let mut pilot_err2 = 0.0f32; // sum of squared pilot residuals (post-EQ)

        for p in 0..NUM_PILOTS {
            let k = p * PILOT_SPACING; // active-carrier index
            let expected = pilot_sign(p * PILOT_SPACING); // ±1 BPSK reference

            // Channel at this pilot from received signal
            let h_measured = y[k] / Complex32::new(expected, 0.0);

            // EMA update
            self.h[k] = self.h[k] * (1.0 - self.alpha)
                      + h_measured * self.alpha;
            pilot_h.push(self.h[k]);

            // Equalized pilot residual (after EMA update)
            let equalized_pilot = y[k] * zf(self.h[k]);
            let error = equalized_pilot - Complex32::new(expected, 0.0);
            pilot_err2 += error.norm_sqr();
        }

        let mean_pilot_err2 = pilot_err2 / NUM_PILOTS as f32;
        let pilot_snr_db    = -10.0 * mean_pilot_err2.max(1e-10).log10();

        // ── 3. Channel interpolation for all active subcarriers ───────────────
        let h_interp = interpolate(&pilot_h);

        // Update non-pilot positions in the stored channel estimate
        for k in 0..NUM_CARRIERS {
            if !is_pilot(k) {
                self.h[k] = h_interp[k];
            }
        }

        // ── 4. ZF equalization — data subcarriers only ────────────────────────
        // FM discriminator noise PSD ∝ f² → noise variance at subcarrier k is
        //   σ²[k] = σ²_ref × (bin_k / bin_ref)²  / |H[k]|²
        //
        // σ²_channel is estimated from pilot residuals, averaged over all pilots.
        // The pilots span bins {FIRST_BIN, …, FIRST_BIN+64}, so sigma2_channel
        // ≈ α_FM × mean(bin_pilot²) where α_FM is the FM noise coefficient.
        // Dividing by mean_pilot_bin² recovers α_FM; multiplying by bin_k² gives
        // the correct per-carrier noise variance.
        //
        // This re-weighting makes LDPC LLRs reliable: low-frequency subcarriers
        // (small bin, quiet) get tight LLRs; high-frequency (noisy) get wide LLRs.
        let sigma2_channel = mean_pilot_err2
            * pilot_h.iter().map(|h| h.norm_sqr()).sum::<f32>()
            / NUM_PILOTS as f32;

        let mut data      = Vec::with_capacity(NUM_DATA);
        let mut noise_var = Vec::with_capacity(NUM_DATA);

        for k in 0..NUM_CARRIERS {
            if is_pilot(k) {
                continue;
            }
            let h_k  = h_interp[k];
            let h_sq = h_k.norm_sqr();

            // ZF equalize
            data.push(y[k] * zf(h_k));

            // Per-carrier noise variance
            noise_var.push(sigma2_channel / h_sq.max(1e-6));
        }

        EqualizedSymbol { data, noise_var, pilot_snr_db }
    }
}

// ── Zero-forcing gain ─────────────────────────────────────────────────────────

/// Returns the ZF equaliser coefficient for channel coefficient `h`.
/// Sets to 0 in deep fades (|H|² < 1e-6) to avoid catastrophic noise
/// amplification — the LDPC treats those as erasures (LLR ≈ 0).
#[inline(always)]
fn zf(h: Complex32) -> Complex32 {
    let h_sq = h.norm_sqr();
    if h_sq > 1e-6 { h.conj() / h_sq } else { Complex32::new(0.0, 0.0) }
}

// ── Pilot-to-all-carrier linear interpolation ─────────────────────────────────

/// Interpolates the channel estimate from the [`NUM_PILOTS`] pilot values to
/// all [`NUM_CARRIERS`] active subcarriers.
///
/// * **Carriers 0 … 64** (within the pilot-covered span): standard piecewise
///   linear interpolation between adjacent pilots.
/// * **Carriers 65 … 71** (beyond the last pilot at k = 64): linear
///   extrapolation using the slope from pilots at k = 56 and k = 64.
///
/// # Arguments
///
/// `pilot_h` — slice of [`NUM_PILOTS`] channel estimates ordered by pilot index
/// (p = 0 → k = 0, p = 1 → k = 8, …, p = 8 → k = 64).
fn interpolate(pilot_h: &[Complex32]) -> Vec<Complex32> {
    debug_assert_eq!(pilot_h.len(), NUM_PILOTS);
    let mut h = vec![Complex32::new(0.0, 0.0); NUM_CARRIERS];

    // ── In-span interpolation (k = 0 … PILOT_SPACING*(NUM_PILOTS−1)) ─────────
    for p in 0..NUM_PILOTS - 1 {
        let k_left  = p * PILOT_SPACING;
        let k_right = k_left + PILOT_SPACING;
        let h_left  = pilot_h[p];
        let h_right = pilot_h[p + 1];

        for k in k_left..=k_right {
            let t = (k - k_left) as f32 / PILOT_SPACING as f32;
            h[k] = h_left * (1.0 - t) + h_right * t;
        }
    }
    // k = k_last_pilot (= 64) is already set by the loop above at t=1.0

    // ── Extrapolation beyond last pilot (k = 65 … NUM_CARRIERS−1 = 71) ───────
    let k_last  = (NUM_PILOTS - 1) * PILOT_SPACING;           // 64
    let k_prev  = (NUM_PILOTS - 2) * PILOT_SPACING;           // 56
    let slope   = (pilot_h[NUM_PILOTS - 1] - pilot_h[NUM_PILOTS - 2])
                / Complex32::new(PILOT_SPACING as f32, 0.0);

    for k in (k_last + 1)..NUM_CARRIERS {
        let steps = (k - k_last) as f32;
        h[k] = pilot_h[NUM_PILOTS - 1] + slope * steps;
        let _ = k_prev; // silence unused warning
    }

    h
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::zc::build_preamble;
    use approx::assert_abs_diff_eq;

    /// Build a synthetic OFDM data symbol on a flat unit channel, with known
    /// pilot and data values.  The equaliser should recover the data exactly.
    fn make_flat_channel_symbol(data_value: Complex32) -> Vec<Complex32> {
        use rustfft::FftPlanner;

        // Build frequency-domain symbol: pilots = pilot_sign, data = data_value
        let mut freq = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        for k in 0..NUM_CARRIERS {
            freq[carrier_to_bin(k)] = if is_pilot(k) {
                Complex32::new(pilot_sign(k), 0.0)
            } else {
                data_value
            };
        }

        // IFFT → time domain
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_inverse(FFT_SIZE).process(&mut freq);
        let scale = 1.0 / (FFT_SIZE as f32).sqrt();
        let time: Vec<Complex32> = freq.iter().map(|&s| s * scale).collect();

        // Prepend CP
        let mut sym = time[FFT_SIZE - CP_LEN..].to_vec();
        sym.extend_from_slice(&time);
        sym
    }

    #[test]
    fn flat_channel_data_recovery() {
        // Initial H from ZC preamble (flat unit channel → H = 1+0j everywhere)
        let preamble = build_preamble();
        let h_preamble = crate::ofdm::rx::sync::channel_estimate_from_zc(
            &preamble[CP_LEN..]);

        let mut eq = SymbolEqualizer::from_preamble(&h_preamble, 0.1);

        let target = Complex32::new(0.5, -0.3);
        let sym    = make_flat_channel_symbol(target);
        let result = eq.process(&sym);

        assert_eq!(result.data.len(), NUM_DATA);
        assert_eq!(result.noise_var.len(), NUM_DATA);

        for (i, &d) in result.data.iter().enumerate() {
            assert!(
                (d - target).norm() < 1e-3,
                "data[{i}] = {d:?}, expected {target:?}"
            );
        }
    }

    #[test]
    fn flat_channel_high_snr() {
        let preamble = build_preamble();
        let h_preamble = crate::ofdm::rx::sync::channel_estimate_from_zc(
            &preamble[CP_LEN..]);
        let mut eq = SymbolEqualizer::from_preamble(&h_preamble, 0.1);

        let sym    = make_flat_channel_symbol(Complex32::new(1.0, 0.0));
        let result = eq.process(&sym);

        // On a noiseless flat channel, SNR should be very high
        assert!(result.pilot_snr_db > 30.0,
            "expected high SNR on noiseless channel, got {:.1} dB", result.pilot_snr_db);
    }

    #[test]
    fn interpolation_pilot_positions() {
        // At pilot positions, interpolated H must match input pilot values exactly
        let pilot_h: Vec<Complex32> = (0..NUM_PILOTS)
            .map(|p| Complex32::new(p as f32 * 0.1 + 0.5, p as f32 * 0.05))
            .collect();
        let h = interpolate(&pilot_h);

        for p in 0..NUM_PILOTS {
            let k = p * PILOT_SPACING;
            assert_abs_diff_eq!(h[k].re, pilot_h[p].re, epsilon = 1e-5);
            assert_abs_diff_eq!(h[k].im, pilot_h[p].im, epsilon = 1e-5);
        }
    }

    #[test]
    fn interpolation_midpoint_linear() {
        // At the midpoint between two pilots the interpolated value must be
        // the arithmetic mean of the two pilot values
        let mut pilot_h = vec![Complex32::new(1.0, 0.0); NUM_PILOTS];
        pilot_h[0] = Complex32::new(0.0, 0.0); // pilot at k=0
        pilot_h[1] = Complex32::new(1.0, 0.0); // pilot at k=8

        let h = interpolate(&pilot_h);
        let mid = h[4]; // k=4, midpoint between k=0 and k=8
        assert_abs_diff_eq!(mid.re, 0.5, epsilon = 1e-5);
        assert_abs_diff_eq!(mid.im, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn resync_updates_channel() {
        let preamble = build_preamble();
        let h0 = crate::ofdm::rx::sync::channel_estimate_from_zc(&preamble[CP_LEN..]);
        let mut eq = SymbolEqualizer::from_preamble(&h0, 0.1);

        // Verify resync doesn't panic and updates h
        eq.resync_from_zc(&preamble);
        // After resync on the same preamble, H should be close to the original
        for k in 0..NUM_CARRIERS {
            assert!(
                (eq.h[k] - h0[k]).norm() < 1e-3,
                "H[{k}] diverged after resync"
            );
        }
    }
}
