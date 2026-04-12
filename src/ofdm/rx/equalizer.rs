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
use crate::ofdm::drm_pilots::{is_drm_pilot, drm_pilot_value, drm_pilot_indices, drm_num_data};
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
    /// Reference channel estimate from the most recent ZC preamble / resync.
    /// Holds the full per-carrier shape including any sharp features (HPF
    /// phase, magnitude dips) that a sparse-pilot linear interpolator could
    /// not reconstruct from pilots alone. Only rewritten by `resync_from_zc`.
    h_ref: Vec<Complex32>,
    /// EMA-smoothed multiplicative correction factor, one value per pilot.
    /// Nominally 1 + 0j after a fresh ZC — tracks slow drift (CPE, soundcard
    /// clock mismatch, oscillator wander) between the ZC and the current
    /// symbol. Pilots measure `correction = y[pilot]/expected / h_ref[pilot]`
    /// which is nearly uniform across the band, so *linear interpolation of
    /// the correction* is accurate even when `h_ref` itself has > 100° of
    /// phase rotation between adjacent pilots.
    pilot_corr: Vec<Complex32>,
    /// EMA coefficient for the pilot correction factor (0 < α ≤ 1).
    alpha: f32,
    /// Estimated per-data-symbol timing drift in samples (positive = late read).
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
            h_ref: h_preamble.to_vec(),
            pilot_corr: vec![Complex32::new(1.0, 0.0); NUM_PILOTS],
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
    /// Fully replaces the reference channel (no EMA blending), resets the
    /// pilot correction factor to unity, and zeroes the intra-group symbol
    /// counter used for drift pre-correction.
    pub fn resync_from_zc(&mut self, ofdm_symbol: &[Complex32]) {
        assert_eq!(ofdm_symbol.len(), SYMBOL_LEN);
        let fft_window = &ofdm_symbol[CP_LEN..];
        self.h_ref = channel_estimate_from_zc(fft_window);
        self.pilot_corr.fill(Complex32::new(1.0, 0.0));
        self.syms_since_resync = 0;
    }

    // ── Main processing ───────────────────────────────────────────────────────

    /// Equalises one data OFDM symbol (CP included).
    ///
    /// # Algorithm — reference-based pilot tracking
    ///
    /// The ZC preamble has already given us a full per-carrier reference
    /// `h_ref[k]`, which captures every sharp feature of the channel
    /// (magnitude dips, fast phase rotations at band edges, etc.) with
    /// unit-subcarrier resolution. Between ZC events the real channel
    /// only *drifts slowly* from that reference — mostly a common phase
    /// rotation from the TX/RX oscillator mismatch.
    ///
    /// Instead of re-deriving `H[k]` from 6 pilots every symbol (which
    /// collapses to a linear interpolation that cannot follow > 100° of
    /// phase in one pilot spacing), we measure the per-pilot correction
    /// factor `c[p] = (y[k]/expected) / h_ref[k]`, EMA-smooth it, linearly
    /// interpolate it across all carriers (linear is accurate because `c[k]`
    /// is nearly uniform) and apply it multiplicatively to the ZC reference:
    ///
    /// ```text
    ///   h_eff[k] = h_ref[k] × linear_interp(c[pilots])[k]
    /// ```
    ///
    /// # Steps
    /// 1. Strip CP, FFT, scale → Y[k].
    /// 2. Pre-rotate for clock drift and common phase error.
    /// 3. At each pilot: compute raw correction factor, EMA-smooth it.
    /// 4. Interpolate the correction factor to all carriers (linear).
    /// 5. `h_eff[k] = h_ref[k] * corr[k]`, ZF-equalize data subcarriers.
    pub fn process(&mut self, ofdm_symbol: &[Complex32]) -> EqualizedSymbol {
        assert_eq!(ofdm_symbol.len(), SYMBOL_LEN);

        // ── 1. Demodulate ────────────────────────────────────────────────────
        let fft_window = &ofdm_symbol[CP_LEN..];
        let mut y = ofdm_demodulate(fft_window); // NUM_CARRIERS complex values

        // ── 1b. Clock-drift phase pre-correction ─────────────────────────────
        let drift_samples = (self.syms_since_resync as f32 + 1.0) * self.timing_drift_per_sym;
        if drift_samples.abs() > 1e-9 {
            use std::f32::consts::PI;
            for k in 0..NUM_CARRIERS {
                let phi = 2.0 * PI * drift_samples * carrier_to_bin(k) as f32
                          / FFT_SIZE as f32;
                let (sin_phi, cos_phi) = phi.sin_cos();
                y[k] *= Complex32::new(cos_phi, -sin_phi);
            }
        }
        self.syms_since_resync += 1;

        // ── 1c. Common Phase Error (CPE) removal ─────────────────────────────
        // Estimate and strip the global per-symbol phase rotation so the
        // pilot-correction EMA only sees slow multiplicative drift.  Rotation
        // is estimated against the *current* effective channel (reference ×
        // correction), not against h_ref alone, so the residue is near zero.
        let cpe: f32 = {
            let sum: Complex32 = (0..NUM_PILOTS)
                .map(|p| {
                    let k      = p * PILOT_SPACING;
                    let h_meas = y[k] / Complex32::new(pilot_sign(k), 0.0);
                    let h_exp  = self.h_ref[k] * self.pilot_corr[p];
                    h_meas * h_exp.conj()
                })
                .fold(Complex32::new(0.0, 0.0), |acc, v| acc + v);
            sum.arg()
        };
        if cpe.abs() > 0.1 {
            let corr = Complex32::from_polar(1.0, -cpe);
            for k in 0..NUM_CARRIERS {
                y[k] *= corr;
            }
        }

        // ── 2. Pilot correction measurement and EMA update ───────────────────
        // For each pilot:
        //   h_measured[p] = y[pilot_k] / expected_sign
        //   c_measured[p] = h_measured[p] / h_ref[pilot_k]     (~ 1 + 0j)
        //   EMA: c[p] ← (1−α)·c[p] + α·c_measured[p]
        //
        // In deep fades the ZC reference can have |h_ref| ≈ 0 at one pilot,
        // in which case the division is ill-conditioned — we clamp to the
        // previous EMA value and flag the pilot as weak.
        let mut pilot_err2 = 0.0_f32;

        for p in 0..NUM_PILOTS {
            let k = p * PILOT_SPACING;
            let expected = Complex32::new(pilot_sign(k), 0.0);

            let y_over_exp = y[k] / expected;
            let h_ref_p    = self.h_ref[k];
            let h_ref_sq   = h_ref_p.norm_sqr();

            if h_ref_sq > 1e-10 {
                // c_measured = y_over_exp / h_ref_p = y_over_exp * conj(h_ref_p) / |h_ref_p|²
                let c_meas = y_over_exp * h_ref_p.conj() / h_ref_sq;
                self.pilot_corr[p] = self.pilot_corr[p] * (1.0 - self.alpha)
                                   + c_meas * self.alpha;
            }
            // else: keep previous correction factor — the pilot is in a fade.

            // Pilot residual: after applying the EMA-updated h_eff, how far
            // are we from the nominal expected value?
            let h_eff_p      = self.h_ref[k] * self.pilot_corr[p];
            let equalized    = y[k] * zf(h_eff_p);
            let err          = equalized - expected;
            pilot_err2      += err.norm_sqr();
        }

        let mean_pilot_err2 = pilot_err2 / NUM_PILOTS as f32;
        let pilot_snr_db    = -10.0 * mean_pilot_err2.max(1e-10).log10();

        // ── 3. Linear interpolation of the correction across all carriers ────
        // `pilot_corr` is smooth (nominally all values near 1+0j) so linear
        // interpolation in Cartesian coordinates is accurate even though
        // `h_ref` itself may have 100°+ phase jumps between pilots.
        let corr_interp = interpolate(&self.pilot_corr);

        // ── 4. Effective channel and ZF equalisation ─────────────────────────
        // h_eff[k] = h_ref[k] × corr[k]
        //
        // Average signal power over pilots (used to scale the noise variance
        // so the LLRs stay in a reasonable range after the ±20 clipping in
        // the demapper).
        let mean_h2: f32 = (0..NUM_PILOTS)
            .map(|p| {
                let k     = p * PILOT_SPACING;
                let h_eff = self.h_ref[k] * self.pilot_corr[p];
                h_eff.norm_sqr()
            })
            .sum::<f32>() / NUM_PILOTS as f32;
        let sigma2_channel = mean_pilot_err2 * mean_h2;

        let mut data      = Vec::with_capacity(NUM_DATA);
        let mut noise_var = Vec::with_capacity(NUM_DATA);

        for k in 0..NUM_CARRIERS {
            if is_pilot(k) {
                continue;
            }
            let h_eff = self.h_ref[k] * corr_interp[k];
            let h_sq  = h_eff.norm_sqr();

            data.push(y[k] * zf(h_eff));
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

// ══════════════════════════════════════════════════════════════════════════════
// ScatteredEqualizer — DRM Mode B scattered-pilot 2D channel estimation
// ══════════════════════════════════════════════════════════════════════════════

/// Channel equaliser using DRM-style scattered pilots.
///
/// Instead of fixed pilot positions, each OFDM symbol has pilots at positions
/// determined by `is_drm_pilot(k, sym_idx)`. The pilot positions rotate every
/// `SCAT_TIME_INT = 3` symbols, covering ALL carriers in one pilot cycle.
///
/// Channel estimation is 2D: frequency interpolation within each symbol +
/// temporal EMA smoothing across symbols. Because pilots are at every 2nd
/// carrier (SCAT_FREQ_INT = 2), the frequency interpolation gap is only 1
/// carrier — dramatically more accurate than the old 8-carrier fixed spacing.
pub struct ScatteredEqualizer {
    /// Full per-carrier channel estimate H[k], EMA-smoothed across symbols.
    h: Vec<Complex32>,
    /// EMA coefficient (0 < α ≤ 1). Higher = faster tracking, noisier.
    alpha: f32,
    /// Timing drift in samples per symbol (from resync measurements).
    timing_drift_per_sym: f32,
    /// Symbol counter since last resync / init.
    syms_processed: usize,
}

impl ScatteredEqualizer {
    /// Create from an initial per-carrier channel estimate (e.g. from ZC preamble).
    pub fn from_initial(h_init: &[Complex32], alpha: f32) -> Self {
        assert_eq!(h_init.len(), NUM_CARRIERS);
        Self {
            h: h_init.to_vec(),
            alpha: alpha.clamp(1e-4, 1.0),
            timing_drift_per_sym: 0.0,
            syms_processed: 0,
        }
    }

    /// Create with a flat unit channel (for loopback or when no ZC is available).
    pub fn flat(alpha: f32) -> Self {
        Self::from_initial(&vec![Complex32::new(1.0, 0.0); NUM_CARRIERS], alpha)
    }

    pub fn set_timing_drift_per_sym(&mut self, drift: f32) {
        self.timing_drift_per_sym = drift;
    }

    /// Re-seed the channel estimate from a ZC or known reference.
    pub fn resync(&mut self, h_new: &[Complex32]) {
        assert_eq!(h_new.len(), NUM_CARRIERS);
        self.h = h_new.to_vec();
        self.syms_processed = 0;
    }

    /// Equalise one OFDM symbol with scattered pilots.
    ///
    /// `sym_idx` determines which carriers are pilots (via `is_pilot_at`).
    /// Returns the equalized data carriers (skipping pilot positions) and
    /// per-carrier noise variance estimates.
    pub fn process(&mut self, ofdm_symbol: &[Complex32], sym_idx: usize) -> EqualizedSymbol {
        assert_eq!(ofdm_symbol.len(), SYMBOL_LEN);

        // ── 1. Strip CP, FFT → Y[k] ─────────────────────────────────────────
        let fft_window = &ofdm_symbol[CP_LEN..];
        let mut y = ofdm_demodulate(fft_window);

        // ── 2. Clock-drift phase pre-correction ─────────────────────────────
        let drift = (self.syms_processed as f32 + 1.0) * self.timing_drift_per_sym;
        if drift.abs() > 1e-9 {
            use std::f32::consts::PI;
            for k in 0..NUM_CARRIERS {
                let phi = 2.0 * PI * drift * carrier_to_bin(k) as f32 / FFT_SIZE as f32;
                y[k] *= Complex32::new(phi.cos(), -phi.sin());
            }
        }
        self.syms_processed += 1;

        // ── 3. CPE (Common Phase Error) removal from pilots ─────────────────
        let pilot_positions = drm_pilot_indices(sym_idx);
        let cpe: f32 = {
            let sum: Complex32 = pilot_positions.iter()
                .map(|&k| {
                    let expected = drm_pilot_value(k, sym_idx);
                    let h_meas = y[k] / expected;
                    h_meas * self.h[k].conj()
                })
                .fold(Complex32::new(0.0, 0.0), |a, v| a + v);
            sum.arg()
        };
        if cpe.abs() > 0.05 {
            let corr = Complex32::from_polar(1.0, -cpe);
            for k in 0..NUM_CARRIERS {
                y[k] *= corr;
            }
        }

        // ── 4. Measure H at pilot positions, EMA update ─────────────────────
        // Fast convergence on the first CONVERGENCE_SYMS symbols after init
        // or resync: use alpha=0.8 instead of the configured (slow) alpha.
        // This mirrors QSSTV's approach of repeating the first 20 blocks —
        // we don't repeat, but we track aggressively so the channel estimate
        // is usable within 3-4 symbols instead of 10+.
        const CONVERGENCE_SYMS: usize = 6;
        let ema_alpha = if self.syms_processed < CONVERGENCE_SYMS {
            0.8_f32
        } else {
            self.alpha
        };

        let mut pilot_err2 = 0.0_f32;
        for &k in &pilot_positions {
            let expected = drm_pilot_value(k, sym_idx);
            let h_meas = y[k] / expected;

            // EMA update at this carrier
            self.h[k] = self.h[k] * (1.0 - ema_alpha) + h_meas * ema_alpha;

            // Pilot residual for SNR estimation
            let eq_pilot = y[k] * zf(self.h[k]);
            let err = eq_pilot - expected;
            pilot_err2 += err.norm_sqr();
        }
        let n_pilots = pilot_positions.len() as f32;
        let mean_pilot_err2 = pilot_err2 / n_pilots.max(1.0);
        let pilot_snr_db = -10.0 * mean_pilot_err2.max(1e-10).log10();

        // ── 5. Frequency interpolation for non-pilot carriers ───────────────
        // With SCAT_FREQ_INT=2, every non-pilot carrier is exactly 1 position
        // away from a pilot. Simple linear interpolation between the two
        // nearest pilot-updated H values.
        //
        // Build sorted pilot positions for this symbol, then interpolate.
        let mut h_interp = self.h.clone();
        {
            let mut ppos: Vec<usize> = pilot_positions.clone();
            ppos.sort();

            for k in 0..NUM_CARRIERS {
                if is_drm_pilot(k, sym_idx) { continue; }

                // Find the nearest pilot below and above
                let below = ppos.iter().rev().find(|&&p| p < k).copied();
                let above = ppos.iter().find(|&&p| p > k).copied();

                h_interp[k] = match (below, above) {
                    (Some(lo), Some(hi)) => {
                        let t = (k - lo) as f32 / (hi - lo) as f32;
                        self.h[lo] * (1.0 - t) + self.h[hi] * t
                    }
                    (Some(lo), None) => self.h[lo],
                    (None, Some(hi)) => self.h[hi],
                    (None, None) => Complex32::new(1.0, 0.0),
                };

                // Also update the stored H for non-pilot positions (for
                // the temporal EMA to have a baseline next time this carrier
                // IS a pilot in a future symbol).
                self.h[k] = h_interp[k];
            }
        }

        // ── 6. MMSE equalisation — data subcarriers only ─────────────────
        // MMSE: x̂ = conj(H)·y / (|H|² + σ²_n)
        // Better than ZF at low SNR: avoids noise amplification in faded carriers.
        // σ²_n estimated from pilot residuals (same as QSSTV's MIN_ABS_H floor).
        let sigma2_noise = mean_pilot_err2.max(1e-6);

        let mut data = Vec::with_capacity(drm_num_data(sym_idx));
        let mut noise_var = Vec::with_capacity(drm_num_data(sym_idx));

        for k in 0..NUM_CARRIERS {
            if is_drm_pilot(k, sym_idx) { continue; }

            let h_k = h_interp[k];
            let h_sq = h_k.norm_sqr();
            // MMSE equalization
            let w = h_k.conj() / (h_sq + sigma2_noise);
            data.push(y[k] * w);
            noise_var.push(sigma2_noise / (h_sq + sigma2_noise));
        }

        EqualizedSymbol { data, noise_var, pilot_snr_db }
    }
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
    fn scattered_eq_flat_channel() {
        use crate::ofdm::tx::ofdm_modulate_scattered;

        let target = Complex32::new(0.7, -0.4);
        // Use high alpha for fast convergence from flat initial estimate.
        // On a real signal, H converges from 1.0 to ~0.5 (half-energy from
        // taking Re() of complex OFDM). After ~8 symbols the EMA stabilises.
        let mut eq = ScatteredEqualizer::flat(0.5);

        for sym_idx in 0..12 {
            let n_data = drm_num_data(sym_idx);
            let data_sc = vec![target; n_data];
            let sym = ofdm_modulate_scattered(&data_sc, sym_idx);

            // Take real part only (matches TX WAV pipeline)
            let sym_real: Vec<Complex32> = sym.iter()
                .map(|s| Complex32::new(s.re, 0.0))
                .collect();

            let result = eq.process(&sym_real, sym_idx);
            assert_eq!(result.data.len(), n_data,
                "data len mismatch at sym {sym_idx}");

            // After ~8 symbols the EMA has converged
            if sym_idx >= 8 {
                for (i, &d) in result.data.iter().enumerate() {
                    let err = (d - target).norm();
                    assert!(err < 0.1,
                        "sym {sym_idx} data[{i}] = {d:?}, expected {target:?}, err={err:.4}");
                }
            }
        }
    }

    #[test]
    fn scattered_eq_snr_flat() {
        use crate::ofdm::tx::ofdm_modulate_scattered;

        let mut eq = ScatteredEqualizer::flat(0.3);
        let mut last_snr = 0.0_f32;

        for sym_idx in 0..10 {
            let n_data = drm_num_data(sym_idx);
            let data_sc = vec![Complex32::new(1.0, 0.0); n_data];
            let sym = ofdm_modulate_scattered(&data_sc, sym_idx);
            let sym_real: Vec<Complex32> = sym.iter()
                .map(|s| Complex32::new(s.re, 0.0))
                .collect();
            let result = eq.process(&sym_real, sym_idx);
            last_snr = result.pilot_snr_db;
        }
        assert!(last_snr > 15.0,
            "expected decent SNR on flat channel, got {last_snr:.1} dB");
    }

    // ── Legacy tests below (old fixed-pilot equalizer — not used by ScatteredEqualizer) ──

    #[test]
    #[ignore = "tests legacy fixed-pilot Equalizer, not the active ScatteredEqualizer"]
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
    #[ignore = "tests legacy fixed-pilot Equalizer"]
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
    #[ignore = "tests legacy fixed-pilot interpolation"]
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
    #[ignore = "tests legacy fixed-pilot interpolation"]
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
        // After resync on the same preamble, h_ref should be close to the original
        for k in 0..NUM_CARRIERS {
            assert!(
                (eq.h_ref[k] - h0[k]).norm() < 1e-3,
                "h_ref[{k}] diverged after resync"
            );
        }
    }
}
