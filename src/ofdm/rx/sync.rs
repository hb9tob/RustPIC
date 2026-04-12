//! ZC correlator, frame synchroniser, CFO estimator and channel estimator.
//!
//! # Processing pipeline
//!
//! ```text
//!  received baseband
//!       │
//!       ▼
//!  ┌──────────────────────────────────────────────┐
//!  │  ZcCorrelator::find_sync()                   │
//!  │                                              │
//!  │  1. Sliding normalised cross-correlation     │
//!  │     with ZC preamble reference               │
//!  │     → best peak position + metric            │
//!  │                                              │
//!  │  2. Verify second ZC at pos + SYMBOL_LEN     │
//!  │     (confirmation gate)                      │
//!  │                                              │
//!  │  3. CFO estimate from dual ZC preambles      │
//!  │     φ = arg(Σ ZC2[n]·ZC1*[n])               │
//!  │     f_cfo = φ · fs / (2π · T_sym)           │
//!  │                                              │
//!  │  4. Channel estimate H[k] from ZC#1          │
//!  │     H_hat[k] = FFT{ZC1_rx}[k] / X_zc[k]    │
//!  └──────────────────────────────────────────────┘
//!       │
//!       ▼  SyncResult { header_start, cfo_hz, channel_est }
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let corr = ZcCorrelator::new(0.45, 0.35);
//! match corr.find_sync(&baseband_samples) {
//!     Ok(sync) => {
//!         correct_cfo(&mut baseband_samples[sync.preamble_start..], sync.cfo_hz);
//!         // strip CP of header symbol
//!         let hdr_win = &baseband_samples[sync.header_start + CP_LEN
//!                                         ..sync.header_start + SYMBOL_LEN];
//!         let hdr = decode_mode_header(hdr_win, &sync.channel_est)?;
//!     }
//!     Err(e) => eprintln!("sync failed: {e}"),
//! }
//! ```

use num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

use crate::ofdm::params::*;
use crate::ofdm::zc::{build_preamble, zc_freq_reference};

// ── Public types ──────────────────────────────────────────────────────────────

/// Outcome of a successful ZC synchronisation.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Index of the first sample of the **first** ZC preamble in the input
    /// buffer.
    pub preamble_start: usize,

    /// Index of the first sample of the **mode-header** OFDM symbol
    /// (= `preamble_start + 2 × SYMBOL_LEN`).
    pub header_start: usize,

    /// Normalised correlation metric of the first ZC peak ∈ [0, 1].
    /// Typical threshold: 0.45.  Perfect channel → 1.0.
    pub metric: f32,

    /// Normalised metric of the second (confirming) ZC preamble.
    pub confirm_metric: f32,

    /// Estimated carrier frequency offset in Hz.
    /// Positive: received signal is above the nominal TX carrier.
    pub cfo_hz: f32,

    /// Per-subcarrier complex channel estimate `H_hat[k]`, derived from ZC#1.
    /// Length = [`NUM_CARRIERS`].  Index 0 = active subcarrier 0 (FFT bin
    /// [`FIRST_BIN`]).
    pub channel_est: Vec<Complex32>,
}

/// Error returned when frame synchronisation fails.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncError {
    /// The input buffer is shorter than the minimum required length.
    BufferTooShort { min_len: usize, got: usize },

    /// No correlation peak exceeded the detection threshold.
    NoPeakFound { best_metric: f32, threshold: f32 },

    /// First ZC found but the confirmation peak at offset `SYMBOL_LEN` was
    /// too weak.
    ConfirmationFailed { first_metric: f32, second_metric: f32, threshold: f32 },
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooShort { min_len, got } =>
                write!(f, "buffer too short: need {min_len}, got {got}"),
            Self::NoPeakFound { best_metric, threshold } =>
                write!(f, "no ZC peak (best={best_metric:.3} < threshold={threshold:.3})"),
            Self::ConfirmationFailed { first_metric, second_metric, threshold } =>
                write!(f, "ZC confirmation failed \
                    (m1={first_metric:.3}, m2={second_metric:.3} < {threshold:.3})"),
        }
    }
}

impl std::error::Error for SyncError {}

// ── ZcCorrelator ──────────────────────────────────────────────────────────────

/// Sliding Zadoff–Chu matched-filter correlator.
///
/// The correlator computes the normalised cross-correlation
///
/// ```text
///   ρ(d) = |Σₙ r[d+n] · p*[n]| / (‖r[d..d+L]‖ · ‖p‖)
/// ```
///
/// at every sample offset `d`, then locates the global maximum.
/// A second peak at `d + SYMBOL_LEN` is required for confirmation.
pub struct ZcCorrelator {
    /// Complex conjugate of the ZC preamble (matched-filter convention).
    /// Length = [`SYMBOL_LEN`].
    preamble_conj: Vec<Complex32>,

    /// Pre-computed reference energy ‖p‖.
    ref_energy: f32,

    /// ZC frequency-domain reference (unit-amplitude complex values for the
    /// 42 active bins).  Conjugated for fast per-bin division.
    zc_freq_conj: Vec<Complex32>,

    /// Primary detection threshold on ρ.
    threshold: f32,

    /// Confirmation threshold for the second ZC preamble (may be slightly
    /// lower to tolerate channel variation between the two symbols).
    confirm_threshold: f32,
}

impl ZcCorrelator {
    /// Creates a new correlator with given thresholds.
    ///
    /// * `threshold` – primary detection gate (typical: 0.40 … 0.55).
    /// * `confirm_threshold` – second ZC gate (typical: 0.30 … 0.45).
    pub fn new(threshold: f32, confirm_threshold: f32) -> Self {
        // We transmit only the real part of the ZC preamble (audio is real),
        // so we match against the real part only.  This makes the normalised
        // correlation reach 1.0 on a perfect loopback instead of ~0.71, and
        // gives the best sensitivity on a real FM channel.
        let preamble_re: Vec<Complex32> = build_preamble()
            .into_iter()
            .map(|s| Complex32::new(s.re, 0.0))
            .collect();
        let ref_energy    = energy(&preamble_re);
        let preamble_conj = preamble_re.iter().map(|s| s.conj()).collect();
        let zc_freq_conj  = zc_freq_reference().into_iter().map(|s| s.conj()).collect();
        Self { preamble_conj, ref_energy, zc_freq_conj, threshold, confirm_threshold }
    }

    // ── Main entry point ──────────────────────────────────────────────────────

    /// Searches `samples` for a valid ZC frame preamble.
    ///
    /// The search window is `[0 … samples.len() − 3·SYMBOL_LEN]` to ensure
    /// both preamble symbols and the mode-header symbol fit inside the buffer.
    pub fn find_sync(&self, samples: &[Complex32]) -> Result<SyncResult, SyncError> {
        // Minimum: ZC#1 + ZC#2 + mode-header, each SYMBOL_LEN samples.
        let min_len = 3 * SYMBOL_LEN;
        if samples.len() < min_len {
            return Err(SyncError::BufferTooShort { min_len, got: samples.len() });
        }

        // ── Stage 1: global maximum of ρ(d) ──────────────────────────────────
        // Search only up to the position where ZC#2 + header still fit.
        let search_end = samples.len() - 2 * SYMBOL_LEN;
        let (best_pos, best_metric) = self.global_peak(samples, search_end);

        if best_metric < self.threshold {
            return Err(SyncError::NoPeakFound {
                best_metric,
                threshold: self.threshold,
            });
        }

        // ── Stage 2: freq-domain confirmation ─────────────────────────────────
        //
        // ── Stage 2: time-domain confirmation + freq-domain rescue ────────────
        const ZC2_SEARCH_RADIUS: usize = 256;
        let mut best_pos       = best_pos;
        let mut best_metric    = best_metric;
        let mut zc2_start;
        let mut confirm_metric;

        // Time-domain confirm search for ZC#2.
        {
            let nom        = best_pos + SYMBOL_LEN;
            let search_lo  = nom.saturating_sub(ZC2_SEARCH_RADIUS);
            let search_hi  = (nom + ZC2_SEARCH_RADIUS + 1)
                                 .min(samples.len().saturating_sub(SYMBOL_LEN));
            let (best_off, best_m2) = (search_lo..search_hi)
                .map(|d| (d, self.single_metric(&samples[d..d + SYMBOL_LEN])))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((nom, 0.0));
            zc2_start      = best_off;
            confirm_metric = best_m2;
        }

        // Swap-back: always try ZC#1 ← best_pos−SYMBOL_LEN, pick best.
        if best_pos >= SYMBOL_LEN {
            let nom2       = best_pos - SYMBOL_LEN;
            let search_lo2 = nom2.saturating_sub(ZC2_SEARCH_RADIUS);
            let search_hi2 = (nom2 + ZC2_SEARCH_RADIUS + 1)
                                 .min(samples.len().saturating_sub(SYMBOL_LEN));
            let (alt_pos, alt_metric) = (search_lo2..search_hi2)
                .map(|d| (d, self.single_metric(&samples[d..d + SYMBOL_LEN])))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((nom2, 0.0));
            let alt_confirm = best_metric;
            let current_score = best_metric.min(confirm_metric);
            let alt_score     = alt_metric.min(alt_confirm);
            if alt_metric >= self.threshold && alt_score > current_score {
                best_pos       = alt_pos;
                best_metric    = alt_metric;
                zc2_start      = best_pos + SYMBOL_LEN;
                confirm_metric = alt_confirm;
                let nom3 = best_pos + SYMBOL_LEN;
                let lo3  = nom3.saturating_sub(ZC2_SEARCH_RADIUS);
                let hi3  = (nom3 + ZC2_SEARCH_RADIUS + 1)
                             .min(samples.len().saturating_sub(SYMBOL_LEN));
                if let Some((zc2s, m2)) = (lo3..hi3)
                    .map(|d| (d, self.single_metric(&samples[d..d + SYMBOL_LEN])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                {
                    zc2_start      = zc2s;
                    confirm_metric = m2;
                }
            }
        }

        // ── Freq-domain rescue ───────────────────────────────────────────────
        // When time-domain confirm is mediocre (below the soft-sync threshold
        // but above the hard floor), the frequency-domain differential-phase
        // coherence metric gets a chance.  It's insensitive to the per-carrier
        // amplitude distortion that degrades the time-domain correlation on
        // NBFM channels (pre-emphasis, HPF rolloff, AGC).  If the freq-domain
        // says the ZC is real, we upgrade the metrics.
        if confirm_metric < SOFT_SYNC_M1_MIN && best_metric >= self.threshold {
            let fm1 = self.freq_metric(&samples[best_pos..best_pos + SYMBOL_LEN]);
            if fm1 >= 0.75 {
                best_metric = best_metric.max(fm1);
                let fm2 = self.freq_metric(&samples[zc2_start..zc2_start + SYMBOL_LEN]);
                if fm2 >= 0.50 {
                    confirm_metric = confirm_metric.max(fm2);
                }
            }
        }

        // Soft-sync fallback: if ZC#2 is too weak to confirm but ZC#1 is very
        // strong (≥ 0.65, well above the nominal 0.35 primary threshold), accept
        // the detection anyway.  This handles the real-world case where a short
        // FM click or a narrowband interferer wipes out a single symbol — the
        // first ZC survives, the second does not.  CFO estimation is skipped
        // (set to 0) because it needs a clean ZC#2, but the clock-tracking
        // path and the re-sync ZCs recover timing for subsequent symbols.
        const SOFT_SYNC_M1_MIN: f32 = 0.65;
        let soft_sync = confirm_metric < self.confirm_threshold
            && best_metric >= SOFT_SYNC_M1_MIN;

        if confirm_metric < self.confirm_threshold && !soft_sync {
            return Err(SyncError::ConfirmationFailed {
                first_metric: best_metric,
                second_metric: confirm_metric,
                threshold: self.confirm_threshold,
            });
        }

        // ── Stage 3: CFO estimation from the two ZC preambles ─────────────────
        // Skipped when the sync is the soft-fallback case — we have no reliable
        // ZC#2 to measure the per-symbol phase rotation against.
        let zc1 = &samples[best_pos..best_pos + SYMBOL_LEN];
        let cfo_hz = if soft_sync {
            0.0
        } else {
            let zc2 = &samples[zc2_start..zc2_start + SYMBOL_LEN];
            estimate_cfo(zc1, zc2)
        };

        // ── Stage 4: channel estimation from ZC#1 ─────────────────────────────
        // Strip the CP: the FFT window starts at CP_LEN.
        let zc1_fft = &zc1[CP_LEN..]; // FFT_SIZE samples
        let channel_est = channel_estimate_from_zc(zc1_fft);

        Ok(SyncResult {
            preamble_start: best_pos,
            header_start: best_pos + 2 * SYMBOL_LEN,
            metric: best_metric,
            confirm_metric,
            cfo_hz,
            channel_est,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Returns `(position, metric)` of the global correlation peak in
    /// `samples[0..search_end]`.
    fn global_peak(&self, samples: &[Complex32], search_end: usize) -> (usize, f32) {
        let mut best_metric = 0.0f32;
        let mut best_pos    = 0usize;
        for d in 0..search_end {
            let m = self.single_metric(&samples[d..d + SYMBOL_LEN]);
            if m > best_metric {
                best_metric = m;
                best_pos    = d;
            }
        }
        (best_pos, best_metric)
    }

    /// Computes the normalised correlation metric ρ for a single window of
    /// length `SYMBOL_LEN`.
    #[inline]
    fn single_metric(&self, window: &[Complex32]) -> f32 {
        debug_assert_eq!(window.len(), SYMBOL_LEN);

        // Cross-correlation (dot product with conjugate reference)
        let corr: Complex32 = window.iter()
            .zip(self.preamble_conj.iter())
            .map(|(&s, &rc)| s * rc) // s · p*(n) = s · (p(n))*
            .sum();

        let win_nrg = energy(window);
        // ρ ∈ [0, 1] by Cauchy–Schwarz
        corr.norm() / (win_nrg * self.ref_energy + 1e-12)
    }

    /// Frequency-domain ZC detection metric: insensitive to per-carrier
    /// amplitude changes (pre-emphasis, HPF rolloff, AGC gain).
    ///
    /// Steps:
    ///   1. FFT the 1024-sample window after stripping the CP.
    ///   2. For each active bin k, compute R[k] = Y[k] × ZC*(k) — this is the
    ///      per-carrier channel estimate if the window really is a ZC.
    ///   3. Measure the **differential phase coherence** between adjacent
    ///      carriers: `Σ R[k+1]·conj(R[k]) / (|R[k+1]|·|R[k]|)`.
    ///
    /// On a ZC through a smooth channel, the differential phase is nearly
    /// constant (= the channel's per-carrier group delay) → coherence ≈ 1.
    /// On random data, differential phases are random → coherence ≈ 0.
    ///
    /// Returns a value in [0, 1].
    fn freq_metric(&self, window: &[Complex32]) -> f32 {
        debug_assert!(window.len() >= SYMBOL_LEN);

        // Strip CP, FFT.
        let fft_in = &window[CP_LEN..CP_LEN + FFT_SIZE];
        let mut buf: Vec<Complex32> = fft_in.to_vec();
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(FFT_SIZE).process(&mut buf);
        let scale = 1.0 / (FFT_SIZE as f32).sqrt();

        // R[k] = Y[bin_k] × ZC*[k] — per-carrier channel if window is ZC.
        let r: Vec<Complex32> = (0..NUM_CARRIERS)
            .map(|k| buf[carrier_to_bin(k)] * scale * self.zc_freq_conj[k])
            .collect();

        // Differential phase coherence: Σ R[k+1]·conj(R[k]) / (|R[k+1]|·|R[k]|)
        let mut sum = Complex32::new(0.0, 0.0);
        let mut count = 0u32;
        for k in 0..NUM_CARRIERS - 1 {
            let mag_prod = r[k].norm() * r[k + 1].norm();
            if mag_prod > 1e-12 {
                sum += r[k + 1] * r[k].conj() / mag_prod;
                count += 1;
            }
        }
        if count == 0 { return 0.0; }
        sum.norm() / count as f32
    }
}

// ── Signal energy ──────────────────────────────────────────────────────────────

/// Computes the RMS energy ‖x‖ = sqrt(Σ|x[n]|²).
#[inline]
fn energy(x: &[Complex32]) -> f32 {
    x.iter().map(|s| s.norm_sqr()).sum::<f32>().sqrt()
}

// ── CFO estimation ─────────────────────────────────────────────────────────────

/// Estimates the carrier frequency offset from **two consecutive identical**
/// OFDM symbols (ZC#1 and ZC#2).
///
/// Because the channel is nearly constant over two adjacent symbols, the
/// phase rotation between them is
///
/// ```text
///   ZC2[n] ≈ ZC1[n] · exp(j·2π·f_cfo·T_sym)
/// ```
///
/// so
///
/// ```text
///   φ = arg( Σ ZC2[n]·ZC1*[n] )
///   f_cfo = φ · fs / (2π · SYMBOL_LEN)
/// ```
///
/// The estimator is unambiguous for |f_cfo| < fs / (2·SYMBOL_LEN) ≈ 13.9 Hz.
pub fn estimate_cfo(zc1: &[Complex32], zc2: &[Complex32]) -> f32 {
    debug_assert_eq!(zc1.len(), zc2.len());
    let cross: Complex32 = zc2.iter()
        .zip(zc1.iter())
        .map(|(&r2, &r1)| r2 * r1.conj())
        .sum();
    let phase = cross.arg(); // ∈ (−π, π]
    phase * SAMPLE_RATE / (2.0 * PI * SYMBOL_LEN as f32)
}

/// Corrects the CFO of `samples` **in-place**.
///
/// Each sample `s[n]` is multiplied by the counter-rotating phasor:
///
/// ```text
///   s'[n] = s[n] · exp(−j·2π·f_cfo·n/fs)
/// ```
///
/// Apply this once on the entire received buffer after sync, before OFDM
/// demodulation.
pub fn correct_cfo(samples: &mut [Complex32], cfo_hz: f32) {
    let phase_inc = -2.0 * PI * cfo_hz / SAMPLE_RATE;
    for (n, s) in samples.iter_mut().enumerate() {
        let phi = phase_inc * n as f32;
        let phasor = Complex32::new(phi.cos(), phi.sin());
        *s *= phasor;
    }
}

// ── Intra-frame timing correction ─────────────────────────────────────────────

/// Searches for the ZC re-sync symbol within a ±`radius` sample window around
/// `expected_pos`.
///
/// Returns `Some((integer_pos, frac))` where:
/// * `integer_pos` — the integer sample index of the best-matching ZC start.
/// * `frac`        — sub-sample fractional offset `∈ (−0.5, 0.5]` obtained by
///   quadratic interpolation of the correlation peak.  Adding it to
///   `integer_pos` gives the estimated ZC position with sub-sample precision.
///
/// Returns `None` if the peak correlation falls below `min_metric`.
///
/// Used by the receiver loop to track clock-drift continuously.
///
/// # Typical usage
///
/// ```text
/// if is_resync_position {
///     if let Some((found, frac)) = find_resync_zc(&samples, expected, 64, 0.20) {
///         let correction = found as f64 + frac as f64 - expected as f64;
///         timing_offset_frac += correction;
///         drift_per_sym = -correction / (RESYNC_PERIOD as f64 + 1.0);
///     }
/// }
/// ```
pub fn find_resync_zc(
    samples:      &[Complex32],
    expected_pos: usize,
    radius:       usize,
    min_metric:   f32,
) -> Option<(usize, f32)> {
    let search_start = expected_pos.saturating_sub(radius);
    let search_end   = (expected_pos + radius + 1)
                           .min(samples.len().saturating_sub(SYMBOL_LEN));

    // Build ZC reference in time domain (CP-stripped, length FFT_SIZE).
    // Use only the real part — we transmit Re(ZC), so this is the correct
    // matched filter.
    let zc_ref: Vec<Complex32> = {
        let zc_freq = zc_freq_reference();
        let mut freq_buf = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        for k in 0..NUM_CARRIERS {
            freq_buf[carrier_to_bin(k)] = zc_freq[k];
        }
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_inverse(FFT_SIZE).process(&mut freq_buf);
        let scale = 1.0 / (FFT_SIZE as f32).sqrt();
        freq_buf.iter().map(|s| Complex32::new(s.re * scale, 0.0)).collect()
    };

    let ref_energy: f32 = zc_ref.iter().map(|s| s.norm_sqr()).sum::<f32>().sqrt();

    // Helper: normalised correlation metric at one integer position.
    let metric_at = |pos: usize| -> f32 {
        let win_start = pos + CP_LEN;
        if win_start + FFT_SIZE > samples.len() { return 0.0; }
        let window = &samples[win_start..win_start + FFT_SIZE];
        let dot: Complex32 = window.iter().zip(zc_ref.iter())
            .map(|(&s, &r)| s * r.conj())
            .sum();
        let win_energy: f32 = window.iter().map(|s| s.norm_sqr()).sum::<f32>().sqrt();
        let denom = win_energy * ref_energy;
        if denom > 1e-12 { dot.norm() / denom } else { 0.0 }
    };

    let mut best_metric = 0.0f32;
    let mut best_pos    = expected_pos;

    for pos in search_start..search_end {
        let m = metric_at(pos);
        if m > best_metric {
            best_metric = m;
            best_pos    = pos;
        }
    }

    if best_metric < min_metric {
        return None;
    }

    // ── Sub-sample interpolation: quadratic fit of peak and its neighbours ──
    // Parabolic interpolation: δ = 0.5 × (m₋₁ − m₊₁) / (m₋₁ − 2m₀ + m₊₁)
    let m_minus = if best_pos > search_start { metric_at(best_pos - 1) } else { 0.0 };
    let m_plus  = if best_pos + 1 < search_end { metric_at(best_pos + 1) } else { 0.0 };
    let denom   = m_minus - 2.0 * best_metric + m_plus;
    let frac    = if denom.abs() > 1e-10 {
        0.5 * (m_minus - m_plus) / denom
    } else {
        0.0f32
    };

    Some((best_pos, frac.clamp(-0.5, 0.5)))
}

// ── Channel estimation ─────────────────────────────────────────────────────────

/// Estimates the per-subcarrier complex channel response H[k] from the FFT
/// window of the received ZC preamble symbol (CP **already removed**).
///
/// Least-squares (zero-forcing) estimate:
/// ```text
///   H_hat[k] = Y_rx[k] / X_zc[k]
/// ```
/// where `X_zc[k]` is the known transmitted ZC in the frequency domain and
/// `Y_rx[k]` is the FFT of the received CP-stripped window.
///
/// Returns `NUM_CARRIERS` complex coefficients, one per active subcarrier.
pub fn channel_estimate_from_zc(fft_window: &[Complex32]) -> Vec<Complex32> {
    debug_assert_eq!(fft_window.len(), FFT_SIZE,
        "pass the CP-stripped FFT window (length {FFT_SIZE})");

    // Forward FFT of received window
    let mut buf = fft_window.to_vec();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    fft.process(&mut buf);
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();

    // Known TX frequency-domain ZC
    let zc_ref = zc_freq_reference();

    (0..NUM_CARRIERS)
        .map(|k| {
            let y = buf[carrier_to_bin(k)] * scale;
            let x = zc_ref[k]; // |x| = 1 (CAZAC)
            // H = Y / X = Y · X*  (since |X|² = 1)
            y * x.conj()
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::zc::build_preamble;
    use approx::assert_abs_diff_eq;

    /// In an AWGN-free channel the correlator must find sync at offset 0
    /// with metric ≈ 1.
    ///
    /// The received signal is the real part of the preamble (as transmitted over
    /// audio with Im = 0), matching the real-part template in the correlator.
    #[test]
    fn perfect_channel_sync() {
        // Simulate the actual received signal: Re(preamble) with Im = 0.
        let preamble_re: Vec<Complex32> = build_preamble()
            .into_iter()
            .map(|s| Complex32::new(s.re, 0.0))
            .collect();
        let mut samples: Vec<Complex32> = Vec::new();
        samples.extend_from_slice(&preamble_re); // ZC#1
        samples.extend_from_slice(&preamble_re); // ZC#2
        samples.extend(std::iter::repeat(Complex32::new(0.0, 0.0)).take(SYMBOL_LEN)); // hdr

        let corr = ZcCorrelator::new(0.40, 0.30);
        let result = corr.find_sync(&samples).expect("sync must succeed");

        assert_eq!(result.preamble_start, 0, "preamble must be at offset 0");
        assert_eq!(result.header_start, 2 * SYMBOL_LEN);
        assert_abs_diff_eq!(result.metric, 1.0, epsilon = 1e-4);
    }

    /// With a known timing offset the correlator must recover the correct
    /// preamble start.
    #[test]
    fn sync_with_timing_offset() {
        let preamble_re: Vec<Complex32> = build_preamble()
            .into_iter().map(|s| Complex32::new(s.re, 0.0)).collect();
        let offset = 137usize;

        let mut samples: Vec<Complex32> =
            vec![Complex32::new(0.0, 0.0); offset];           // leading guard
        samples.extend_from_slice(&preamble_re);              // ZC#1
        samples.extend_from_slice(&preamble_re);              // ZC#2
        samples.extend(
            std::iter::repeat(Complex32::new(0.0, 0.0)).take(SYMBOL_LEN));

        let corr = ZcCorrelator::new(0.40, 0.30);
        let result = corr.find_sync(&samples).expect("sync must succeed");
        assert_eq!(result.preamble_start, offset);
    }

    /// Zero CFO must be estimated as ~ 0 Hz.
    #[test]
    fn cfo_zero_estimate() {
        let preamble = build_preamble();
        let cfo = estimate_cfo(&preamble, &preamble);
        assert_abs_diff_eq!(cfo, 0.0, epsilon = 0.1);
    }

    /// A known CFO must be recovered with low error.
    ///
    /// In a real receiver the CFO causes a continuous phase ramp across ALL
    /// samples.  ZC#1 occupies samples [0..SYMBOL_LEN) and ZC#2 occupies
    /// [SYMBOL_LEN..2·SYMBOL_LEN), so both must be rotated by the full ramp.
    #[test]
    fn cfo_recovery() {
        let preamble = build_preamble();
        let true_cfo = 5.0f32; // Hz
        let phase_per_sample = 2.0 * PI * true_cfo / SAMPLE_RATE;

        // ZC#1: samples 0 … SYMBOL_LEN-1
        let zc1: Vec<Complex32> = preamble.iter().enumerate().map(|(n, &s)| {
            let phi = phase_per_sample * n as f32;
            s * Complex32::new(phi.cos(), phi.sin())
        }).collect();

        // ZC#2: samples SYMBOL_LEN … 2·SYMBOL_LEN-1
        let zc2: Vec<Complex32> = preamble.iter().enumerate().map(|(n, &s)| {
            let phi = phase_per_sample * (SYMBOL_LEN + n) as f32;
            s * Complex32::new(phi.cos(), phi.sin())
        }).collect();

        let cfo_est = estimate_cfo(&zc1, &zc2);
        assert_abs_diff_eq!(cfo_est, true_cfo, epsilon = 0.5);
    }

    /// Channel estimation on a flat-fading AWGN-free channel must return
    /// coefficients equal to the applied channel.
    #[test]
    fn channel_estimate_flat_fading() {
        // Simulate a flat complex channel gain
        let h_true = Complex32::new(0.7, 0.3);

        // "Received" ZC: apply channel, strip CP
        let preamble = build_preamble();
        let zc_rx: Vec<Complex32> = preamble[CP_LEN..] // strip CP
            .iter()
            .map(|&s| s * h_true)
            .collect();

        let h_est = channel_estimate_from_zc(&zc_rx);
        assert_eq!(h_est.len(), NUM_CARRIERS);
        for (k, &h) in h_est.iter().enumerate() {
            assert!(
                (h.re - h_true.re).abs() < 1e-4,
                "H_re mismatch at carrier {k}: got {}", h.re
            );
            assert!(
                (h.im - h_true.im).abs() < 1e-4,
                "H_im mismatch at carrier {k}: got {}", h.im
            );
        }
    }
}
