//! OFDM system parameters — DRM Robustness Mode B, 2.5 kHz audio bandwidth.
//!
//! Mirrors the ITU-R BS.1514 / ETSI ES 201 980 DRM standard (Mode B, SO1)
//! operating natively at 48 kHz — no resampling required.
//!
//! # Subcarrier layout
//!
//! ```text
//! FFT size  = 1024,  fs = 48 000 Hz  →  Δf = 46.875 Hz / bin
//!
//! Active band : bins 7 … 53  (≈ 328 Hz … 2484 Hz)
//! 47 subcarriers, pilots every 8th (k = 0, 8, 16, 24, 32, 40 → 6 pilots)
//!
//! Subcarrier frequencies (Hz):
//!   bins 7, 8, …, 53  →  328, 375, …, 2484 Hz
//!
//! FIRST_BIN = 7 keeps the signal above the FM sub-audio / CTCSS range
//! (typically < 300 Hz) and is parameterizable via the constant below.
//! ```
//!
//! # Symbol timing (DRM Mode B)
//!
//! ```text
//! Tu  = 1024 samples = 21.33 ms  (useful symbol)
//! Tg  =  256 samples =  5.33 ms  (guard interval / cyclic prefix, ratio 1/4)
//! Ts  = 1280 samples = 26.67 ms  (total symbol period)
//! ```
//!
//! # Super-frame structure
//!
//! ```text
//! ┌──────┬──────┬────────┬── … ──┬── re-sync ────┬── … ──┬─────┐
//! │ ZC#1 │ ZC#2 │ header │ data  │ ZC (every 12) │ data  │ EOT │
//! └──────┴──────┴────────┴── … ──┴───────────────┴── … ──┴─────┘
//! ```

// ── FFT / CP ──────────────────────────────────────────────────────────────────

/// Number of complex samples in one OFDM FFT window (without CP).
/// DRM Mode B: Tu = 1024 / 48000 = 21.33 ms.
pub const FFT_SIZE: usize = 1024;

/// Cyclic-prefix length in samples.
/// DRM Mode B guard ratio: Tg/Tu = 1/4  →  256 samples = 5.33 ms.
pub const CP_LEN: usize = 256;

/// Total OFDM symbol length (FFT + CP) in samples.
pub const SYMBOL_LEN: usize = FFT_SIZE + CP_LEN; // 1280

// ── Timing / frequency ────────────────────────────────────────────────────────

/// Audio sample rate in Hz.  Native 48 kHz — no resampling required.
pub const SAMPLE_RATE: f32 = 48_000.0;

/// Subcarrier spacing in Hz  (fs / FFT_SIZE = 48000 / 1024 = 46.875 Hz).
pub const SUBCARRIER_SPACING: f32 = SAMPLE_RATE / FFT_SIZE as f32; // 46.875 Hz

// ── Active subcarrier mapping ─────────────────────────────────────────────────

/// FFT bin index of the first active subcarrier (≈ 328 Hz).
///
/// Keeping the signal above 300 Hz avoids the FM sub-audio / CTCSS range.
/// Parameterizable: increase to push the lower edge higher.
pub const FIRST_BIN: usize = 7;

/// Total number of active subcarriers (data + pilots).
/// Covers bins 7 … 53  (≈ 328 Hz … 2484 Hz).
pub const NUM_CARRIERS: usize = 47;

/// FFT bin index of the last active subcarrier (inclusive), ≈ 2484 Hz.
pub const LAST_BIN: usize = FIRST_BIN + NUM_CARRIERS - 1; // 53

// ── Pilots ────────────────────────────────────────────────────────────────────

/// Distance between two consecutive pilot subcarriers (in active-carrier index).
/// Pilots are always BPSK-modulated with the known PN sequence.
pub const PILOT_SPACING: usize = 8;

/// Number of pilot subcarriers per OFDM symbol.
/// Pilot active-carrier indices: 0, 8, 16, 24, 32, 40  (6 pilots).
///
/// Formula: number of k in [0, NUM_CARRIERS) where k % PILOT_SPACING == 0.
pub const NUM_PILOTS: usize = (NUM_CARRIERS - 1) / PILOT_SPACING + 1; // 6

/// Number of data subcarriers per OFDM symbol.
pub const NUM_DATA: usize = NUM_CARRIERS - NUM_PILOTS; // 41

// ── ZC preamble ───────────────────────────────────────────────────────────────

/// Zadoff–Chu root.  Must be coprime with `ZC_LEN`.
/// 47 is prime so any root 1…46 is coprime; 25 chosen to match prior design.
pub const ZC_ROOT: u32 = 25;

/// ZC sequence length (= number of active subcarriers for a full-band preamble).
pub const ZC_LEN: usize = NUM_CARRIERS; // 47

// ── Super-frame / re-sync ─────────────────────────────────────────────────────

/// A single re-sync ZC symbol is inserted every `RESYNC_PERIOD` data symbols.
pub const RESYNC_PERIOD: usize = 12;

/// Minimum number of data symbols remaining before inserting a re-sync ZC.
pub const RESYNC_MIN_REMAIN: usize = 6;

// ── Subcarrier index helpers ──────────────────────────────────────────────────

/// Maps active-subcarrier index `k ∈ [0, NUM_CARRIERS)` to the absolute FFT bin.
#[inline(always)]
pub const fn carrier_to_bin(k: usize) -> usize {
    FIRST_BIN + k
}

/// Maps absolute FFT bin to active-subcarrier index.
/// Returns `None` if the bin lies outside the active band.
#[inline(always)]
pub const fn bin_to_carrier(bin: usize) -> Option<usize> {
    if bin >= FIRST_BIN && bin <= LAST_BIN {
        Some(bin - FIRST_BIN)
    } else {
        None
    }
}

/// Returns `true` if the active-subcarrier index `k` is a pilot.
#[inline(always)]
pub const fn is_pilot(k: usize) -> bool {
    k % PILOT_SPACING == 0
}

/// FM pre-emphasis gain for active-carrier index `k` (0 … NUM_CARRIERS−1).
///
/// Returns `f_k / f_0 = carrier_to_bin(k) / FIRST_BIN`, proportional to the
/// subcarrier frequency.  Can be used to compensate the FM discriminator's
/// parabolic noise PSD (∝ f²) by boosting high-frequency subcarriers at TX.
///
/// Not currently applied — kept as a utility for future FT-847 use
/// (`--preemph` CLI flag).
#[inline(always)]
#[allow(dead_code)]
pub fn preemphasis_gain(k: usize) -> f32 {
    carrier_to_bin(k) as f32 / FIRST_BIN as f32
}

/// Returns the signed BPSK value (+1 or −1) for pilot `k` using a simple
/// m-sequence–derived pattern.
///
/// The same sequence must be used by the TX when inserting pilots.
/// Only the first `NUM_PILOTS = 6` values are used; the 9-chip register
/// ensures well-distributed signs.
pub fn pilot_sign(k: usize) -> f32 {
    const PN9: [f32; 9] = [1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0];
    let pilot_idx = k / PILOT_SPACING;
    PN9[pilot_idx % PN9.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_consistency() {
        assert_eq!(SYMBOL_LEN, FFT_SIZE + CP_LEN);
        assert_eq!(LAST_BIN, FIRST_BIN + NUM_CARRIERS - 1);
        assert_eq!(NUM_PILOTS + NUM_DATA, NUM_CARRIERS);
        // Active band must fit inside the positive-frequency half of the FFT
        assert!(LAST_BIN < FFT_SIZE / 2,
            "last bin {LAST_BIN} must be < Nyquist bin {}", FFT_SIZE / 2);
        // Guard interval ratio ≥ 1/4  (DRM Mode B requirement)
        assert!(CP_LEN * 4 >= FFT_SIZE,
            "CP/FFT = {}/{} < 1/4", CP_LEN, FFT_SIZE);
        // First bin must be above 300 Hz (FM sub-audio / CTCSS safe margin)
        let first_hz = FIRST_BIN as f32 * SUBCARRIER_SPACING;
        assert!(first_hz >= 300.0,
            "first carrier {first_hz:.1} Hz is below 300 Hz CTCSS margin");
    }

    #[test]
    fn pilot_positions() {
        let pilots: Vec<usize> = (0..NUM_CARRIERS).filter(|&k| is_pilot(k)).collect();
        assert_eq!(pilots, vec![0, 8, 16, 24, 32, 40]);
        assert_eq!(pilots.len(), NUM_PILOTS);
    }

    #[test]
    fn carrier_frequencies() {
        // First carrier ≈ 328 Hz, last ≈ 2484 Hz
        let f_first = carrier_to_bin(0) as f32 * SUBCARRIER_SPACING;
        let f_last  = carrier_to_bin(NUM_CARRIERS - 1) as f32 * SUBCARRIER_SPACING;
        assert!((f_first - 328.125).abs() < 0.1);
        assert!((f_last  - 2484.375).abs() < 0.1);
    }
}
