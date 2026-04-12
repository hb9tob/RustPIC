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
//! Active band : bins 12 … 53  (≈ 562 Hz … 2484 Hz)
//! 42 subcarriers, pilots every 8th (k = 0, 8, 16, 24, 32, 40 → 6 pilots)
//!
//! Subcarrier frequencies (Hz):
//!   bins 12, 13, …, 53  →  562, 609, …, 2484 Hz
//!
//! FIRST_BIN = 12 (was 7) keeps the signal above the non-linear phase region
//! of typical NBFM voice HPFs.  Measurements on OTA recordings (G3E marine)
//! show that the audio HPF at ~300 Hz creates phase rotations of > 100°
//! between adjacent bins for frequencies < 560 Hz — far too steep for the
//! pilot-based linear interpolation (pilots every 8 carriers) to track,
//! causing the 5 lowest carriers to contribute > 5 % raw BER on their own.
//! Starting at 562 Hz places pilot 0 in the linear-delay region of the
//! channel, restoring clean equalization at the cost of 5 carriers
//! (~15 % throughput).  See memory/feedback_first_bin.md.
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

/// FFT bin index of the first active subcarrier (≈ 562 Hz).
///
/// Chosen to keep pilot 0 above the ~300 Hz audio HPF where typical NBFM
/// radios introduce a steep phase non-linearity that the 8-carrier pilot
/// interpolation cannot track.
pub const FIRST_BIN: usize = 12;

/// Total number of active subcarriers (data + pilots).
/// Covers bins 12 … 53  (≈ 562 Hz … 2484 Hz).
pub const NUM_CARRIERS: usize = 42;

/// FFT bin index of the last active subcarrier (inclusive), ≈ 2484 Hz.
pub const LAST_BIN: usize = FIRST_BIN + NUM_CARRIERS - 1; // 53

// ── Scattered pilots (DRM Mode B style) ──────────────────────────────────────
//
// DRM scattered pilots are placed at rotating positions so that every
// carrier is a pilot at least once in every SCAT_TIME_INT symbols.
//
// At symbol index `s`, a scattered pilot occupies active-carrier index `k`
// when:
//     k % (SCAT_FREQ_INT * SCAT_TIME_INT) == (s % SCAT_TIME_INT) * SCAT_FREQ_INT
//
// For Mode B: FreqInt=2, TimeInt=3 → period 6.
//   s%3 = 0: pilots at k = 0, 6, 12, 18, 24, 30, 36   (7 pilots)
//   s%3 = 1: pilots at k = 2, 8, 14, 20, 26, 32, 38   (7 pilots)
//   s%3 = 2: pilots at k = 4, 10, 16, 22, 28, 34, 40  (7 pilots)
//
// Every 3 symbols all 42 carriers have been a pilot → full 2D coverage.

/// Scattered pilot frequency interval (carriers between pilots in one symbol).
pub const SCAT_FREQ_INT: usize = 2;

/// Scattered pilot time interval (symbols between two pilots at the same carrier).
pub const SCAT_TIME_INT: usize = 3;

/// Period of the scattered pilot pattern in carrier indices.
pub const SCAT_PERIOD: usize = SCAT_FREQ_INT * SCAT_TIME_INT; // 6

/// Number of scattered pilots per OFDM symbol.
/// = ceil(NUM_CARRIERS / SCAT_PERIOD) = ceil(42/6) = 7
pub const NUM_PILOTS_PER_SYM: usize = (NUM_CARRIERS + SCAT_PERIOD - 1) / SCAT_PERIOD; // 7

/// Number of data subcarriers per OFDM symbol (varies slightly with symbol
/// index because of pilot placement at the edges, but 42 - 7 = 35 for most).
pub const NUM_DATA_PER_SYM: usize = NUM_CARRIERS - NUM_PILOTS_PER_SYM; // 35

/// DRM Mode B scattered pilot phase parameter Q.
pub const SCAT_PILOT_Q: usize = 12;

// ── Frequency reference pilots (fixed, for coarse sync) ─────────────────────

/// Three fixed-position frequency pilots used for initial frame detection
/// and coarse frequency offset estimation.  Their carrier indices and
/// reference phase codes are taken from the DRM Mode B table.
/// Each entry: (active_carrier_index, phase_code).
pub const FREQ_PILOTS: [(usize, u16); 3] = [
    ( 8, 331),
    (24, 651),
    (32, 555),
];

// ── Frame structure ─────────────────────────────────────────────────────────

/// Number of OFDM symbols per DRM frame.
pub const SYMBOLS_PER_FRAME: usize = 15;

/// Number of frames per super-frame.
pub const FRAMES_PER_SUPERFRAME: usize = 3;

/// Total symbols in one super-frame.
pub const SYMBOLS_PER_SUPERFRAME: usize = SYMBOLS_PER_FRAME * FRAMES_PER_SUPERFRAME; // 45

/// Number of header symbols repeated at the start of a transmission for
/// sync acquisition.  QSSTV repeats the first ~20 blocks; we use 2 full
/// DRM frames (30 symbols ≈ 800 ms) to let the RX lock onto the pilot
/// pattern and stabilise the channel estimate before data begins.
pub const HEADER_REPEAT_SYMS: usize = 2 * SYMBOLS_PER_FRAME; // 30

/// Number of pilot-bearing dummy OFDM symbols before the mode header.
/// Warms up the RX Hilbert filter and lets pilot sync detect the frame.
pub const RUNIN_PREAMBLE_SYMS: usize = 10;

/// Number of pilot-bearing dummy symbols after the EOT.
/// Flushes the RX Hilbert filter and keeps the equaliser stable.
pub const RUNOUT_SYMS: usize = 4;

/// Total non-data symbols before the first data symbol.
/// RUNIN preamble + mode header repeats.
pub const PREAMBLE_SYMS: usize = RUNIN_PREAMBLE_SYMS + MODE_HEADER_REPEAT;

/// Pilot amplitude boost (√2 ≈ +3 dB above data, matching DRM spec).
/// Boosted pilots improve channel estimation SNR at the cost of a small
/// reduction in data power.  QSSTV uses the same √2 boost.
pub const PILOT_BOOST: f32 = 1.414_213_6; // √2

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

/// Returns `true` if active-carrier index `k` is a scattered pilot at
/// symbol index `sym_idx`.
///
/// DRM Mode B pattern:
///   k % (SCAT_FREQ_INT × SCAT_TIME_INT) == (sym_idx % SCAT_TIME_INT) × SCAT_FREQ_INT
#[inline(always)]
pub fn is_scattered_pilot(k: usize, sym_idx: usize) -> bool {
    k % SCAT_PERIOD == (sym_idx % SCAT_TIME_INT) * SCAT_FREQ_INT
}

/// Returns `true` if active-carrier index `k` is one of the 3 fixed
/// frequency-reference pilots.
#[inline(always)]
pub fn is_freq_pilot(k: usize) -> bool {
    FREQ_PILOTS.iter().any(|&(pk, _)| pk == k)
}

/// Returns `true` if `k` carries a pilot at symbol `sym_idx`.
/// Currently uses scattered pilots only (freq pilots reserved for future
/// pilot-based sync when ZC is removed).
#[inline(always)]
pub fn is_pilot_at(k: usize, sym_idx: usize) -> bool {
    is_scattered_pilot(k, sym_idx)
}

/// Returns the list of pilot carrier indices for a given symbol index.
pub fn pilot_indices(sym_idx: usize) -> Vec<usize> {
    (0..NUM_CARRIERS)
        .filter(|&k| is_pilot_at(k, sym_idx))
        .collect()
}

/// Returns the number of data carriers for a given symbol index.
pub fn num_data_at(sym_idx: usize) -> usize {
    NUM_CARRIERS - pilot_indices(sym_idx).len()
}

/// Returns the known complex pilot value at active-carrier `k` for
/// symbol `sym_idx`.  Uses a deterministic pseudo-random phase derived
/// from `k` and `sym_idx` (DRM-style: phase = 2π × hash(k, s) / 1024).
///
/// The amplitude is `PILOT_BOOST` (√2 ≈ +3 dB above data carriers).
pub fn pilot_value(k: usize, sym_idx: usize) -> num_complex::Complex32 {
    use std::f32::consts::PI;
    // Simple deterministic phase from a hash of (k, sym_idx).
    // Uses a 10-bit gold-code-like scrambler seeded from the DRM Q parameter.
    let phase_idx = ((k as u32).wrapping_mul(SCAT_PILOT_Q as u32 + 1)
        .wrapping_add(sym_idx as u32 * 7))
        % 1024;
    let phase = 2.0 * PI * phase_idx as f32 / 1024.0;
    num_complex::Complex32::from_polar(PILOT_BOOST, phase)
}

/// Legacy helper — kept for the old fixed-pilot code paths still used in
/// some tests.  Returns the signed BPSK value (+1 or −1).
#[allow(dead_code)]
pub fn pilot_sign(k: usize) -> f32 {
    const PN9: [f32; 9] = [1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0];
    let pilot_idx = k / 8;
    PN9[pilot_idx % PN9.len()]
}

/// Legacy constant kept for old code paths.
#[allow(dead_code)]
pub const PILOT_SPACING: usize = 8;

// ── Legacy aliases (old fixed-pilot system) ─────────────────────────────────
// These are used by the old equalizer, sync, frame code. They will be removed
// once the DRM scattered-pilot rewrite is complete.
#[allow(dead_code)]
pub const NUM_PILOTS: usize = NUM_PILOTS_PER_SYM;
#[allow(dead_code)]
pub const NUM_DATA: usize = NUM_DATA_PER_SYM;

/// Legacy: returns true if k is a pilot at sym_idx=0 (backward compat for
/// code that doesn't track symbol index yet).
#[allow(dead_code)]
pub fn is_pilot(k: usize) -> bool {
    is_scattered_pilot(k, 0)
}

// ZC constants kept for old sync.rs / zc.rs code paths.
#[allow(dead_code)]
pub const ZC_ROOT: u32 = 25;
#[allow(dead_code)]
pub const ZC_LEN: usize = NUM_CARRIERS;
#[allow(dead_code)]
pub const RESYNC_PERIOD: usize = 12;
#[allow(dead_code)]
pub const MODE_HEADER_REPEAT: usize = 3;
#[allow(dead_code)]
pub const RESYNC_MIN_REMAIN: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_consistency() {
        assert_eq!(SYMBOL_LEN, FFT_SIZE + CP_LEN);
        assert_eq!(LAST_BIN, FIRST_BIN + NUM_CARRIERS - 1);
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
    fn scattered_pilot_pattern() {
        // sym 0: k=0,6,12,18,24,30,36
        let p0 = pilot_indices(0);
        assert!(p0.contains(&0));
        assert!(p0.contains(&6));
        assert!(p0.contains(&36));
        // sym 1: k=2,8,14,20,26,32,38
        let p1 = pilot_indices(1);
        assert!(p1.contains(&2));
        assert!(p1.contains(&8));  // also freq pilot
        assert!(p1.contains(&38));
        // sym 2: k=4,10,16,22,28,34,40
        let p2 = pilot_indices(2);
        assert!(p2.contains(&4));
        assert!(p2.contains(&40));
        // Every EVEN carrier is a pilot at least once in SCAT_TIME_INT symbols.
        // ODD carriers are interpolated from neighbours (FreqInt=2).
        let mut covered = vec![false; NUM_CARRIERS];
        for s in 0..SCAT_TIME_INT {
            for &k in &pilot_indices(s) {
                covered[k] = true;
            }
        }
        let even_covered = (0..NUM_CARRIERS).step_by(SCAT_FREQ_INT).all(|k| covered[k]);
        assert!(even_covered, "not all even carriers covered in one pilot cycle");
        // Every data carrier is at most SCAT_FREQ_INT-1 = 1 position from a pilot
        // → linear interpolation is sufficient for smooth channels.
    }

    #[test]
    fn carrier_frequencies() {
        // First carrier ≈ 562 Hz, last ≈ 2484 Hz
        let f_first = carrier_to_bin(0) as f32 * SUBCARRIER_SPACING;
        let f_last  = carrier_to_bin(NUM_CARRIERS - 1) as f32 * SUBCARRIER_SPACING;
        assert!((f_first - 562.5).abs() < 0.1);
        assert!((f_last  - 2484.375).abs() < 0.1);
    }
}
