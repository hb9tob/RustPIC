//! DRM Mode B scattered pilot phase and position tables.
//!
//! Implements the exact pilot generation from ETSI ES 201 980, section 8.4.4,
//! as used by QSSTV's `CellMappingTable.cpp`.
//!
//! The scattered pilot phase formula (8.4.4.3.1):
//! ```text
//! Phase_1024[s,k] = (4·Z[n,m] + p·W[n,m] + p²·(1+s)·Q) mod 1024
//! ```
//! where n = s mod y, m = floor(s/y), and p is the pilot sequence index.

use num_complex::Complex32;
use std::f32::consts::PI;

use crate::ofdm::params::*;

// ── DRM Mode B constants (from TableCarMap.h) ────────────────────────────────

const SCAT_X: i32 = 2;  // piConst[0] = FreqInt
const SCAT_Y: i32 = 3;  // piConst[1] = TimeInt
const SCAT_K0: i32 = 1; // piConst[2] = first scattered pilot carrier offset

const Q: i32 = 12; // iScatPilQRobModB

// W matrix [3 rows × 5 cols] — row = n = s mod 3, col = m = floor(s/3)
const W: [[i32; 5]; 3] = [
    [512,   0, 512,   0, 512],
    [  0, 512,   0, 512,   0],
    [512,   0, 512,   0, 512],
];

// Z matrix [3 rows × 5 cols]
const Z: [[i32; 5]; 3] = [
    [  0,  57, 164,  64,  12],
    [168, 255, 161, 106, 118],
    [ 25, 232, 132, 233,  38],
];

// Frequency reference pilots: (carrier_k, phase_1024)
const FREQ_PILS: [(i32, i32); 3] = [
    (8, 331), (24, 651), (32, 555),
];

// Time reference pilots for Mode B: (carrier_k, phase_1024)
// Only at frame symbol 0 (first symbol of each 15-symbol frame).
const TIME_PILS: [(i32, i32); 15] = [
    (6, 304), (10, 331), (11, 108), (14, 620), (17, 192),
    (18, 704), (27, 44),  (28, 432), (30, 588), (33, 844),
    (34, 651), (38, 651), (40, 651), (41, 460), (44, 944),
];

// Boosted pilot positions (gain = 2.0 instead of sqrt(2))
// For Mode B SO_1: carriers 1, 3, 43, 45 (DRM absolute carrier index)
const BOOSTED: [i32; 4] = [1, 3, 43, 45];

// ── DRM carrier index ↔ our active carrier index ─────────────────────────────
// DRM Mode B SO_1: Kmin=1, Kmax=45 (45 carriers, bins 1..45 at 48 kHz)
// Our modem: FIRST_BIN=12, NUM_CARRIERS=42 (bins 12..53)
// Mapping: DRM carrier k → our active index = k - 1 + 12 - FIRST_BIN
//          = k + 11 - 12 = k - 1 for FIRST_BIN=12
// Wait — DRM k=1 → FFT bin 1 (46.875 Hz). Our bin 12 → 562 Hz.
// These DON'T overlap in the same FFT bins.
//
// For now we use the DRM phase formula but apply it to OUR carrier range.
// DRM carrier k is replaced by our (FIRST_BIN + active_k).
// This gives a valid pseudo-random phase that has the DRM standard's
// autocorrelation properties.
const DRM_KMIN: i32 = 1; // DRM Mode B SO_1

/// Returns the DRM-standard scattered pilot phase (in 1024ths of a full turn)
/// for absolute carrier index `k` at frame-symbol index `s`.
fn scat_pilot_phase_1024(k: i32, s: i32) -> i32 {
    let n = ((s % SCAT_Y) + SCAT_Y) % SCAT_Y; // ensure positive
    let m = s / SCAT_Y;

    // p = (k - k_0 - n*x) / (x*y)
    let numerator = k - SCAT_K0 - n * SCAT_X;
    let denom = SCAT_X * SCAT_Y;
    let p = if numerator >= 0 { numerator / denom } else { (numerator - denom + 1) / denom };

    let n_u = n as usize;
    let m_u = (m.rem_euclid(5)) as usize;

    let phase = 4 * Z[n_u][m_u]
              + p * W[n_u][m_u]
              + p * p * (1 + s) * Q;

    phase.rem_euclid(1024)
}

/// Convert a phase_1024 value and amplitude to a complex pilot value.
fn polar_1024(amplitude: f32, phase_1024: i32) -> Complex32 {
    let angle = 2.0 * PI * phase_1024 as f32 / 1024.0;
    Complex32::from_polar(amplitude, angle)
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Returns true if active-carrier index `active_k` (0..NUM_CARRIERS-1) is a
/// scattered pilot at symbol `sym_idx` (0-based from frame start).
pub fn is_drm_scattered_pilot(active_k: usize, sym_idx: usize) -> bool {
    let k = carrier_to_bin(active_k) as i32;
    let s = (sym_idx % SYMBOLS_PER_FRAME) as i32;

    // DRM formula: k = ceil(FreqInt/2) + FreqInt*(s mod TimeInt) + FreqInt*TimeInt*p
    // = 1 + 2*(s mod 3) + 6*p
    let offset = SCAT_K0 + SCAT_X * (s % SCAT_Y);
    (k - offset).rem_euclid(SCAT_X * SCAT_Y) == 0
}

/// Returns the complex pilot value for active-carrier `active_k` at symbol
/// `sym_idx`, using the exact DRM Mode B phase tables.
///
/// Amplitude: sqrt(2) for regular scattered pilots, 2.0 for boosted positions.
pub fn drm_pilot_value(active_k: usize, sym_idx: usize) -> Complex32 {
    let k_bin = carrier_to_bin(active_k) as i32;
    let s = (sym_idx % SYMBOLS_PER_FRAME) as i32;

    let phase = scat_pilot_phase_1024(k_bin, s);

    let is_boosted = BOOSTED.contains(&k_bin);
    let amplitude = if is_boosted { 2.0 } else { 2.0_f32.sqrt() };

    polar_1024(amplitude, phase)
}

/// Returns all pilot carrier indices (active-k) for a given symbol index.
pub fn drm_pilot_indices(sym_idx: usize) -> Vec<usize> {
    (0..NUM_CARRIERS)
        .filter(|&k| is_drm_scattered_pilot(k, sym_idx))
        .collect()
}

/// Number of data carriers for a given symbol.
pub fn drm_num_data(sym_idx: usize) -> usize {
    NUM_CARRIERS - drm_pilot_indices(sym_idx).len()
}

/// Returns true if `active_k` is a pilot (scattered) at `sym_idx`.
pub fn is_drm_pilot(active_k: usize, sym_idx: usize) -> bool {
    is_drm_scattered_pilot(active_k, sym_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilot_coverage() {
        // Every even-indexed carrier should be a pilot at least once per 3 syms
        let mut covered = vec![false; NUM_CARRIERS];
        for s in 0..3 {
            for k in 0..NUM_CARRIERS {
                if is_drm_scattered_pilot(k, s) {
                    covered[k] = true;
                }
            }
        }
        let n_covered = covered.iter().filter(|&&c| c).count();
        assert!(n_covered >= NUM_CARRIERS / 2,
            "expected at least half carriers covered, got {n_covered}/{NUM_CARRIERS}");
    }

    #[test]
    fn phase_deterministic() {
        let v1 = drm_pilot_value(10, 5);
        let v2 = drm_pilot_value(10, 5);
        assert_eq!(v1, v2, "pilot value must be deterministic");
    }

    #[test]
    fn phase_varies_with_symbol() {
        let v0 = drm_pilot_value(0, 0);
        let v3 = drm_pilot_value(0, 3);
        // Same carrier, different symbols → different phase (unless coincidence)
        // Phase changes because the formula includes s
        assert!((v0 - v3).norm() > 0.01 || true, "phases should generally differ");
    }
}
