//! OFDM system parameters — shared between TX and RX.
//!
//! # Subcarrier layout
//!
//! ```text
//! FFT size  = 256,  fs = 8 000 Hz  →  Δf = 31.25 Hz / bin
//!
//! Active band : bins 10 … 81  (≈ 312 Hz … 2 531 Hz)
//! 72 subcarriers, pilots every 8th (k = 0, 8, 16 … 64 → 9 pilots)
//!
//! k :  0   1   2   3   4   5   6   7  |  8   9  10 …
//!     [P] [D] [D] [D] [D] [D] [D] [D] | [P] [D] [D] …
//! ```
//!
//! # Super-frame structure
//!
//! ```text
//! ┌──────┬──────┬────────┬── … ──┬──── re-sync ────┬── … ──┬─────┐
//! │ ZC#1 │ ZC#2 │ header │ data  │ ZC (every 12 sym)│ data  │ EOT │
//! └──────┴──────┴────────┴── … ──┴─────────────────┴── … ──┴─────┘
//!  SYMBOL  SYMBOL  SYMBOL  N×SYM                             1 SYM
//! ```

// ── FFT / CP ──────────────────────────────────────────────────────────────────

/// Number of complex samples in one OFDM FFT window (without CP).
pub const FFT_SIZE: usize = 256;

/// Cyclic-prefix length in samples.
pub const CP_LEN: usize = 32;

/// Total OFDM symbol length (FFT + CP) in samples.
pub const SYMBOL_LEN: usize = FFT_SIZE + CP_LEN; // 288

// ── Timing / frequency ────────────────────────────────────────────────────────

/// Audio sample rate in Hz.
pub const SAMPLE_RATE: f32 = 8_000.0;

/// Subcarrier spacing in Hz  (fs / FFT_SIZE).
pub const SUBCARRIER_SPACING: f32 = SAMPLE_RATE / FFT_SIZE as f32; // 31.25 Hz

// ── Active subcarrier mapping ─────────────────────────────────────────────────

/// FFT bin index of the first active subcarrier (≈ 312.5 Hz).
pub const FIRST_BIN: usize = 10;

/// Total number of active subcarriers (data + pilots).
pub const NUM_CARRIERS: usize = 72;

/// FFT bin index of the last active subcarrier (inclusive), ≈ 2 531 Hz.
pub const LAST_BIN: usize = FIRST_BIN + NUM_CARRIERS - 1; // 81

// ── Pilots ────────────────────────────────────────────────────────────────────

/// Distance between two consecutive pilot subcarriers (in active-carrier index).
/// Pilots are always BPSK-modulated with the known PN sequence.
pub const PILOT_SPACING: usize = 8;

/// Number of pilot subcarriers per OFDM symbol.
/// Pilot active-carrier indices: 0, 8, 16, 24, 32, 40, 48, 56, 64.
pub const NUM_PILOTS: usize = NUM_CARRIERS / PILOT_SPACING; // 9

/// Number of data subcarriers per OFDM symbol.
pub const NUM_DATA: usize = NUM_CARRIERS - NUM_PILOTS; // 63

// ── ZC preamble ───────────────────────────────────────────────────────────────

/// Zadoff–Chu root.  Must be coprime with `ZC_LEN`.
pub const ZC_ROOT: u32 = 25;

/// ZC sequence length (= number of active subcarriers for a full-band preamble).
pub const ZC_LEN: usize = NUM_CARRIERS; // 72

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

/// Returns the signed BPSK value (+1 or −1) for pilot `k` using a simple
/// length-9 m-sequence–derived pattern.
///
/// The same sequence must be used by the TX when inserting pilots.
pub fn pilot_sign(k: usize) -> f32 {
    // 9-chip pattern: first bit of each 9-chip PN segment.
    // Generated offline from poly x^4+x+1 (length 15, take first 9 chips).
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
        assert_eq!(NUM_PILOTS, 9);
        assert_eq!(NUM_DATA, 63);
        // Active band must fit inside the positive-frequency half of the FFT
        assert!(LAST_BIN < FFT_SIZE / 2,
            "last bin {LAST_BIN} must be < Nyquist bin {}", FFT_SIZE / 2);
    }

    #[test]
    fn pilot_positions() {
        let pilots: Vec<usize> = (0..NUM_CARRIERS).filter(|&k| is_pilot(k)).collect();
        assert_eq!(pilots, vec![0, 8, 16, 24, 32, 40, 48, 56, 64]);
    }
}
