//! Soft-output LLR demapper for all supported constellations.
//!
//! Computes max-log LLR values for input to the LDPC decoder.
//!
//! # Constellation overview
//!
//! | Modulation | Bits/sym | Normalization d   | Axes                    |
//! |------------|----------|-------------------|-------------------------|
//! | BPSK       | 1        | 1                 | I ∈ {±1}               |
//! | QPSK       | 2        | 1/√2              | I,Q ∈ {±d}             |
//! | 16-QAM     | 4        | 1/√10             | I,Q ∈ {±1d, ±3d}       |
//! | 32-QAM     | 5        | 1/√26             | I ∈ {±1,±3,±5,±7}·d   |
//! |            |          |                   | Q ∈ {±1,±3}·d          |
//! | 64-QAM     | 6        | 1/√42             | I,Q ∈ {±1,±3,±5,±7}·d |
//!
//! All constellations are **Gray-coded**: adjacent symbols differ in exactly
//! one bit, minimising BER at moderate SNR.
//!
//! # LLR convention
//!
//! `LLR(bᵢ) = log P(bᵢ=0|y) / P(bᵢ=1|y)`:
//! * LLR > 0 → bit more likely 0
//! * LLR < 0 → bit more likely 1
//! * |LLR| → confidence
//!
//! Computed with the **max-log** (min-distance) approximation:
//!
//! ```text
//! LLR(bᵢ) ≈ (min_{s: bᵢ=1} |y−s|²  −  min_{s: bᵢ=0} |y−s|²) / σ²
//! ```
//!
//! # Bit ordering per symbol
//!
//! Bits are ordered MSB-first within each symbol.  For QPSK: b0 encodes I, b1
//! encodes Q.  For rectangular constellations: I-axis bits come before Q-axis
//! bits.

use num_complex::Complex32;

use crate::ofdm::rx::mode_detect::Modulation;

// ── Normalization constants ───────────────────────────────────────────────────

/// BPSK: s ∈ {±1}, average power = 1.
const BPSK_D: f32 = 1.0;

/// QPSK: s ∈ {±d ± jd}, d = 1/√2, average power = 1.
const QPSK_D: f32 = std::f32::consts::FRAC_1_SQRT_2; // ≈ 0.7071

/// 16-QAM: I,Q ∈ {±1,±3}·d, d = 1/√10, average power = 1.
const QAM16_D: f32 = 0.316_227_77; // 1/√10

/// 32-QAM (8×4 rectangular): I ∈ {±1,±3,±5,±7}·d, Q ∈ {±1,±3}·d.
/// d = 1/√26 → average power = (21+5)d² = 1.
const QAM32_D: f32 = 0.196_116_13; // 1/√26

/// 64-QAM: I,Q ∈ {±1,±3,±5,±7}·d, d = 1/√42, average power = 1.
const QAM64_D: f32 = 0.154_303_35; // 1/√42

// ── Public API ────────────────────────────────────────────────────────────────

/// Computes soft LLR values for a slice of equalized complex symbols.
///
/// # Arguments
///
/// * `symbols`    — equalized data subcarrier values from [`SymbolEqualizer`].
///                  Length = `N`.
/// * `noise_var`  — per-symbol post-equalization noise variance σ²_eq.
///                  Length = `N`.  Use a uniform value if not available.
/// * `modulation` — constellation to use for demapping.
///
/// # Returns
///
/// A `Vec<f32>` of length `N × bits_per_symbol` with LLR values, interleaved
/// by symbol: `[llr_b0_sym0, llr_b1_sym0, …, llr_b0_sym1, llr_b1_sym1, …]`.
///
/// [`SymbolEqualizer`]: crate::ofdm::rx::equalizer::SymbolEqualizer
pub fn demap(symbols: &[Complex32], noise_var: &[f32], modulation: Modulation) -> Vec<f32> {
    assert_eq!(symbols.len(), noise_var.len());
    let bps      = modulation.bits_per_symbol();
    let pts      = constellation_points(modulation);
    let mut llrs = Vec::with_capacity(symbols.len() * bps);

    for (&y, &nv) in symbols.iter().zip(noise_var.iter()) {
        // Clamp noise variance to avoid division by zero
        let nv_safe = nv.max(1e-10);
        let sym_llrs = max_log_llr(y, &pts, bps, nv_safe);
        llrs.extend_from_slice(&sym_llrs);
    }
    llrs
}

/// Variant of [`demap`] with a single uniform noise variance for all symbols.
pub fn demap_uniform(symbols: &[Complex32], noise_var: f32, modulation: Modulation) -> Vec<f32> {
    let nv = vec![noise_var; symbols.len()];
    demap(symbols, &nv, modulation)
}

// ── Max-log LLR engine ────────────────────────────────────────────────────────

/// Computes `n_bits` LLR values for one received complex symbol `y` using the
/// max-log approximation.
///
/// `constellation` is a slice of `(symbol_point, gray_code)` pairs where
/// `gray_code` is a `u32` bitmask with bit 0 = LSB and bit `n_bits−1` = MSB.
fn max_log_llr(
    y:             Complex32,
    constellation: &[(Complex32, u32)],
    n_bits:        usize,
    noise_var:     f32,
) -> Vec<f32> {
    let inv_var = noise_var.recip();
    (0..n_bits)
        .map(|bit| {
            // MSB-first: bit 0 corresponds to the most-significant bit position.
            let mask = 1u32 << (n_bits - 1 - bit);
            let mut d_zero = f32::INFINITY;
            let mut d_one  = f32::INFINITY;
            for &(s, code) in constellation {
                let dist = (y - s).norm_sqr();
                if code & mask == 0 { d_zero = d_zero.min(dist); }
                else                { d_one  = d_one.min(dist);  }
            }
            // LLR > 0 → bit is more likely 0 (numerator = prob of bit=0)
            (d_one - d_zero) * inv_var
        })
        .collect()
}

// ── Constellation builders ────────────────────────────────────────────────────

/// Returns the constellation points for `modulation` as `(symbol, gray_code)`.
pub(crate) fn constellation_points(m: Modulation) -> Vec<(Complex32, u32)> {
    match m {
        Modulation::Bpsk  => bpsk_points(),
        Modulation::Qpsk  => qpsk_points(),
        Modulation::Qam16 => qam_points(&pam4_gray(QAM16_D), &pam4_gray(QAM16_D), 2),
        Modulation::Qam32 => qam_points(&pam8_gray(QAM32_D), &pam4_gray(QAM32_D), 3),
        Modulation::Qam64 => qam_points(&pam8_gray(QAM64_D), &pam8_gray(QAM64_D), 3),
    }
}

/// BPSK: +1 → bit 0 = 0,  −1 → bit 0 = 1.
fn bpsk_points() -> Vec<(Complex32, u32)> {
    vec![
        (Complex32::new( BPSK_D, 0.0), 0b0),
        (Complex32::new(-BPSK_D, 0.0), 0b1),
    ]
}

/// QPSK Gray-coded:
///
/// ```text
///   Q
///   │  10 │ 00
///   ──────┼──────  I
///   11 │ 01
/// ```
/// b0 = I polarity (0→I>0, 1→I<0), b1 = Q polarity (0→Q>0, 1→Q<0).
fn qpsk_points() -> Vec<(Complex32, u32)> {
    let d = QPSK_D;
    vec![
        (Complex32::new( d,  d), 0b00),
        (Complex32::new( d, -d), 0b01),
        (Complex32::new(-d,  d), 0b10),
        (Complex32::new(-d, -d), 0b11),
    ]
}

/// Rectangular QAM constellation from two PAM sub-constellations.
///
/// `pam_i` — PAM levels for the I axis with `i_bits` bits.
/// `pam_q` — PAM levels for the Q axis with `n_bits − i_bits` bits.
/// `i_bits` — number of bits for the I axis (MSBs of the combined code word).
fn qam_points(
    pam_i:  &[(f32, u32)],
    pam_q:  &[(f32, u32)],
    i_bits: u32,
) -> Vec<(Complex32, u32)> {
    let q_bits = (pam_q.len() as f32).log2() as u32; // bits for Q axis
    let mut pts = Vec::with_capacity(pam_i.len() * pam_q.len());
    for &(i_val, i_code) in pam_i {
        for &(q_val, q_code) in pam_q {
            let code = (i_code << q_bits) | q_code;
            pts.push((Complex32::new(i_val, q_val), code));
        }
    }
    let _ = i_bits; // used implicitly via left-shift above
    pts
}

/// 4-level Gray-coded PAM: {+3d, +1d, −1d, −3d} → {00, 01, 11, 10}.
fn pam4_gray(d: f32) -> Vec<(f32, u32)> {
    vec![
        ( 3.0 * d, 0b00),
        ( 1.0 * d, 0b01),
        (-1.0 * d, 0b11),
        (-3.0 * d, 0b10),
    ]
}

/// 8-level Gray-coded PAM: {+7d,+5d,+3d,+1d,−1d,−3d,−5d,−7d}
///                        → {000,001,011,010,110,111,101,100}.
fn pam8_gray(d: f32) -> Vec<(f32, u32)> {
    vec![
        ( 7.0 * d, 0b000),
        ( 5.0 * d, 0b001),
        ( 3.0 * d, 0b011),
        ( 1.0 * d, 0b010),
        (-1.0 * d, 0b110),
        (-3.0 * d, 0b111),
        (-5.0 * d, 0b101),
        (-7.0 * d, 0b100),
    ]
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const SNR_DB: f32 = 20.0; // high SNR for deterministic hard-decision tests

    fn snr_to_noise_var(snr_db: f32) -> f32 {
        10.0f32.powf(-snr_db / 10.0)
    }

    // ── Constellation sanity ─────────────────────────────────────────────────

    /// Every constellation must have exactly 2^bps points.
    #[test]
    fn constellation_sizes() {
        for &m in Modulation::all_ordered() {
            let expected = 1usize << m.bits_per_symbol();
            let got      = constellation_points(m).len();
            assert_eq!(got, expected, "{m} should have {expected} points, got {got}");
        }
    }

    /// Every bit position must have exactly half the points with that bit = 0
    /// and half with bit = 1 (balanced Gray code).
    #[test]
    fn constellation_balanced_bits() {
        for &m in Modulation::all_ordered() {
            let pts  = constellation_points(m);
            let bps  = m.bits_per_symbol();
            let half = pts.len() / 2;
            for bit in 0..bps {
                let mask   = 1u32 << (bps - 1 - bit);
                let zeros  = pts.iter().filter(|&&(_, c)| c & mask == 0).count();
                assert_eq!(zeros, half,
                    "{m} bit {bit}: {zeros} zeros ≠ {half}");
            }
        }
    }

    /// Average power of every constellation must be ≈ 1.
    #[test]
    fn constellation_unit_power() {
        for &m in Modulation::all_ordered() {
            let pts  = constellation_points(m);
            let pwr  = pts.iter().map(|(s, _)| s.norm_sqr()).sum::<f32>()
                     / pts.len() as f32;
            assert_abs_diff_eq!(pwr, 1.0, epsilon = 1e-4,
                // message
            );
        }
    }

    // ── LLR sign correctness (noiseless) ─────────────────────────────────────

    /// Transmit every constellation point, check that the LLR signs match the
    /// transmitted bits (noiseless → confident decisions).
    #[test]
    fn llr_sign_all_constellations() {
        let nv = snr_to_noise_var(SNR_DB);
        for &m in Modulation::all_ordered() {
            let pts = constellation_points(m);
            let bps = m.bits_per_symbol();
            for &(s, code) in &pts {
                let llrs = max_log_llr(s, &pts, bps, nv);
                for bit in 0..bps {
                    let mask    = 1u32 << (bps - 1 - bit);
                    let tx_bit  = (code & mask != 0) as u8;
                    let llr_pos = llrs[bit] >= 0.0; // true → decided 0
                    let correct = if tx_bit == 0 { llr_pos } else { !llr_pos };
                    assert!(
                        correct,
                        "{m} bit {bit}: tx={tx_bit}, LLR={:.3}",
                        llrs[bit]
                    );
                }
            }
        }
    }

    /// On the noiseless AWGN channel a transmitted symbol must yield LLRs
    /// all with the correct sign (hard decision = transmitted bits).
    #[test]
    fn demap_bpsk_correct_sign() {
        let nv = snr_to_noise_var(SNR_DB);
        // +1 → bit 0 (LLR > 0)
        let llrs = demap_uniform(&[Complex32::new(1.0, 0.0)], nv, Modulation::Bpsk);
        assert_eq!(llrs.len(), 1);
        assert!(llrs[0] > 0.0, "LLR for +1 must be positive, got {}", llrs[0]);

        // -1 → bit 1 (LLR < 0)
        let llrs = demap_uniform(&[Complex32::new(-1.0, 0.0)], nv, Modulation::Bpsk);
        assert!(llrs[0] < 0.0, "LLR for -1 must be negative, got {}", llrs[0]);
    }

    /// LLR magnitude should scale inversely with noise variance.
    #[test]
    fn llr_scales_with_snr() {
        let sym  = Complex32::new(1.0, 0.0);
        let bps  = Modulation::Bpsk.bits_per_symbol();
        let pts  = constellation_points(Modulation::Bpsk);
        let low_snr  = max_log_llr(sym, &pts, bps, 1.0);
        let high_snr = max_log_llr(sym, &pts, bps, 0.01);
        assert!(
            high_snr[0].abs() > low_snr[0].abs(),
            "higher SNR must yield larger |LLR| magnitude"
        );
    }

    /// Output length = num_symbols × bits_per_symbol.
    #[test]
    fn output_length() {
        let n   = 10usize;
        let syms: Vec<Complex32> = (0..n).map(|_| Complex32::new(1.0, 0.0)).collect();
        let nv  = vec![0.1f32; n];
        for &m in Modulation::all_ordered() {
            let llrs = demap(&syms, &nv, m);
            assert_eq!(llrs.len(), n * m.bits_per_symbol(),
                "{m} output length mismatch");
        }
    }

    // ── QPSK separability ────────────────────────────────────────────────────

    /// QPSK b0 must depend only on I, b1 only on Q.
    #[test]
    fn qpsk_separability() {
        let nv  = snr_to_noise_var(SNR_DB);
        let pts = constellation_points(Modulation::Qpsk);

        // Positive I, positive Q → b0=0, b1=0 → both LLRs > 0
        let y = Complex32::new(QPSK_D, QPSK_D);
        let l = max_log_llr(y, &pts, 2, nv);
        assert!(l[0] > 0.0, "QPSK b0 should be +, got {}", l[0]);
        assert!(l[1] > 0.0, "QPSK b1 should be +, got {}", l[1]);

        // Negative I, positive Q → b0=1, b1=0 → LLR[0]<0, LLR[1]>0
        let y = Complex32::new(-QPSK_D, QPSK_D);
        let l = max_log_llr(y, &pts, 2, nv);
        assert!(l[0] < 0.0, "QPSK b0 should be -, got {}", l[0]);
        assert!(l[1] > 0.0, "QPSK b1 should be +, got {}", l[1]);
    }

    // ── Gray code adjacency ───────────────────────────────────────────────────

    /// Adjacent constellation points (nearest neighbours in Euclidean distance)
    /// must differ in exactly one bit (Gray code property).
    #[test]
    fn gray_code_nearest_neighbor() {
        for &m in &[Modulation::Qpsk, Modulation::Qam16, Modulation::Qam64] {
            let pts = constellation_points(m);
            let bps = m.bits_per_symbol();
            for &(s_a, c_a) in &pts {
                // Find nearest neighbour
                let (_s_b, c_b) = pts.iter()
                    .filter(|&&(s, _)| (s - s_a).norm_sqr() > 1e-8)
                    .min_by(|&&(sa, _), &&(sb, _)| {
                        (sa - s_a).norm_sqr().partial_cmp(&(sb - s_a).norm_sqr()).unwrap()
                    })
                    .copied()
                    .unwrap();
                let hamming = (c_a ^ c_b).count_ones();
                assert_eq!(
                    hamming, 1,
                    "{m}: neighbour of {c_a:0bps$b} is {c_b:0bps$b} (Hamming = {hamming})",
                    bps = bps
                );
            }
        }
    }
}
