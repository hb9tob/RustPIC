//! OFDM transmitter: bit → constellation → OFDM symbol.
//!
//! Provides the TX counterpart to [`crate::ofdm::rx`]:
//! * [`bits_to_symbol`] — maps `bps` bits (MSB first) to a complex constellation point.
//! * [`ofdm_modulate`]  — IFFT + CP for one data symbol with pilots inserted.
//! * [`ofdm_modulate_all_carriers`] — IFFT + CP for one header/preamble symbol
//!   where all [`NUM_CARRIERS`] active subcarriers carry given values.

pub mod frame;

use num_complex::Complex32;
use rustfft::FftPlanner;

use crate::ofdm::params::*;
use crate::ofdm::rx::{
    demapper::constellation_points,
    mode_detect::Modulation,
};

// ── Bit-to-symbol mapper ───────────────────────────────────────────────────────

/// Maps `modulation.bits_per_symbol()` bits (MSB first) to the corresponding
/// Gray-coded constellation point.
///
/// Bit ordering: `bits[0]` is the MSB of the Gray code word.
///
/// # Panics
///
/// Panics if `bits.len() < modulation.bits_per_symbol()`.
pub fn bits_to_symbol(bits: &[u8], modulation: Modulation) -> Complex32 {
    let bps = modulation.bits_per_symbol();
    debug_assert!(bits.len() >= bps);

    // Build Gray code integer from bits (MSB first)
    let gray: u32 = bits[..bps]
        .iter()
        .fold(0u32, |acc, &b| (acc << 1) | u32::from(b & 1));

    // Look up constellation point
    let pts = constellation_points(modulation);
    pts.iter()
        .find(|&&(_, code)| code == gray)
        .map(|&(s, _)| s)
        .unwrap_or(Complex32::new(0.0, 0.0))
}

// ── OFDM data symbol ──────────────────────────────────────────────────────────

/// Modulates [`NUM_DATA`] complex data subcarrier values into one OFDM symbol.
///
/// Pilots are inserted automatically at positions `k = 0, 8, 16, …, 64` using
/// the PN sequence from [`crate::ofdm::params::pilot_sign`].
///
/// Returns [`SYMBOL_LEN`] time-domain samples (cyclic prefix prepended).
pub fn ofdm_modulate(data_subcarriers: &[Complex32]) -> Vec<Complex32> {
    debug_assert_eq!(data_subcarriers.len(), NUM_DATA);

    let mut freq = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

    let mut data_idx = 0usize;
    for k in 0..NUM_CARRIERS {
        let bin = carrier_to_bin(k);
        if is_pilot(k) {
            freq[bin] = Complex32::new(pilot_sign(k), 0.0);
        } else {
            freq[bin] = data_subcarriers[data_idx];
            data_idx += 1;
        }
    }

    ifft_and_cp(freq)
}

/// Modulates an entire [`NUM_CARRIERS`]-length vector onto all active
/// subcarriers (no pilots — caller provides all values).
///
/// Used for the **mode header** symbol where `encode_mode_header` returns
/// 72 BPSK values for all active subcarriers (pilots and data alike).
///
/// Returns [`SYMBOL_LEN`] time-domain samples (cyclic prefix prepended).
pub fn ofdm_modulate_all_carriers(subcarriers: &[Complex32]) -> Vec<Complex32> {
    debug_assert_eq!(subcarriers.len(), NUM_CARRIERS);

    let mut freq = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    for k in 0..NUM_CARRIERS {
        freq[carrier_to_bin(k)] = subcarriers[k];
    }

    ifft_and_cp(freq)
}

// ── IFFT + cyclic prefix ──────────────────────────────────────────────────────

/// Applies the [`FFT_SIZE`]-point inverse FFT, normalises, prepends the cyclic
/// prefix, and returns [`SYMBOL_LEN`] complex samples.
fn ifft_and_cp(mut freq: Vec<Complex32>) -> Vec<Complex32> {
    debug_assert_eq!(freq.len(), FFT_SIZE);

    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_inverse(FFT_SIZE).process(&mut freq);
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();
    let time: Vec<Complex32> = freq.iter().map(|s| s * scale).collect();

    // Cyclic prefix = last CP_LEN samples of the time-domain symbol
    let mut sym = time[FFT_SIZE - CP_LEN..].to_vec();
    sym.extend_from_slice(&time);
    sym
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// bits_to_symbol must be the inverse of the demapper's hard-decision:
    /// for every constellation point, the recovered bits must equal the TX bits.
    #[test]
    fn bits_to_symbol_roundtrip() {
        use crate::ofdm::rx::demapper::demap_uniform;

        for &m in Modulation::all_ordered() {
            let bps = m.bits_per_symbol();
            let pts = constellation_points(m);

            for &(s, gray_code) in &pts {
                // Bits from Gray code (MSB first)
                let mut tx_bits = vec![0u8; bps];
                for i in 0..bps {
                    tx_bits[i] = ((gray_code >> (bps - 1 - i)) & 1) as u8;
                }

                // TX: bits → symbol
                let sym = bits_to_symbol(&tx_bits, m);
                assert_abs_diff_eq!(sym.re, s.re, epsilon = 1e-5);
                assert_abs_diff_eq!(sym.im, s.im, epsilon = 1e-5);

                // RX: symbol → LLR → hard bits
                let llrs = demap_uniform(&[sym], 1e-4, m);
                for (bit_idx, &llr) in llrs.iter().enumerate() {
                    let hard = u8::from(llr < 0.0);
                    assert_eq!(hard, tx_bits[bit_idx],
                        "{m} bit {bit_idx}: tx={} decoded={}",
                        tx_bits[bit_idx], hard);
                }
            }
        }
    }

    /// After ofdm_modulate → ofdm_demodulate (RX), the data subcarriers
    /// must be recovered exactly on a flat channel.
    #[test]
    fn tx_rx_ofdm_symbol_roundtrip() {
        use crate::ofdm::rx::ofdm_demodulate;

        let data: Vec<Complex32> = (0..NUM_DATA)
            .map(|i| Complex32::new((i % 4) as f32 * 0.5 - 0.75, 0.0))
            .collect();

        let sym = ofdm_modulate(&data);
        assert_eq!(sym.len(), SYMBOL_LEN);

        // Strip CP and demodulate
        let rx_carriers = ofdm_demodulate(&sym[CP_LEN..]);
        assert_eq!(rx_carriers.len(), NUM_CARRIERS);

        // Check data subcarriers (skip pilots)
        let mut data_idx = 0;
        for k in 0..NUM_CARRIERS {
            if !is_pilot(k) {
                assert_abs_diff_eq!(rx_carriers[k].re, data[data_idx].re, epsilon = 1e-4);
                assert_abs_diff_eq!(rx_carriers[k].im, data[data_idx].im, epsilon = 1e-4);
                data_idx += 1;
            }
        }
    }

    /// ofdm_modulate_all_carriers → ofdm_demodulate must recover all carriers.
    #[test]
    fn all_carriers_roundtrip() {
        use crate::ofdm::rx::ofdm_demodulate;

        let carriers: Vec<Complex32> = (0..NUM_CARRIERS)
            .map(|i| Complex32::new(if i % 2 == 0 { 1.0 } else { -1.0 }, 0.0))
            .collect();

        let sym = ofdm_modulate_all_carriers(&carriers);
        let rx  = ofdm_demodulate(&sym[CP_LEN..]);

        for k in 0..NUM_CARRIERS {
            assert_abs_diff_eq!(rx[k].re, carriers[k].re, epsilon = 1e-4,
                // message
            );
        }
    }
}
