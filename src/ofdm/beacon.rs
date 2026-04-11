//! Beacon frame — prepended to every RustPIC transmission.
//!
//! Structure (all at 8 kHz baseband):
//! ```text
//! ┌─────────────────────┬──────┬──────┬─────────────────┐
//! │  Tone 1 kHz ×10 sym │ ZC#1 │ ZC#2 │ ANN × 5 symbols │  → data frames
//! └─────────────────────┴──────┴──────┴─────────────────┘
//!   360 ms               36 ms  36 ms   180 ms
//!   (VOX activation)     (ZcCorrelator detects this pair)
//!                                       (callsign + mode + filename)
//! ```
//!
//! Total: 17 × SYMBOL_LEN samples = 612 ms.
//! The data ZC pair begins at `beacon_preamble_pos + 7 × SYMBOL_LEN`.
//!
//! ## ANN bit layout (5 × 72 bits = 360 bits, MSB first across all symbols)
//! ```text
//! bits   0-15 : magic  0xDA1D
//! bits  16-23 : text byte length N  (max 40)
//! bits  24-343: text (N bytes UTF-8, zero-padded to 40 bytes = 320 bits)
//! bits 344-359: CRC-16/CCITT of (magic_2B + len_1B + text_NB)
//! ```
//!
//! The 1 kHz tone activates VOX-switched transmitters without triggering the
//! ZC correlator (no ZC autocorrelation structure).

use num_complex::Complex32;
use std::f32::consts::PI;

use crate::ofdm::{
    params::{CP_LEN, FFT_SIZE, NUM_CARRIERS, SAMPLE_RATE, SYMBOL_LEN, carrier_to_bin},
    rx::mode_detect::crc16_ccitt,
    zc::build_preamble,
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of 1 kHz tone symbols before the ZC pair.
pub const BEACON_TONE_SYMS: usize = 10;

/// Number of BPSK OFDM symbols carrying the announcement text.
/// 8 × NUM_CARRIERS = 8 × 47 = 376 bits ≥ 360 bits required (magic+len+text+crc).
pub const BEACON_ANN_SYMS: usize = 8;

/// Total symbols in the beacon block (tone + 2×ZC + ANN).
pub const BEACON_TOTAL_SYMS: usize = BEACON_TONE_SYMS + 2 + BEACON_ANN_SYMS; // 17

/// Number of symbols to skip from the beacon ZC pair position to reach the
/// data ZC pair (2 ZC + 5 ANN).
pub const BEACON_SKIP_TO_DATA: usize = 2 + BEACON_ANN_SYMS; // 7

const BEACON_MAGIC: u16 = 0xDA1D;
const BEACON_TEXT_BYTES: usize = 40; // max announcement text length
const TONE_HZ: f32 = 1000.0;

// ── Beacon info ───────────────────────────────────────────────────────────────

/// Information decoded from a beacon frame.
#[derive(Debug, Clone)]
pub struct BeaconInfo {
    /// Human-readable announcement: "DE <CALL> <MODE> <FILE>"
    pub text: String,
}

// ── TX — build ────────────────────────────────────────────────────────────────

/// Builds the complete beacon block (BEACON_TOTAL_SYMS × SYMBOL_LEN samples).
///
/// `callsign`, `filename` and `mode_str` are combined into the announcement:
/// `"DE {callsign} {mode_str} {filename}"`, truncated to 40 UTF-8 bytes.
pub fn build_beacon(callsign: &str, filename: &str, mode_str: &str) -> Vec<Complex32> {
    let text = format!("DE {callsign} {mode_str} {filename}");
    let text_bytes = text.as_bytes();
    let n = text_bytes.len().min(BEACON_TEXT_BYTES);

    // ── Pack bits: magic(16) + len(8) + text(320) + crc(16) = 360 bits ────────
    let mut crc_input = Vec::with_capacity(3 + n);
    crc_input.push((BEACON_MAGIC >> 8) as u8);
    crc_input.push(BEACON_MAGIC as u8);
    crc_input.push(n as u8);
    crc_input.extend_from_slice(&text_bytes[..n]);
    let crc = crc16_ccitt(&crc_input);

    let mut bits = [0u8; BEACON_ANN_SYMS * NUM_CARRIERS]; // 360 bits
    let mut pos = 0usize;

    let push_byte = |bits: &mut [u8], pos: &mut usize, byte: u8| {
        for b in 0..8 {
            bits[*pos] = (byte >> (7 - b)) & 1;
            *pos += 1;
        }
    };

    push_byte(&mut bits, &mut pos, (BEACON_MAGIC >> 8) as u8);
    push_byte(&mut bits, &mut pos, BEACON_MAGIC as u8);
    push_byte(&mut bits, &mut pos, n as u8);
    for i in 0..BEACON_TEXT_BYTES {
        push_byte(&mut bits, &mut pos, if i < n { text_bytes[i] } else { 0 });
    }
    push_byte(&mut bits, &mut pos, (crc >> 8) as u8);
    push_byte(&mut bits, &mut pos, crc as u8);
    // remaining bits stay 0 (pos should be exactly 360 here)

    // ── 1 kHz tone symbols ────────────────────────────────────────────────────
    let tone_samples = BEACON_TONE_SYMS * SYMBOL_LEN;
    let mut out = Vec::with_capacity(BEACON_TOTAL_SYMS * SYMBOL_LEN);
    for i in 0..tone_samples {
        let s = (2.0 * PI * TONE_HZ * i as f32 / SAMPLE_RATE).sin();
        out.push(Complex32::new(s, 0.0));
    }

    // ── ZC#1 + ZC#2 (beacon preamble — identical symbols like in data frames) ──
    let preamble = build_preamble();
    out.extend_from_slice(&preamble); // ZC#1
    out.extend_from_slice(&preamble); // ZC#2

    // ── ANN symbols (BPSK on all 72 carriers) ─────────────────────────────────
    for sym_idx in 0..BEACON_ANN_SYMS {
        let carriers: Vec<Complex32> = (0..NUM_CARRIERS).map(|k| {
            let bit = bits[sym_idx * NUM_CARRIERS + k];
            Complex32::new(if bit == 1 { 1.0 } else { -1.0 }, 0.0)
        }).collect();
        let sym = ofdm_modulate_all_carriers_local(&carriers);
        out.extend_from_slice(&sym);
    }

    out
}

// ── RX — decode ───────────────────────────────────────────────────────────────

/// Tries to decode a beacon announcement from the symbols that follow the
/// detected ZC pair.
///
/// `ann_samples` must be exactly `BEACON_ANN_SYMS × SYMBOL_LEN` samples
/// (the raw time-domain samples of the 5 ANN symbols, CP included).
/// `channel_est` is the `NUM_CARRIERS`-length channel estimate from the
/// preceding ZC pair (same as passed to `decode_mode_header`).
///
/// Returns `None` if the magic or CRC is wrong.
pub fn try_decode_beacon(
    ann_samples:  &[Complex32],
    channel_est:  &[Complex32],
) -> Option<BeaconInfo> {
    if ann_samples.len() < BEACON_ANN_SYMS * SYMBOL_LEN { return None; }
    if channel_est.len() != NUM_CARRIERS { return None; }

    // Demodulate and equalize each ANN symbol → hard bit decisions.
    let mut bits = [0u8; BEACON_ANN_SYMS * NUM_CARRIERS];
    for sym_idx in 0..BEACON_ANN_SYMS {
        let sym_start = sym_idx * SYMBOL_LEN;
        let fft_win = &ann_samples[sym_start + CP_LEN..sym_start + SYMBOL_LEN];
        let subcarriers = ofdm_demodulate_local(fft_win);
        for k in 0..NUM_CARRIERS {
            let eq = zf_eq(subcarriers[k], channel_est[k]);
            bits[sym_idx * NUM_CARRIERS + k] = u8::from(eq.re >= 0.0);
        }
    }

    // Unpack header fields.
    let magic = bits_to_u16(&bits[0..16]);
    if magic != BEACON_MAGIC { return None; }

    let n     = bits_to_u8(&bits[16..24]) as usize;
    if n > BEACON_TEXT_BYTES { return None; }

    let text_start = 24;
    let text_end   = text_start + BEACON_TEXT_BYTES * 8;
    let crc_start  = text_end;

    let mut text_bytes = [0u8; BEACON_TEXT_BYTES];
    for i in 0..BEACON_TEXT_BYTES {
        text_bytes[i] = bits_to_u8(&bits[text_start + i * 8..text_start + i * 8 + 8]);
    }
    let crc_rx = bits_to_u16(&bits[crc_start..crc_start + 16]);

    // Verify CRC.
    let mut crc_input = Vec::with_capacity(3 + n);
    crc_input.push((BEACON_MAGIC >> 8) as u8);
    crc_input.push(BEACON_MAGIC as u8);
    crc_input.push(n as u8);
    crc_input.extend_from_slice(&text_bytes[..n]);
    let crc_calc = crc16_ccitt(&crc_input);
    if crc_rx != crc_calc { return None; }

    let text = String::from_utf8_lossy(&text_bytes[..n]).into_owned();
    Some(BeaconInfo { text })
}

// ── Local DSP helpers (duplicated to avoid circular deps) ─────────────────────

fn ofdm_modulate_all_carriers_local(subcarriers: &[Complex32]) -> Vec<Complex32> {
    use rustfft::FftPlanner;
    debug_assert_eq!(subcarriers.len(), NUM_CARRIERS);
    let mut freq = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    for k in 0..NUM_CARRIERS {
        freq[carrier_to_bin(k)] = subcarriers[k];
    }
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_inverse(FFT_SIZE).process(&mut freq);
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();
    let time: Vec<Complex32> = freq.iter().map(|s| s * scale).collect();
    let mut sym = time[FFT_SIZE - CP_LEN..].to_vec();
    sym.extend_from_slice(&time);
    sym
}

fn ofdm_demodulate_local(fft_window: &[Complex32]) -> Vec<Complex32> {
    use rustfft::FftPlanner;
    debug_assert_eq!(fft_window.len(), FFT_SIZE);
    let mut buf = fft_window.to_vec();
    FftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE).process(&mut buf);
    let scale = 1.0 / (FFT_SIZE as f32).sqrt();
    (0..NUM_CARRIERS).map(|k| buf[carrier_to_bin(k)] * scale).collect()
}

#[inline]
fn zf_eq(y: Complex32, h: Complex32) -> Complex32 {
    let h2 = h.norm_sqr();
    if h2 < 1e-12 { y } else { Complex32::new((y * h.conj()).re / h2, (y * h.conj()).im / h2) }
}

fn bits_to_u8(bits: &[u8]) -> u8 {
    bits.iter().take(8).fold(0u8, |acc, &b| (acc << 1) | (b & 1))
}

fn bits_to_u16(bits: &[u8]) -> u16 {
    bits.iter().take(16).fold(0u16, |acc, &b| (acc << 1) | u16::from(b & 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::params::SYMBOL_LEN;

    #[test]
    fn beacon_encode_decode_loopback() {
        let beacon = build_beacon("HB9TOB", "photo.jpg", "QPSK R3/4");

        // Find where the ZC pair and ANN symbols start
        let zc_start = BEACON_TONE_SYMS * SYMBOL_LEN;
        let ann_start = zc_start + 2 * SYMBOL_LEN;
        let ann_end   = ann_start + BEACON_ANN_SYMS * SYMBOL_LEN;

        // Derive a channel estimate from ZC#1 (as the real RX would)
        use crate::ofdm::rx::sync::channel_estimate_from_zc;
        use crate::ofdm::params::CP_LEN;
        let zc1_fft = &beacon[zc_start + CP_LEN..zc_start + SYMBOL_LEN];
        let ch_est = channel_estimate_from_zc(zc1_fft);

        let info = try_decode_beacon(&beacon[ann_start..ann_end], &ch_est)
            .expect("beacon decode failed");
        assert!(info.text.contains("HB9TOB"), "callsign missing: {}", info.text);
        assert!(info.text.contains("photo.jpg"), "filename missing: {}", info.text);
        println!("Decoded: {}", info.text);
    }
}
