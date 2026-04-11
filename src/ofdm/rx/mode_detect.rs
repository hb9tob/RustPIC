//! Mode header decoder.
//!
//! The **mode header** is the OFDM symbol immediately following the two ZC
//! preambles.  All [`NUM_CARRIERS`] = 42 active subcarriers carry BPSK symbols
//! that encode the transmission parameters for the super-frame.
//!
//! # Bit layout  (47 BPSK symbols = 47 bits)
//!
//! ```text
//!  Bits   Field                         Width  Notes
//!  ─────  ──────────────────────────   ─────  ──────────────────────────────────────────
//!   0–2   modulation                     3    see [`Modulation`]  (MSB first)
//!   3–4   ldpc_rate                      2    see [`LdpcRate`]
//!   5–6   rs_level                       2    see [`RsLevel`]  (0=L0, 1=L1, 2=L2)
//!   7     has_resync                     1    1 = periodic re-sync ZC present in frame
//!   8–19  total_packet_count            12    12-bit field (max 4095 RS packets)
//!  20–31  packet_offset                 12    12-bit field (max 4095)
//!  32–46  crc15                         15    lower 15 bits of CRC-16/CCITT over bits 0–31
//! ```
//!
//! The 12-bit `total_packet_count` supports up to 4095 RS packets (≥ 500 KB for all
//! RS levels).  The 12-bit `packet_offset` accommodates all valid frame offsets.
//!
//! # Decoding pipeline
//!
//! ```text
//!  header FFT window (CP stripped)
//!        │
//!        ▼  FFT(1024) → scale
//!  Y[k], k=0..46
//!        │
//!        ▼  ZF equalisation  Y[k] / H_hat[k]
//!  X̂[k]
//!        │
//!        ▼  BPSK hard decision  sign(Re(X̂[k]))
//!  47 bits
//!        │
//!        ▼  field extraction + CRC check
//!  ModeHeader
//! ```
//!
//! # Point-to-multipoint note
//!
//! The system is **simplex / broadcast**: no ACK or ARQ is possible.
//! The receiver decodes whatever it receives.  If the CRC fails the frame
//! is discarded.  The best achievable modulation at the RX can be *displayed*
//! (estimated from pilot SNR) but cannot be fed back to the TX.

use num_complex::Complex32;

use crate::fec::rs::RsLevel;
use crate::ofdm::params::*;
use crate::ofdm::rx::ofdm_demodulate;


// ── Public enumerations ────────────────────────────────────────────────────────

/// Constellation / modulation order used for the data payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Modulation {
    /// 1 bit/symbol.  Most robust, lowest throughput.
    Bpsk  = 0,
    /// 2 bits/symbol.
    Qpsk  = 1,
    /// 4 bits/symbol.
    Qam16 = 2,
    /// 5 bits/symbol.
    Qam32 = 3,
    /// 6 bits/symbol.  Highest throughput, least robust.
    Qam64 = 4,
}

impl Modulation {
    /// Number of coded bits carried by one data subcarrier symbol.
    pub fn bits_per_symbol(self) -> usize {
        match self {
            Self::Bpsk  => 1,
            Self::Qpsk  => 2,
            Self::Qam16 => 4,
            Self::Qam32 => 5,
            Self::Qam64 => 6,
        }
    }

    /// Minimum SNR (dB, approximate) required for reliable demodulation.
    pub fn min_snr_db(self) -> f32 {
        match self {
            Self::Bpsk  =>  4.0,
            Self::Qpsk  =>  7.0,
            Self::Qam16 => 12.0,
            Self::Qam32 => 15.0,
            Self::Qam64 => 18.0,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Bpsk),
            1 => Some(Self::Qpsk),
            2 => Some(Self::Qam16),
            3 => Some(Self::Qam32),
            4 => Some(Self::Qam64),
            _ => None,
        }
    }

    /// All variants ordered from most to least robust.
    pub fn all_ordered() -> &'static [Modulation] {
        &[Self::Bpsk, Self::Qpsk, Self::Qam16, Self::Qam32, Self::Qam64]
    }
}

impl std::fmt::Display for Modulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Bpsk  => "BPSK",
            Self::Qpsk  => "QPSK",
            Self::Qam16 => "16-QAM",
            Self::Qam32 => "32-QAM",
            Self::Qam64 => "64-QAM",
        })
    }
}

/// LDPC code rate used for the data payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LdpcRate {
    /// Rate 1/2 — 50 % information, most protection.
    R1_2 = 0,
    /// Rate 2/3.
    R2_3 = 1,
    /// Rate 3/4.
    R3_4 = 2,
    /// Rate 5/6 — least protection, highest throughput.
    R5_6 = 3,
}

impl LdpcRate {
    /// Returns the code rate as a float (info bits / coded bits).
    pub fn rate(self) -> f32 {
        match self {
            Self::R1_2 => 0.5,
            Self::R2_3 => 2.0 / 3.0,
            Self::R3_4 => 0.75,
            Self::R5_6 => 5.0 / 6.0,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::R1_2),
            1 => Some(Self::R2_3),
            2 => Some(Self::R3_4),
            3 => Some(Self::R5_6),
            _ => None,
        }
    }
}

impl std::fmt::Display for LdpcRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::R1_2 => "1/2",
            Self::R2_3 => "2/3",
            Self::R3_4 => "3/4",
            Self::R5_6 => "5/6",
        })
    }
}

// ── ModeHeader ─────────────────────────────────────────────────────────────────

/// Decoded transmission parameters for the current super-frame.
#[derive(Debug, Clone)]
pub struct ModeHeader {
    /// Constellation used for data subcarriers.
    pub modulation: Modulation,
    /// LDPC code rate.
    pub ldpc_rate: LdpcRate,
    /// Reed-Solomon protection level (determines RS_K, RS_2T, and interleave M).
    pub rs_level: RsLevel,
    /// Whether periodic re-sync ZC symbols are present in the data block.
    pub has_resync: bool,
    /// Total RS packets in the **whole transmission** (across all super-frames).
    /// Encoded as 12-bit value (max 4095).
    pub total_packet_count: u16,
    /// Index of the first RS packet carried by **this** super-frame.
    /// `0` for the first (or only) super-frame.
    /// Encoded as 12-bit value (max 4095).
    pub packet_offset: u16,
    /// CRC-16 check result (`true` = header integrity verified).
    pub crc_ok: bool,
}

impl std::fmt::Display for ModeHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "mode={} ldpc={} rs={} pkts={}/{} resync={} crc={}",
            self.modulation, self.ldpc_rate, self.rs_level,
            self.packet_offset, self.total_packet_count,
            self.has_resync, if self.crc_ok { "OK" } else { "FAIL" }
        )
    }
}

// ── SNR-based modulation advisory (display only, no feedback to TX) ───────────

/// Advisory: given a measured pilot SNR (dB), returns the highest modulation
/// order that *could* be decoded reliably.
///
/// This is for **display only** — the system is simplex, there is no feedback
/// channel to the TX.
pub fn max_decodable_modulation(pilot_snr_db: f32) -> Modulation {
    for &m in Modulation::all_ordered().iter().rev() {
        if pilot_snr_db >= m.min_snr_db() {
            return m;
        }
    }
    Modulation::Bpsk // always possible (or frame will not decode anyway)
}

// ── Errors ─────────────────────────────────────────────────────────────────────

/// Error returned when mode-header decoding fails.
#[derive(Debug, Clone, PartialEq)]
pub enum ModeError {
    /// `fft_window` length ≠ [`FFT_SIZE`].
    BadFftWindowLen { got: usize },
    /// `channel_est` length ≠ [`NUM_CARRIERS`].
    BadChannelEstLen { got: usize },
    /// Decoded modulation code is out of range.
    InvalidModulation(u8),
    /// Decoded LDPC rate code is out of range.
    InvalidLdpcRate(u8),
    /// Decoded RS level code is out of range.
    InvalidRsLevel(u8),
    /// CRC mismatch — header bits are likely corrupted.
    CrcFailed { computed: u16, received: u16 },
}

impl std::fmt::Display for ModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadFftWindowLen { got } =>
                write!(f, "FFT window len {got} ≠ {FFT_SIZE}"),
            Self::BadChannelEstLen { got } =>
                write!(f, "channel_est len {got} ≠ {NUM_CARRIERS}"),
            Self::InvalidModulation(v) =>
                write!(f, "invalid modulation code {v}"),
            Self::InvalidLdpcRate(v) =>
                write!(f, "invalid LDPC rate code {v}"),
            Self::InvalidRsLevel(v) =>
                write!(f, "invalid RS level code {v}"),
            Self::CrcFailed { computed, received } =>
                write!(f, "CRC-16 failed: computed=0x{computed:04X} received=0x{received:04X}"),
        }
    }
}

impl std::error::Error for ModeError {}

// ── Public decode function ─────────────────────────────────────────────────────

/// Decodes the mode-header OFDM symbol.
///
/// # Arguments
///
/// * `fft_window` – [`FFT_SIZE`] time-domain samples of the header symbol with
///   the **cyclic prefix already stripped** (`samples[header_start + CP_LEN ..]`).
/// * `channel_est` – [`NUM_CARRIERS`] complex channel coefficients `H_hat[k]`
///   as returned by [`channel_estimate_from_zc`].
///
/// # Returns
///
/// [`ModeHeader`] on success, [`ModeError`] otherwise.
///
/// [`channel_estimate_from_zc`]: crate::ofdm::rx::sync::channel_estimate_from_zc
pub fn decode_mode_header(
    fft_window:  &[Complex32],
    channel_est: &[Complex32],
) -> Result<ModeHeader, ModeError> {
    decode_mode_header_repeated(std::slice::from_ref(&fft_window), channel_est)
}

/// Decodes the mode header from **one or more** repeated BPSK OFDM symbols.
///
/// When MODE_HEADER_REPEAT > 1, the TX emits several identical copies of the
/// mode-header symbol back-to-back. This routine equalises each copy against
/// the same ZC-derived channel estimate, sums the equalised complex values
/// per carrier (soft combining — gives ~√N SNR gain because the noise is
/// approximately uncorrelated between symbols while the signal adds
/// coherently), then hard-decides the BPSK bits and verifies the CRC-10.
///
/// A single-symbol input is identical to the legacy single-pass decode.
pub fn decode_mode_header_repeated(
    fft_windows: &[&[Complex32]],
    channel_est: &[Complex32],
) -> Result<ModeHeader, ModeError> {
    if fft_windows.is_empty() {
        return Err(ModeError::BadFftWindowLen { got: 0 });
    }
    if channel_est.len() != NUM_CARRIERS {
        return Err(ModeError::BadChannelEstLen { got: channel_est.len() });
    }
    for w in fft_windows {
        if w.len() != FFT_SIZE {
            return Err(ModeError::BadFftWindowLen { got: w.len() });
        }
    }

    // Sum the equalised complex values across all repetitions.
    let mut combined = vec![Complex32::new(0.0, 0.0); NUM_CARRIERS];
    for window in fft_windows {
        let subcarriers = ofdm_demodulate(window);
        for (i, (&y, &h)) in subcarriers.iter().zip(channel_est.iter()).enumerate() {
            combined[i] += zf_equalise(y, h);
        }
    }

    // ── BPSK hard decisions → NUM_CARRIERS bits ───────────────────────────────
    let bits: Vec<u8> = combined.iter()
        .map(|c| u8::from(c.re >= 0.0))
        .collect();

    // ── Field extraction ──────────────────────────────────────────────────────
    let modulation_code    = bits_to_u8(&bits[0..3]);
    let ldpc_rate_code     = bits_to_u8(&bits[3..5]);
    let rs_level_code      = bits_to_u8(&bits[5..7]);
    let has_resync         = bits[7] == 1;
    let total_packet_count = bits_to_u16(&bits[8..20]);   // 12-bit field (max 4095)
    let packet_offset      = bits_to_u16(&bits[20..32]);  // 12-bit field (max 4095)
    let crc_received       = bits_to_u16(&bits[32..42]);  // 10-bit CRC (low 10 of CRC-16)

    let modulation = Modulation::from_u8(modulation_code)
        .ok_or(ModeError::InvalidModulation(modulation_code))?;
    let ldpc_rate = LdpcRate::from_u8(ldpc_rate_code)
        .ok_or(ModeError::InvalidLdpcRate(ldpc_rate_code))?;
    let rs_level = RsLevel::from_u8(rs_level_code)
        .ok_or(ModeError::InvalidRsLevel(rs_level_code))?;

    // ── CRC-10: lower 10 bits of CRC-16/CCITT over bits 0–31 ─────────────────
    let payload_bytes = bits_to_bytes(&bits[0..32]);
    let crc_computed  = crc16_ccitt(&payload_bytes) & 0x03FF;
    let crc_ok        = crc_computed == crc_received;

    if !crc_ok {
        return Err(ModeError::CrcFailed {
            computed: crc_computed,
            received: crc_received,
        });
    }

    Ok(ModeHeader { modulation, ldpc_rate, rs_level, has_resync, total_packet_count, packet_offset, crc_ok })
}

// ── Zero-forcing equaliser ─────────────────────────────────────────────────────

/// Applies a zero-forcing (ZF) single-tap equaliser: `X̂ = Y / H`.
///
/// When `H` is very small (deep fade) the output is set to zero to avoid
/// noise amplification.
#[inline(always)]
fn zf_equalise(y: Complex32, h: Complex32) -> Complex32 {
    let denom = h.norm_sqr();
    if denom > 1e-6 {
        y * h.conj() / denom
    } else {
        Complex32::new(0.0, 0.0) // deep fade — mark as erasure
    }
}

// ── Bit / byte helpers ─────────────────────────────────────────────────────────

/// Packs up to 8 bits (MSB first) into a `u8`.
fn bits_to_u8(bits: &[u8]) -> u8 {
    bits.iter().fold(0u8, |acc, &b| (acc << 1) | (b & 1))
}

/// Packs exactly 16 bits (MSB first) into a `u16`.
fn bits_to_u16(bits: &[u8]) -> u16 {
    bits.iter().fold(0u16, |acc, &b| (acc << 1) | u16::from(b & 1))
}

/// Groups a bit slice into bytes (MSB first), zero-padding the last byte.
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| chunk.iter().fold(0u8, |acc, &b| (acc << 1) | (b & 1)))
        .collect()
}

// ── Mode header encoder (TX side helper — keeps TX/RX in sync) ────────────────

/// Encodes a [`ModeHeader`] into [`NUM_CARRIERS`] BPSK symbols (+1 / −1) ready
/// to be placed on the active subcarriers of the header OFDM symbol.
///
/// Layout: 32 data bits + 15-bit CRC (lower 15 bits of CRC-16/CCITT) = 47 bits.
///
/// This is the **TX** counterpart to [`decode_mode_header`], kept here so the
/// bit layout is defined in a single place.
pub fn encode_mode_header(hdr: &ModeHeader) -> Vec<f32> {
    let mut bits = vec![0u8; NUM_CARRIERS]; // 47 bits

    // Field packing (MSB first)
    pack_bits_u8 (&mut bits[0..3],   hdr.modulation as u8,  3);
    pack_bits_u8 (&mut bits[3..5],   hdr.ldpc_rate  as u8,  2);
    pack_bits_u8 (&mut bits[5..7],   hdr.rs_level   as u8,  2);
    bits[7] = u8::from(hdr.has_resync);
    pack_bits    (&mut bits[8..20],  hdr.total_packet_count, 12);  // 12-bit field
    pack_bits    (&mut bits[20..32], hdr.packet_offset,      12);  // 12-bit field

    // CRC-15: lower 15 bits of CRC-16/CCITT over first 4 bytes (bits 0–31)
    let payload_bytes = bits_to_bytes(&bits[0..32]);
    let crc = crc16_ccitt(&payload_bytes) & 0x03FF;
    pack_bits(&mut bits[32..42], crc, 10);

    // Map 0 → −1, 1 → +1
    bits.iter().map(|&b| if b == 1 { 1.0f32 } else { -1.0f32 }).collect()
}

fn pack_bits(dst: &mut [u8], val: u16, nbits: usize) {
    for i in 0..nbits {
        dst[i] = ((val >> (nbits - 1 - i)) & 1) as u8;
    }
}

fn pack_bits_u8(dst: &mut [u8], val: u8, nbits: usize) {
    pack_bits(dst, val as u16, nbits);
}

// ── CRC-16 / CCITT ────────────────────────────────────────────────────────────

/// CRC-16/CCITT — poly 0x1021, init 0xFFFF, no reflection, no final XOR.
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    const POLY: u16 = 0x1021;
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ POLY
            } else {
                crc << 1
            };
        }
    }
    crc
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::rx::sync::channel_estimate_from_zc;
    use crate::ofdm::zc::{build_preamble, zc_freq_to_time_domain};
    use num_complex::Complex32;

    /// Encode a header, modulate it as BPSK on a flat channel, then decode.
    #[test]
    fn encode_decode_roundtrip() {
        let hdr_tx = ModeHeader {
            modulation:         Modulation::Qam16,
            ldpc_rate:          LdpcRate::R3_4,
            rs_level:           RsLevel::L1,
            has_resync:         true,
            total_packet_count: 42,
            packet_offset:      7,
            crc_ok:             true,
        };

        // Encode → NUM_CARRIERS BPSK symbols (+1/−1)
        let bpsk_symbols = encode_mode_header(&hdr_tx);
        assert_eq!(bpsk_symbols.len(), NUM_CARRIERS);

        // Map to complex subcarriers (BPSK: real axis only)
        let freq_symbols: Vec<Complex32> = bpsk_symbols.iter()
            .map(|&s| Complex32::new(s, 0.0))
            .collect();

        // Build OFDM symbol (IFFT → CP prepend)
        let time_sym = zc_freq_to_time_domain(&freq_symbols);

        // Flat channel H = 1 → channel_est from ZC
        let zc_preamble = build_preamble();
        let zc_fft_win = &zc_preamble[crate::ofdm::params::CP_LEN..]; // strip CP
        let h_est = channel_estimate_from_zc(zc_fft_win);

        // Decode: strip CP of the header symbol
        let hdr_fft_win = &time_sym[CP_LEN..];
        let hdr_rx = decode_mode_header(hdr_fft_win, &h_est)
            .expect("decode must succeed on flat channel");

        assert_eq!(hdr_rx.modulation,         hdr_tx.modulation);
        assert_eq!(hdr_rx.ldpc_rate,          hdr_tx.ldpc_rate);
        assert_eq!(hdr_rx.has_resync,         hdr_tx.has_resync);
        assert_eq!(hdr_rx.rs_level,            hdr_tx.rs_level);
        assert_eq!(hdr_rx.total_packet_count, hdr_tx.total_packet_count);
        assert_eq!(hdr_rx.packet_offset,      hdr_tx.packet_offset);
        assert!(hdr_rx.crc_ok);
    }

    #[test]
    fn crc16_known_vector() {
        // CRC-16/CCITT for ASCII "123456789" = 0x29B1
        let data = b"123456789";
        assert_eq!(crc16_ccitt(data), 0x29B1);
    }

    #[test]
    fn max_decodable_modulation_boundaries() {
        assert_eq!(max_decodable_modulation(3.0),  Modulation::Bpsk);
        assert_eq!(max_decodable_modulation(4.0),  Modulation::Bpsk);
        assert_eq!(max_decodable_modulation(7.0),  Modulation::Qpsk);
        assert_eq!(max_decodable_modulation(12.0), Modulation::Qam16);
        assert_eq!(max_decodable_modulation(15.0), Modulation::Qam32);
        assert_eq!(max_decodable_modulation(18.0), Modulation::Qam64);
        assert_eq!(max_decodable_modulation(25.0), Modulation::Qam64);
    }

    #[test]
    fn all_modulations_encodable() {
        for &m in Modulation::all_ordered() {
            for &r in &[LdpcRate::R1_2, LdpcRate::R2_3, LdpcRate::R3_4, LdpcRate::R5_6] {
                for &rs in &[RsLevel::L0, RsLevel::L1, RsLevel::L2] {
                    let hdr = ModeHeader {
                        modulation: m, ldpc_rate: r,
                        rs_level: rs,
                        has_resync: false,
                        total_packet_count: 100,
                        packet_offset: 0,
                        crc_ok: true,
                    };
                    let syms = encode_mode_header(&hdr);
                    assert_eq!(syms.len(), NUM_CARRIERS);
                    for &s in &syms {
                        assert!(s == 1.0 || s == -1.0, "BPSK symbol must be ±1, got {s}");
                    }
                }
            }
        }
    }
}
