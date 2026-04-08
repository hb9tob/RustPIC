//! Full super-frame receiver: sync → CFO correction → equalizer → demapper
//! → LDPC → RS → EOT/CRC32.
//!
//! # Usage
//!
//! ```rust,ignore
//! // 1. Locate the frame in the sample buffer
//! let corr   = ZcCorrelator::new(0.45, 0.35);
//! let sync   = corr.find_sync(&samples)?;
//! correct_cfo(&mut samples[sync.preamble_start..], sync.cfo_hz);
//!
//! // 2. Decode the mode header
//! let hdr_win = &samples[sync.header_start + CP_LEN
//!                         ..sync.header_start + SYMBOL_LEN];
//! let header  = decode_mode_header(hdr_win, &sync.channel_est)?;
//!
//! // 3. Set up the frame receiver
//! let mut rx  = FrameReceiver::new(&header, &sync.channel_est);
//!
//! // 4. Feed OFDM symbols from the data region (one at a time, SYMBOL_LEN samples each)
//! let mut pos = sync.header_start + SYMBOL_LEN;
//! loop {
//!     let sym = &samples[pos..pos + SYMBOL_LEN];
//!     match rx.push_symbol(sym) {
//!         PushResult::NeedMore             => {}
//!         PushResult::FrameComplete(frame) => { /* use frame.payload */ break; }
//!         PushResult::Error(e)             => { eprintln!("{e}"); break; }
//!     }
//!     pos += SYMBOL_LEN;
//! }
//! ```
//!
//! # Packet structure
//!
//! ```text
//!  payload bytes
//!       │  (RS_K = 191 bytes per RS packet)
//!       ▼
//!  RS(255,191) encoder → codeword[255 bytes = 2040 bits]
//!       │  (packet_count RS packets in the super-frame)
//!       ▼
//!  LDPC encoder  N=252 bits per block
//!       │  (⌈2040/K⌉ LDPC blocks per RS packet)
//!       ▼
//!  OFDM modulation (NUM_DATA=63 data subcarriers)
//!       │
//!       ▼  (one re-sync ZC every RESYNC_PERIOD=12 data symbols if has_resync)
//!  EOT BPSK symbol — CRC32 in first 32 data subcarriers (MSB first)
//! ```
//!
//! # Info-bit convention
//!
//! For all four LDPC rates, **bits `0..k` are the information bits** and
//! `bits k..n` are the parity bits.  The TX encoder places the payload in
//! bit positions `0..k`; the RX extracts the same positions after BP decoding.

use num_complex::Complex32;

use crate::fec::ldpc::{LdpcCode, LdpcDecoder};
use crate::fec::rs::{RsCodec, RS_K, RS_N};
use crate::ofdm::params::*;
use crate::ofdm::rx::{
    demapper::demap,
    equalizer::SymbolEqualizer,
    mode_detect::{ModeHeader, Modulation},
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// LDPC codeword length (bits) — same for all four rates.
const LDPC_N: usize = 252;

/// RS codeword length in bits.
const RS_CODEWORD_BITS: usize = RS_N * 8; // 255 × 8 = 2040

// ── Helper: LDPC blocks per RS codeword ───────────────────────────────────────

/// Returns how many LDPC blocks (each carrying `k` info bits) are needed to
/// carry exactly one RS codeword of [`RS_CODEWORD_BITS`] bits.
///
/// The last block may carry fewer than `k` payload bits — the tail is padding.
fn ldpc_blocks_per_rs(k: usize) -> usize {
    (RS_CODEWORD_BITS + k - 1) / k
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Result returned by [`FrameReceiver::push_symbol`].
pub enum PushResult {
    /// More symbols needed.
    NeedMore,
    /// All packets received and EOT processed.
    FrameComplete(FrameResult),
    /// Unrecoverable error.
    Error(FrameError),
}

/// Decoded super-frame payload.
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Concatenated RS-decoded payload bytes.
    ///
    /// Length = `packet_count × RS_K` (with failed packets zero-filled).
    pub payload: Vec<u8>,

    /// Number of RS packets that decoded without error.
    pub packets_ok: u16,

    /// `true` if the EOT CRC32 matches the assembled payload.
    pub crc32_ok: bool,

    /// Average pilot SNR over all received data symbols (dB).
    pub pilot_snr_db: f32,
}

/// Error returned when the frame cannot be completed.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameError {
    /// The sample buffer is too short to hold the expected number of symbols.
    UnexpectedEndOfData,
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEndOfData =>
                write!(f, "frame receiver: buffer ended before all symbols were received"),
        }
    }
}

impl std::error::Error for FrameError {}

// ── FrameReceiver ─────────────────────────────────────────────────────────────

/// Stateful receiver for one super-frame.
///
/// Initialised from the [`ModeHeader`] and channel estimate obtained by
/// [`crate::ofdm::rx::sync::ZcCorrelator`] / [`crate::ofdm::rx::mode_detect::decode_mode_header`].
/// The caller feeds OFDM symbols one at a time with [`push_symbol`].
///
/// [`push_symbol`]: FrameReceiver::push_symbol
pub struct FrameReceiver {
    // ── Configuration ─────────────────────────────────────────────────────────
    header:            ModeHeader,
    ldpc_k:            usize,
    blocks_per_rs:     usize,
    total_ldpc_blocks: usize,
    total_data_syms:   usize,

    // ── Processing units ──────────────────────────────────────────────────────
    equalizer: SymbolEqualizer,
    code:      LdpcCode,
    rs:        RsCodec,

    // ── Accumulation buffers ──────────────────────────────────────────────────
    /// Raw LLRs from the demapper, waiting for a complete LDPC block.
    llr_buf: Vec<f32>,

    /// LDPC info bits accumulated for the current RS packet.
    /// Flushed into `rs_payload` after every `blocks_per_rs` LDPC blocks.
    info_bits: Vec<u8>,

    /// Decoded payload bytes (RS_K per successful RS packet, 0-filled otherwise).
    rs_payload: Vec<u8>,

    // ── Symbol / block counters ────────────────────────────────────────────────
    /// Total data symbols fed so far (re-sync ZC symbols not counted).
    data_syms_received: usize,

    /// Position within the current RESYNC_PERIOD group (reset at each re-sync ZC).
    syms_in_group: usize,

    /// Number of LDPC blocks decoded (across all RS packets).
    ldpc_blocks_decoded: usize,

    /// RS packets that decoded successfully.
    rs_packets_ok: u16,

    // ── SNR bookkeeping ────────────────────────────────────────────────────────
    pilot_snr_sum:   f32,
    pilot_snr_count: usize,
}

impl FrameReceiver {
    /// Creates a new `FrameReceiver` for the super-frame described by `header`.
    ///
    /// * `header`      — decoded mode header (modulation, LDPC rate, packet_count, …).
    /// * `channel_est` — [`NUM_CARRIERS`] complex channel coefficients from the ZC preamble.
    /// * `alpha`       — EMA tracking coefficient for the channel equalizer (0.05 … 0.3).
    pub fn new(header: &ModeHeader, channel_est: &[Complex32], alpha: f32) -> Self {
        let code         = LdpcCode::for_rate(header.ldpc_rate);
        let ldpc_k       = code.k;
        let blocks_per_rs = ldpc_blocks_per_rs(ldpc_k);
        let total_ldpc_blocks = header.packet_count as usize * blocks_per_rs;
        let bits_per_ofdm    = NUM_DATA * header.modulation.bits_per_symbol();
        let total_coded_bits = total_ldpc_blocks * LDPC_N;
        // Ceiling division: may produce slightly more bits than needed (padding).
        let total_data_syms  = (total_coded_bits + bits_per_ofdm - 1) / bits_per_ofdm;

        Self {
            header:            header.clone(),
            ldpc_k,
            blocks_per_rs,
            total_ldpc_blocks,
            total_data_syms,
            equalizer: SymbolEqualizer::from_preamble(channel_est, alpha),
            code,
            rs:        RsCodec::new(),
            llr_buf:   Vec::new(),
            info_bits: Vec::new(),
            rs_payload: Vec::new(),
            data_syms_received:  0,
            syms_in_group:       0,
            ldpc_blocks_decoded: 0,
            rs_packets_ok:       0,
            pilot_snr_sum:       0.0,
            pilot_snr_count:     0,
        }
    }

    /// Total number of OFDM symbols this receiver expects before returning
    /// [`PushResult::FrameComplete`].
    ///
    /// Includes data symbols, optional re-sync ZC symbols, and the EOT symbol.
    pub fn expected_symbol_count(&self) -> usize {
        let resync_syms = if self.header.has_resync {
            self.total_data_syms / RESYNC_PERIOD
        } else {
            0
        };
        self.total_data_syms + resync_syms + 1 // +1 for EOT
    }

    // ── Main entry point ──────────────────────────────────────────────────────

    /// Processes one OFDM symbol (`SYMBOL_LEN` samples, CP included).
    ///
    /// Returns [`PushResult::NeedMore`] until the last symbol (EOT) is received,
    /// at which point [`PushResult::FrameComplete`] is returned.
    pub fn push_symbol(&mut self, ofdm_symbol: &[Complex32]) -> PushResult {
        debug_assert_eq!(ofdm_symbol.len(), SYMBOL_LEN);

        // ── All data symbols received → this is the EOT ───────────────────────
        if self.data_syms_received >= self.total_data_syms {
            return self.process_eot(ofdm_symbol);
        }

        // ── Re-sync ZC guard ─────────────────────────────────────────────────
        if self.header.has_resync && self.syms_in_group == RESYNC_PERIOD {
            self.equalizer.resync_from_zc(ofdm_symbol);
            self.syms_in_group = 0;
            return PushResult::NeedMore;
        }

        // ── Regular data symbol ───────────────────────────────────────────────
        let eq = self.equalizer.process(ofdm_symbol);
        self.pilot_snr_sum   += eq.pilot_snr_db;
        self.pilot_snr_count += 1;

        let llrs = demap(&eq.data, &eq.noise_var, self.header.modulation);
        self.llr_buf.extend_from_slice(&llrs);
        self.data_syms_received += 1;
        self.syms_in_group      += 1;

        self.drain_ldpc_blocks();

        PushResult::NeedMore
    }

    // ── LDPC block draining ───────────────────────────────────────────────────

    /// Decodes complete LDPC blocks from `self.llr_buf` and accumulates their
    /// info bits.  When `blocks_per_rs` blocks are ready, decodes the RS packet.
    fn drain_ldpc_blocks(&mut self) {
        while self.llr_buf.len() >= LDPC_N
            && self.ldpc_blocks_decoded < self.total_ldpc_blocks
        {
            // Consume exactly LDPC_N LLRs
            let block_llrs: Vec<f32> = self.llr_buf.drain(..LDPC_N).collect();

            // Decode LDPC — decoder borrows &self.code in its own scope
            let decoded_bits: Vec<u8> = {
                let decoder = LdpcDecoder::new(&self.code, 50, 0.75);
                decoder.decode(&block_llrs).bits
            };

            // Extract info bits (convention: bits[0..k])
            self.info_bits.extend_from_slice(&decoded_bits[..self.ldpc_k]);
            self.ldpc_blocks_decoded += 1;

            // When a complete set of blocks covers one RS codeword, decode RS
            if self.ldpc_blocks_decoded % self.blocks_per_rs == 0 {
                self.decode_rs_packet();
            }
        }
    }

    // ── RS packet decoding ────────────────────────────────────────────────────

    /// Packs `blocks_per_rs × ldpc_k` accumulated info bits into an RS codeword
    /// and decodes it.  On success, appends RS_K payload bytes.  On failure,
    /// appends RS_K zero bytes (erasure placeholder).
    fn decode_rs_packet(&mut self) {
        let total_info_bits = self.blocks_per_rs * self.ldpc_k;
        debug_assert!(self.info_bits.len() >= total_info_bits);

        // Drain all bits contributed by the last blocks_per_rs LDPC blocks
        let bits: Vec<u8> = self.info_bits.drain(..total_info_bits).collect();

        // Pack the first RS_CODEWORD_BITS bits into RS_N bytes (MSB first)
        let mut codeword = [0u8; RS_N];
        for (byte_idx, byte) in codeword.iter_mut().enumerate() {
            for bit_pos in 0..8usize {
                let bit_idx = byte_idx * 8 + bit_pos;
                if bit_idx < RS_CODEWORD_BITS && bit_idx < bits.len() && bits[bit_idx] != 0 {
                    *byte |= 1 << (7 - bit_pos);
                }
            }
        }

        // RS decode (no erasures — we decode optimistically; RS handles errors)
        match self.rs.decode(&codeword, &[]) {
            Ok(info_bytes) => {
                self.rs_payload.extend_from_slice(&info_bytes);
                self.rs_packets_ok += 1;
            }
            Err(_) => {
                // RS failed: zero-fill so the payload slice remains aligned
                self.rs_payload
                    .extend(std::iter::repeat(0u8).take(RS_K));
            }
        }
    }

    // ── EOT symbol ────────────────────────────────────────────────────────────

    /// Processes the end-of-transmission OFDM symbol.
    ///
    /// The EOT symbol is always BPSK.  The CRC32 of the assembled payload is
    /// transmitted in the first 32 data subcarrier positions (MSB first).
    fn process_eot(&mut self, ofdm_symbol: &[Complex32]) -> PushResult {
        // Safety flush: complete any pending LDPC block with zero-padding
        while self.ldpc_blocks_decoded < self.total_ldpc_blocks {
            while self.llr_buf.len() < LDPC_N {
                self.llr_buf.push(0.0); // neutral LLR (50/50)
            }
            self.drain_ldpc_blocks();
        }

        // Equalize the EOT symbol with BPSK demapper
        let eq      = self.equalizer.process(ofdm_symbol);
        let eot_llr = demap(&eq.data, &eq.noise_var, Modulation::Bpsk);

        // Hard-decision on the first 32 subcarriers → 32-bit CRC
        let received_crc = {
            let mut acc = 0u32;
            for (i, &llr) in eot_llr.iter().take(32).enumerate() {
                let bit = u32::from(llr < 0.0); // LLR < 0 → bit 1
                acc |= bit << (31 - i);
            }
            acc
        };

        let computed_crc = crc32_ieee(&self.rs_payload);
        let crc32_ok = received_crc == computed_crc;

        let pilot_snr_db = if self.pilot_snr_count > 0 {
            self.pilot_snr_sum / self.pilot_snr_count as f32
        } else {
            0.0
        };

        PushResult::FrameComplete(FrameResult {
            payload:    std::mem::take(&mut self.rs_payload),
            packets_ok: self.rs_packets_ok,
            crc32_ok,
            pilot_snr_db,
        })
    }
}

// ── CRC-32/ISO-HDLC (IEEE 802.3) ─────────────────────────────────────────────

/// CRC-32/ISO-HDLC: poly 0xEDB88320 (bit-reflected), init 0xFFFFFFFF,
/// input/output reflected, final XOR 0xFFFFFFFF.
///
/// This is the standard Ethernet / zlib / PNG CRC-32.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 { crc = (crc >> 1) ^ POLY; }
            else             { crc >>= 1; }
        }
    }
    !crc
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::{
        rx::{
            mode_detect::{LdpcRate, ModeHeader, Modulation},
            sync::channel_estimate_from_zc,
        },
        zc::build_preamble,
    };
    use rustfft::FftPlanner;

    // ── Helper: build a modulated OFDM data symbol from a bit slice ───────────

    /// Maps LDPC-coded bits to OFDM symbols and returns SYMBOL_LEN time-domain
    /// samples.  Uses BPSK with pilots inserted.
    fn bits_to_ofdm_symbol(bits: &[u8], bps: usize) -> Vec<Complex32> {
        // Build the frequency-domain vector: pilots + data subcarriers
        let mut freq = vec![Complex32::new(0.0, 0.0); FFT_SIZE];

        // Insert pilots
        for k in 0..NUM_CARRIERS {
            if is_pilot(k) {
                let s = pilot_sign(k);
                freq[carrier_to_bin(k)] = Complex32::new(s, 0.0);
            }
        }

        // Map bits to data subcarriers (BPSK-only for simplicity)
        // Bits must be aligned to bps per data subcarrier
        let mut data_idx = 0usize;
        let mut bit_cursor = 0usize;
        for k in 0..NUM_CARRIERS {
            if is_pilot(k) { continue; }
            if bit_cursor + bps <= bits.len() {
                // BPSK: use first bit only (LLR sign test)
                let b = bits[bit_cursor];
                freq[carrier_to_bin(k)] = Complex32::new(if b == 0 { 1.0 } else { -1.0 }, 0.0);
            } else {
                freq[carrier_to_bin(k)] = Complex32::new(1.0, 0.0); // padding
            }
            bit_cursor += bps;
            data_idx += 1;
        }
        let _ = data_idx;

        // IFFT
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_inverse(FFT_SIZE).process(&mut freq);
        let scale = 1.0 / (FFT_SIZE as f32).sqrt();
        let time: Vec<Complex32> = freq.iter().map(|s| s * scale).collect();

        // Prepend CP
        let mut sym = time[FFT_SIZE - CP_LEN..].to_vec();
        sym.extend_from_slice(&time);
        sym
    }

    /// Flat unit channel: channel_est from the real ZC preamble (H = 1 everywhere).
    fn flat_channel_est() -> Vec<Complex32> {
        let preamble = build_preamble();
        channel_estimate_from_zc(&preamble[CP_LEN..])
    }

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn ldpc_blocks_per_rs_sensible() {
        for &rate in &[LdpcRate::R1_2, LdpcRate::R2_3, LdpcRate::R3_4, LdpcRate::R5_6] {
            let code = LdpcCode::for_rate(rate);
            let b    = ldpc_blocks_per_rs(code.k);
            assert!(b >= 1, "must need at least 1 LDPC block per RS codeword");
            let carried = b * code.k;
            assert!(carried >= RS_CODEWORD_BITS,
                "rate {rate}: {b} blocks × k={} = {carried} bits < {RS_CODEWORD_BITS}",
                code.k);
            // The excess (padding) must be less than one full LDPC block
            assert!(carried - RS_CODEWORD_BITS < code.k,
                "rate {rate}: excess {excess} >= k={k}",
                excess = carried - RS_CODEWORD_BITS, k = code.k);
        }
    }

    #[test]
    fn expected_symbol_count_no_resync() {
        let header = ModeHeader {
            modulation:   Modulation::Qam16,   // 4 bps → 252 bits/symbol
            ldpc_rate:    LdpcRate::R3_4,      // k=189, blocks_per_rs=11
            has_resync:   false,
            packet_count: 1,
            crc_ok:       true,
        };
        let code = LdpcCode::for_rate(header.ldpc_rate);
        let h_est = flat_channel_est();
        let rx = FrameReceiver::new(&header, &h_est, 0.1);

        // 1 RS packet × 11 LDPC blocks × 252 bits / (63 × 4 bps) = 11 sym → +1 EOT
        let blocks_per_rs = ldpc_blocks_per_rs(code.k);
        let total_coded   = 1 * blocks_per_rs * LDPC_N;
        let bits_per_ofdm = NUM_DATA * 4;
        let expected_data = (total_coded + bits_per_ofdm - 1) / bits_per_ofdm;
        assert_eq!(rx.total_data_syms, expected_data);
        assert_eq!(rx.expected_symbol_count(), expected_data + 1);
    }

    #[test]
    fn expected_symbol_count_with_resync() {
        let header = ModeHeader {
            modulation:   Modulation::Bpsk,
            ldpc_rate:    LdpcRate::R1_2,  // k=126, blocks_per_rs=17
            has_resync:   true,
            packet_count: 1,
            crc_ok:       true,
        };
        let h_est = flat_channel_est();
        let rx    = FrameReceiver::new(&header, &h_est, 0.1);
        let resync_syms = rx.total_data_syms / RESYNC_PERIOD;
        assert_eq!(rx.expected_symbol_count(), rx.total_data_syms + resync_syms + 1);
    }

    #[test]
    fn crc32_known_vector() {
        // CRC32 of "123456789" = 0xCBF43926 (IEEE 802.3)
        assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
    }

    /// Full loopback on a flat noiseless channel using the all-zero payload.
    ///
    /// All-zero payload → all-zero RS codeword (valid) → all-zero LDPC info bits
    /// → all-zero LDPC codeword (the zero vector satisfies every parity check).
    /// The BPSK channel maps 0 → +1, recovered with very high LLR → BP converges
    /// to all-zeros immediately → RS decodes → CRC32 matches.
    #[test]
    fn loopback_bpsk_r12_one_packet() {
        use crate::fec::ldpc::LdpcCode;

        let ldpc_rate  = LdpcRate::R1_2;
        let modulation = Modulation::Bpsk;
        let bps        = modulation.bits_per_symbol(); // 1

        let code = LdpcCode::for_rate(ldpc_rate);
        let k    = code.k;  // 126
        let bpr  = ldpc_blocks_per_rs(k); // ⌈2040/126⌉ = 17

        // All-zero payload: RS encode of zeros = all-zero codeword.
        let payload = vec![0u8; RS_K];

        // All-zero RS codeword bits → all-zero LDPC blocks (zero is always a
        // valid LDPC codeword — every parity check sums to 0 over GF(2)).
        let all_coded_bits = vec![0u8; bpr * LDPC_N];
        assert_eq!(all_coded_bits.len(), bpr * LDPC_N);

        // ── Build OFDM symbols ────────────────────────────────────────────────
        let bits_per_ofdm = NUM_DATA * bps;
        let total_data_syms = (all_coded_bits.len() + bits_per_ofdm - 1) / bits_per_ofdm;

        let mut ofdm_syms: Vec<Vec<Complex32>> = Vec::new();
        for sym_idx in 0..total_data_syms {
            let start = sym_idx * bits_per_ofdm;
            let end   = (start + bits_per_ofdm).min(all_coded_bits.len());
            let sym_bits = &all_coded_bits[start..end];
            ofdm_syms.push(bits_to_ofdm_symbol(sym_bits, bps));
        }

        // EOT symbol: CRC32 of payload in first 32 data subcarriers
        let crc = crc32_ieee(&payload);
        let mut eot_bits = vec![0u8; NUM_DATA];
        for i in 0..32 {
            eot_bits[i] = ((crc >> (31 - i)) & 1) as u8;
        }
        ofdm_syms.push(bits_to_ofdm_symbol(&eot_bits, 1));

        // ── Feed to FrameReceiver ─────────────────────────────────────────────
        let header = ModeHeader {
            modulation,
            ldpc_rate,
            has_resync:   false,
            packet_count: 1,
            crc_ok:       true,
        };
        let h_est = flat_channel_est();
        let mut rx = FrameReceiver::new(&header, &h_est, 0.1);

        let mut result = None;
        for sym in &ofdm_syms {
            match rx.push_symbol(sym) {
                PushResult::NeedMore             => {}
                PushResult::FrameComplete(frame) => { result = Some(frame); break; }
                PushResult::Error(e)             => panic!("frame error: {e}"),
            }
        }

        let frame = result.expect("FrameReceiver should have returned FrameComplete");
        assert!(frame.crc32_ok, "EOT CRC32 must pass on noiseless loopback");
        assert_eq!(frame.packets_ok, 1, "RS packet must decode successfully");
        assert_eq!(&frame.payload[..RS_K], &payload[..],
            "decoded payload must match original");
    }
}
