//! Super-frame assembler (TX side).
//!
//! Assembles the complete sample stream for one super-frame:
//!
//! ```text
//! ┌──────┬──────┬────────┬─ data ──┬── re-sync ──┬─ data ──┬─────┐
//! │ ZC#1 │ ZC#2 │ header │ D×12   │ ZC (resync) │ D×…    │ EOT │
//! └──────┴──────┴────────┴─────────┴─────────────┴─────────┴─────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let config = FrameConfig {
//!     modulation: Modulation::Qam16,
//!     ldpc_rate:  LdpcRate::R3_4,
//!     has_resync: true,
//! };
//! let samples = build_frame(&image_bytes, &config);
//! // feed `samples` to the audio output / radio
//! ```

use num_complex::Complex32;

use crate::fec::{ldpc::LdpcCode, rs::RsCodec};
use crate::fec::rs::RS_K;
use crate::ofdm::{
    params::*,
    rx::{
        frame::{crc32_ieee, ldpc_blocks_per_rs},
        mode_detect::{encode_mode_header, LdpcRate, ModeHeader, Modulation},
    },
    tx::{bits_to_symbol, ofdm_modulate, ofdm_modulate_all_carriers},
    zc::build_preamble,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// TX configuration for a super-frame.
#[derive(Debug, Clone)]
pub struct FrameConfig {
    /// Constellation for data subcarriers.
    pub modulation: Modulation,
    /// LDPC code rate.
    pub ldpc_rate:  LdpcRate,
    /// Insert a re-sync ZC symbol every [`RESYNC_PERIOD`] data symbols.
    pub has_resync: bool,
}

// ── Frame builder ─────────────────────────────────────────────────────────────

/// Encodes `payload` bytes and assembles the complete super-frame sample
/// stream.
///
/// # Encoding chain
///
/// ```text
/// payload  ─┬─ pad to multiple of RS_K ─┐
///            │  RS(255,191) encode        │  per RS packet
///            │  flatten to bits           │
///            │  pad to blocks_per_rs × k  │
///            └─ LDPC encode (×blocks_per_rs) ─► accumulated coded bits
///
/// coded bits ─► OFDM data symbols (NUM_DATA × bps bits each)
/// ```
///
/// # Frame structure
///
/// ZC#1 │ ZC#2 │ mode header │ [D×RESYNC_PERIOD │ ZC_resync]* │ D×tail │ EOT
///
/// The EOT OFDM symbol carries the CRC32 of `payload` on the first 32 data
/// subcarriers as BPSK (MSB first); the remaining subcarriers are set to +1.
///
/// # Packet count
///
/// `packet_count` in the mode header = `⌈payload.len() / RS_K⌉`.
/// The last RS packet is zero-padded to [`RS_K`] bytes if needed.
pub fn build_frame(payload: &[u8], config: &FrameConfig) -> Vec<Complex32> {
    let rs      = RsCodec::new();
    let code    = LdpcCode::for_rate(config.ldpc_rate);
    let k       = code.k;
    let bpr     = ldpc_blocks_per_rs(k); // LDPC blocks per RS codeword
    let bps     = config.modulation.bits_per_symbol();

    // ── 1. RS-encode + LDPC-encode all payload packets ────────────────────────
    let rs_k = RS_K;
    let packet_count = payload.len().div_ceil(rs_k).max(1);
    let mut all_coded_bits: Vec<u8> = Vec::new();

    for pkt in 0..packet_count {
        let start = pkt * rs_k;
        let end   = (start + rs_k).min(payload.len());

        // Zero-pad last packet to RS_K bytes
        let mut pkt_data = vec![0u8; rs_k];
        pkt_data[..end - start].copy_from_slice(&payload[start..end]);

        // RS encode → 255-byte codeword
        let rs_cw = rs.encode(&pkt_data);

        // Flatten RS codeword to bits (MSB first)
        let mut cw_bits: Vec<u8> = rs_cw.iter()
            .flat_map(|&b| (0..8usize).map(move |i| (b >> (7 - i)) & 1))
            .collect();
        cw_bits.resize(bpr * k, 0); // pad to bpr × k (zero padding after RS bits)

        // LDPC encode each block
        for blk in 0..bpr {
            let info = &cw_bits[blk * k..(blk + 1) * k];
            let codeword = code.encode(info); // n = LDPC_N bits
            all_coded_bits.extend_from_slice(&codeword);
        }
    }

    // ── 2. Slice into OFDM data symbols ───────────────────────────────────────
    let bits_per_ofdm = NUM_DATA * bps;
    // Pad coded bits to a multiple of bits_per_ofdm
    let total_data_syms = all_coded_bits.len().div_ceil(bits_per_ofdm);
    all_coded_bits.resize(total_data_syms * bits_per_ofdm, 0);

    let data_ofdm_syms: Vec<Vec<Complex32>> = (0..total_data_syms)
        .map(|i| {
            let bits = &all_coded_bits[i * bits_per_ofdm..(i + 1) * bits_per_ofdm];
            let data_sc: Vec<Complex32> = (0..NUM_DATA)
                .map(|sym| bits_to_symbol(&bits[sym * bps..], config.modulation))
                .collect();
            ofdm_modulate(&data_sc)
        })
        .collect();

    // ── 3. Build re-sync ZC symbol ────────────────────────────────────────────
    // Single ZC (same as ZC#1 of the preamble = first SYMBOL_LEN samples).
    let resync_sym: Vec<Complex32> = build_preamble()[..SYMBOL_LEN].to_vec();

    // ── 4. Assemble sample stream ─────────────────────────────────────────────
    let mut samples: Vec<Complex32> = Vec::new();

    // Preamble: ZC#1 + ZC#2
    let preamble = build_preamble();
    samples.extend_from_slice(&preamble); // ZC#1
    samples.extend_from_slice(&preamble); // ZC#2

    // Mode header
    let header = ModeHeader {
        modulation:   config.modulation,
        ldpc_rate:    config.ldpc_rate,
        has_resync:   config.has_resync,
        packet_count: packet_count as u16,
        crc_ok:       true,
    };
    let hdr_bpsk = encode_mode_header(&header);
    let hdr_sc: Vec<Complex32> = hdr_bpsk.iter()
        .map(|&s| Complex32::new(s, 0.0))
        .collect();
    samples.extend_from_slice(&ofdm_modulate_all_carriers(&hdr_sc));

    // Data symbols interleaved with optional re-sync ZCs
    for (data_sym_idx, sym) in data_ofdm_syms.iter().enumerate() {
        // Insert re-sync ZC BEFORE the first symbol of each new group
        // (after the first group of RESYNC_PERIOD data symbols)
        if config.has_resync && data_sym_idx > 0 && data_sym_idx % RESYNC_PERIOD == 0 {
            samples.extend_from_slice(&resync_sym);
        }
        samples.extend_from_slice(sym);
    }

    // EOT symbol: CRC32 in first 32 data subcarriers (BPSK, MSB first).
    // CRC is computed over the zero-padded payload (packet_count × RS_K bytes)
    // so the RX can recompute it directly over rs_payload without knowing the
    // original payload length.
    let crc = {
        let padded_len = packet_count * rs_k;
        let mut padded = vec![0u8; padded_len];
        padded[..payload.len()].copy_from_slice(payload);
        crc32_ieee(&padded)
    };
    let eot_sc: Vec<Complex32> = (0..NUM_DATA)
        .map(|i| {
            if i < 32 {
                let bit = (crc >> (31 - i)) & 1;
                Complex32::new(if bit == 0 { 1.0 } else { -1.0 }, 0.0)
            } else {
                Complex32::new(1.0, 0.0) // padding → bit 0
            }
        })
        .collect();
    samples.extend_from_slice(&ofdm_modulate(&eot_sc));

    samples
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::rx::{
        frame::{FrameReceiver, PushResult},
        mode_detect::decode_mode_header,
        sync::{correct_cfo, ZcCorrelator},
    };

    /// Full loopback: build_frame → feed through the RX pipeline →
    /// verify payload, CRC32, and packet count.
    fn loopback(payload: &[u8], config: &FrameConfig) {
        // ── TX ────────────────────────────────────────────────────────────────
        let mut samples = build_frame(payload, config);

        // ── RX sync ───────────────────────────────────────────────────────────
        let corr = ZcCorrelator::new(0.40, 0.30);
        let sync = corr.find_sync(&samples)
            .unwrap_or_else(|e| panic!("sync failed: {e}"));

        correct_cfo(&mut samples[sync.preamble_start..], sync.cfo_hz);

        // Mode header
        let hdr_start  = sync.header_start;
        let hdr_win    = &samples[hdr_start + CP_LEN..hdr_start + SYMBOL_LEN];
        let header     = decode_mode_header(hdr_win, &sync.channel_est)
            .unwrap_or_else(|e| panic!("header decode failed: {e}"));

        assert_eq!(header.modulation,   config.modulation);
        assert_eq!(header.ldpc_rate,    config.ldpc_rate);
        assert_eq!(header.has_resync,   config.has_resync);

        // Frame receiver
        let mut rx  = FrameReceiver::new(&header, &sync.channel_est, 0.1);
        let data_start = hdr_start + SYMBOL_LEN;
        let n_expected = rx.expected_symbol_count();
        let mut frame_result = None;

        for i in 0..n_expected {
            let pos = data_start + i * SYMBOL_LEN;
            let sym = &samples[pos..pos + SYMBOL_LEN];
            match rx.push_symbol(sym) {
                PushResult::NeedMore             => {}
                PushResult::FrameComplete(frame) => { frame_result = Some(frame); break; }
                PushResult::Error(e)             => panic!("frame rx error: {e}"),
            }
        }

        let frame = frame_result.expect("FrameReceiver must return FrameComplete");

        assert!(frame.crc32_ok,
            "CRC32 failed for {:?}/{:?}", config.modulation, config.ldpc_rate);
        assert_eq!(frame.packets_ok, header.packet_count,
            "not all RS packets decoded");

        // Verify payload bytes (truncate to original length)
        let recovered = &frame.payload[..payload.len()];
        assert_eq!(recovered, payload,
            "payload mismatch for {:?}/{:?}", config.modulation, config.ldpc_rate);
    }

    #[test]
    fn loopback_bpsk_r12_no_resync() {
        let payload = vec![0u8; RS_K]; // all-zero → trivial LDPC codewords
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Bpsk,
            ldpc_rate:  LdpcRate::R1_2,
            has_resync: false,
        });
    }

    #[test]
    fn loopback_qpsk_r34_no_resync() {
        let payload = vec![0u8; RS_K];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Qpsk,
            ldpc_rate:  LdpcRate::R3_4,
            has_resync: false,
        });
    }

    #[test]
    fn loopback_16qam_r56_no_resync() {
        let payload = vec![0u8; RS_K];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Qam16,
            ldpc_rate:  LdpcRate::R5_6,
            has_resync: false,
        });
    }

    #[test]
    fn loopback_bpsk_r12_with_resync() {
        let payload = vec![0u8; RS_K];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Bpsk,
            ldpc_rate:  LdpcRate::R1_2,
            has_resync: true,
        });
    }

    /// Non-zero payload: verifies that the RS+LDPC layers actually encode/decode
    /// non-trivial data correctly.
    #[test]
    fn loopback_nonzero_payload_bpsk() {
        // 191-byte pattern: 0xAA, 0x55, alternating
        let payload: Vec<u8> = (0..RS_K).map(|i| if i % 2 == 0 { 0xAAu8 } else { 0x55u8 }).collect();
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Bpsk,
            ldpc_rate:  LdpcRate::R1_2,
            has_resync: false,
        });
    }
}
