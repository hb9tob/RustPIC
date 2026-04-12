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

use crate::fec::{ldpc::LdpcCode, rs::{rs_interleave_m, RsCodec, RsLevel}};
use crate::ofdm::{
    params::*,
    rx::{
        frame::{blocks_per_rs_group, crc32_ieee, max_packets_per_frame},
        mode_detect::{encode_mode_header, LdpcRate, ModeHeader, Modulation},
    },
    scrambler::G3ruhScrambler,
    tx::{bits_to_symbol, ofdm_modulate, ofdm_modulate_all_carriers, ofdm_modulate_scattered},
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
    /// RS protection level (3 choices, trading throughput for error correction).
    pub rs_level:   RsLevel,
    /// Insert a re-sync ZC symbol every [`RESYNC_PERIOD`] data symbols.
    pub has_resync: bool,
}

// ── Frame builder ─────────────────────────────────────────────────────────────

/// Encodes `payload` bytes into one or more super-frame sample streams.
///
/// For payloads small enough to fit in a single super-frame this returns a
/// one-element `Vec`; larger payloads are split automatically so that no frame
/// exceeds the clock-drift budget (see [`max_packets_per_frame`]).
///
/// # Encoding chain (per super-frame)
///
/// ```text
/// frame_payload  ─► RS(255,191) per packet ─► flatten to bits
///                ─► LDPC encode per block ─► OFDM data symbols
/// ```
///
/// # Frame structure
///
/// ZC#1 │ ZC#2 │ mode header │ [D×RESYNC_PERIOD │ ZC_resync]* │ D×tail │ EOT
pub fn build_transmission(payload: &[u8], config: &FrameConfig) -> Vec<Vec<Complex32>> {
    let (_, rs_k, _)       = config.rs_level.params();
    let total_packet_count = payload.len().div_ceil(rs_k).max(1);
    let max_per            = max_packets_per_frame(
        config.modulation, config.ldpc_rate, config.rs_level);

    let mut frames         = Vec::new();
    let mut pkt_offset     = 0usize;

    while pkt_offset < total_packet_count {
        let packets_this  = (total_packet_count - pkt_offset).min(max_per);
        let byte_start    = pkt_offset * rs_k;
        let byte_end      = ((pkt_offset + packets_this) * rs_k).min(payload.len());
        let frame_payload = if byte_start < payload.len() {
            &payload[byte_start..byte_end]
        } else {
            &[]
        };
        frames.push(build_frame_internal(
            frame_payload,
            config,
            pkt_offset         as u16,
            total_packet_count as u16,
            packets_this,
        ));
        pkt_offset += packets_this;
    }

    frames
}

/// Convenience wrapper for single-frame transmissions.
///
/// Panics if `payload` requires more than one super-frame (use
/// [`build_transmission`] for large payloads).
pub fn build_frame(payload: &[u8], config: &FrameConfig) -> Vec<Complex32> {
    let mut frames = build_transmission(payload, config);
    assert_eq!(
        frames.len(), 1,
        "payload too large for a single super-frame — use build_transmission()"
    );
    frames.remove(0)
}

// ── Internal frame builder ────────────────────────────────────────────────────

fn build_frame_internal(
    payload:            &[u8],
    config:             &FrameConfig,
    packet_offset:      u16,
    total_packet_count: u16,
    packets_this_frame: usize,
) -> Vec<Complex32> {
    let code                    = LdpcCode::for_rate(config.ldpc_rate);
    let k                       = code.k;
    let (rs_n, rs_k, rs_2t)    = config.rs_level.params();
    let rs                      = RsCodec::for_level(config.rs_level);
    let m                       = rs_interleave_m(rs_2t, k);
    let bpg                     = blocks_per_rs_group(k, rs_n, m);
    let bps                     = config.modulation.bits_per_symbol();
    let groups                  = packets_this_frame.div_ceil(m);

    // ── 1. RS-encode + M-interleave + LDPC-encode all groups ─────────────────
    //
    // Layout for one group (M RS codewords):
    //   flat[b] = codeword[b % M][b / M]   (byte b of the flat stream)
    // This spreads a single failed LDPC block across M RS codewords, keeping
    // the per-codeword erasure count ≤ ⌈k/(8·M)⌉ ≤ RS_2T.
    let mut all_coded_bits: Vec<u8> = Vec::new();

    for grp in 0..groups {
        let pkt_base = grp * m;

        // Encode M RS codewords (padding codewords are all-zero encoded).
        let mut codewords: Vec<Vec<u8>> = Vec::with_capacity(m);
        for i in 0..m {
            let pkt_idx = pkt_base + i;
            let mut pkt_data = vec![0u8; rs_k];
            if pkt_idx < packets_this_frame {
                let byte_start = pkt_idx * rs_k;
                let byte_end   = (byte_start + rs_k).min(payload.len());
                if byte_start < payload.len() {
                    pkt_data[..byte_end - byte_start]
                        .copy_from_slice(&payload[byte_start..byte_end]);
                }
            }
            codewords.push(rs.encode(&pkt_data));
        }

        // Interleave: flat[b] = codeword[b%M][b/M]
        let total_bytes = m * rs_n;
        let mut flat = vec![0u8; total_bytes];
        for b in 0..total_bytes {
            flat[b] = codewords[b % m][b / m];
        }

        // Convert flat bytes → bits, zero-pad to bpg×k bits
        let mut flat_bits: Vec<u8> = flat.iter()
            .flat_map(|&b| (0..8usize).map(move |i| (b >> (7 - i)) & 1))
            .collect();
        flat_bits.resize(bpg * k, 0);

        // Encode bpg LDPC blocks
        for blk in 0..bpg {
            let info     = &flat_bits[blk * k..(blk + 1) * k];
            let codeword = code.encode(info);
            all_coded_bits.extend_from_slice(&codeword);
        }
    }

    // ── 2. Scramble coded bits (PRBS-15) then pad ─────────────────────────────
    // With scattered pilots, each data symbol has NUM_DATA_PER_SYM = 35
    // data carriers (constant because we use scattered-only, no freq pilots).
    let bits_per_ofdm   = NUM_DATA_PER_SYM * bps;
    let total_data_syms = all_coded_bits.len().div_ceil(bits_per_ofdm);
    let padded_len      = total_data_syms * bits_per_ofdm;

    let mut prbs = G3ruhScrambler::new();
    all_coded_bits.resize(padded_len, 0);
    prbs.scramble_bits(&mut all_coded_bits);

    // ── RUNIN: emit all data symbols TWICE ──────────────────────────────────
    // First pass  (RUNIN): trains the scattered-pilot equaliser on real data.
    // Second pass (DATA):  same LDPC blocks, now with a converged estimator.
    //
    // The RX processes BOTH passes. Failed LDPC blocks from the RUNIN get a
    // second chance on the DATA pass (soft-combine LLRs if both copies exist).
    // QSSTV does 24 RUNIN + 10 RUNOUT segments; we do a full 1× repeat.
    //
    // sym_idx is CONTINUOUS across both passes so the scattered pilot pattern
    // rotates correctly: pass1 uses sym_idx 5..5+D, pass2 uses 5+D..5+2D.
    let preamble_syms = 2 + MODE_HEADER_REPEAT;

    // Build OFDM symbols for one complete pass (bits → constellation → IFFT).
    let build_pass = |sym_offset: usize| -> Vec<Vec<Complex32>> {
        (0..total_data_syms)
            .map(|i| {
                let bits    = &all_coded_bits[i * bits_per_ofdm..(i + 1) * bits_per_ofdm];
                let n_data  = NUM_DATA_PER_SYM;
                let data_sc: Vec<Complex32> = (0..n_data)
                    .map(|s| bits_to_symbol(&bits[s * bps..], config.modulation))
                    .collect();
                ofdm_modulate_scattered(&data_sc, preamble_syms + sym_offset + i)
            })
            .collect()
    };
    let pass1_syms = build_pass(0);                     // RUNIN
    let pass2_syms = build_pass(total_data_syms);       // DATA

    // ── 3. Re-sync ZC symbol ─────────────────────────────────────────────────
    let resync_sym: Vec<Complex32> = build_preamble()[..SYMBOL_LEN].to_vec();

    // ── 4. Assemble sample stream ─────────────────────────────────────────────
    let mut samples: Vec<Complex32> = Vec::new();

    let preamble = build_preamble();
    samples.extend_from_slice(&preamble); // ZC#1
    samples.extend_from_slice(&preamble); // ZC#2

    let header = ModeHeader {
        modulation:         config.modulation,
        ldpc_rate:          config.ldpc_rate,
        rs_level:           config.rs_level,
        has_resync:         config.has_resync,
        total_packet_count,
        packet_offset,
        crc_ok:             true,
    };
    let hdr_bpsk = encode_mode_header(&header);
    let hdr_sc: Vec<Complex32> = hdr_bpsk.iter()
        .map(|&s| Complex32::new(s, 0.0))
        .collect();
    // Emit MODE_HEADER_REPEAT identical copies of the mode-header OFDM symbol
    // back-to-back. The RX equalises each independently, majority-votes the
    // bits, then verifies the CRC-10 on the voted result.
    let mode_hdr_sym = ofdm_modulate_all_carriers(&hdr_sc);
    for _ in 0..MODE_HEADER_REPEAT {
        samples.extend_from_slice(&mode_hdr_sym);
    }

    // ── 4b. RUNIN pass (first copy) ─────────────────────────────────────────
    // Same data as the main pass — trains the equaliser on real pilots +
    // gives the LDPC a first shot at each block.
    for (i, sym) in pass1_syms.iter().enumerate() {
        if config.has_resync && i > 0 && i % RESYNC_PERIOD == 0 {
            samples.extend_from_slice(&resync_sym);
        }
        samples.extend_from_slice(sym);
    }

    // ── 4c. DATA pass (second copy) ──────────────────────────────────────────
    for (i, sym) in pass2_syms.iter().enumerate() {
        if config.has_resync && i > 0 && i % RESYNC_PERIOD == 0 {
            samples.extend_from_slice(&resync_sym);
        }
        samples.extend_from_slice(sym);
    }

    // EOT: CRC32 over groups×M×rs_k bytes — same layout the RX assembles
    // (real packets zero-padded, padding packets zero-filled).
    let crc = {
        let padded_total = groups * m * rs_k;
        let mut padded   = vec![0u8; padded_total];
        let copy_len     = payload.len().min(padded_total);
        padded[..copy_len].copy_from_slice(&payload[..copy_len]);
        crc32_ieee(&padded)
    };
    let eot_sc: Vec<Complex32> = (0..NUM_DATA)
        .map(|i| {
            if i < 32 {
                let bit = (crc >> (31 - i)) & 1;
                Complex32::new(if bit == 0 { 1.0 } else { -1.0 }, 0.0)
            } else {
                Complex32::new(1.0, 0.0)
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
    use crate::{
        fec::rs::RsLevel,
        ofdm::rx::{
            frame::{FrameReceiver, PushResult},
            mode_detect::decode_mode_header,
            sync::{correct_cfo, ZcCorrelator},
        },
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

        // Mode header — MODE_HEADER_REPEAT identical copies, soft-combined.
        let hdr_start  = sync.header_start;
        let hdr_windows: Vec<&[Complex32]> = (0..MODE_HEADER_REPEAT)
            .map(|r| {
                let off = hdr_start + r * SYMBOL_LEN;
                &samples[off + CP_LEN..off + SYMBOL_LEN]
            })
            .collect();
        let header     = crate::ofdm::rx::mode_detect::decode_mode_header_repeated(
                &hdr_windows, &sync.channel_est)
            .unwrap_or_else(|e| panic!("header decode failed: {e}"));

        assert_eq!(header.modulation, config.modulation);
        assert_eq!(header.ldpc_rate,  config.ldpc_rate);
        assert_eq!(header.rs_level,   config.rs_level);
        assert_eq!(header.has_resync, config.has_resync);

        // Frame receiver
        let mut rx  = FrameReceiver::new(&header, &sync.channel_est, 0.1);
        let data_start = hdr_start + MODE_HEADER_REPEAT * SYMBOL_LEN;
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
        // packets_ok counts only real (non-padding) packets
        assert_eq!(frame.packets_ok, header.total_packet_count,
            "not all RS packets decoded");

        // Verify payload bytes (truncate to original length)
        let recovered = &frame.payload[..payload.len()];
        assert_eq!(recovered, payload,
            "payload mismatch for {:?}/{:?}", config.modulation, config.ldpc_rate);
    }

    #[test]
    fn loopback_bpsk_r12_no_resync() {
        let (_, rs_k, _) = RsLevel::L1.params();
        let payload = vec![0u8; rs_k]; // all-zero → trivial LDPC codewords
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Bpsk,
            ldpc_rate:  LdpcRate::R1_2,
            rs_level:   RsLevel::L1,
            has_resync: false,
        });
    }

    #[test]
    fn loopback_qpsk_r34_no_resync() {
        let (_, rs_k, _) = RsLevel::L1.params();
        let payload = vec![0u8; rs_k];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Qpsk,
            ldpc_rate:  LdpcRate::R3_4,
            rs_level:   RsLevel::L1,
            has_resync: false,
        });
    }

    #[test]
    fn loopback_16qam_r56_no_resync() {
        let (_, rs_k, _) = RsLevel::L1.params();
        let payload = vec![0u8; rs_k];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Qam16,
            ldpc_rate:  LdpcRate::R5_6,
            rs_level:   RsLevel::L1,
            has_resync: false,
        });
    }

    #[test]
    fn loopback_bpsk_r12_with_resync() {
        let (_, rs_k, _) = RsLevel::L1.params();
        let payload = vec![0u8; rs_k];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Bpsk,
            ldpc_rate:  LdpcRate::R1_2,
            rs_level:   RsLevel::L1,
            has_resync: true,
        });
    }

    /// Non-zero payload: verifies that the RS+LDPC layers actually encode/decode
    /// non-trivial data correctly.
    #[test]
    fn loopback_nonzero_payload_bpsk() {
        let (_, rs_k, _) = RsLevel::L1.params();
        // 191-byte pattern: 0xAA, 0x55, alternating
        let payload: Vec<u8> = (0..rs_k).map(|i| if i % 2 == 0 { 0xAAu8 } else { 0x55u8 }).collect();
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Bpsk,
            ldpc_rate:  LdpcRate::R1_2,
            rs_level:   RsLevel::L1,
            has_resync: false,
        });
    }
}
