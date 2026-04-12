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
    // sym_idx is CONTINUOUS across the whole frame so the scattered pilot
    // pattern rotates correctly.
    // Layout: RUNIN_PREAMBLE(10) + mode_header(3) + pass1(D) + pass2(D) + EOT + RUNOUT(4)
    let preamble_syms = PREAMBLE_SYMS;

    // Build OFDM symbols for one complete pass (bits → constellation → IFFT).
    let build_pass = |sym_offset: usize| -> Vec<Vec<Complex32>> {
        (0..total_data_syms)
            .map(|i| {
                let sym_idx = preamble_syms + sym_offset + i;
                let n_data  = crate::ofdm::drm_pilots::drm_num_data(sym_idx);
                let bits    = &all_coded_bits[i * bits_per_ofdm..(i + 1) * bits_per_ofdm];
                // n_data may exceed NUM_DATA_PER_SYM by 1 for some sym phases;
                // extra carriers get zero-padded.
                let data_sc: Vec<Complex32> = (0..n_data)
                    .map(|s| {
                        let bit_start = s * bps;
                        if bit_start + bps <= bits.len() {
                            bits_to_symbol(&bits[bit_start..], config.modulation)
                        } else {
                            Complex32::new(0.0, 0.0) // padding
                        }
                    })
                    .collect();
                ofdm_modulate_scattered(&data_sc, sym_idx)
            })
            .collect()
    };
    let pass1_syms = build_pass(0);                     // RUNIN
    let pass2_syms = build_pass(total_data_syms);       // DATA

    // ── 3. Assemble sample stream ────────────────────────────────────────────
    let mut samples: Vec<Complex32> = Vec::new();

    // ── 3a. RUNIN preamble: pilot-bearing OFDM symbols before the mode
    // header.  These warm up the RX Hilbert filter on real OFDM signal
    // (not silence) and let the pilot sync detect the frame early.
    // Like QSSTV's RUNIN segments, but using all-zero data carriers with
    // proper scattered pilots so the RX equaliser can start tracking H.
    for i in 0..RUNIN_PREAMBLE_SYMS {
        let n_data = crate::ofdm::drm_pilots::drm_num_data(i);
        let dummy_data = vec![Complex32::new(0.0, 0.0); n_data];
        samples.extend_from_slice(&ofdm_modulate_scattered(&dummy_data, i));
    }

    let header = ModeHeader {
        modulation:         config.modulation,
        ldpc_rate:          config.ldpc_rate,
        rs_level:           config.rs_level,
        has_resync:         false,
        total_packet_count,
        packet_offset,
        crc_ok:             true,
    };
    let hdr_bits = crate::ofdm::rx::mode_detect::encode_mode_header_bits(&header);
    // Emit MODE_HEADER_REPEAT copies of the mode-header as QPSK symbols with
    // scattered pilots.  Each symbol carries 35 data carriers × 2 bits = 70 bits;
    // the 42-bit header is placed in the first 21 QPSK symbols, rest is zero-padded.
    for r in 0..MODE_HEADER_REPEAT {
        let sym_idx = RUNIN_PREAMBLE_SYMS + r;
        let n_data = crate::ofdm::drm_pilots::drm_num_data(sym_idx);
        let data_sc: Vec<Complex32> = (0..n_data)
            .map(|s| {
                let bit_idx = s * 2;
                let b0 = hdr_bits.get(bit_idx).copied().unwrap_or(0);
                let b1 = hdr_bits.get(bit_idx + 1).copied().unwrap_or(0);
                // QPSK: (2b-1)/√2 for each component
                let scale = std::f32::consts::FRAC_1_SQRT_2;
                Complex32::new(
                    if b0 == 0 { scale } else { -scale },
                    if b1 == 0 { scale } else { -scale },
                )
            })
            .collect();
        samples.extend_from_slice(&ofdm_modulate_scattered(&data_sc, sym_idx));
    }

    // ── 3b. RUNIN pass (first copy) ─────────────────────────────────────────
    // Same data as the main pass — trains the equaliser on real pilots +
    // gives the LDPC a first shot at each block.
    for sym in &pass1_syms {
        samples.extend_from_slice(sym);
    }

    // ── 3c. DATA pass (second copy) ──────────────────────────────────────────
    for sym in &pass2_syms {
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
    let eot_sym_idx = preamble_syms + 2 * total_data_syms;
    let n_eot_data = crate::ofdm::drm_pilots::drm_num_data(eot_sym_idx);
    let crc_bits = n_eot_data.min(32); // transmit as many CRC bits as we have carriers
    let eot_sc: Vec<Complex32> = (0..n_eot_data)
        .map(|i| {
            if i < crc_bits {
                let bit = (crc >> (31 - i)) & 1;
                Complex32::new(if bit == 0 { 1.0 } else { -1.0 }, 0.0)
            } else {
                Complex32::new(1.0, 0.0)
            }
        })
        .collect();
    samples.extend_from_slice(&ofdm_modulate_scattered(&eot_sc, eot_sym_idx));

    // ── RUNOUT: pilot-bearing dummy symbols after the EOT ─────────────────
    // Ensures the RX Hilbert filter and equaliser have valid pilot data
    // all the way through the last decoded symbol (like QSSTV's RUNOUT).
    for i in 0..RUNOUT_SYMS {
        let idx = eot_sym_idx + 1 + i;
        let n_data = crate::ofdm::drm_pilots::drm_num_data(idx);
        let dummy = vec![Complex32::new(0.0, 0.0); n_data];
        samples.extend_from_slice(&ofdm_modulate_scattered(&dummy, idx));
    }

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
            equalizer::ScatteredEqualizer,
            mode_detect::decode_mode_header_from_eq,
            pilot_sync::scan_for_pilots,
        },
    };

    /// Full loopback: build_frame → pilot sync → decode → verify payload.
    ///
    /// No ZC — sync is entirely from scattered pilots (CP correlation +
    /// pilot phase matching).
    fn loopback(payload: &[u8], config: &FrameConfig) {
        // ── TX ────────────────────────────────────────────────────────────────
        let samples = build_frame(payload, config);

        // ── RX: pilot-based sync ──────────────────────────────────────────────
        // scan_for_pilots finds the first pilot-bearing symbol.  The RUNIN
        // preamble has sym_idx 0..RUNIN_PREAMBLE_SYMS-1.  We scan from the
        // start — the first match will be a RUNIN preamble symbol, from which
        // we can navigate to the mode header and data.
        let ps = scan_for_pilots(&samples, 0, 0.3, 0.7)
            .expect("pilot sync should find OFDM symbols");

        // Navigate from detected symbol to the mode header and data.
        // sym_idx 0..9 = RUNIN preamble, 10..12 = mode header (no pilots),
        // 13+ = data.
        // From any detected sym_idx < PREAMBLE_SYMS (RUNIN preamble range),
        // the mode header is at RUNIN_PREAMBLE_SYMS symbols from frame start.
        let frame_start = ps.symbol_pos - ps.sym_idx * SYMBOL_LEN;
        let hdr_start = frame_start + RUNIN_PREAMBLE_SYMS * SYMBOL_LEN;
        let first_data_pos = frame_start + PREAMBLE_SYMS * SYMBOL_LEN;

        // Decode mode header — QPSK with scattered pilots through equalizer.
        let mut hdr_eq = ScatteredEqualizer::from_initial(&ps.channel_est, 0.5);
        let mut eq_vecs: Vec<Vec<Complex32>> = Vec::new();
        for r in 0..MODE_HEADER_REPEAT {
            let off = hdr_start + r * SYMBOL_LEN;
            if off + SYMBOL_LEN > samples.len() { break; }
            let sym_idx = RUNIN_PREAMBLE_SYMS + r;
            let eq = hdr_eq.process(&samples[off..off + SYMBOL_LEN], sym_idx);
            eq_vecs.push(eq.data);
        }
        let eq_refs: Vec<&[Complex32]> = eq_vecs.iter().map(|v| v.as_slice()).collect();
        let header = decode_mode_header_from_eq(&eq_refs)
            .unwrap_or_else(|e| panic!("header decode failed: {e}"));

        assert_eq!(header.modulation, config.modulation);
        assert_eq!(header.ldpc_rate,  config.ldpc_rate);
        assert_eq!(header.rs_level,   config.rs_level);

        // Frame receiver
        let mut rx = FrameReceiver::new(&header, &ps.channel_est, 0.1);
        let n_expected = rx.expected_symbol_count();
        let mut frame_result = None;

        for i in 0..n_expected {
            let pos = first_data_pos + i * SYMBOL_LEN;
            if pos + SYMBOL_LEN > samples.len() { break; }
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
        assert_eq!(frame.packets_ok, header.total_packet_count,
            "not all RS packets decoded");

        let recovered = &frame.payload[..payload.len()];
        assert_eq!(recovered, payload,
            "payload mismatch for {:?}/{:?}", config.modulation, config.ldpc_rate);
    }

    #[test]
    fn loopback_bpsk_r12() {
        let (_, rs_k, _) = RsLevel::L1.params();
        let payload = vec![0u8; rs_k]; // all-zero → trivial LDPC codewords
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Bpsk,
            ldpc_rate:  LdpcRate::R1_2,
            rs_level:   RsLevel::L1,
        });
    }

    #[test]
    fn loopback_qpsk_r34() {
        let (_, rs_k, _) = RsLevel::L1.params();
        let payload = vec![0u8; rs_k];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Qpsk,
            ldpc_rate:  LdpcRate::R3_4,
            rs_level:   RsLevel::L1,
        });
    }

    #[test]
    fn loopback_16qam_r56() {
        let (_, rs_k, _) = RsLevel::L1.params();
        let payload = vec![0u8; rs_k];
        loopback(&payload, &FrameConfig {
            modulation: Modulation::Qam16,
            ldpc_rate:  LdpcRate::R5_6,
            rs_level:   RsLevel::L1,
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
        });
    }
}
