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
use crate::fec::rs::{rs_interleave_m, RsCodec, RsLevel};
use crate::ofdm::params::*;
use crate::ofdm::rx::{
    demapper::demap,
    equalizer::ScatteredEqualizer,
    mode_detect::{LdpcRate, ModeHeader, Modulation},
};
use crate::ofdm::scrambler::G3ruhScrambler;

// ── Constants ─────────────────────────────────────────────────────────────────

/// LDPC codeword length (bits) — same for all four rates (IEEE 802.11n, z = 81).
const LDPC_N: usize = 1944;

/// Maximum OFDM data symbols per super-frame.
///
/// At ±20 ppm and SYMBOL_LEN = 288, the per-symbol clock drift is
/// 288 × 20×10⁻⁶ ≈ 0.00576 samples.  The CP absorbs up to CP/2 = 16 samples,
/// reached after ≈ 2 778 symbols.  1 400 symbols keeps the drift below 8 samples
/// (half that), providing comfortable margin.
pub const MAX_DATA_SYMS_PER_FRAME: usize = 1_400;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns how many LDPC blocks are needed to carry `m` complete RS codewords
/// of `rs_n` bytes each.
///
/// The last block may be partially padded.
pub(crate) fn blocks_per_rs_group(ldpc_k: usize, rs_n: usize, m: usize) -> usize {
    (m * rs_n * 8).div_ceil(ldpc_k)
}

/// Maximum RS packets that fit within [`MAX_DATA_SYMS_PER_FRAME`] OFDM data
/// symbols for the given modulation, LDPC rate, and RS level.
///
/// The result is always a multiple of the interleave factor M so that complete
/// RS groups fill each super-frame.
///
/// Both the TX (`build_transmission`) and RX (`FrameReceiver`) must call this
/// with identical arguments to agree on the per-frame packet boundary.
pub fn max_packets_per_frame(
    modulation: Modulation,
    ldpc_rate:  LdpcRate,
    rs_level:   RsLevel,
) -> usize {
    let bps        = modulation.bits_per_symbol();
    let code       = LdpcCode::for_rate(ldpc_rate);
    let (rs_n, _, rs_2t) = rs_level_params(rs_level);
    let m          = rs_interleave_m(rs_2t, code.k);
    let bpg        = blocks_per_rs_group(code.k, rs_n, m);
    let syms       = (bpg * LDPC_N).div_ceil(NUM_DATA * bps);
    let groups     = (MAX_DATA_SYMS_PER_FRAME / syms).max(1);
    groups * m
}

/// Returns `(rs_n, rs_k, rs_2t)` for the given level (convenience alias).
#[inline]
pub(crate) fn rs_level_params(level: RsLevel) -> (usize, usize, usize) {
    let (n, k, two_t) = level.params();
    (n, k, two_t)
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

/// Per-frame channel and codec statistics.
#[derive(Debug, Clone)]
pub struct FrameMetrics {
    /// Estimated channel BER: fraction of bits flipped by the channel
    /// (measured as hard-decision errors corrected by LDPC / total coded bits).
    pub channel_ber: f32,

    /// Number of LDPC blocks that converged (all parity checks satisfied).
    pub ldpc_converged: u32,

    /// Total LDPC blocks decoded in this frame.
    pub ldpc_total: u32,

    /// Total RS byte errors corrected across all RS packets.
    pub rs_errors_corrected: u32,

    /// Total RS erasures consumed (from failed LDPC blocks) across all RS packets.
    pub rs_erasures_used: u32,

    /// Minimum RS correction budget remaining as a fraction of `RS_2T = 64`,
    /// taken over all successfully decoded RS packets.
    /// Value 1.0 means no budget was used; 0.0 means budget was exactly exhausted.
    /// Stays at 1.0 if no RS packet succeeded.
    pub rs_margin_frac: f32,

    /// Indices of LDPC blocks that did not converge (0-based, empty if all converged).
    pub failing_block_indices: Vec<u32>,
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

    /// Channel and codec statistics for this frame.
    pub metrics: FrameMetrics,
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
    header:              ModeHeader,
    ldpc_k:              usize,
    /// RS codeword length (255 bytes).
    rs_n:                usize,
    /// RS info bytes per codeword.
    rs_k:                usize,
    /// RS parity bytes per codeword (2t).
    rs_2t:               usize,
    /// Number of RS codewords processed in parallel (interleave factor).
    m_interleave:        usize,
    /// LDPC blocks per RS group (= blocks needed for m_interleave RS codewords).
    blocks_per_group:    usize,
    total_ldpc_blocks:   usize,
    total_data_syms:     usize,
    /// RS packets actually in this frame (for capping rs_packets_ok).
    packets_this_frame:  usize,

    // ── Processing units ──────────────────────────────────────────────────────
    equalizer:    ScatteredEqualizer,
    code:         LdpcCode,
    rs:           RsCodec,
    descrambler:  G3ruhScrambler,

    // ── Accumulation buffers ──────────────────────────────────────────────────
    llr_buf:       Vec<f32>,
    pass1_llrs:    Vec<f32>,  // LLR from RUNIN pass, soft-combined with pass2
    data_syms_per_pass: usize,
    info_bits:     Vec<u8>,
    rs_payload:    Vec<u8>,

    // ── Symbol / block counters ────────────────────────────────────────────────
    warmup_remaining:    usize,
    total_syms_fed:      usize, // warmup + data, for sym_idx pilot pattern
    data_syms_received:  usize,
    syms_in_group:       usize,
    ldpc_blocks_decoded: usize,
    rs_packets_ok:       u16,

    // ── SNR bookkeeping ────────────────────────────────────────────────────────
    pilot_snr_sum:   f32,
    pilot_snr_count: usize,

    // ── Metrics accumulation ───────────────────────────────────────────────────
    ldpc_block_converged: Vec<bool>,
    channel_bit_errors:   u32,
    channel_bit_total:    u32,
    rs_errors_corrected:  u32,
    rs_erasures_used:     u32,
    rs_min_margin_frac:   f32,
}

impl FrameReceiver {
    /// Creates a new `FrameReceiver` for the super-frame described by `header`.
    ///
    /// * `header`      — decoded mode header (modulation, LDPC rate, packet_count, …).
    /// * `channel_est` — [`NUM_CARRIERS`] complex channel coefficients from the ZC preamble.
    /// * `alpha`       — EMA tracking coefficient for the channel equalizer (0.05 … 0.3).
    pub fn new(header: &ModeHeader, channel_est: &[Complex32], alpha: f32) -> Self {
        let code             = LdpcCode::for_rate(header.ldpc_rate);
        let ldpc_k           = code.k;
        let (rs_n, rs_k, rs_2t) = rs_level_params(header.rs_level);
        let m_interleave     = rs_interleave_m(rs_2t, ldpc_k);
        let blocks_per_group = blocks_per_rs_group(ldpc_k, rs_n, m_interleave);

        // RS packets in THIS super-frame
        let remaining          = header.total_packet_count
                                    .saturating_sub(header.packet_offset) as usize;
        let packets_this_frame = remaining.min(
            max_packets_per_frame(header.modulation, header.ldpc_rate, header.rs_level)
        );
        let groups_this_frame  = packets_this_frame.div_ceil(m_interleave);

        let total_ldpc_blocks = groups_this_frame * blocks_per_group;
        let bits_per_ofdm     = NUM_DATA_PER_SYM * header.modulation.bits_per_symbol();
        let total_coded_bits  = total_ldpc_blocks * LDPC_N;
        let data_syms_per_pass = total_coded_bits.div_ceil(bits_per_ofdm);
        // TX sends 2 passes (RUNIN + DATA), so total data symbols = 2×.
        let total_data_syms   = 2 * data_syms_per_pass;

        Self {
            header:             header.clone(),
            ldpc_k,
            rs_n,
            rs_k,
            rs_2t,
            m_interleave,
            blocks_per_group,
            total_ldpc_blocks,
            total_data_syms,
            packets_this_frame,
            equalizer:    ScatteredEqualizer::from_initial(channel_est, alpha),
            code,
            rs:           RsCodec::for_level(header.rs_level),
            descrambler:  G3ruhScrambler::new(),
            llr_buf:      Vec::new(),
            pass1_llrs:   Vec::new(),
            data_syms_per_pass,
            info_bits:    Vec::new(),
            rs_payload: Vec::new(),
            warmup_remaining:    0,  // match TX WARMUP_SYMS
            total_syms_fed:      0,
            data_syms_received:  0,
            syms_in_group:       0,
            ldpc_blocks_decoded: 0,
            rs_packets_ok:       0,
            pilot_snr_sum:       0.0,
            pilot_snr_count:     0,
            ldpc_block_converged: Vec::new(),
            channel_bit_errors:   0,
            channel_bit_total:    0,
            rs_errors_corrected:  0,
            rs_erasures_used:     0,
            rs_min_margin_frac:   1.0,
        }
    }

    /// Total OFDM data symbols (not counting resync ZC or EOT).
    pub fn total_data_syms(&self) -> usize { self.total_data_syms }

    /// Sets the per-data-symbol timing drift used for clock-drift phase
    /// pre-correction inside the channel equalizer.
    ///
    /// Call this after each resync ZC is located in the receiver loop:
    /// ```text
    ///   drift_per_sym = −correction / (RESYNC_PERIOD + 1)
    /// ```
    /// where `correction = found_pos_fractional − expected_pos`.
    pub fn set_timing_drift_per_sym(&mut self, drift_per_sym: f32) {
        self.equalizer.set_timing_drift_per_sym(drift_per_sym);
    }

    /// Total number of OFDM symbols this receiver expects before returning
    /// [`PushResult::FrameComplete`].
    ///
    /// Includes data symbols and the EOT symbol.
    pub fn expected_symbol_count(&self) -> usize {
        self.total_data_syms + 1 // data + EOT
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

        // ── Warm-up symbols: EQ training only, no LDPC ────────────────────────
        // The TX prepends WARMUP_SYMS copies of the first data symbols so the
        // scattered-pilot equaliser converges before real data begins.
        // We process them for channel tracking but discard their LLRs.
        if self.warmup_remaining > 0 {
            let warmup_idx = 10 - self.warmup_remaining; // 0, 1, 2, ...
            let sym_idx = PREAMBLE_SYMS + warmup_idx;
            let _eq = self.equalizer.process(ofdm_symbol, sym_idx);
            self.warmup_remaining -= 1;
            return PushResult::NeedMore;
        }

        // ── Regular data symbol ───────────────────────────────────────────────
        // sym_idx = MODE_HEADER_REPEAT + data_syms_received, matching the TX's
        // ofdm_modulate_scattered(data, MODE_HEADER_REPEAT + i).
        let sym_idx = PREAMBLE_SYMS + self.data_syms_received;
        let eq = self.equalizer.process(ofdm_symbol, sym_idx);
        // Only accumulate SNR metrics from pass2 (converged equaliser).
        let in_pass2 = self.data_syms_received >= self.data_syms_per_pass;
        if in_pass2 {
            self.pilot_snr_sum   += eq.pilot_snr_db;
            self.pilot_snr_count += 1;
        }

        if std::env::var("FRAME_DEBUG").is_ok() {
            let (mean_re, mean_im, mean_amp) = if !eq.data.is_empty() {
                let n = eq.data.len() as f32;
                let re = eq.data.iter().map(|c| c.re.abs()).sum::<f32>() / n;
                let im = eq.data.iter().map(|c| c.im.abs()).sum::<f32>() / n;
                let amp = (eq.data.iter().map(|c| c.norm_sqr()).sum::<f32>() / n).sqrt();
                (re, im, amp)
            } else { (0.0, 0.0, 0.0) };
            eprintln!("  [frame.rs] sym={} grp={} snr={:.1}dB amp={:.3} mean|Re|={:.3} mean|Im|={:.3} Re/Im={:.2}",
                self.data_syms_received, self.syms_in_group, eq.pilot_snr_db,
                mean_amp, mean_re, mean_im, mean_re / (mean_im + 1e-9));
        }

        let mut llrs = demap(&eq.data, &eq.noise_var, self.header.modulation);
        // Truncate to bits_per_ofdm: some symbols have 1 extra data carrier
        // due to the scattered pilot pattern (e.g. 29 vs 28 data carriers).
        let bits_per_ofdm = NUM_DATA_PER_SYM * self.header.modulation.bits_per_symbol();
        llrs.truncate(bits_per_ofdm);
        self.data_syms_received += 1;
        self.total_syms_fed     += 1;
        self.syms_in_group      += 1;

        // Pass 1 (RUNIN): EQ training only — don't accumulate LLR or drain.
        // The equaliser converges on the real scattered pilots. All decode
        // happens in pass2 when the estimator is stable.
        if self.data_syms_received <= self.data_syms_per_pass {
            if self.data_syms_received == self.data_syms_per_pass {
                // Boundary: reset for pass2.
                self.syms_in_group = 0;
                self.pilot_snr_sum = 0.0;
                self.pilot_snr_count = 0;
            }
            return PushResult::NeedMore;
        }

        // Pass 2 (DATA): equaliser converged, decode for real.
        self.llr_buf.extend_from_slice(&llrs);
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
            // Consume exactly LDPC_N LLRs and descramble (G3RUH PRBS)
            let mut block_llrs: Vec<f32> = self.llr_buf.drain(..LDPC_N).collect();
            self.descrambler.descramble_llrs(&mut block_llrs);

            // Hard-decision bits before LDPC (for BER estimation)
            let hard_bits: Vec<u8> = block_llrs.iter()
                .map(|&l| u8::from(l < 0.0))
                .collect();

            // Decode LDPC — decoder borrows &self.code in its own scope
            if std::env::var("FRAME_DEBUG").is_ok() {
                let n = block_llrs.len() as f32;
                let mean_abs = block_llrs.iter().map(|l| l.abs()).sum::<f32>() / n;
                let min_abs  = block_llrs.iter().map(|l| l.abs()).fold(f32::INFINITY, f32::min);
                let max_abs  = block_llrs.iter().map(|l| l.abs()).fold(0.0f32, f32::max);
                let near_zero = block_llrs.iter().filter(|l| l.abs() < 1.0).count();
                let wrong_sign = hard_bits.iter().zip(block_llrs.iter())
                    .filter(|(&h, &l)| (h == 1) == (l > 0.0))  // h=1→bit=1, expect l<0
                    .count();
                eprintln!("  [ldpc] blk={} mean|LLR|={:.2} min={:.2} max={:.2} near0={} bad_sign={}/{}",
                    self.ldpc_blocks_decoded, mean_abs, min_abs, max_abs,
                    near_zero, wrong_sign, block_llrs.len());
            }
            let decode_result = {
                let decoder = LdpcDecoder::new(&self.code, 50, 0.75);
                decoder.decode(&block_llrs)
            };
            let decoded_bits = decode_result.bits;

            // Accumulate channel BER (bits the decoder had to flip)
            let bit_errors: u32 = hard_bits.iter().zip(decoded_bits.iter())
                .filter(|(&h, &d)| h != d)
                .count() as u32;
            self.channel_bit_errors += bit_errors;
            self.channel_bit_total  += LDPC_N as u32;

            // Track convergence
            self.ldpc_block_converged.push(decode_result.converged);

            // Extract info bits from the positions identified during code construction
            for &col in &self.code.info_cols {
                self.info_bits.push(decoded_bits[col]);
            }
            self.ldpc_blocks_decoded += 1;

            // When blocks_per_group LDPC blocks complete, decode one RS group (M packets)
            if self.ldpc_blocks_decoded % self.blocks_per_group == 0 {
                self.decode_rs_group();
            }
        }
    }

    // ── RS group decoding (M codewords in parallel) ───────────────────────────

    /// Decodes one RS group: `blocks_per_group` LDPC blocks → `m_interleave` RS
    /// codewords (byte-interleaved) → appends `m_interleave × rs_k` payload bytes.
    ///
    /// Erasures are distributed across codewords via the interleave mapping so
    /// each LDPC block failure causes at most `⌈ldpc_k/(8·M)⌉` erasures per
    /// RS codeword.
    fn decode_rs_group(&mut self) {
        let m         = self.m_interleave;
        let rs_n      = self.rs_n;
        let rs_k      = self.rs_k;
        let rs_2t     = self.rs_2t;
        let ldpc_k    = self.ldpc_k;
        let total_bits = self.blocks_per_group * ldpc_k;
        let total_bytes = m * rs_n; // flat interleaved byte stream length

        debug_assert!(self.info_bits.len() >= total_bits);
        let bits: Vec<u8> = self.info_bits.drain(..total_bits).collect();

        // Convert bits → flat interleaved byte stream
        let mut flat = vec![0u8; total_bytes];
        for (byte_idx, byte) in flat.iter_mut().enumerate() {
            for bit_pos in 0..8usize {
                let bit_idx = byte_idx * 8 + bit_pos;
                if bit_idx < bits.len() && bits[bit_idx] != 0 {
                    *byte |= 1 << (7 - bit_pos);
                }
            }
        }

        // Deinterleave: flat[b] → codeword[b % M][b / M]
        let mut codewords: Vec<Vec<u8>> = (0..m).map(|_| vec![0u8; rs_n]).collect();
        for (b, &byte) in flat.iter().enumerate() {
            codewords[b % m][b / m] = byte;
        }

        // Compute erasure sets per RS codeword from failed LDPC blocks
        let blk_end   = self.ldpc_blocks_decoded;
        let blk_start = blk_end - self.blocks_per_group;
        let mut erasure_sets: Vec<std::collections::BTreeSet<usize>> =
            (0..m).map(|_| std::collections::BTreeSet::new()).collect();

        for (offset, &converged) in self.ldpc_block_converged[blk_start..blk_end].iter().enumerate() {
            if !converged {
                let b_start = offset * ldpc_k / 8;
                let b_end   = ((offset + 1) * ldpc_k).div_ceil(8).min(total_bytes);
                for b in b_start..b_end {
                    let cw_idx   = b % m;
                    let byte_pos = b / m;
                    if byte_pos < rs_n {
                        erasure_sets[cw_idx].insert(byte_pos);
                    }
                }
            }
        }

        // Decode each RS codeword; track which RS packets are "real" (vs padding)
        let group_idx  = (self.ldpc_blocks_decoded - 1) / self.blocks_per_group;
        let pkt_base   = group_idx * m;

        for i in 0..m {
            let is_real = pkt_base + i < self.packets_this_frame;
            let era: Vec<usize> = erasure_sets[i].iter().copied().collect();

            match self.rs.decode(&codewords[i], &era) {
                Ok((info_bytes, stats)) => {
                    self.rs_payload.extend_from_slice(&info_bytes);
                    if is_real {
                        self.rs_packets_ok       += 1;
                        self.rs_errors_corrected += stats.errors_corrected as u32;
                        self.rs_erasures_used    += stats.erasures_used as u32;
                        let budget_used = 2 * stats.errors_corrected + stats.erasures_used;
                        let margin = 1.0 - budget_used as f32 / rs_2t as f32;
                        if margin < self.rs_min_margin_frac { self.rs_min_margin_frac = margin; }
                    }
                }
                Err(_) => {
                    self.rs_payload.extend(std::iter::repeat(0u8).take(rs_k));
                }
            }
        }
    }

    // ── EOT symbol ────────────────────────────────────────────────────────────

    /// Processes the end-of-transmission OFDM symbol.
    ///
    /// The EOT symbol is always BPSK.  The CRC32 of the assembled payload is
    /// transmitted in the first 32 data subcarrier positions (MSB first).
    fn process_eot(&mut self, ofdm_symbol: &[Complex32]) -> PushResult {
        // Safety flush: complete any partial LDPC blocks with zero-padding
        while self.ldpc_blocks_decoded < self.total_ldpc_blocks {
            while self.llr_buf.len() < LDPC_N {
                self.llr_buf.push(0.0);
            }
            self.drain_ldpc_blocks();
        }

        // Equalize the EOT symbol with BPSK demapper
        let eot_sym_idx = PREAMBLE_SYMS + self.data_syms_received;
        let eq      = self.equalizer.process(ofdm_symbol, eot_sym_idx);
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

        let failing_block_indices: Vec<u32> = self.ldpc_block_converged.iter()
            .enumerate()
            .filter(|(_, &c)| !c)
            .map(|(i, _)| i as u32)
            .collect();

        let metrics = FrameMetrics {
            channel_ber: if self.channel_bit_total > 0 {
                self.channel_bit_errors as f32 / self.channel_bit_total as f32
            } else {
                0.0
            },
            ldpc_converged:      self.ldpc_block_converged.iter().filter(|&&c| c).count() as u32,
            ldpc_total:          self.ldpc_block_converged.len() as u32,
            rs_errors_corrected: self.rs_errors_corrected,
            rs_erasures_used:    self.rs_erasures_used,
            rs_margin_frac:      self.rs_min_margin_frac,
            failing_block_indices,
        };

        PushResult::FrameComplete(FrameResult {
            payload:    std::mem::take(&mut self.rs_payload),
            packets_ok: self.rs_packets_ok,
            crc32_ok,
            pilot_snr_db,
            metrics,
        })
    }
}

// ── TransmissionReceiver ──────────────────────────────────────────────────────

/// Result returned by [`TransmissionReceiver::push_frame`].
pub enum TxPushResult {
    /// More super-frames expected.
    NeedMoreFrames,
    /// All super-frames received; full transmission assembled.
    Complete(TransmissionResult),
}

/// Assembled result of a complete multi-super-frame transmission.
#[derive(Debug, Clone)]
pub struct TransmissionResult {
    /// Concatenated RS-decoded payload, `total_packet_count × RS_K` bytes.
    pub payload: Vec<u8>,
    /// Total RS packets in the whole transmission.
    pub total_packet_count: u16,
    /// RS packets that decoded successfully (across all super-frames).
    pub packets_ok: u16,
    /// `true` if every super-frame passed its per-frame CRC32 check.
    pub all_crc32_ok: bool,
}

/// Reassembles a multi-super-frame transmission from individual [`FrameResult`]s.
///
/// Create with the `total_packet_count` and `rs_k` from the first super-frame's
/// header, then call [`push_frame`] for each decoded super-frame in order.
///
/// [`push_frame`]: TransmissionReceiver::push_frame
pub struct TransmissionReceiver {
    total_packet_count: u16,
    rs_k:               usize,
    payload:            Vec<u8>,
    packets_ok:         u16,
    all_crc32_ok:       bool,
}

impl TransmissionReceiver {
    /// Creates a new receiver for a transmission of `total_packet_count` RS packets,
    /// where each RS packet carries `rs_k` info bytes.
    pub fn new(total_packet_count: u16, rs_k: usize) -> Self {
        Self {
            total_packet_count,
            rs_k,
            payload:      vec![0u8; total_packet_count as usize * rs_k],
            packets_ok:   0,
            all_crc32_ok: true,
        }
    }

    /// Returns `true` if every super-frame pushed so far passed its CRC32 check.
    pub fn is_all_crc_ok(&self) -> bool { self.all_crc32_ok }

    /// Number of RS packets successfully decoded so far.
    pub fn packets_received(&self) -> u16 { self.packets_ok }

    /// Adds one decoded super-frame.
    pub fn push_frame(&mut self, result: FrameResult, header: &ModeHeader) -> TxPushResult {
        let rs_k        = self.rs_k;
        let byte_offset = header.packet_offset as usize * rs_k;
        let copy_len    = result.payload.len().min(
            self.payload.len().saturating_sub(byte_offset)
        );
        if copy_len > 0 {
            self.payload[byte_offset..byte_offset + copy_len]
                .copy_from_slice(&result.payload[..copy_len]);
        }
        self.packets_ok   += result.packets_ok;
        self.all_crc32_ok &= result.crc32_ok;

        let remaining = self.total_packet_count
                            .saturating_sub(header.packet_offset) as usize;
        let packets_this_frame = remaining.min(
            max_packets_per_frame(header.modulation, header.ldpc_rate, header.rs_level)
        );
        let next_offset = header.packet_offset as usize + packets_this_frame;

        if next_offset >= self.total_packet_count as usize {
            TxPushResult::Complete(TransmissionResult {
                payload:            std::mem::take(&mut self.payload),
                total_packet_count: self.total_packet_count,
                packets_ok:         self.packets_ok,
                all_crc32_ok:       self.all_crc32_ok,
            })
        } else {
            TxPushResult::NeedMoreFrames
        }
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
    use crate::{
        fec::rs::RsLevel,
        ofdm::{
            rx::{
                mode_detect::{LdpcRate, ModeHeader, Modulation},
                sync::channel_estimate_from_zc,
            },
            zc::build_preamble,
        },
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
    fn blocks_per_rs_group_sensible() {
        for &rate in &[LdpcRate::R1_2, LdpcRate::R2_3, LdpcRate::R3_4, LdpcRate::R5_6] {
            for &level in &[RsLevel::L0, RsLevel::L1, RsLevel::L2] {
                let code  = LdpcCode::for_rate(rate);
                let (rs_n, _, rs_2t) = level.params();
                let m     = rs_interleave_m(rs_2t, code.k);
                let bpg   = blocks_per_rs_group(code.k, rs_n, m);
                assert!(bpg >= 1, "rate={rate:?} level={level}: need ≥1 block");
                let carried = bpg * code.k;
                let needed  = m * rs_n * 8;
                assert!(carried >= needed,
                    "rate={rate:?} level={level}: {bpg}×k={} = {carried} < {needed}",
                    code.k);
                // Waste must be < one full LDPC block
                assert!(carried - needed < code.k,
                    "rate={rate:?} level={level}: waste {} ≥ k={}",
                    carried - needed, code.k);
            }
        }
    }

    #[test]
    fn expected_symbol_count() {
        let header = ModeHeader {
            modulation:         Modulation::Qam16,
            ldpc_rate:          LdpcRate::R3_4,
            rs_level:           RsLevel::L1,
            has_resync:         false,
            total_packet_count: 1,
            packet_offset:      0,
            crc_ok:             true,
        };
        let code = LdpcCode::for_rate(header.ldpc_rate);
        let (rs_n, _, rs_2t) = header.rs_level.params();
        let m   = rs_interleave_m(rs_2t, code.k);
        let bpg = blocks_per_rs_group(code.k, rs_n, m);
        let groups = 1usize.div_ceil(m);
        let total_coded   = groups * bpg * LDPC_N;
        let bits_per_ofdm = NUM_DATA * 4;
        let expected_data = total_coded.div_ceil(bits_per_ofdm);
        let h_est = flat_channel_est();
        let rx = FrameReceiver::new(&header, &h_est, 0.1);
        // total_data_syms = 2 × data_per_pass (RUNIN + DATA)
        assert_eq!(rx.total_data_syms, 2 * expected_data);
        assert_eq!(rx.expected_symbol_count(), 2 * expected_data + 1);
    }

    #[test]
    fn crc32_known_vector() {
        // CRC32 of "123456789" = 0xCBF43926 (IEEE 802.3)
        assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
    }

    /// Full loopback on a flat noiseless channel using the all-zero payload.
    ///
    /// All-zero payload → all-zero RS codewords (valid) → all-zero LDPC info bits
    /// → all-zero LDPC codeword (the zero vector satisfies every parity check).
    /// The BPSK channel maps 0 → +1, recovered with very high LLR → BP converges
    /// to all-zeros immediately → RS decodes → CRC32 matches.
    ///
    /// 802.11n z=81, R1/2: k=972, M=rs_interleave_m(64,972)=2,
    /// bpg=blocks_per_rs_group(972,255,2)=5 → 5×1944=9720 coded bits → 155 OFDM syms.
    #[test]
    fn loopback_bpsk_r12_one_packet() {
        let ldpc_rate  = LdpcRate::R1_2;
        let modulation = Modulation::Bpsk;
        let rs_level   = RsLevel::L1;
        let bps        = modulation.bits_per_symbol(); // 1

        let code = LdpcCode::for_rate(ldpc_rate);
        let k    = code.k; // 972 (802.11n z=81, R1/2)
        let (rs_n, rs_k, rs_2t) = rs_level.params(); // (255, 191, 64)
        let m    = rs_interleave_m(rs_2t, k);         // 2
        let bpg  = blocks_per_rs_group(k, rs_n, m);   // 5

        // 1 real RS packet → 1 group (ceil(1/M) = 1).
        // The group carries M=2 RS codewords; only the first is "real".
        // All-zero payload → all-zero RS codewords → all-zero LDPC info bits
        // → all-zero LDPC codewords (zero satisfies every parity check).
        let total_coded_bits = bpg * LDPC_N; // 5 × 1944 = 9720

        // ── Build OFDM symbols (2 passes: RUNIN + DATA) ─────────────────────
        let bits_per_ofdm = NUM_DATA * bps; // 35
        let data_syms_per_pass = total_coded_bits.div_ceil(bits_per_ofdm);
        let preamble_syms = PREAMBLE_SYMS;

        // Scramble exactly as TX does
        let mut all_coded_bits = vec![0u8; data_syms_per_pass * bits_per_ofdm];
        G3ruhScrambler::new().scramble_bits(&mut all_coded_bits);

        let bits_to_scattered = |bits: &[u8], sym_idx: usize| -> Vec<Complex32> {
            let n_data = crate::ofdm::drm_pilots::drm_num_data(sym_idx);
            let data_sc: Vec<Complex32> = (0..n_data)
                .map(|s| {
                    let b = bits.get(s * bps).copied().unwrap_or(0);
                    Complex32::new(if b == 0 { 1.0 } else { -1.0 }, 0.0)
                })
                .collect();
            crate::ofdm::tx::ofdm_modulate_scattered(&data_sc, sym_idx)
        };

        let mut ofdm_syms: Vec<Vec<Complex32>> = Vec::new();
        // Pass 1 (RUNIN)
        for i in 0..data_syms_per_pass {
            let start = i * bits_per_ofdm;
            let end   = (start + bits_per_ofdm).min(all_coded_bits.len());
            ofdm_syms.push(bits_to_scattered(&all_coded_bits[start..end], preamble_syms + i));
        }
        // Pass 2 (DATA) — identical bits, different sym_idx
        for i in 0..data_syms_per_pass {
            let start = i * bits_per_ofdm;
            let end   = (start + bits_per_ofdm).min(all_coded_bits.len());
            ofdm_syms.push(bits_to_scattered(&all_coded_bits[start..end], preamble_syms + data_syms_per_pass + i));
        }

        // EOT CRC32
        let rs_payload_bytes = m * rs_k;
        let crc = crc32_ieee(&vec![0u8; rs_payload_bytes]);
        let eot_sym_idx = preamble_syms + 2 * data_syms_per_pass;
        let n_eot_data = crate::ofdm::drm_pilots::drm_num_data(eot_sym_idx);
        let mut eot_data: Vec<Complex32> = (0..n_eot_data)
            .map(|i| {
                if i < 32 {
                    let bit = ((crc >> (31 - i)) & 1) as u8;
                    Complex32::new(if bit == 0 { 1.0 } else { -1.0 }, 0.0)
                } else {
                    Complex32::new(1.0, 0.0)
                }
            })
            .collect();
        ofdm_syms.push(crate::ofdm::tx::ofdm_modulate_scattered(&eot_data, eot_sym_idx));

        // ── Feed to FrameReceiver ─────────────────────────────────────────────
        let header = ModeHeader {
            modulation,
            ldpc_rate,
            rs_level,
            has_resync:         false,
            total_packet_count: 1,
            packet_offset:      0,
            crc_ok:             true,
        };
        let h_est = flat_channel_est();
        let mut rx = FrameReceiver::new(&header, &h_est, 0.1);

        assert_eq!(rx.total_data_syms, 2 * data_syms_per_pass,
            "FrameReceiver and test disagree on data symbol count");

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
        assert_eq!(frame.packets_ok, 1, "only the real RS packet counts");
        assert_eq!(&frame.payload[..rs_k], &vec![0u8; rs_k][..],
            "decoded payload must match original");
    }
}
