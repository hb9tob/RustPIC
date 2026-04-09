//! RustPIC channel simulation runner.
//!
//! Sweeps SNR and reports per-point statistics: channel BER, LDPC convergence,
//! RS correction stats, and end-to-end CRC32 success rate.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --bin simtest -- [OPTIONS]
//!
//! OPTIONS:
//!   --snr-min F      Minimum SNR in dB                [default: 0.0]
//!   --snr-max F      Maximum SNR in dB                [default: 20.0]
//!   --snr-step F     SNR step in dB                   [default: 2.0]
//!   --ppm F          Max clock offset (ppm)           [default: 20.0]
//!   --guard N        Max guard samples                [default: 1000]
//!   --runs N         Frames per SNR point             [default: 10]
//!   --rate RATE      LDPC rate: 1/2 2/3 3/4 5/6       [default: 1/2]
//!   --mod MOD        Modulation: bpsk qpsk 16qam 32qam 64qam  [default: bpsk]
//!   --payload-size N Payload bytes (≥1)               [default: 191]
//!   --resync         Enable re-sync ZC symbols
//! ```

use rand::{rngs::StdRng, SeedableRng};

use rustpic::{
    fec::rs::RS_K,
    ofdm::{
        params::{CP_LEN, RESYNC_PERIOD, SYMBOL_LEN},
        rx::{
            frame::{FrameReceiver, PushResult, TransmissionReceiver, TxPushResult,
                    max_packets_per_frame},
            mode_detect::{decode_mode_header, LdpcRate, Modulation},
            sync::{correct_cfo, find_resync_zc, ZcCorrelator},
        },
        tx::frame::{build_transmission, FrameConfig},
    },
    sim::SimChannel,
};

// ── CLI config ────────────────────────────────────────────────────────────────

struct Config {
    snr_min:      f32,
    snr_max:      f32,
    snr_step:     f32,
    ppm:          f32,
    guard:        usize,
    runs:         usize,
    ldpc_rate:    LdpcRate,
    modulation:   Modulation,
    payload_size: usize,
    has_resync:   bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            snr_min:      0.0,
            snr_max:      20.0,
            snr_step:     2.0,
            ppm:          20.0,
            guard:        1000,
            runs:         10,
            ldpc_rate:    LdpcRate::R1_2,
            modulation:   Modulation::Bpsk,
            payload_size: RS_K,
            has_resync:   false,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg  = Config::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--snr-min"  => { i += 1; cfg.snr_min      = args[i].parse().expect("--snr-min"); }
            "--snr-max"  => { i += 1; cfg.snr_max       = args[i].parse().expect("--snr-max"); }
            "--snr-step" => { i += 1; cfg.snr_step      = args[i].parse().expect("--snr-step"); }
            "--ppm"      => { i += 1; cfg.ppm           = args[i].parse().expect("--ppm"); }
            "--guard"    => { i += 1; cfg.guard         = args[i].parse().expect("--guard"); }
            "--runs"     => { i += 1; cfg.runs          = args[i].parse().expect("--runs"); }
            "--payload-size" => { i += 1; cfg.payload_size = args[i].parse().expect("--payload-size"); }
            "--resync"   => { cfg.has_resync = true; }
            "--rate" => {
                i += 1;
                cfg.ldpc_rate = match args[i].as_str() {
                    "1/2" => LdpcRate::R1_2,
                    "2/3" => LdpcRate::R2_3,
                    "3/4" => LdpcRate::R3_4,
                    "5/6" => LdpcRate::R5_6,
                    r => { eprintln!("unknown rate: {r}"); std::process::exit(1); }
                };
            }
            "--mod" => {
                i += 1;
                cfg.modulation = match args[i].as_str() {
                    "bpsk"  => Modulation::Bpsk,
                    "qpsk"  => Modulation::Qpsk,
                    "16qam" => Modulation::Qam16,
                    "32qam" => Modulation::Qam32,
                    "64qam" => Modulation::Qam64,
                    m => { eprintln!("unknown modulation: {m}"); std::process::exit(1); }
                };
            }
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            a => { eprintln!("unknown argument: {a}  (try --help)"); std::process::exit(1); }
        }
        i += 1;
    }
    cfg
}

fn print_help() {
    println!("RustPIC channel simulation");
    println!();
    println!("USAGE: simtest [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --snr-min F        Minimum SNR in dB              [default: 0.0]");
    println!("  --snr-max F        Maximum SNR in dB              [default: 20.0]");
    println!("  --snr-step F       SNR step in dB                 [default: 2.0]");
    println!("  --ppm F            Max clock offset ppm           [default: 20.0]");
    println!("  --guard N          Max guard samples              [default: 1000]");
    println!("  --runs N           Frames per SNR point           [default: 10]");
    println!("  --rate RATE        LDPC rate: 1/2 2/3 3/4 5/6     [default: 1/2]");
    println!("  --mod MOD          Modulation: bpsk qpsk 16qam 32qam 64qam  [default: bpsk]");
    println!("  --payload-size N   Payload bytes                  [default: 191]");
    println!("  --resync           Enable re-sync ZC symbols");
}

// ── Per-frame result ──────────────────────────────────────────────────────────

struct FrameEntry {
    ber:       f32,
    ldpc_ok:   f32, // fraction of LDPC blocks converged
    ldpc_fail: f32,
    rs_ok:     f32, // fraction of RS packets decoded
    rs_fail:   f32,
    rs_margin: f32, // min RS budget remaining (fraction)
    crc_ok:    f32, // 1.0 or 0.0
}

// ── TX→channel→RX: full multi-super-frame transmission ───────────────────────

fn run_transmission(
    payload:   &[u8],
    frame_cfg: &FrameConfig,
    channel:   &SimChannel,
    rng:       &mut StdRng,
) -> Option<FrameEntry> {
    let tx_frames = build_transmission(payload, frame_cfg);
    let total_pkt = payload.len().div_ceil(RS_K).max(1) as u16;
    let mut tx_rx = TransmissionReceiver::new(total_pkt);

    // Aggregate metrics across super-frames
    let mut sum_ber:           f32 = 0.0;
    let mut sum_ldpc_ok:       f32 = 0.0;
    let mut sum_ldpc_total:    u32 = 0;
    let mut sum_rs_ok:         u32 = 0;
    let mut sum_rs_total:      u32 = 0;
    let mut min_rs_margin:     f32 = 1.0;
    let mut frames_processed:  usize = 0;

    for tx_frame in &tx_frames {
        // Channel impairments (independent per super-frame: fresh guard + noise)
        let mut rx_buf = channel.apply(tx_frame, rng);

        // Sync — limit search window to the preamble region only.
        // Without this limit, find_sync searches the entire frame and can
        // mistake a resync ZC (identical to the preamble) for ZC#1.
        let sync_window_len = (channel.guard_samples_max + 3 * SYMBOL_LEN + 64)
            .min(rx_buf.len());
        let corr = ZcCorrelator::new(0.40, 0.30);
        let sync  = match corr.find_sync(&rx_buf[..sync_window_len]) {
            Ok(s) => s,
            Err(_) => return None,
        };
        correct_cfo(&mut rx_buf[sync.preamble_start..], sync.cfo_hz);

        // Mode header
        let hdr_start = sync.header_start;
        if hdr_start + SYMBOL_LEN > rx_buf.len() { return None; }
        let hdr_win = &rx_buf[hdr_start + CP_LEN..hdr_start + SYMBOL_LEN];
        let header  = decode_mode_header(hdr_win, &sync.channel_est).ok()?;

        // Frame receiver — with timing tracking at resync ZC positions
        let mut rx         = FrameReceiver::new(&header, &sync.channel_est, 0.1);
        let n_expected     = rx.expected_symbol_count();

        // Zero-pad the buffer so the nominal positions (computed with rate_est=1.0)
        // always fall within bounds.  Clock drift compresses the resampled buffer
        // slightly; without padding the EOT position can be 1–15 samples past the
        // end, which would force clamping and create a phase discontinuity versus
        // the EMA-estimated channel.  A few zero samples beyond the EOT are harmless.
        {
            let nominal_end = sync.preamble_start + (3 + n_expected) * SYMBOL_LEN;
            if rx_buf.len() < nominal_end {
                rx_buf.resize(nominal_end, num_complex::Complex32::new(0.0, 0.0));
            }
        }
        let max_per        = max_packets_per_frame(header.modulation, header.ldpc_rate);
        let remaining      = header.total_packet_count
                                .saturating_sub(header.packet_offset) as usize;
        let pkts_this      = remaining.min(max_per);

        // Track the current read position (in samples) within rx_buf.
        //
        // Clock-drift tracking strategy:
        //   At each resync ZC we find its exact integer position (found_int) and
        //   update rate_est from the absolute origin for long-term accuracy.
        //
        //   Data symbols after a resync are placed at found_int + s×SYMBOL_LEN
        //   (integer steps from the integer ZC position), NOT at
        //   round(found_exact + s×SL_eff).  The reason: when found_frac ≈ ±0.5,
        //   round(found_exact + SL_eff) can differ from found_int + SYMBOL_LEN by
        //   1 sample, causing a ~1-sample timing discontinuity between the ZC
        //   channel estimate and the first data symbol.  At bin 81, 1 sample ≈
        //   114° — catastrophic for 16-QAM.  Using integer steps avoids this.
        //   The residual drift (0.003 samples/sym at 10 ppm) is handled by the
        //   pilot EMA.
        let total_data   = rx.total_data_syms();
        // Absolute sample index of ZC#1 in rx_buf (origin of all timing math).
        let origin: i64  = sync.preamble_start as i64;
        // Estimated RX/TX sample-rate ratio (updated after each resync ZC).
        let mut rate_est: f64 = 1.0;
        // k_sym: absolute symbol count from ZC#1 (0 = ZC#1, 1 = ZC#2, 2 = header,
        // 3 = first data symbol, …).
        let mut k_sym: i64    = 3; // loop enters at first data symbol
        // Mirror FrameReceiver's syms_in_group counter to detect resync slots.
        let mut syms_in_group: usize = 0;
        let mut data_received:  usize = 0;
        let mut frame_result = None;
        // Integer position of the most recent resync ZC (None = use rate_est from origin).
        let mut last_resync_int: Option<(usize, i64)> = None; // (found_int, k_sym_at_resync)
        let debug2 = std::env::var("SIMTEST_DEBUG2").is_ok();

        for _sym_idx in 0..n_expected {
            // Determine symbol type, mirroring FrameReceiver::push_symbol logic.
            let is_eot    = data_received >= total_data;
            let is_resync = !is_eot
                && header.has_resync
                && syms_in_group == RESYNC_PERIOD;

            // Predicted position of the current symbol.
            //
            // For data symbols after a resync: step by exactly SYMBOL_LEN from the
            // integer ZC position.  This avoids the rounding-boundary discontinuity
            // described above.  For the pre-resync stretch and the resync ZC itself
            // (found by correlator search), use rate_est from the absolute origin.
            let actual_pos = if let Some((ri, rk)) = last_resync_int {
                // Integer steps from the last found ZC position.
                ri + (k_sym - rk) as usize * SYMBOL_LEN
            } else {
                ((origin as f64
                  + k_sym as f64 * SYMBOL_LEN as f64 / rate_est)
                 .round() as i64)
                .max(0) as usize
            };
            // Buffer was pre-padded to nominal_end; actual_pos should always be valid.

            let sym_pos = if is_resync {
                // Find the actual ZC position with sub-sample precision.
                if let Some((found_int, found_frac)) =
                    find_resync_zc(&rx_buf, actual_pos, 64, 0.20)
                {
                    let found_exact = found_int as f64 + found_frac as f64;

                    // Update rate estimate from absolute origin for long-term accuracy.
                    let nominal = k_sym as f64 * SYMBOL_LEN as f64;
                    let observed = found_exact - origin as f64;
                    if observed > 0.0 {
                        rate_est = nominal / observed;
                    }

                    // When found_frac < 0, found_int = ceil(true_pos): the FFT
                    // window would extend past the ZC DFT boundary into the next
                    // symbol's CP, causing ISI in the channel estimate.  Shift
                    // down by one sample so we always use floor(true_pos), which
                    // reads into the ZC's own cyclic prefix instead (safe).
                    let zc_pos = if found_frac < 0.0 && found_int > 0 {
                        found_int - 1
                    } else {
                        found_int
                    };

                    // Record the ZC position; subsequent data symbols will
                    // be positioned relative to it with integer steps.
                    last_resync_int = Some((zc_pos, k_sym));

                    // The simtest corrects timing in the time domain; no extra
                    // frequency-domain drift correction needed.
                    rx.set_timing_drift_per_sym(0.0);

                    if debug2 && data_received + 6 >= total_data {
                        eprintln!("  resync k_sym={k_sym} actual_pos={actual_pos} found_int={found_int} found_frac={found_frac:.3} zc_pos={zc_pos} data_received={data_received}/{total_data}");
                    }
                    zc_pos
                } else {
                    if debug2 {
                        eprintln!("  resync k_sym={k_sym} actual_pos={actual_pos} FIND_FAILED data_received={data_received}/{total_data}");
                    }
                    actual_pos
                }
            } else {
                if debug2 && data_received + 6 >= total_data {
                    let rms: f32 = rx_buf[actual_pos..actual_pos+SYMBOL_LEN]
                        .iter().map(|s| s.norm_sqr()).sum::<f32>() / SYMBOL_LEN as f32;
                    // Also check nominal (no-drift) position
                    let nominal_pos = ((origin as f64 + k_sym as f64 * SYMBOL_LEN as f64).round() as i64).max(0) as usize;
                    let rms_nom: f32 = if nominal_pos + SYMBOL_LEN <= rx_buf.len() {
                        rx_buf[nominal_pos..nominal_pos+SYMBOL_LEN]
                            .iter().map(|s| s.norm_sqr()).sum::<f32>() / SYMBOL_LEN as f32
                    } else { 0.0 };
                    eprintln!("  data  k_sym={k_sym} sym_pos={actual_pos}(nom={nominal_pos}) data_idx={data_received}/{total_data} rms={:.4} rms_nom={:.4} is_eot={is_eot}",
                        rms.sqrt(), rms_nom.sqrt());
                }
                actual_pos
            };

            if sym_pos + SYMBOL_LEN > rx_buf.len() { return None; }
            match rx.push_symbol(&rx_buf[sym_pos..sym_pos + SYMBOL_LEN]) {
                PushResult::NeedMore             => {}
                PushResult::FrameComplete(frame) => { frame_result = Some(frame); break; }
                PushResult::Error(_)             => return None,
            }

            // Update local state mirrors
            if is_resync {
                syms_in_group = 0;
            } else if !is_eot {
                data_received  += 1;
                syms_in_group  += 1;
            }
            k_sym += 1;
        }

        let frame = frame_result?;
        let m     = &frame.metrics;

        // Accumulate metrics
        sum_ber       += m.channel_ber;
        sum_ldpc_ok   += m.ldpc_converged as f32;
        sum_ldpc_total += m.ldpc_total;
        sum_rs_ok     += frame.packets_ok as u32;
        sum_rs_total  += pkts_this as u32;
        if m.rs_margin_frac < min_rs_margin { min_rs_margin = m.rs_margin_frac; }
        frames_processed += 1;

        if std::env::var("SIMTEST_DEBUG").is_ok() && !frame.crc32_ok {
            eprintln!("DBG crc fail: frame={} pkt_offset={} pkts={} ldpc_conv={}/{} rs_ok={}/{} margin={:.2} failing_blocks={:?}",
                frames_processed - 1, header.packet_offset, pkts_this,
                m.ldpc_converged, m.ldpc_total, frame.packets_ok, pkts_this,
                m.rs_margin_frac, m.failing_block_indices);
        }
        if debug2 && !frame.crc32_ok {
            eprintln!("  -> CRC FAIL frame={} origin={origin} total_data={total_data}", frames_processed - 1);
        }

        let is_complete = matches!(tx_rx.push_frame(frame, &header), TxPushResult::Complete(_));
        if is_complete { break; }
    }

    if frames_processed == 0 { return None; }

    // Retrieve the assembled result (if Complete was reached)
    // For the stats we use the accumulated per-frame metrics.
    let n = frames_processed as f32;
    let ldpc_ok = if sum_ldpc_total > 0 { sum_ldpc_ok / sum_ldpc_total as f32 } else { 0.0 };
    let rs_ok   = if sum_rs_total   > 0 { sum_rs_ok as f32 / sum_rs_total as f32 } else { 0.0 };
    // all_crc_ok() is tracked inside tx_rx; expose via a helper or read field
    let crc_ok  = tx_rx.is_all_crc_ok();

    Some(FrameEntry {
        ber:       sum_ber / n,
        ldpc_ok,
        ldpc_fail: 1.0 - ldpc_ok,
        rs_ok,
        rs_fail:   1.0 - rs_ok,
        rs_margin: min_rs_margin,
        crc_ok:    if crc_ok { 1.0 } else { 0.0 },
    })
}

// ── Accumulator ───────────────────────────────────────────────────────────────

#[derive(Default)]
struct Acc {
    ber:       f32,
    ldpc_ok:   f32,
    ldpc_fail: f32,
    rs_ok:     f32,
    rs_fail:   f32,
    rs_margin: f32,
    crc_ok:    f32,
    count:     usize,
}

impl Acc {
    fn add(&mut self, e: &FrameEntry) {
        self.ber       += e.ber;
        self.ldpc_ok   += e.ldpc_ok;
        self.ldpc_fail += e.ldpc_fail;
        self.rs_ok     += e.rs_ok;
        self.rs_fail   += e.rs_fail;
        self.rs_margin += e.rs_margin;
        self.crc_ok    += e.crc_ok;
        self.count     += 1;
    }

    fn avg(&self, x: f32) -> f32 { if self.count > 0 { x / self.count as f32 } else { 0.0 } }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    let frame_cfg = FrameConfig {
        modulation: cfg.modulation,
        ldpc_rate:  cfg.ldpc_rate,
        has_resync: cfg.has_resync,
    };

    // Deterministic non-trivial payload (pseudo-random bytes)
    let payload: Vec<u8> = (0..cfg.payload_size)
        .map(|i| ((i.wrapping_mul(167).wrapping_add(13)) & 0xFF) as u8)
        .collect();

    // ── Header ────────────────────────────────────────────────────────────────
    println!("RustPIC channel simulation");
    println!(
        "  modulation: {:?}  LDPC: {:?}  resync: {}",
        cfg.modulation, cfg.ldpc_rate, cfg.has_resync
    );
    println!(
        "  ppm ±{:.0}  guard 0..{}  {}/SNR  payload {} B",
        cfg.ppm, cfg.guard, cfg.runs, cfg.payload_size
    );
    println!();

    let hdr = format!(
        "{:>8}  {:>8}  {:>9}  {:>10}  {:>7}  {:>8}  {:>11}  {:>9}",
        "SNR(dB)", "ch.BER%", "LDPC_ok%", "LDPC_fail%",
        "RS_ok%", "RS_fail%", "RS_margin%", "CRC32_ok%"
    );
    println!("{hdr}");
    println!("{}", "-".repeat(hdr.len()));

    // ── SNR sweep ─────────────────────────────────────────────────────────────
    let mut snr = cfg.snr_min;
    while snr <= cfg.snr_max + 1e-3 {
        let channel = SimChannel {
            snr_db:            snr,
            clock_ppm_max:     cfg.ppm,
            guard_samples_max: cfg.guard,
        };

        let mut acc        = Acc::default();
        let mut sync_fails = 0usize;
        // Seed from quantised SNR for reproducibility
        let mut rng = StdRng::seed_from_u64(0xC0DE_BEEFu64 ^ (snr * 100.0) as u64);

        for _ in 0..cfg.runs {
            match run_transmission(&payload, &frame_cfg, &channel, &mut rng) {
                Some(e) => acc.add(&e),
                None    => sync_fails += 1,
            }
        }

        if acc.count > 0 {
            let sync_note = if sync_fails > 0 {
                format!("  [{sync_fails} sync/decode fail]")
            } else {
                String::new()
            };
            println!(
                "{:>8.1}  {:>8.2}  {:>9.1}  {:>10.1}  {:>7.1}  {:>8.1}  {:>11.1}  {:>9.1}{}",
                snr,
                acc.avg(acc.ber) * 100.0,
                acc.avg(acc.ldpc_ok) * 100.0,
                acc.avg(acc.ldpc_fail) * 100.0,
                acc.avg(acc.rs_ok) * 100.0,
                acc.avg(acc.rs_fail) * 100.0,
                acc.avg(acc.rs_margin) * 100.0,
                acc.avg(acc.crc_ok) * 100.0,
                sync_note,
            );
        } else {
            println!("{:>8.1}  (all {} frames: sync/decode failure)", snr, cfg.runs);
        }

        snr += cfg.snr_step;
    }
}
