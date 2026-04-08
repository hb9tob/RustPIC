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
//!   --mod MOD        Modulation: bpsk qpsk 16qam      [default: bpsk]
//!   --payload-size N Payload bytes (≥1)               [default: 191]
//!   --resync         Enable re-sync ZC symbols
//! ```

use rand::{rngs::StdRng, SeedableRng};

use rustpic::{
    fec::rs::RS_K,
    ofdm::{
        params::{CP_LEN, SYMBOL_LEN},
        rx::{
            frame::{FrameReceiver, PushResult},
            mode_detect::{decode_mode_header, LdpcRate, Modulation},
            sync::{correct_cfo, ZcCorrelator},
        },
        tx::frame::{build_frame, FrameConfig},
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
    println!("  --mod MOD          Modulation: bpsk qpsk 16qam    [default: bpsk]");
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

// ── Single TX→channel→RX run ─────────────────────────────────────────────────

fn run_one(
    payload:    &[u8],
    frame_cfg:  &FrameConfig,
    channel:    &SimChannel,
    rng:        &mut StdRng,
) -> Option<FrameEntry> {
    // TX
    let tx = build_frame(payload, frame_cfg);

    // Channel impairments
    let mut rx_buf = channel.apply(&tx, rng);

    // Sync
    let corr = ZcCorrelator::new(0.40, 0.30);
    let sync  = corr.find_sync(&rx_buf).ok()?;
    correct_cfo(&mut rx_buf[sync.preamble_start..], sync.cfo_hz);

    // Mode header
    let hdr_start = sync.header_start;
    if hdr_start + SYMBOL_LEN > rx_buf.len() { return None; }
    let hdr_win = &rx_buf[hdr_start + CP_LEN..hdr_start + SYMBOL_LEN];
    let header  = decode_mode_header(hdr_win, &sync.channel_est).ok()?;

    // Frame receiver
    let mut rx       = FrameReceiver::new(&header, &sync.channel_est, 0.1);
    let data_start   = hdr_start + SYMBOL_LEN;
    let n_expected   = rx.expected_symbol_count();
    let mut result   = None;

    for i in 0..n_expected {
        let pos = data_start + i * SYMBOL_LEN;
        if pos + SYMBOL_LEN > rx_buf.len() { return None; }
        match rx.push_symbol(&rx_buf[pos..pos + SYMBOL_LEN]) {
            PushResult::NeedMore             => {}
            PushResult::FrameComplete(frame) => { result = Some(frame); break; }
            PushResult::Error(_)             => return None,
        }
    }

    let frame = result?;
    let m     = &frame.metrics;

    let ldpc_ok   = if m.ldpc_total > 0 { m.ldpc_converged as f32 / m.ldpc_total as f32 } else { 0.0 };
    let rs_total  = header.packet_count as f32;
    let rs_ok     = if rs_total > 0.0 { frame.packets_ok as f32 / rs_total } else { 0.0 };

    Some(FrameEntry {
        ber:       m.channel_ber,
        ldpc_ok,
        ldpc_fail: 1.0 - ldpc_ok,
        rs_ok,
        rs_fail:   1.0 - rs_ok,
        rs_margin: m.rs_margin_frac,
        crc_ok:    if frame.crc32_ok { 1.0 } else { 0.0 },
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
            match run_one(&payload, &frame_cfg, &channel, &mut rng) {
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
