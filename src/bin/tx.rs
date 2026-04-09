//! RustPIC transmitter — encodes a file into a 48 kHz 16-bit stereo WAV.
//!
//! USAGE: tx --input <file> --output <file.wav> [OPTIONS]
//!
//! OPTIONS:
//!   --input  <file>      File to transmit (required)
//!   --output <file.wav>  Output WAV file  (required)
//!   --mod    <mod>       bpsk | qpsk | 16qam | 32qam | 64qam  [default: qpsk]
//!   --rate   <rate>      1/2 | 2/3 | 3/4 | 5/6                [default: 3/4]
//!   --rs     <0|1|2>     RS protection level                   [default: 1]
//!   --resync             Insert re-sync ZC symbols

use std::io::Write as IoWrite;

use rustpic::{
    fec::rs::RsLevel,
    ofdm::{
        rx::{
            frame::max_packets_per_frame,
            mode_detect::{LdpcRate, Modulation},
        },
        tx::frame::{build_transmission, FrameConfig},
    },
};

// ── Constants ─────────────────────────────────────────────────────────────────

const FS_BASE:   u32 = 8_000;
const FS_OUT:    u32 = 48_000;
const UP_FACTOR: usize = (FS_OUT / FS_BASE) as usize; // 6

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Cfg {
    input:      String,
    output:     String,
    modulation: Modulation,
    ldpc_rate:  LdpcRate,
    rs_level:   RsLevel,
    resync:     bool,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            input:      String::new(),
            output:     String::new(),
            modulation: Modulation::Qpsk,
            ldpc_rate:  LdpcRate::R3_4,
            rs_level:   RsLevel::L1,
            resync:     false,
        }
    }
}

fn parse_args() -> Cfg {
    let mut cfg = Cfg::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input"  => { i += 1; cfg.input  = args[i].clone(); }
            "--output" => { i += 1; cfg.output = args[i].clone(); }
            "--resync" => { cfg.resync = true; }
            "--mod"  => {
                i += 1;
                cfg.modulation = match args[i].as_str() {
                    "bpsk"  => Modulation::Bpsk,
                    "qpsk"  => Modulation::Qpsk,
                    "16qam" => Modulation::Qam16,
                    "32qam" => Modulation::Qam32,
                    "64qam" => Modulation::Qam64,
                    m => die(&format!("unknown modulation: {m}")),
                };
            }
            "--rate" => {
                i += 1;
                cfg.ldpc_rate = match args[i].as_str() {
                    "1/2" => LdpcRate::R1_2,
                    "2/3" => LdpcRate::R2_3,
                    "3/4" => LdpcRate::R3_4,
                    "5/6" => LdpcRate::R5_6,
                    r => die(&format!("unknown rate: {r}")),
                };
            }
            "--rs" => {
                i += 1;
                cfg.rs_level = match args[i].as_str() {
                    "0" => RsLevel::L0,
                    "1" => RsLevel::L1,
                    "2" => RsLevel::L2,
                    l => die(&format!("unknown rs level: {l}")),
                };
            }
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            a => die(&format!("unknown argument: {a}  (try --help)")),
        }
        i += 1;
    }
    if cfg.input.is_empty()  { die("--input is required"); }
    if cfg.output.is_empty() { die("--output is required"); }
    cfg
}

fn print_help() {
    println!("RustPIC TX — encode a file into a 48 kHz 16-bit stereo WAV");
    println!();
    println!("USAGE: tx --input <file> --output <out.wav> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --input  <file>      File to transmit (required)");
    println!("  --output <out.wav>   Output WAV file  (required)");
    println!("  --mod    <mod>       bpsk | qpsk | 16qam | 32qam | 64qam  [default: qpsk]");
    println!("  --rate   <rate>      1/2 | 2/3 | 3/4 | 5/6                [default: 3/4]");
    println!("  --rs     <0|1|2>     RS protection level                   [default: 1]");
    println!("  --resync             Insert re-sync ZC symbols");
}

fn die(msg: &str) -> ! {
    eprintln!("tx: {msg}");
    std::process::exit(1);
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    let raw = std::fs::read(&cfg.input)
        .unwrap_or_else(|e| die(&format!("cannot read '{}': {e}", cfg.input)));

    // Payload header: [4 bytes: data_len LE] [1 byte: name_len] [name bytes] [data bytes]
    let name = std::path::Path::new(&cfg.input)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("data");
    let name_bytes = name.as_bytes();
    let name_len   = name_bytes.len().min(255) as u8;
    let mut payload = Vec::with_capacity(5 + name_len as usize + raw.len());
    payload.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    payload.push(name_len);
    payload.extend_from_slice(&name_bytes[..name_len as usize]);
    payload.extend_from_slice(&raw);
    let payload = payload;

    let frame_cfg = FrameConfig {
        modulation: cfg.modulation,
        ldpc_rate:  cfg.ldpc_rate,
        rs_level:   cfg.rs_level,
        has_resync: cfg.resync,
    };

    let frames  = build_transmission(&payload, &frame_cfg);
    let n_frames = frames.len();
    let (_, rs_k, _) = cfg.rs_level.params();
    let pkt_per_frame = max_packets_per_frame(cfg.modulation, cfg.ldpc_rate, cfg.rs_level);

    // Flatten all super-frames; take only the real part (baseband audio).
    let samples_f: Vec<f32> = frames.iter()
        .flat_map(|f| f.iter().map(|s| s.re))
        .collect();
    let n_base_samples = samples_f.len();

    // Normalize so peak hits 90 % of i16 full-scale.
    let peak = samples_f.iter().map(|&x| x.abs()).fold(0f32, f32::max);
    let scale = if peak > 1e-9 { 0.9 * 32767.0 / peak } else { 32767.0 };

    // Upsample 8 kHz → 48 kHz (linear interpolation).
    let upsampled = upsample_linear(&samples_f, UP_FACTOR);

    // Convert to i16.
    let samples_i16: Vec<i16> = upsampled
        .iter()
        .map(|&x| (x * scale).round().clamp(-32768.0, 32767.0) as i16)
        .collect();

    write_wav_stereo(&cfg.output, &samples_i16, FS_OUT)
        .unwrap_or_else(|e| die(&format!("cannot write '{}': {e}", cfg.output)));

    let duration_s = n_base_samples as f64 / FS_BASE as f64;
    let wav_bytes  = std::fs::metadata(&cfg.output).map(|m| m.len()).unwrap_or(0);

    println!("RustPIC TX");
    println!("  Input    : {} ({} bytes)", cfg.input, raw.len());
    println!("  Mode     : {}  {}  RS-L{}{}",
        mod_label(cfg.modulation),
        rate_label(cfg.ldpc_rate),
        match cfg.rs_level { RsLevel::L0 => "0", RsLevel::L1 => "1", RsLevel::L2 => "2" },
        if cfg.resync { "  resync" } else { "" });
    println!("  RS pkts  : {} total, ≤ {}/frame, rs_k={}", (raw.len() + 4).div_ceil(rs_k), pkt_per_frame, rs_k);
    println!("  Frames   : {}", n_frames);
    println!("  Duration : {:.1} s  ({} samples @ {} Hz)", duration_s, n_base_samples, FS_BASE);
    println!("  Output   : {}  ({} bytes, 48 kHz 16-bit stereo)", cfg.output, wav_bytes);
}

// ── Audio helpers ─────────────────────────────────────────────────────────────

/// Nearest-neighbour upsampler: repeat each sample `factor` times.
///
/// Combined with the box-average downsampler used by the receiver,
/// this round-trip is exact (nearest-neighbour × box-average = identity),
/// avoiding the non-linear phase distortion introduced by linear interpolation.
fn upsample_linear(src: &[f32], factor: usize) -> Vec<f32> {
    src.iter().flat_map(|&x| std::iter::repeat(x).take(factor)).collect()
}

// ── WAV writer ────────────────────────────────────────────────────────────────

/// Write a 16-bit stereo PCM WAV (both channels carry the same mono signal).
fn write_wav_stereo(path: &str, samples: &[i16], rate: u32) -> std::io::Result<()> {
    let channels:    u16 = 2;
    let bits:        u16 = 16;
    let block_align: u16 = channels * (bits / 8);
    let byte_rate:   u32 = rate * block_align as u32;
    let data_bytes:  u32 = samples.len() as u32 * block_align as u32;
    let chunk_size:  u32 = 36 + data_bytes;

    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    wle32(&mut f, chunk_size)?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    wle32(&mut f, 16)?;
    wle16(&mut f, 1)?;           // PCM
    wle16(&mut f, channels)?;
    wle32(&mut f, rate)?;
    wle32(&mut f, byte_rate)?;
    wle16(&mut f, block_align)?;
    wle16(&mut f, bits)?;
    f.write_all(b"data")?;
    wle32(&mut f, data_bytes)?;
    let mut buf = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        let le = s.to_le_bytes();
        buf.extend_from_slice(&le); // left
        buf.extend_from_slice(&le); // right
    }
    f.write_all(&buf)
}

#[inline] fn wle16(f: &mut std::fs::File, v: u16) -> std::io::Result<()> { f.write_all(&v.to_le_bytes()) }
#[inline] fn wle32(f: &mut std::fs::File, v: u32) -> std::io::Result<()> { f.write_all(&v.to_le_bytes()) }

// ── Label helpers ─────────────────────────────────────────────────────────────

fn mod_label(m: Modulation) -> &'static str {
    match m {
        Modulation::Bpsk  => "BPSK",
        Modulation::Qpsk  => "QPSK",
        Modulation::Qam16 => "16-QAM",
        Modulation::Qam32 => "32-QAM",
        Modulation::Qam64 => "64-QAM",
    }
}
fn rate_label(r: LdpcRate) -> &'static str {
    match r {
        LdpcRate::R1_2 => "R1/2",
        LdpcRate::R2_3 => "R2/3",
        LdpcRate::R3_4 => "R3/4",
        LdpcRate::R5_6 => "R5/6",
    }
}
