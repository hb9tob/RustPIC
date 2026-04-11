//! RustPIC transmitter — encodes a file into a 48 kHz 16-bit stereo WAV.
//!
//! USAGE: tx --input <file> --output <file.wav> [OPTIONS]
//!
//! OPTIONS:
//!   --input    <file>    File to transmit (required)
//!   --output   <file>    Output WAV file  (required)
//!   --mod      <mod>     bpsk | qpsk | 16qam | 32qam | 64qam  [default: qpsk]
//!   --rate     <rate>    1/2 | 2/3 | 3/4 | 5/6                [default: 3/4]
//!   --rs       <0|1|2>   RS protection level                   [default: 1]
//!   --no-resync          Disable re-sync ZC symbols (not recommended for BPSK/QPSK)
//!   --papr   <dB>        Soft-clip threshold dB above RMS  [default: 12]
//!                        Raises RMS ~5.7 dB vs raw OFDM to match FM-calibrated tools.
//!                        Lower values clip more aggressively (more RMS, more ICI).
//!   --no-papr            Disable PAPR clipping

use std::io::Write as IoWrite;

use rustpic::{
    fec::rs::RsLevel,
    ofdm::{
        beacon::build_beacon,
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
    callsign:   String,
    modulation: Modulation,
    ldpc_rate:  LdpcRate,
    rs_level:   RsLevel,
    resync:     bool,
    papr_clip:  Option<f32>,  // soft-clip threshold in dB above RMS; None = no clipping
    gain_db:    f32,           // additional gain applied after peak normalisation (dB)
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            input:      String::new(),
            output:     String::new(),
            callsign:   String::new(),
            modulation: Modulation::Qpsk,
            ldpc_rate:  LdpcRate::R3_4,
            rs_level:   RsLevel::L1,
            resync:     true,  // on by default — required for BPSK/QPSK over independent soundcards
            papr_clip:  None,
            gain_db:    0.0,
        }
    }
}

fn parse_args() -> Cfg {
    let mut cfg = Cfg::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input"     => { i += 1; cfg.input    = args[i].clone(); }
            "--output"    => { i += 1; cfg.output   = args[i].clone(); }
            "--callsign"  => { i += 1; cfg.callsign = args[i].to_uppercase(); }
            "--resync"    => { cfg.resync = true; }
            "--no-resync" => { cfg.resync = false; }
            "--papr" => {
                i += 1;
                cfg.papr_clip = Some(args[i].parse::<f32>()
                    .unwrap_or_else(|_| die("--papr expects a dB value, e.g. --papr 12")));
            }
            "--no-papr" => { cfg.papr_clip = None; }
            "--gain" => {
                i += 1;
                cfg.gain_db = args[i].parse::<f32>()
                    .unwrap_or_else(|_| die("--gain expects a dB value, e.g. --gain 6"));
            }
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
    if cfg.input.is_empty()    { die("--input is required"); }
    if cfg.output.is_empty()   { die("--output is required"); }
    if cfg.callsign.is_empty() { die("--callsign is required"); }
    cfg
}

fn print_help() {
    println!("RustPIC TX — encode a file into a 48 kHz 16-bit stereo WAV");
    println!();
    println!("USAGE: tx --input <file> --output <out.wav> --callsign <CALL> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --input     <file>   File to transmit (required)");
    println!("  --output    <wav>    Output WAV file  (required)");
    println!("  --callsign  <CALL>   Station callsign (required, e.g. HB9TOB)");
    println!("  --mod       <mod>    bpsk | qpsk | 16qam | 32qam | 64qam  [default: qpsk]");
    println!("  --rate      <rate>   1/2 | 2/3 | 3/4 | 5/6                [default: 3/4]");
    println!("  --rs        <0|1|2>  RS protection level                   [default: 1]");
    println!("  --no-resync          Disable re-sync ZC symbols (on by default)");
    println!("  --papr  <dB>         PAPR soft-clip threshold dB above RMS [default: 12]");
    println!("  --no-papr            Disable PAPR clipping");
    println!("  --gain  <dB>         Additional audio gain in dB, e.g. --gain 6  (positive = louder)");
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

    // Build beacon (1 kHz tone + ZC pair + ANN with callsign/mode/filename).
    let mode_str = format!("{} {}", mod_label(cfg.modulation), rate_label(cfg.ldpc_rate));
    let beacon = build_beacon(&cfg.callsign, &name, &mode_str);

    // Flatten beacon + all super-frames; take only the real part (baseband audio).
    let samples_f: Vec<f32> = beacon.iter().map(|s| s.re)
        .chain(frames.iter().flat_map(|f| f.iter().map(|s| s.re)))
        .collect();
    let n_base_samples = samples_f.len();

    // Optional soft-clip PAPR reduction (tanh-based, operates at 8 kHz).
    // Computes RMS over the whole signal, then clips at `clip_db` above RMS.
    // This raises the average level relative to the peak, increasing average
    // FM deviation to match single-carrier or DRM-style modulations.
    let samples_f = if let Some(clip_db) = cfg.papr_clip {
        papr_clip(&samples_f, clip_db)
    } else {
        samples_f
    };

    // Normalize so peak hits 90 % of i16 full-scale.
    let peak = samples_f.iter().map(|&x| x.abs()).fold(0f32, f32::max);
    let mut scale = if peak > 1e-9 { 0.9 * 32767.0 / peak } else { 32767.0 };

    // Optional additional gain (positive = louder, may cause soft clipping via clamp).
    if cfg.gain_db != 0.0 {
        scale *= 10_f32.powf(cfg.gain_db / 20.0);
    }

    // Upsample 8 kHz → 48 kHz (nearest-neighbour).
    let upsampled = upsample_linear(&samples_f, UP_FACTOR);

    // Convert to i16 (clamp handles occasional peaks when --gain is used).
    let samples_i16: Vec<i16> = upsampled
        .iter()
        .map(|&x| (x * scale).round().clamp(-32768.0, 32767.0) as i16)
        .collect();

    write_wav_stereo(&cfg.output, &samples_i16, FS_OUT)
        .unwrap_or_else(|e| die(&format!("cannot write '{}': {e}", cfg.output)));

    let duration_s = n_base_samples as f64 / FS_BASE as f64;
    let wav_bytes  = std::fs::metadata(&cfg.output).map(|m| m.len()).unwrap_or(0);
    use rustpic::ofdm::{beacon::BEACON_TOTAL_SYMS, params::SYMBOL_LEN as SYM};
    let beacon_ms = BEACON_TOTAL_SYMS * SYM * 1000 / FS_BASE as usize;

    println!("RustPIC TX");
    println!("  Callsign : {}", cfg.callsign);
    println!("  Input    : {} ({} bytes)", cfg.input, raw.len());
    println!("  Mode     : {}  {}  RS-L{}{}{}{}",
        mod_label(cfg.modulation),
        rate_label(cfg.ldpc_rate),
        match cfg.rs_level { RsLevel::L0 => "0", RsLevel::L1 => "1", RsLevel::L2 => "2" },
        if cfg.resync { "  resync" } else { "" },
        if let Some(db) = cfg.papr_clip { format!("  papr-clip={db}dB") } else { String::new() },
        if cfg.gain_db != 0.0 { format!("  gain={:+.1}dB", cfg.gain_db) } else { String::new() });
    println!("  RS pkts  : {} total, ≤ {}/frame, rs_k={}", (raw.len() + 4).div_ceil(rs_k), pkt_per_frame, rs_k);
    println!("  Beacon   : {} ms (1 kHz VOX tone + ZC + indicatif)", beacon_ms);
    println!("  Frames   : {}", n_frames);
    println!("  Duration : {:.1} s  ({} samples @ {} Hz, beacon included)", duration_s, n_base_samples, FS_BASE);
    println!("  Output   : {}  ({} bytes, 48 kHz 16-bit stereo)", cfg.output, wav_bytes);
}

// ── Audio helpers ─────────────────────────────────────────────────────────────

/// Soft-clip (tanh) PAPR reduction.
///
/// Computes the RMS of `samples`, then applies `x → thr·tanh(x/thr)`
/// where `thr = rms × 10^(clip_db/20)`.
///
/// Example: `clip_db = 6` → clips at 2×RMS → PAPR ≤ ~6 dB after renorm.
/// This raises the RMS (and therefore the average FM deviation) by several dB,
/// compensating for OFDM's inherently high PAPR.
fn papr_clip(samples: &[f32], clip_db: f32) -> Vec<f32> {
    let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 1e-9 { return samples.to_vec(); }
    let thr = rms * 10_f32.powf(clip_db / 20.0);
    samples.iter().map(|&x| thr * (x / thr).tanh()).collect()
}

/// Band-limited upsampler: zero-insert by `factor` then apply a Kaiser-windowed
/// sinc LPF at fc = 4 kHz (48 kHz rate).
///
/// ## Why this matters
///
/// ZOH (sample-repeat) upsampling leaves spectral images at k × 8 kHz in the
/// 48 kHz output.  The nearest image falls at 5469–8312 Hz, only −8 dB below
/// our 312–2531 Hz signal.  If the FM radio's audio input passes this range
/// (common in data/WFM modes), the image is transmitted over the air.  The FM
/// discriminator then returns it with 12–18 dB more noise (f² PSD), and the
/// RX anti-alias FIR rejects it by only ~9 dB at 5469 Hz.  The aliased image
/// degrades every subcarrier it folds onto.
///
/// This filter (51 taps, Kaiser β = 4.0, fc = 4 kHz @ 48 kHz, gain = 6)
/// attenuates the first image by > 45 dB while keeping 0–2531 Hz flat
/// (< 0.1 dB ripple).  Group delay = 25 samples at 48 kHz ≈ 4 samples at
/// 8 kHz — well within the cyclic prefix (32 samples at 8 kHz).
fn upsample_linear(src: &[f32], factor: usize) -> Vec<f32> {
    // 51-tap Kaiser (β=4.0) lowpass, fc=4000 Hz @ 48000 Hz, pre-scaled ×6.
    // Computed with scipy.signal.firwin(51, 4000/24000, window=('kaiser',4.0))
    // then multiplied by 6 to restore unity passband gain after zero-insertion.
    #[allow(clippy::excessive_precision)]
    const H: [f32; 51] = [
         0.00339190,  0.00000000, -0.00632466, -0.01420392, -0.02078061,
        -0.02239275, -0.01585454,  0.00000000,  0.02306087,  0.04756134,
         0.06497979,  0.06625534,  0.04487754,  0.00000000, -0.06146642,
        -0.12469687, -0.16919125, -0.17318569, -0.11930367,  0.00000000,
         0.17877731,  0.39694688,  0.62317635,  0.82084101,  0.95572857,
         1.00360694,
         0.95572857,  0.82084101,  0.62317635,  0.39694688,  0.17877731,
         0.00000000, -0.11930367, -0.17318569, -0.16919125, -0.12469687,
        -0.06146642,  0.00000000,  0.04487754,  0.06625534,  0.06497979,
         0.04756134,  0.02306087,  0.00000000, -0.01585454, -0.02239275,
        -0.02078061, -0.01420392, -0.00632466,  0.00000000,  0.00339190,
    ];
    const NTAPS: usize = 51;
    const HALF:  usize = (NTAPS - 1) / 2; // = 25, group delay in 48 kHz samples

    let n_out = src.len() * factor;

    (0..n_out).map(|out_idx| {
        // The zero-inserted upsampled stream is non-zero only at multiples of
        // `factor`.  For each output sample, iterate over filter taps and
        // accumulate only where the (shifted) tap position aligns with an input.
        H.iter().enumerate().map(|(tap, &hk)| {
            // Guard against underflow: out_idx + HALF >= tap is always true
            // when out_idx >= 0 and tap <= HALF (first half of filter), but
            // we need to check for the second half.
            if tap > out_idx + HALF { return 0.0; }
            let shifted = out_idx + HALF - tap;
            if shifted % factor == 0 {
                let src_idx = shifted / factor;
                if src_idx < src.len() { hk * src[src_idx] } else { 0.0 }
            } else {
                0.0
            }
        }).sum()
    }).collect()
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
