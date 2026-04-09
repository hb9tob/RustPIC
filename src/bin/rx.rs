//! RustPIC receiver — decodes all transmissions found in a 48 kHz 16-bit stereo WAV.
//!
//! USAGE: rx --input <file.wav> [--outdir <dir>]
//!
//! OPTIONS:
//!   --input  <file.wav>  WAV file to scan (required)
//!   --outdir <dir>       Directory for decoded files  [default: .]

use num_complex::Complex32;

use rustpic::{
    ofdm::{
        params::{CP_LEN, RESYNC_PERIOD, SYMBOL_LEN},
        rx::{
            frame::{
                FrameReceiver, PushResult, TransmissionReceiver, TxPushResult,
                max_packets_per_frame,
            },
            mode_detect::{decode_mode_header, LdpcRate, ModeHeader, Modulation},
            sync::{correct_cfo, find_resync_zc, ZcCorrelator},
        },
    },
};

// ── Constants ─────────────────────────────────────────────────────────────────

const FS_IN:       u32   = 48_000;
const FS_BASE:     u32   = 8_000;
const DOWN_FACTOR: usize = (FS_IN / FS_BASE) as usize; // 6

/// Scan window width used when sliding through unknown audio looking for a preamble.
/// 4×SYMBOL_LEN (1152 samples) is large enough for ZcCorrelator and small enough
/// that the first resync ZC (15 symbols after the true preamble) never falls inside.
const SCAN_WINDOW: usize = 4 * SYMBOL_LEN;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Cfg {
    input:  String,
    outdir: String,
}

fn parse_args() -> Cfg {
    let mut input  = String::new();
    let mut outdir = String::from(".");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input"  => { i += 1; input  = args[i].clone(); }
            "--outdir" => { i += 1; outdir = args[i].clone(); }
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            a => die(&format!("unknown argument: {a}  (try --help)")),
        }
        i += 1;
    }
    if input.is_empty() { die("--input is required"); }
    Cfg { input, outdir }
}

fn print_help() {
    println!("RustPIC RX — decode all transmissions found in a 48 kHz WAV");
    println!();
    println!("USAGE: rx --input <in.wav> [--outdir <dir>]");
    println!();
    println!("OPTIONS:");
    println!("  --input  <file.wav>  WAV file to scan (required)");
    println!("  --outdir <dir>       Output directory for decoded files  [default: .]");
    println!();
    println!("The WAV must be 48 kHz or 8 kHz, 16-bit PCM (mono or stereo).");
    println!("Multiple transmissions in the same file are all decoded.");
}

fn die(msg: &str) -> ! {
    eprintln!("rx: {msg}");
    std::process::exit(1);
}

// ── Per-frame display stats ───────────────────────────────────────────────────

struct FrameStat {
    header:       ModeHeader,
    pilot_snr_db: f32,
    ch_ber_pct:   f32,
    ldpc_ok_pct:  f32,
    rs_ok:        u16,
    rs_total:     usize,
    rs_margin:    f32,
    crc32_ok:     bool,
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    // ── Load WAV ──────────────────────────────────────────────────────────────
    let (wav_samples, wav_rate) = read_wav_mono(&cfg.input)
        .unwrap_or_else(|e| die(&format!("cannot read '{}': {e}", cfg.input)));

    // ── Resample to 8 kHz ─────────────────────────────────────────────────────
    let samples_f32: Vec<f32> = match wav_rate {
        r if r == FS_BASE => wav_samples,
        r if r == FS_IN   => downsample_fir(&wav_samples, DOWN_FACTOR),
        r => die(&format!("unsupported sample rate {r} Hz (expected 48000 or 8000)")),
    };

    // ── Convert to Complex32 (imaginary = 0) ──────────────────────────────────
    let audio: Vec<Complex32> = samples_f32.iter()
        .map(|&x| Complex32::new(x, 0.0))
        .collect();

    let duration_s = audio.len() as f64 / FS_BASE as f64;
    println!("RustPIC RX");
    println!("  WAV      : {} ({} kHz, {:.1} s)",
        cfg.input, wav_rate / 1000, duration_s);

    // ── Scan and decode all super-frames ────────────────────────────────────
    let mut search_start:  usize = 0;
    let mut tx_rx:         Option<TransmissionReceiver> = None;
    let mut frame_stats:   Vec<FrameStat> = Vec::new();
    let mut n_files:       usize = 0;       // transmissions successfully decoded
    let mut n_frames_seen: usize = 0;       // total super-frames across all transmissions

    let corr = ZcCorrelator::new(0.40, 0.30);
    let mut next_progress = audio.len() / 10;

    while search_start + 3 * SYMBOL_LEN <= audio.len() {
        // Progress indicator for long audio files (every ~10%).
        if search_start >= next_progress {
            let pct = search_start * 100 / audio.len();
            eprint!("\r  Scanning  : {pct:3}% ({:.1} s)…    ", search_start as f64 / FS_BASE as f64);
            next_progress = search_start + audio.len() / 10;
        }

        let window_end = (search_start + SCAN_WINDOW).min(audio.len());
        let sync = match corr.find_sync(&audio[search_start..window_end]) {
            Ok(s)  => s,
            Err(_) => { search_start += SYMBOL_LEN; continue; }
        };

        let abs_preamble = search_start + sync.preamble_start;

        // CFO correction on a working copy starting at the preamble.
        let mut buf: Vec<Complex32> = audio[abs_preamble..].to_vec();
        correct_cfo(&mut buf, sync.cfo_hz);

        // Decode mode header.
        let hdr_rel = sync.header_start - sync.preamble_start;
        if hdr_rel + SYMBOL_LEN > buf.len() {
            search_start += SYMBOL_LEN;
            continue;
        }
        let hdr_win = &buf[hdr_rel + CP_LEN..hdr_rel + SYMBOL_LEN];
        let header  = match decode_mode_header(hdr_win, &sync.channel_est) {
            Ok(h)  => h,
            Err(e) => {
                eprintln!("  [frame {}] header decode error: {e}", frame_stats.len());
                search_start += SYMBOL_LEN;
                continue;
            }
        };

        // Initialise TransmissionReceiver from the first decoded header.
        if tx_rx.is_none() {
            let (_, rs_k, _) = header.rs_level.params();
            tx_rx = Some(TransmissionReceiver::new(header.total_packet_count, rs_k));
        }

        // Prepare frame receiver.
        let mut rx       = FrameReceiver::new(&header, &sync.channel_est, 0.1);
        let n_expected   = rx.expected_symbol_count();
        let total_data   = rx.total_data_syms();
        let pkts_this    = {
            let remaining = header.total_packet_count
                                .saturating_sub(header.packet_offset) as usize;
            remaining.min(max_packets_per_frame(
                header.modulation, header.ldpc_rate, header.rs_level))
        };

        // Pre-pad buffer against clock-drift tail truncation.
        {
            let nominal_end = (3 + n_expected) * SYMBOL_LEN;
            if buf.len() < nominal_end {
                buf.resize(nominal_end, Complex32::new(0.0, 0.0));
            }
        }

        // Symbol-by-symbol processing with drift tracking (mirrors simtest).
        let mut rate_est:      f64   = 1.0;
        let mut k_sym:         i64   = 3; // 0=ZC1 1=ZC2 2=hdr 3=first data
        let mut syms_in_group: usize = 0;
        let mut data_received: usize = 0;
        let mut last_resync: Option<(usize, i64)> = None;
        let mut frame_result = None;

        for _ in 0..n_expected {
            let is_eot    = data_received >= total_data;
            let is_resync = !is_eot && header.has_resync && syms_in_group == RESYNC_PERIOD;

            let actual_pos = if let Some((ri, rk)) = last_resync {
                ri + (k_sym - rk) as usize * SYMBOL_LEN
            } else {
                ((k_sym as f64 * SYMBOL_LEN as f64 / rate_est).round() as i64)
                    .max(0) as usize
            };

            let sym_pos = if is_resync {
                if let Some((found_int, found_frac)) =
                    find_resync_zc(&buf, actual_pos, 64, 0.20)
                {
                    let found_exact = found_int as f64 + found_frac as f64;
                    let nominal     = k_sym as f64 * SYMBOL_LEN as f64;
                    if found_exact > 0.0 { rate_est = nominal / found_exact; }
                    let zc_pos = if found_frac < 0.0 && found_int > 0 {
                        found_int - 1
                    } else {
                        found_int
                    };
                    last_resync = Some((zc_pos, k_sym));
                    rx.set_timing_drift_per_sym(0.0);
                    zc_pos
                } else {
                    actual_pos
                }
            } else {
                actual_pos
            };

            if sym_pos + SYMBOL_LEN > buf.len() { break; }

            match rx.push_symbol(&buf[sym_pos..sym_pos + SYMBOL_LEN]) {
                PushResult::NeedMore             => {}
                PushResult::FrameComplete(frame) => { frame_result = Some(frame); break; }
                PushResult::Error(_)             => break,
            }

            if is_resync {
                syms_in_group = 0;
            } else if !is_eot {
                data_received  += 1;
                syms_in_group  += 1;
            }
            k_sym += 1;
        }

        let frame = match frame_result {
            Some(f) => f,
            None => {
                eprintln!("  [frame {}] incomplete — skipping", frame_stats.len());
                search_start = abs_preamble + SYMBOL_LEN;
                continue;
            }
        };

        let m = &frame.metrics;
        frame_stats.push(FrameStat {
            header:       header.clone(),
            pilot_snr_db: frame.pilot_snr_db,
            ch_ber_pct:   m.channel_ber * 100.0,
            ldpc_ok_pct:  if m.ldpc_total > 0 {
                m.ldpc_converged as f32 / m.ldpc_total as f32 * 100.0
            } else { 0.0 },
            rs_ok:        frame.packets_ok,
            rs_total:     pkts_this,
            rs_margin:    m.rs_margin_frac * 100.0,
            crc32_ok:     frame.crc32_ok,
        });

        // Push frame to assembler.
        match tx_rx.as_mut().unwrap().push_frame(frame, &header) {
            TxPushResult::Complete(result) => {
                eprint!("\r                                          \r");
                print_frame_table(&frame_stats, n_files);
                n_frames_seen += frame_stats.len();

                match extract_payload(&result.payload) {
                    Ok((name, data)) => {
                        let path = unique_path(&cfg.outdir, &name);
                        match std::fs::write(&path, data) {
                            Ok(()) => {
                                println!("  Saved     : {} ({} bytes)", path, data.len());
                                n_files += 1;
                            }
                            Err(e) => eprintln!("  Write error: {path}: {e}"),
                        }
                    }
                    Err(e) => eprintln!("  Payload error: {e}"),
                }

                // Reset for next transmission.
                tx_rx = None;
                frame_stats.clear();
                next_progress = (search_start + audio.len() / 10).min(audio.len());
            }
            TxPushResult::NeedMoreFrames => {}
        }

        search_start = abs_preamble + (3 + n_expected) * SYMBOL_LEN;
    }

    // ── End-of-audio summary ──────────────────────────────────────────────────
    eprint!("\r                                          \r");

    // Print stats for any partially-decoded in-progress transmission.
    if !frame_stats.is_empty() {
        print_frame_table(&frame_stats, n_files);
        n_frames_seen += frame_stats.len();
        eprintln!("  (transmission incomplete — {}/{} packets received)",
            tx_rx.as_ref().map(|t| t.packets_received()).unwrap_or(0),
            frame_stats.last().map(|s| s.header.total_packet_count).unwrap_or(0));
    }

    println!();
    if n_files == 0 && n_frames_seen == 0 {
        eprintln!("rx: no transmissions found in '{}'", cfg.input);
        std::process::exit(1);
    }
    println!("  Decoded  : {n_files} file(s)  ({n_frames_seen} super-frame(s) total)");
    if n_files == 0 { std::process::exit(1); }
}

// ── Audio helpers ─────────────────────────────────────────────────────────────

/// 13-tap Hamming-windowed sinc LPF, fc = 4 kHz @ 48 kHz, group delay = 6 samples.
///
/// H[0] = H[12] = 0 (sinc zeros): the filter is purely backward-looking when applied
/// at output sample boundaries, so no future-symbol content bleeds into the FFT window.
/// Group delay = 6 samples at 48 kHz = 1 sample at 8 kHz — absorbed by the CP (32 samp).
fn downsample_fir(src: &[f32], factor: usize) -> Vec<f32> {
    #[allow(clippy::excessive_precision)]
    const H: [f32; 13] = [
        0.000000,
        0.005340, 0.025310, 0.067900, 0.125750, 0.176970,
        0.197480,
        0.176970, 0.125750, 0.067900, 0.025310, 0.005340,
        0.000000,
    ];
    const HALF: usize = H.len() / 2; // = 6
    let n_out = src.len() / factor;
    (0..n_out).map(|n| {
        let center = n * factor;
        H.iter().enumerate().map(|(k, &hk)| {
            // center - k + HALF: tap k looks back (k > HALF) or forward (k < HALF).
            // With H[0]=H[12]=0 the two forward taps contribute nothing.
            let idx = center as isize - k as isize + HALF as isize;
            if idx >= 0 && (idx as usize) < src.len() { hk * src[idx as usize] } else { 0.0 }
        }).sum()
    }).collect()
}

// ── Payload helpers ───────────────────────────────────────────────────────────

/// Decode payload: `[4 bytes: data_len] [1 byte: name_len] [name] [data]`
fn extract_payload(p: &[u8]) -> Result<(String, &[u8]), String> {
    if p.len() < 5 {
        return Err(format!("payload too short ({} bytes)", p.len()));
    }
    let data_len = u32::from_le_bytes([p[0], p[1], p[2], p[3]]) as usize;
    let name_len = p[4] as usize;
    if p.len() < 5 + name_len {
        return Err(format!("name_len={name_len} overruns payload ({} bytes)", p.len()));
    }
    let name = std::str::from_utf8(&p[5..5 + name_len])
        .unwrap_or("data")
        .to_string();
    // Sanitise: keep only the basename, replace path separators.
    let name = std::path::Path::new(&name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("data")
        .replace(['/', '\\', '\0'], "_");
    let name = if name.is_empty() { "data".to_string() } else { name };

    let data_start = 5 + name_len;
    let end = (data_start + data_len).min(p.len());
    Ok((name, &p[data_start..end]))
}

/// Return a path that does not yet exist: `<dir>/<name>`, then `<dir>/<stem>_2.<ext>`, etc.
fn unique_path(dir: &str, name: &str) -> String {
    let base = format!("{dir}/{name}");
    if !std::path::Path::new(&base).exists() { return base; }
    let stem = std::path::Path::new(name)
        .file_stem().and_then(|s| s.to_str()).unwrap_or(name);
    let ext  = std::path::Path::new(name)
        .extension().and_then(|s| s.to_str()).map(|e| format!(".{e}")).unwrap_or_default();
    for n in 2u32.. {
        let candidate = format!("{dir}/{stem}_{n}{ext}");
        if !std::path::Path::new(&candidate).exists() { return candidate; }
    }
    base
}

fn print_frame_table(stats: &[FrameStat], tx_idx: usize) {
    println!();
    println!("  Transmission #{tx_idx}");
    println!("  {:>5}  {:>8}  {:>5}  {:>6}  {:>9}  {:>5}/{:<5}  {:>9}  {:>5}",
        "Frame", "Mode", "SNRdB", "BER%", "LDPC_ok%",
        "RS_ok", "tot", "RS_marg%", "CRC");
    println!("  {}", "-".repeat(75));

    let mut total_rs_ok:  u32 = 0;
    let mut total_rs_tot: u32 = 0;
    let mut n_ok = 0usize;

    for (idx, s) in stats.iter().enumerate() {
        let mode = format!("{} {}", mod_label(s.header.modulation), rate_label(s.header.ldpc_rate));
        println!("  {:>5}  {:>8}  {:>5.1}  {:>6.2}  {:>9.1}  {:>5}/{:<5}  {:>9.1}  {:>5}",
            idx, mode,
            s.pilot_snr_db, s.ch_ber_pct, s.ldpc_ok_pct,
            s.rs_ok, s.rs_total, s.rs_margin,
            if s.crc32_ok { "OK" } else { "FAIL" });
        total_rs_ok  += s.rs_ok as u32;
        total_rs_tot += s.rs_total as u32;
        if s.crc32_ok { n_ok += 1; }
    }

    println!();
    println!("  Frames   : {} super-frame(s)  ({}/{} CRC32 OK)", stats.len(), n_ok, stats.len());
    println!("  RS pkts  : {}/{} ok  ({:.1}%)",
        total_rs_ok, total_rs_tot,
        if total_rs_tot > 0 { 100.0 * total_rs_ok as f32 / total_rs_tot as f32 } else { 0.0 });
}

// ── WAV reader ────────────────────────────────────────────────────────────────

fn read_wav_mono(path: &str) -> std::io::Result<(Vec<f32>, u32)> {
    use std::io::{BufReader, Read, Seek, SeekFrom};

    let file   = std::fs::File::open(path)?;
    let mut rd = BufReader::new(file);

    let mut tag = [0u8; 4];
    rd.read_exact(&mut tag)?;
    if &tag != b"RIFF" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "not a RIFF file"));
    }
    let _ = read_u32_le(&mut rd)?; // file size
    rd.read_exact(&mut tag)?;
    if &tag != b"WAVE" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "RIFF is not WAVE"));
    }

    let mut sample_rate: u32 = 0;
    let mut channels:    u16 = 0;
    let mut bits:        u16 = 0;
    let mut data:        Vec<u8> = Vec::new();

    loop {
        let mut id = [0u8; 4];
        if rd.read_exact(&mut id).is_err() { break; }
        let size = read_u32_le(&mut rd)?;

        match &id {
            b"fmt " => {
                let fmt = read_u16_le(&mut rd)?;
                if fmt != 1 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unsupported WAV format {fmt} (only PCM=1)")));
                }
                channels    = read_u16_le(&mut rd)?;
                sample_rate = read_u32_le(&mut rd)?;
                let _ = read_u32_le(&mut rd)?; // byte rate
                let _ = read_u16_le(&mut rd)?; // block align
                bits        = read_u16_le(&mut rd)?;
                if size > 16 {
                    rd.seek(SeekFrom::Current((size - 16) as i64))?;
                }
            }
            b"data" => {
                data.resize(size as usize, 0);
                rd.read_exact(&mut data)?;
                break;
            }
            _ => { rd.seek(SeekFrom::Current(size as i64 + (size & 1) as i64))?; }
        }
    }

    if sample_rate == 0 || channels == 0 || bits == 0 || data.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "incomplete WAV"));
    }

    let bps  = (bits / 8) as usize;
    let frsz = bps * channels as usize;
    let n    = data.len() / frsz;
    let mut out = Vec::with_capacity(n);

    for f in 0..n {
        let mut sum = 0f32;
        for ch in 0..channels as usize {
            let o = f * frsz + ch * bps;
            sum += match bits {
                8  => (data[o] as f32 - 128.0) / 128.0,
                16 => i16::from_le_bytes([data[o], data[o+1]]) as f32 / 32768.0,
                32 => i32::from_le_bytes(
                    [data[o], data[o+1], data[o+2], data[o+3]]) as f32 / 2_147_483_648.0,
                b  => return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unsupported bit depth {b}"))),
            };
        }
        out.push(sum / channels as f32);
    }

    Ok((out, sample_rate))
}

fn read_u16_le<R: std::io::Read>(r: &mut R) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32_le<R: std::io::Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

// ── Label helpers ─────────────────────────────────────────────────────────────

fn mod_label(m: Modulation) -> &'static str {
    match m {
        Modulation::Bpsk  => "BPSK",
        Modulation::Qpsk  => "QPSK",
        Modulation::Qam16 => "16QAM",
        Modulation::Qam32 => "32QAM",
        Modulation::Qam64 => "64QAM",
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
