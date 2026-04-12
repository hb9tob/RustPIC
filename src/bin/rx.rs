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
        params::{CP_LEN, MODE_HEADER_REPEAT, PREAMBLE_SYMS, RUNIN_PREAMBLE_SYMS, SAMPLE_RATE, SYMBOL_LEN},
        rx::{
            frame::{
                FrameReceiver, PushResult, TransmissionReceiver, TxPushResult,
                max_packets_per_frame,
            },
            hilbert::HilbertFilter,
            mode_detect::{decode_mode_header_from_eq, LdpcRate, ModeHeader, Modulation},
            pilot_sync::{scan_for_pilots, cp_correlation},
        },
    },
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Scan window width: 4×SYMBOL_LEN is large enough for ZcCorrelator and small
/// enough that the first resync ZC never falls inside the initial search window.
// ── CLI ───────────────────────────────────────────────────────────────────────

struct Cfg {
    input:      String,
    outdir:     String,
    pilot_sync: bool,
}

fn parse_args() -> Cfg {
    let mut input      = String::new();
    let mut outdir     = String::from(".");
    let mut pilot_sync = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input"      => { i += 1; input  = args[i].clone(); }
            "--outdir"     => { i += 1; outdir = args[i].clone(); }
            "--pilot-sync" => { pilot_sync = true; }
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            a => die(&format!("unknown argument: {a}  (try --help)")),
        }
        i += 1;
    }
    if input.is_empty() { die("--input is required"); }
    Cfg { input, outdir, pilot_sync }
}

fn print_help() {
    println!("RustPIC RX — decode all transmissions found in a 48 kHz WAV");
    println!();
    println!("USAGE: rx --input <in.wav> [--outdir <dir>]");
    println!();
    println!("OPTIONS:");
    println!("  --input  <file.wav>  WAV file to scan (required)");
    println!("  --outdir <dir>       Output directory for decoded files  [default: .]");
    println!("  --pilot-sync         Use pilot-based sync instead of ZC correlator");
    println!();
    println!("The WAV must be 48 kHz 16-bit PCM (mono or stereo).");
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

    // Verify sample rate.
    if wav_rate != SAMPLE_RATE as u32 {
        die(&format!("unsupported sample rate {wav_rate} Hz (expected 48000)"));
    }

    // ── AGC: block-wise RMS normalisation ──────────────────────────────────────
    // Normalise the audio to a target RMS of 0.2 (matching the TX's OFDM level)
    // using overlapping blocks of ~100 ms (4800 samples at 48 kHz).  This
    // removes the receiver's volume-setting dependence: whether the soundcard
    // is at 20 % or 80 %, the correlator and equaliser see the same amplitude.
    //
    // The gain is applied per-sample with linear interpolation between block
    // boundaries to avoid discontinuities.  A minimum RMS floor prevents
    // noise-only silence from being amplified to infinity.
    let wav_samples = {
        const AGC_BLOCK: usize = 4800;       // ~100 ms at 48 kHz
        const TARGET_RMS: f32  = 0.20;
        const MIN_RMS: f32     = 1e-4;       // don't amplify pure silence

        let n = wav_samples.len();
        if n < AGC_BLOCK {
            wav_samples
        } else {
            // Compute per-block RMS and the resulting gain.
            let n_blocks = n.div_ceil(AGC_BLOCK);
            let mut gains = Vec::with_capacity(n_blocks);
            for b in 0..n_blocks {
                let start = b * AGC_BLOCK;
                let end   = (start + AGC_BLOCK).min(n);
                let rms   = (wav_samples[start..end].iter()
                    .map(|&x| x * x).sum::<f32>() / (end - start) as f32)
                    .sqrt();
                gains.push(if rms > MIN_RMS { TARGET_RMS / rms } else { 1.0 });
            }

            // Apply gain with linear interpolation between block centres.
            let mut out = wav_samples;
            for i in 0..n {
                let block_f  = i as f32 / AGC_BLOCK as f32 - 0.5;
                let b0       = (block_f.floor() as i32).clamp(0, n_blocks as i32 - 1) as usize;
                let b1       = (b0 + 1).min(n_blocks - 1);
                let t        = (block_f - b0 as f32).clamp(0.0, 1.0);
                let gain     = gains[b0] * (1.0 - t) + gains[b1] * t;
                out[i] *= gain;
            }
            out
        }
    };

    // ── Hilbert transform → analytic signal (complex baseband) ─────────────
    // Converts the real audio to a complex analytic signal using a 153-tap
    // Nuttall-windowed FIR Hilbert filter (same approach as QSSTV DRM RX).
    // The analytic signal has only positive-frequency content: pilot phases
    // are preserved without conjugate-spectrum folding, and CFO estimation
    // works correctly in the complex domain (essential for future SSB).
    let audio: Vec<Complex32> = {
        let mut hf = HilbertFilter::new();
        // Pad with silence to compensate for Hilbert filter group delay (76 samples)
        // and avoid truncating the last OFDM symbol.
        let mut padded = wav_samples;
        padded.extend(vec![0.0_f32; SYMBOL_LEN]);
        hf.process(&padded)
    };

    let duration_s = audio.len() as f64 / SAMPLE_RATE as f64;
    println!("RustPIC RX");
    println!("  WAV      : {} ({} kHz, {:.1} s)",
        cfg.input, wav_rate / 1000, duration_s);

    // ── Scan and decode all super-frames ────────────────────────────────────
    // Pilot-based sync: scan with CP correlation + scattered pilot matching.
    // No ZC preamble — the mode header is right before the first data symbol.
    let mut search_start:  usize = 0;
    let mut tx_rx:         Option<TransmissionReceiver> = None;
    let mut frame_stats:   Vec<FrameStat> = Vec::new();
    let mut n_files:       usize = 0;
    let mut n_frames_seen: usize = 0;

    let mut next_progress = audio.len() / 10;

    while search_start + 3 * SYMBOL_LEN <= audio.len() {
        if search_start >= next_progress {
            let pct = search_start * 100 / audio.len();
            eprint!("\r  Scanning  : {pct:3}% ({:.1} s)…    ", search_start as f64 / SAMPLE_RATE as f64);
            next_progress = search_start + audio.len() / 10;
        }

        // ── Pilot-based sync ─────────────────────────────────────────────────
        let ps = match scan_for_pilots(&audio, search_start, 0.3, 0.85) {
            Some(r) => r,
            None => { break; } // no more OFDM symbols found
        };

        if std::env::var("SYNC_DEBUG").is_ok() {
            eprintln!("  [pilot] FOUND at {:.3}s: sym_pos={} sym_idx={} metric={:.3} cfo={:.1}Hz",
                ps.symbol_pos as f64 / SAMPLE_RATE as f64,
                ps.symbol_pos, ps.sym_idx, ps.metric, ps.cfo_hz);
        }
        if std::env::var("CH_DEBUG").is_ok() {
            eprint!("  [ch] |H| dB:");
            for h in ps.channel_est.iter() {
                eprint!(" {:+5.1}", 20.0 * (h.norm() + 1e-12).log10());
            }
            eprintln!();
        }

        // Navigate from the detected symbol to the frame start.
        // sym_idx is modulo SYMBOLS_PER_FRAME → ambiguous.  Try each
        // candidate frame_start (stepping back by SYMBOLS_PER_FRAME)
        // and pick the one where the mode header QPSK CRC passes.
        let base_back = ps.sym_idx * SYMBOL_LEN;
        let mut frame_start = ps.symbol_pos.saturating_sub(base_back);
        let mut hdr_start = frame_start + RUNIN_PREAMBLE_SYMS * SYMBOL_LEN;
        let mut first_data_pos = frame_start + PREAMBLE_SYMS * SYMBOL_LEN;

        // ── CFO correction on the full frame (RUNIN + header + data) ─────
        if first_data_pos + SYMBOL_LEN > audio.len() {
            search_start = ps.symbol_pos + SYMBOL_LEN;
            continue;
        }
        let frame_end = (first_data_pos + 300 * SYMBOL_LEN).min(audio.len());
        let mut buf: Vec<Complex32> = audio[frame_start..frame_end].to_vec();
        if ps.cfo_hz.abs() > 0.1 {
            let omega = -2.0 * std::f32::consts::PI * ps.cfo_hz / SAMPLE_RATE;
            for (i, s) in buf.iter_mut().enumerate() {
                let phase = omega * i as f32;
                *s *= Complex32::from_polar(1.0, phase);
            }
        }

        // ── Equalizer warm-up: feed RUNIN preamble + mode header ─────────
        // The 10 RUNIN preamble symbols have scattered pilots → the equalizer
        // converges on the real channel response BEFORE we decode the mode
        // header or any data.  This is the key to surviving NBFM channels.
        use rustpic::ofdm::rx::equalizer::ScatteredEqualizer;
        let mut eq = ScatteredEqualizer::from_initial(&ps.channel_est, 0.3);

        // Feed RUNIN preamble (10 symbols with pilots, dummy data)
        for i in 0..RUNIN_PREAMBLE_SYMS {
            let off = i * SYMBOL_LEN;
            if off + SYMBOL_LEN > buf.len() { break; }
            let _ = eq.process(&buf[off..off + SYMBOL_LEN], i);
        }

        // Decode mode header with the now-converged equalizer
        let hdr_offset = RUNIN_PREAMBLE_SYMS * SYMBOL_LEN;
        let mut eq_data_vecs: Vec<Vec<Complex32>> = Vec::new();
        for r in 0..MODE_HEADER_REPEAT {
            let off = hdr_offset + r * SYMBOL_LEN;
            if off + SYMBOL_LEN > buf.len() { break; }
            let sym_idx = RUNIN_PREAMBLE_SYMS + r;
            let result = eq.process(&buf[off..off + SYMBOL_LEN], sym_idx);
            eq_data_vecs.push(result.data);
        }
        let eq_refs: Vec<&[Complex32]> = eq_data_vecs.iter().map(|v| v.as_slice()).collect();
        let header = match decode_mode_header_from_eq(&eq_refs) {
            Ok(h) => h,
            Err(_) => {
                search_start = ps.symbol_pos + SYMBOL_LEN;
                continue;
            }
        };

        if tx_rx.is_none() {
            let (_, rs_k, _) = header.rs_level.params();
            tx_rx = Some(TransmissionReceiver::new(header.total_packet_count, rs_k));
        }

        // Create FrameReceiver with the converged channel estimate from
        // the equalizer (not the raw pilot_sync estimate).
        let converged_h: Vec<Complex32> = (0..rustpic::ofdm::params::NUM_CARRIERS)
            .map(|k| eq.h_at(k))
            .collect();
        let mut rx = FrameReceiver::new(&header, &converged_h, 0.3);
        let n_expected   = rx.expected_symbol_count();
        let pkts_this    = {
            let remaining = header.total_packet_count
                                .saturating_sub(header.packet_offset) as usize;
            remaining.min(max_packets_per_frame(
                header.modulation, header.ldpc_rate, header.rs_level))
        };

        // Symbol-by-symbol processing — simple sequential, no resync ZCs.
        // Offsets are relative to buf (which starts at frame_start).
        let data_offset_in_buf = PREAMBLE_SYMS * SYMBOL_LEN;
        let mut frame_result = None;
        for i in 0..n_expected {
            let sym_pos = data_offset_in_buf + i * SYMBOL_LEN;
            if sym_pos + SYMBOL_LEN > buf.len() { break; }
            match rx.push_symbol(&buf[sym_pos..sym_pos + SYMBOL_LEN]) {
                PushResult::NeedMore             => {}
                PushResult::FrameComplete(frame) => { frame_result = Some(frame); break; }
                PushResult::Error(_)             => break,
            }
        }

        let frame = match frame_result {
            Some(f) => f,
            None => {
                eprintln!("  [frame {}] incomplete — skipping", frame_stats.len());
                search_start = ps.symbol_pos + SYMBOL_LEN;
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
                    Ok((name, data, crc_ok)) => {
                        let path = unique_path(&cfg.outdir, &name);
                        match std::fs::write(&path, data) {
                            Ok(()) => {
                                let crc_tag = if crc_ok { "CRC32 OK" } else { "CRC32 FAIL" };
                                println!("  Saved     : {} ({} bytes, {})", path, data.len(), crc_tag);
                                if crc_ok {
                                    n_files += 1;
                                }
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

        search_start = first_data_pos + n_expected * SYMBOL_LEN;
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

// ── Payload helpers ───────────────────────────────────────────────────────────

/// Decodes the TX payload layout.
///
/// Layout:
///
/// ```text
///   [META × 3]  [raw data bytes]  [4B trailing CRC32]
/// ```
///
/// Each META block is 55 bytes:
///
/// ```text
///   4B magic "RPIC"  4B data_len  1B name_len  40B name  4B data_crc32  2B crc16
/// ```
///
/// Because three identical copies of META are spread across the
/// FEC-protected payload (LDPC + RS + M-way byte interleave), even if a
/// single RS group fails completely at least one copy normally survives —
/// each copy is self-validated with a CRC-16 so we just pick the first
/// one that passes.
///
/// Returns `Ok((filename, data_slice, full_crc_ok))`.
fn extract_payload(p: &[u8]) -> Result<(String, &[u8], bool), String> {
    const META_MAGIC: u32 = 0x52504943;
    const META_LEN: usize  = 55;
    const NAME_LEN_MAX: usize = 40;

    if p.len() < META_LEN {
        return Err(format!("payload too short ({} bytes)", p.len()));
    }

    // Parse a single META block.  Returns None if its internal CRC-16 fails
    // or the magic word is wrong.
    let parse_meta = |block: &[u8]| -> Option<(u32, u8, [u8; NAME_LEN_MAX], u32)> {
        if block.len() < META_LEN { return None; }
        // Verify CRC-16 on the first 53 bytes.
        let crc_body   = &block[..53];
        let crc_stored = u16::from_le_bytes([block[53], block[54]]);
        if rustpic::ofdm::rx::mode_detect::crc16_ccitt(crc_body) != crc_stored {
            return None;
        }
        let magic     = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        if magic != META_MAGIC { return None; }
        let data_len  = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let name_len  = block[8];
        let mut name_bytes = [0u8; NAME_LEN_MAX];
        name_bytes.copy_from_slice(&block[9..9 + NAME_LEN_MAX]);
        let data_crc32 = u32::from_le_bytes([block[49], block[50], block[51], block[52]]);
        Some((data_len, name_len, name_bytes, data_crc32))
    };

    let mut chosen: Option<(u32, u8, [u8; NAME_LEN_MAX], u32)> = None;
    for i in 0..3 {
        let off = i * META_LEN;
        if off + META_LEN > p.len() { break; }
        if let Some(meta) = parse_meta(&p[off..off + META_LEN]) {
            chosen = Some(meta);
            break;
        }
    }
    let (data_len_u32, name_len, name_bytes, data_crc_expected) = chosen
        .ok_or_else(|| "all 3 META copies failed CRC-16 check".to_string())?;

    let data_len = data_len_u32 as usize;
    let name_len = (name_len as usize).min(NAME_LEN_MAX);
    let name = std::str::from_utf8(&name_bytes[..name_len])
        .unwrap_or("data")
        .to_string();
    // Sanitise: keep only the basename, replace path separators.
    let name = std::path::Path::new(&name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("data")
        .replace(['/', '\\', '\0'], "_");
    let name = if name.is_empty() { "data".to_string() } else { name };

    let data_start = 3 * META_LEN;
    let data_end   = data_start + data_len;
    let crc_end    = data_end + 4;

    if p.len() < data_end {
        return Err(format!(
            "data overruns payload: data_end={data_end} > p.len()={}",
            p.len()
        ));
    }

    let data = &p[data_start..data_end];

    // Verify the trailing CRC-32 if present (authoritative full-payload check).
    let trailing_ok = if p.len() >= crc_end {
        let stored = u32::from_le_bytes([
            p[data_end], p[data_end + 1], p[data_end + 2], p[data_end + 3],
        ]);
        rustpic::ofdm::rx::frame::crc32_ieee(&p[..data_end]) == stored
    } else {
        false
    };

    // Secondary check: the META's own CRC-32 over the raw data.
    let data_crc_ok = rustpic::ofdm::rx::frame::crc32_ieee(data) == data_crc_expected;

    Ok((name, data, trailing_ok && data_crc_ok))
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
