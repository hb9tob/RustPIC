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

use num_complex::Complex32;
use rustfft::FftPlanner;

use rustpic::{
    fec::rs::RsLevel,
    ofdm::{
        beacon::build_beacon,
        params::{CP_LEN, FFT_SIZE, SAMPLE_RATE, SYMBOL_LEN},
        rx::{
            frame::{crc32_ieee, max_packets_per_frame},
            mode_detect::{crc16_ccitt, LdpcRate, Modulation},
        },
        tx::frame::{build_transmission, FrameConfig},
    },
};

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
    deemph_tau: Option<f32>,  // de-emphasis time constant in µs; None = disabled
    bandpass:   bool,          // post-OFDM bandpass filter (mirrors QSSTV CDRMBandpassFilt)
    smooth_tw:  usize,         // raised-cosine boundary taper width in samples (0 = off)
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
            papr_clip:  Some(12.0),  // default on — soft tanh clip at RMS+12 dB
            gain_db:    0.0,
            deemph_tau: None,
            bandpass:   false,
            smooth_tw:  0,
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
            "--deemph" => {
                i += 1;
                cfg.deemph_tau = Some(args[i].parse::<f32>()
                    .unwrap_or_else(|_| die("--deemph expects time constant in µs, e.g. --deemph 750")));
            }
            "--no-deemph" => { cfg.deemph_tau = None; }
            "--bpf"       => { cfg.bandpass = true; }
            "--no-bpf"    => { cfg.bandpass = false; }
            "--smooth-tw" => {
                i += 1;
                cfg.smooth_tw = args[i].parse::<usize>()
                    .unwrap_or_else(|_| die("--smooth-tw expects a non-negative integer"));
            }
            "--no-smooth" => { cfg.smooth_tw = 0; }
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
    println!("  --deemph <µs>        Apply TX de-emphasis before FM radio microphone input.");
    println!("                       Use 750 for G3E marine / European NBFM (ITU-R),");
    println!("                       530 for NBFM 75µs (North America).");
    println!("                       Prevents FM deviation clipping caused by pre-emphasis");
    println!("                       boosting high-frequency OFDM carriers by up to 21 dB.");
    println!("  --no-deemph          Disable TX de-emphasis (default: off)");
    println!("  --no-bpf             Disable post-OFDM bandpass filter (default: off).");
    println!("                       The bandpass confines energy to ~280-2620 Hz, matching");
    println!("                       QSSTV HamDRM's CDRMBandpassFilt.  Prevents out-of-band");
    println!("                       spectral leakage from feeding the FM pre-emphasis region.");
    println!("  --smooth-tw <N>      Raised-cosine symbol-boundary taper width in samples");
    println!("                       (default: 32, 0 = off).  Kills the inter-symbol time-");
    println!("                       domain discontinuity that creates broadband out-of-band");
    println!("                       leakage when the FM pre-emphasis amplifies high carriers.");
    println!("  --no-smooth          Disable symbol-boundary smoothing (equivalent to --smooth-tw 0).");
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

    // Payload layout — defence in depth:
    //
    //   [META × 3]  [raw data bytes]  [4B trailing CRC32]
    //
    // META is a fixed 64-byte block that carries the critical metadata
    // (magic, data_len, filename, data_crc32).  It is repeated 3 times at
    // the start of the payload so that even if a whole RS group fails on
    // the RX side, at least one META copy survives somewhere in the
    // M-interleaved bytes.  Each copy has its own internal CRC-16 so the
    // RX can pick the first valid copy without needing majority vote.
    //
    // The trailing CRC-32 covers everything (all three META blocks + the
    // data) and is the authoritative full-payload integrity check.
    //
    // All of this sits INSIDE the LDPC + RS + M-interleave chain, so the
    // effective protection is the same as the user-selected LDPC rate on
    // the data bytes — but the triple-META layout makes the single point
    // of failure (losing the filename / data_len / data_crc32) disappear.
    const META_MAGIC: u32 = 0x52504943; // "RPIC"
    const META_NAME_BYTES: usize = 40;   // fixed-size name field
    const META_LEN: usize = 4 + 4 + 1 + META_NAME_BYTES + 4 + 2; // 55 bytes
    let _ = META_LEN;

    let name = std::path::Path::new(&cfg.input)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("data");
    let name_bytes = name.as_bytes();
    let name_len   = name_bytes.len().min(META_NAME_BYTES) as u8;
    let data_crc32 = crc32_ieee(&raw);

    // Build a single META block (55 bytes) with its own CRC-16.
    let build_meta = || -> Vec<u8> {
        let mut m = Vec::with_capacity(55);
        m.extend_from_slice(&META_MAGIC.to_le_bytes());          // 4
        m.extend_from_slice(&(raw.len() as u32).to_le_bytes());  // 4
        m.push(name_len);                                         // 1
        let mut padded_name = [0u8; META_NAME_BYTES];
        padded_name[..name_len as usize].copy_from_slice(&name_bytes[..name_len as usize]);
        m.extend_from_slice(&padded_name);                        // 40
        m.extend_from_slice(&data_crc32.to_le_bytes());           // 4
        // Internal CRC-16/CCITT over the preceding 53 bytes.
        let crc16 = crc16_ccitt(&m);
        m.extend_from_slice(&crc16.to_le_bytes());                // 2
        m
    };

    let meta_block = build_meta();
    assert_eq!(meta_block.len(), 55);

    let mut payload = Vec::with_capacity(3 * 55 + raw.len() + 4);
    // Three copies of META.
    for _ in 0..3 {
        payload.extend_from_slice(&meta_block);
    }
    // The raw user data.
    payload.extend_from_slice(&raw);
    // Authoritative trailing CRC-32 over everything above.
    let trailing_crc = crc32_ieee(&payload);
    payload.extend_from_slice(&trailing_crc.to_le_bytes());
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
    let mut samples_f: Vec<f32> = beacon.iter().map(|s| s.re)
        .chain(frames.iter().flat_map(|f| f.iter().map(|s| s.re)))
        .collect();
    let n_base_samples = samples_f.len();

    // OFDM symbol-boundary smoothing — kills the broadband spectral leakage
    // caused by time-domain discontinuities at every SYMBOL_LEN boundary.
    //
    // QSSTV suppresses this leakage with a post-OFDM FIR bandpass filter
    // (drmtx/common/util/Utilities.cpp:127 CDRMBandpassFilt), but a long FIR
    // smears the data-frame ZC#1 with preceding ANN content and breaks our
    // time-domain ZC correlator (see memory/feedback_bandpass_zc.md).  A
    // per-symbol raised-cosine taper achieves the same spectral result with no
    // smearing across symbol boundaries: the last T_w samples of each symbol
    // go to zero, and the first T_w samples of the next symbol rise from zero,
    // so the transition is a smooth zero crossing.
    if cfg.smooth_tw > 0 {
        smooth_symbol_boundaries(&mut samples_f, cfg.smooth_tw);
    }

    // Optional TX de-emphasis (inverse pre-emphasis) for FM radio use.
    // The FM radio's internal pre-emphasis boosts the OFDM carriers by up to
    // 21 dB at 2484 Hz (G3E marine, τ=750 µs), causing FM deviation clipping
    // and severe ICI.  Applying de-emphasis here cancels that boost so the
    // modulator sees a flat spectrum.  The ZF equalizer at RX handles the
    // residual channel shape (RX radio de-emphasis slope is visible in H[k]).
    let samples_f = if let Some(tau_us) = cfg.deemph_tau {
        apply_deemph(&samples_f, tau_us, SAMPLE_RATE as usize)
    } else {
        samples_f
    };

    // Optional soft-clip PAPR reduction (tanh-based).
    let samples_f = if let Some(clip_db) = cfg.papr_clip {
        papr_clip(&samples_f, clip_db)
    } else {
        samples_f
    };

    // Post-OFDM bandpass filter confining energy to the active carrier band.
    // Mirrors QSSTV CDRMBandpassFilt in drmtx/common/util/Utilities.cpp:127 —
    // QSSTV's Mode B uses a Nuttall-windowed 2.5 kHz lowpass shifted to the DRM
    // IF centre.  Our real-baseband equivalent is a raised-cosine bandpass over
    // 280…2620 Hz that passes bins 7…53 (328…2484 Hz) flat and suppresses
    // everything else.  Applied last so CP discontinuities and PAPR-clip spurs
    // do not leak into the FM pre-emphasis region and create ICI.
    let samples_f = if cfg.bandpass {
        apply_bandpass(&samples_f, SAMPLE_RATE as usize)
    } else {
        samples_f
    };

    // Normalize so peak hits 90 % of i16 full-scale.
    let peak = samples_f.iter().map(|&x| x.abs()).fold(0f32, f32::max);
    let mut scale = if peak > 1e-9 { 0.9 * 32767.0 / peak } else { 32767.0 };

    // Optional additional gain.
    if cfg.gain_db != 0.0 {
        scale *= 10_f32.powf(cfg.gain_db / 20.0);
    }

    // Convert to i16 — signal is already at 48 kHz natively, no resampling needed.
    let samples_i16: Vec<i16> = samples_f
        .iter()
        .map(|&x| (x * scale).round().clamp(-32768.0, 32767.0) as i16)
        .collect();

    write_wav_stereo(&cfg.output, &samples_i16, SAMPLE_RATE as u32)
        .unwrap_or_else(|e| die(&format!("cannot write '{}': {e}", cfg.output)));

    let duration_s = n_base_samples as f64 / SAMPLE_RATE as f64;
    let wav_bytes  = std::fs::metadata(&cfg.output).map(|m| m.len()).unwrap_or(0);
    use rustpic::ofdm::{beacon::BEACON_TOTAL_SYMS, params::SYMBOL_LEN as SYM};
    let beacon_ms = BEACON_TOTAL_SYMS * SYM * 1000 / SAMPLE_RATE as usize;

    println!("RustPIC TX");
    println!("  Callsign : {}", cfg.callsign);
    println!("  Input    : {} ({} bytes)", cfg.input, raw.len());
    println!("  Mode     : {}  {}  RS-L{}{}{}{}{}",
        mod_label(cfg.modulation),
        rate_label(cfg.ldpc_rate),
        match cfg.rs_level { RsLevel::L0 => "0", RsLevel::L1 => "1", RsLevel::L2 => "2" },
        if cfg.resync { "  resync" } else { "" },
        if let Some(db) = cfg.papr_clip { format!("  papr-clip={db}dB") } else { String::new() },
        if let Some(tau) = cfg.deemph_tau { format!("  deemph={tau}µs") } else { String::new() },
        if cfg.gain_db != 0.0 { format!("  gain={:+.1}dB", cfg.gain_db) } else { String::new() });
    println!("  BPF      : {}", if cfg.bandpass { "280-2620 Hz (QSSTV-style)" } else { "off" });
    println!("  Smooth   : {}", if cfg.smooth_tw > 0 { format!("{} samples raised-cosine taper", cfg.smooth_tw) } else { "off".to_string() });
    println!("  RS pkts  : {} total, ≤ {}/frame, rs_k={}", (raw.len() + 4).div_ceil(rs_k), pkt_per_frame, rs_k);
    println!("  Beacon   : {} ms (1 kHz VOX tone + ZC + indicatif)", beacon_ms);
    println!("  Frames   : {}", n_frames);
    println!("  Duration : {:.1} s  ({} samples @ {} Hz, beacon included)", duration_s, n_base_samples, SAMPLE_RATE as u32);
    println!("  Output   : {}  ({} bytes, 48 kHz 16-bit stereo)", cfg.output, wav_bytes);
}

// ── Audio helpers ─────────────────────────────────────────────────────────────

/// Per-symbol frequency-domain de-emphasis (inverse pre-emphasis) for OFDM.
///
/// A time-domain IIR filter would introduce inter-symbol transients at every
/// symbol boundary, reducing the ZC preamble's matched-filter metric below the
/// detection threshold.  Applying de-emphasis in the frequency domain avoids
/// this: each OFDM symbol is de-emphasized independently with no inter-symbol
/// memory, so there are no transients.
///
/// For each SYMBOL_LEN = CP_LEN + FFT_SIZE block:
///   1. FFT of the 1024-sample useful window.
///   2. Multiply each bin k by |H_de(f_k)| = f₀/√(f₀²+f_k²).
///   3. IFFT → regenerate cyclic prefix from the new window tail.
///
/// The gain profile is the amplitude response of H(s) = 1/(1+s·τ),
/// the inverse of the FM radio's internal pre-emphasis filter.
/// After the radio applies its pre-emphasis, the OFDM spectrum is flat.
/// The ZF equaliser at RX already corrects for the residual RX de-emphasis slope.
///
/// Typical values: τ = 750 µs (G3E marine / ITU-R European NBFM, f₀ = 212 Hz)
///                 τ = 530 µs (North-American NBFM,              f₀ = 300 Hz)
fn apply_deemph(samples: &[f32], tau_us: f32, fs: usize) -> Vec<f32> {
    let tau = tau_us * 1e-6;
    let f0  = 1.0 / (2.0 * std::f32::consts::PI * tau);   // corner freq [Hz]
    let df  = fs as f32 / FFT_SIZE as f32;                 // bin spacing [Hz]

    // Pre-compute de-emphasis gain for every FFT bin (DC → Nyquist and back).
    // Bin k → positive freq k·df; bin (FFT_SIZE−k) → negative freq k·df.
    let gains: Vec<f32> = (0..FFT_SIZE).map(|k| {
        let freq = if k <= FFT_SIZE / 2 { k as f32 } else { (FFT_SIZE - k) as f32 } * df;
        f0 / (f0 * f0 + freq * freq).sqrt()    // |H_de(freq)| = f0/√(f0²+f²)
    }).collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft  = planner.plan_fft_forward(FFT_SIZE);
    let ifft = planner.plan_fft_inverse(FFT_SIZE);
    let scale = 1.0 / FFT_SIZE as f32;

    let mut out = samples.to_vec();

    // Process each complete OFDM symbol (CP + FFT window).
    let mut sym = 0;
    while sym + SYMBOL_LEN <= out.len() {
        let win_start = sym + CP_LEN;
        let win_end   = sym + SYMBOL_LEN;

        // Forward FFT.
        let mut buf: Vec<Complex32> = out[win_start..win_end]
            .iter().map(|&x| Complex32::new(x, 0.0)).collect();
        fft.process(&mut buf);

        // Apply de-emphasis amplitude gains (real filter → symmetric gains).
        for (b, &g) in buf.iter_mut().zip(gains.iter()) {
            *b *= g;
        }

        // Inverse FFT and write back (real part only, normalised).
        ifft.process(&mut buf);
        for (i, b) in buf.iter().enumerate() {
            out[win_start + i] = b.re * scale;
        }

        // Regenerate CP from the tail of the new FFT window.
        let cp_src = win_end - CP_LEN;
        out.copy_within(cp_src..win_end, sym);

        sym += SYMBOL_LEN;
    }

    out
}

/// Per-symbol raised-cosine boundary taper.
///
/// Multiplies each `SYMBOL_LEN`-aligned block by a Tukey-like envelope: the
/// first `t_w` samples rise from 0 to 1 through a raised-cosine, the middle is
/// flat at 1, and the last `t_w` samples fall from 1 to 0.  This eliminates
/// the time-domain discontinuity at every symbol boundary — the source of the
/// broadband spectral leakage we want to keep out of the FM pre-emphasis
/// region that creates ICI after the FM modulator.
///
/// The taper sits inside the CP for the first `t_w` samples (the CP is unused
/// by the RX FFT window, so no data is lost there) and attenuates the last
/// `t_w` samples of the FFT window for the falling edge.  With `t_w` ≪
/// `FFT_SIZE`, the resulting bin leakage is negligible (ratio ≈ (t_w/FFT)²),
/// and the ZC correlation drops from 1.00 to ≈ 1 − 1.5·t_w/SYMBOL_LEN ≈ 0.96
/// for t_w = 32 — well above the 0.35 sync threshold.
///
/// **Assumption**: `samples` is laid out as back-to-back `SYMBOL_LEN` blocks
/// starting at index 0.  Both the beacon and every data super-frame satisfy
/// this; the RX does not need to change.
fn smooth_symbol_boundaries(samples: &mut [f32], t_w: usize) {
    if t_w == 0 || t_w * 2 >= SYMBOL_LEN || samples.len() < SYMBOL_LEN {
        return;
    }

    // Raised-cosine rising edge: window[n] = (1 − cos(π·n/t_w))/2, n ∈ [0, t_w).
    let window: Vec<f32> = (0..t_w)
        .map(|n| 0.5 - 0.5 * (std::f32::consts::PI * n as f32 / t_w as f32).cos())
        .collect();

    let n_syms = samples.len() / SYMBOL_LEN;
    for k in 0..n_syms {
        let start = k * SYMBOL_LEN;
        let end   = start + SYMBOL_LEN;
        // Rising taper on the first t_w samples (inside the CP — harmless).
        for n in 0..t_w {
            samples[start + n] *= window[n];
        }
        // Falling taper on the last t_w samples (end of the FFT window).
        for n in 0..t_w {
            samples[end - t_w + n] *= window[t_w - 1 - n];
        }
    }
}

/// Linear-phase Kaiser-windowed FIR bandpass, passband 300…2600 Hz.
///
/// Mirrors QSSTV's CDRMBandpassFilt (drmtx/common/util/Utilities.cpp:127),
/// which applies a Nuttall-windowed 2.5 kHz lowpass modulated to the DRM IF
/// centre on the complex-baseband OFDM signal.  RustPIC emits real audio
/// directly in baseband, so the equivalent is a real bandpass covering the
/// active carrier band (328…2484 Hz) with small guard margins.
///
/// Without this filter, CP discontinuities, PAPR-clip spurs, and spectral
/// skirts spill energy below 300 Hz and above 2500 Hz.  Once that signal
/// goes through an FM transmitter with 75 µs pre-emphasis, the high-frequency
/// spurs are boosted up to +21 dB and saturate the FM modulator — producing
/// the ICI that raises the raw channel BER we observe OTA.
///
/// Implementation: a 201-tap symmetric FIR built as `ideal_BP × Kaiser(β=4.55)`
/// (≈50 dB stopband), applied by direct `O(N·L)` convolution.  The output is
/// aligned with the input (centre tap at `(N-1)/2`) so downstream sync still
/// finds the beacon/ZC at the expected positions.
fn apply_bandpass(samples: &[f32], fs: usize) -> Vec<f32> {
    const N: usize = 201;                 // filter length (odd for symmetry)
    const MID: usize = (N - 1) / 2;       // group delay = 100 samples

    // Normalised cutoff frequencies (cycles/sample).
    let f_lo = 300.0_f32 / fs as f32;
    let f_hi = 2600.0_f32 / fs as f32;

    // Modified Bessel function I₀, series form.
    fn i0(x: f32) -> f32 {
        let mut sum  = 1.0_f32;
        let mut term = 1.0_f32;
        let half_x   = x * 0.5;
        for k in 1..30 {
            let r = half_x / k as f32;
            term *= r * r;
            sum  += term;
            if term < 1e-12 * sum { break; }
        }
        sum
    }

    // Kaiser window, β = 4.55 → ~50 dB stopband attenuation.
    let beta    = 4.55_f32;
    let i0_beta = i0(beta);

    let mut taps = [0.0_f32; N];
    for n in 0..N {
        let m     = n as f32 - MID as f32;
        // Ideal bandpass impulse response.
        let ideal = if m == 0.0 {
            2.0 * (f_hi - f_lo)
        } else {
            let pm  = std::f32::consts::PI * m;
            ((2.0 * f_hi * pm).sin() - (2.0 * f_lo * pm).sin()) / pm
        };
        // Kaiser window.
        let r = m / MID as f32;
        let w = i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
        taps[n] = ideal * w;
    }

    // Direct convolution, centred (aligns output with input, zero-padded edges).
    let l    = samples.len();
    let mut out = vec![0.0_f32; l];
    for i in 0..l {
        let mut s = 0.0_f32;
        let k_min = MID.saturating_sub(i);
        let k_max = (MID + l - i).min(N);
        for k in k_min..k_max {
            let idx = i + k - MID;
            s += taps[k] * samples[idx];
        }
        out[i] = s;
    }
    out
}

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
