# RustPIC

Digital image transmission over narrow-band FM voice repeaters — written in Rust.

RustPIC encodes images into an OFDM audio stream that fits within a standard 2.5 kHz FM channel.
It is designed for simplex point-to-multipoint links (e.g. amateur radio repeaters) with no return
channel and no ARQ.

---

## Architecture

```
Image → WebP encode → payload bytes
           │
           ▼
     RS(255,k) outer FEC   3 selectable protection levels (L0/L1/L2)
           │  M-way byte interleave
           ▼
     LDPC (1/2…5/6)        inner FEC — adaptive rate per link SNR, IEEE 802.11n z=81 (n=1944)
           │
           ▼
     G3RUH scrambler       17-bit LFSR, seed 0x1FFFF
           │
           ▼
     OFDM modulator        FFT=1024, CP=256, fs=48 kHz, 42 subcarriers (562–2484 Hz)
           │                Δf = 46.875 Hz, symbol period 26.67 ms
           ▼
     FM transmitter        audio-in to any NBFM transceiver (VHF/UHF voice repeaters)
```

### Super-frame structure

```
┌──────┬──────┬────────┬─ data ──┬── re-sync ──┬─ data ──┬─────┐
│ ZC#1 │ ZC#2 │ header │ D×12    │ ZC (resync) │ D×…     │ EOT │
└──────┴──────┴────────┴─────────┴─────────────┴─────────┴─────┘
```

| Field | Details |
|---|---|
| Preamble | Two identical Zadoff–Chu symbols (`ZC_ROOT=25`, `ZC_LEN=42`), one `SYMBOL_LEN=1280` each |
| Mode header | 42 BPSK symbols + 10-bit CRC — encodes modulation, LDPC rate, RS level, packet count/offset |
| Data symbols | BPSK / QPSK / 16-QAM / 32-QAM / 64-QAM — 36 data + 6 pilot subcarriers per symbol |
| Re-sync ZC | On by default, inserted every 12 data symbols — maintains timing over long frames |
| EOT | CRC-32/IEEE-802.3 of the payload in the first 32 data subcarriers (BPSK) |

### OFDM parameters

RustPIC mirrors the ETSI DRM Mode B timing at native 48 kHz, with a narrower
active band tuned for NBFM voice repeaters.

| Parameter | Value |
|---|---|
| Sample rate (`SAMPLE_RATE`) | **48 000 Hz** (native, no resampling) |
| FFT size (`FFT_SIZE`) | 1024 |
| Cyclic prefix (`CP_LEN`) | 256 samples |
| Symbol length (`SYMBOL_LEN`) | 1280 samples = **26.67 ms** |
| Subcarrier spacing (`Δf`) | **46.875 Hz** (= 48 000 / 1024) |
| **Active subcarriers** | **42** (`NUM_CARRIERS`, bins 12…53) |
| Pilot subcarriers | 6 (active indices 0, 8, 16, 24, 32, 40) |
| Data subcarriers | 36 per OFDM symbol |
| First active carrier | **562.5 Hz** (`FIRST_BIN = 12`) |
| Last active carrier  | **2484.4 Hz** (`LAST_BIN = 53`) |
| Audio bandwidth | 1.9 kHz |

`FIRST_BIN = 12` keeps pilot 0 above the phase-non-linear region of a typical
NBFM radio's microphone-input high-pass filter (≈ 300 Hz corner). On early
on-the-air tests starting at 328 Hz, the pilot-based linear interpolation
could not track the HPF's phase rotation (up to −127°/carrier in the first
few bins) and lost roughly 5 % raw BER to misdecoded low carriers. Starting
at 562.5 Hz places every pilot in the channel's linear-delay region.

Large payloads are split across multiple super-frames. The mode header carries a 12-bit
`packet_offset` field so the receiver can reassemble them in order (max 4095 RS packets ≈ 760 KB).

### FEC

#### Inner — LDPC (IEEE 802.11n, z = 81, n = 1944)

| Rate | k (info bits) | Code rate |
|------|--------------|-----------|
| 1/2  | 972          | 0.500     |
| 2/3  | 1 296        | 0.667     |
| 3/4  | 1 458        | 0.750     |
| 5/6  | 1 620        | 0.833     |

QC-LDPC based on IEEE 802.11n-2009 Annex R (z=81, n_b=24, PEG-optimised shifts).
Scaled min-sum BP decoder (50 iterations, α=0.75).
Failed LDPC blocks are forwarded to the RS layer as erasures.

#### Outer — Reed–Solomon, 3 selectable levels

| Level | RS params   | rs_k | Parity (2t) | Max payload (4095 pkts) |
|-------|-------------|------|-------------|------------------------|
| L0    | RS(255,239) | 239  | 16 bytes    | ~956 KB                |
| L1    | RS(255,191) | 191  | 64 bytes    | ~764 KB  *(default)*   |
| L2    | RS(255,127) | 127  | 128 bytes   | ~508 KB                |

RS codewords are **M-way byte-interleaved**: a single failed LDPC block spreads its
`⌈k/8⌉` affected bytes across M RS codewords, keeping the per-codeword erasure count
within the RS budget.  M is derived automatically for every rate/level pair.

For all RS levels and LDPC rates, the interleave depth M is chosen so that **1 failed LDPC
block per RS group** is correctable.  With bpg=5 this means 20% of LDPC blocks per group
can be declared failed and still recovered; bpg=6 (R5/6 L1) allows 16.7%.

### Modulations

| Mode | bps | Min SNR | Use case |
|------|-----|---------|----------|
| BPSK | 1 | ~4 dB | Very weak signals |
| QPSK | 2 | ~7 dB | Marginal links |
| 16-QAM | 4 | ~12 dB | Normal links |
| 32-QAM | 5 | ~15 dB | Good links |
| 64-QAM | 6 | ~21 dB | Excellent links |

### Throughput

Gross OFDM symbol rate: 37.5 symbols/s (1 symbol every 26.67 ms). With
36 data subcarriers per symbol the gross information rates before FEC are:

| Modulation | bits/symbol | Gross rate |
|---|---|---|
| BPSK   | 36  | ~135 B/s |
| QPSK   | 72  | ~270 B/s |
| 16-QAM | 144 | ~540 B/s |
| 32-QAM | 180 | ~675 B/s |
| 64-QAM | 216 | ~810 B/s |

After LDPC inner code (rate 1/2 … 5/6), RS outer code (L0/L1/L2) and
M-way byte interleaving, the net payload throughput is roughly 0.35 … 0.8 ×
the gross rate depending on the protection level.

---

## Crate layout

```
src/
├── lib.rs
├── main.rs
├── sim.rs                        channel simulation
├── bin/
│   └── simtest.rs                CLI SNR-sweep runner
├── fec/
│   ├── ldpc.rs                   QC-LDPC encoder + scaled min-sum BP decoder
│   └── rs.rs                     RS encoder/decoder, RsLevel (L0/L1/L2), M-interleave helper
└── ofdm/
    ├── params.rs                 system constants
    ├── zc.rs                     Zadoff–Chu preamble
    ├── rx/
    │   ├── sync.rs               ZC correlator, CFO estimator, channel estimator, resync
    │   ├── mode_detect.rs        mode-header encoder/decoder (modulation/rate/rs_level)
    │   ├── equalizer.rs          pilot-based EMA channel equalizer with drift compensation
    │   ├── demapper.rs           soft LLR demapper (BPSK/QPSK/16/32/64-QAM)
    │   └── frame.rs              FrameReceiver — full RX pipeline, M-interleaved RS groups
    └── tx/
        ├── mod.rs                OFDM modulator, constellation mapper
        └── frame.rs              build_transmission — multi-frame TX assembler
```

---

## Building

```bash
cargo build --release
cargo test
```

Dependencies: `rustfft`, `num-complex`, `thiserror`, `tracing`, `rand`.

---

## Channel simulation

`simtest` sweeps SNR and reports channel BER, LDPC convergence, RS correction margin,
and end-to-end CRC32 success rate, with optional clock-offset and guard-noise impairments.

```bash
cargo run --release --bin simtest -- --help
```

```
OPTIONS:
  --snr-min F        Minimum SNR in dB              [default: 0.0]
  --snr-max F        Maximum SNR in dB              [default: 20.0]
  --snr-step F       SNR step in dB                 [default: 2.0]
  --ppm F            Max clock offset ppm           [default: 20.0]
  --guard N          Max guard samples              [default: 1000]
  --runs N           Frames per SNR point           [default: 10]
  --rate RATE        LDPC rate: 1/2 2/3 3/4 5/6     [default: 1/2]
  --mod MOD          Modulation: bpsk qpsk 16qam 32qam 64qam  [default: bpsk]
  --rs-level N       RS protection: 0=L0 1=L1 2=L2  [default: 1]
  --payload-size N   Payload bytes                  [default: 191]
  --resync           Enable re-sync ZC symbols
```

Example — 64-QAM R3/4 RS-L1, 100 KB payload, ±10 ppm clock offset, resync enabled:

```
$ cargo run --release --bin simtest -- \
    --mod 64qam --rate 3/4 --rs-level 1 --payload-size 100000 \
    --snr-min 14 --snr-max 22 --snr-step 1 --ppm 10 --runs 20 --resync

 SNR(dB)   ch.BER%   LDPC_ok%  LDPC_fail%   RS_ok%  RS_fail%   RS_margin%  CRC32_ok%
------------------------------------------------------------------------------------
    14.0      3.42       91.0         9.0     88.3      11.7          9.5       35.0
    15.0      2.70       95.3         4.7     93.6       6.4         36.2       40.0
    16.0      1.59       99.7         0.3     99.8       0.2         75.2       85.0
    17.0      0.97       99.9         0.1    100.0       0.0         94.3       95.0
    18.0      0.57      100.0         0.0    100.0       0.0         99.1      100.0
    19.0      0.27      100.0         0.0    100.0       0.0        100.0      100.0
    20.0      0.15      100.0         0.0    100.0       0.0        100.0      100.0
    21.0      0.00      100.0         0.0    100.0       0.0        100.0      100.0
    22.0      0.00      100.0         0.0    100.0       0.0        100.0      100.0
```

---

## TX/RX command-line tools

### Transmitter — `tx`

```bash
cargo run --release --bin tx -- --input <file> --output <out.wav> --callsign <CALL> [OPTIONS]
```

| Option | Values | Default |
|--------|--------|---------|
| `--callsign` | e.g. `HB9TOB` | *required* |
| `--mod` | `bpsk` `qpsk` `16qam` `32qam` `64qam` | `qpsk` |
| `--rate` | `1/2` `2/3` `3/4` `5/6` | `3/4` |
| `--rs` | `0` `1` `2` | `1` (L1) |
| `--no-resync` | flag | resync **on** by default |

Every transmission is preceded by a beacon (~640 ms) that:
- plays a 1 kHz tone for 10 OFDM symbol periods (≈ 267 ms) to activate VOX-switched transmitters
- follows with a ZC preamble + 9 BPSK OFDM symbols carrying `"DE <CALL> <mode> <filename>"`
- lets the TX chain and repeater squelch fully open before data starts

Re-sync ZC symbols are inserted every 12 data symbols by default (`--no-resync` to disable).
They allow the receiver to correct accumulated clock drift between independent soundcards — critical
for long frames (BPSK/QPSK) where even 100 ppm offset accumulates to multiple samples.

The filename of `--input` is also embedded in the data payload so the receiver
restores the original filename automatically.

### Receiver — `rx`

```bash
cargo run --release --bin rx -- --input <file.wav> [--outdir <dir>]
```

The receiver scans the **entire** WAV for RustPIC preambles and decodes every
transmission it finds, writing each file to `--outdir` (default: `.`) under its
original filename.  Multiple transmissions in sequence — separated by silence,
speech, or noise — are all decoded in one pass.  Progress is shown on stderr.

### Audio pipeline

The OFDM chain runs natively at 48 kHz — no resampling is required. The TX
writes 16-bit stereo WAV files (both channels identical) and the RX accepts
48 kHz mono or stereo input.

A symbol-boundary raised-cosine taper (32-sample default, `--smooth-tw`)
smooths the time-domain transition between consecutive OFDM symbols so the
CP discontinuity no longer leaks broadband energy into the FM audio stage.

---

## Status

Work in progress. The full codec stack (TX + RX + simulation) is implemented and
tested through a realistic NBFM channel model (audio HPF 300 Hz + LPF 3 kHz +
GNU Radio-style pre-emphasis / de-emphasis τ = 75 µs + optional hard-clip
deviation limiter). Image encode/decode, Hamlib integration, and the web
interface are not yet wired up.

Pre-built binaries (Windows + Linux static) are attached to each GitHub
release: https://github.com/hb9tob/RustPIC/releases

---

## License

MIT
