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
     OFDM modulator        FFT=256, CP=32, fs=8 kHz, 72 subcarriers (312–2531 Hz)
           │
           ▼
     FM transmitter        via Hamlib CAT+PTT, software FM pre-emphasis 50 µs
```

### Super-frame structure

```
┌──────┬──────┬────────┬─ data ──┬── re-sync ──┬─ data ──┬─────┐
│ ZC#1 │ ZC#2 │ header │ D×12   │ ZC (resync) │ D×…    │ EOT │
└──────┴──────┴────────┴─────────┴─────────────┴─────────┴─────┘
```

| Field | Details |
|---|---|
| Preamble | Two identical Zadoff–Chu symbols (ZC_ROOT=25, SYMBOL_LEN=288) |
| Mode header | BPSK + CRC-16/CCITT, 32-bit diversity repeat — encodes modulation, LDPC rate, RS level, packet count/offset |
| Data symbols | BPSK / QPSK / 16-QAM / 32-QAM / 64-QAM — 63 data + 9 pilot subcarriers per symbol |
| Re-sync ZC | Optional, inserted every 12 data symbols — maintains timing over long frames |
| EOT | CRC-32/IEEE-802.3 of the payload in the first 32 data subcarriers (BPSK) |

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

### Packets per super-frame (RS level L1)

| Modulation | LDPC rate | M | bpg | Pkts/frame | Payload/frame |
|------------|-----------|---|-----|-----------|---------------|
| BPSK       | R1/2      | 2 | 5   | 18        | ~3.4 KB       |
| QPSK       | R2/3      | 3 | 5   | 51        | ~9.7 KB       |
| 16-QAM     | R3/4      | 3 | 5   | 105       | ~20 KB        |
| 64-QAM     | R3/4      | 3 | 5   | 159       | ~30 KB        |
| 64-QAM     | R5/6      | 4 | 6   | 180       | ~34 KB        |

### Net payload throughput (asymptotic, per RS level)

| Modulation | LDPC rate | L0       | L1       | L2      |
|------------|-----------|----------|----------|---------|
| BPSK       | R1/2      | ~101 B/s | ~68 B/s  | ~38 B/s |
| QPSK       | R2/3      | ~263 B/s | ~204 B/s | ~114 B/s|
| 16-QAM     | R3/4      | ~604 B/s | ~408 B/s | ~294 B/s|
| 32-QAM     | R3/4      | ~759 B/s | ~513 B/s | ~371 B/s|
| 64-QAM     | R3/4      | ~905 B/s | ~612 B/s | ~441 B/s|
| 64-QAM     | R5/6      | ~981 B/s | ~685 B/s | ~441 B/s|

Time to transmit 100 KB with 64-QAM: ~1.7 min (L0 R5/6) · ~2.4 min (L1 R5/6) · ~3.8 min (L2 R3/4).
Time with BPSK R1/2: ~16.5 min (L0) · ~24 min (L1) · ~44 min (L2).

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
cargo run --release --bin tx -- --input <file> --output <out.wav> [OPTIONS]
```

| Option | Values | Default |
|--------|--------|---------|
| `--mod` | `bpsk` `qpsk` `16qam` `32qam` `64qam` | `qpsk` |
| `--rate` | `1/2` `2/3` `3/4` `5/6` | `3/4` |
| `--rs` | `0` `1` `2` | `1` (L1) |
| `--resync` | flag | off |

The output WAV is 48 kHz 16-bit stereo.  The filename of `--input` is embedded in the
payload so the receiver can restore it without any metadata on the command line.

### Receiver — `rx`

```bash
cargo run --release --bin rx -- --input <file.wav> [--outdir <dir>]
```

The receiver scans the **entire** WAV for RustPIC preambles and decodes every
transmission it finds, writing each file to `--outdir` (default: `.`) under its
original filename.  Multiple transmissions in sequence — separated by silence,
speech, or noise — are all decoded in one pass.  Progress is shown on stderr.

### Audio pipeline

| Direction | Operation | Details |
|-----------|-----------|---------|
| TX 8→48 kHz | Nearest-neighbour upsample | Repeat × 6; no phase distortion |
| RX 48→8 kHz | 13-tap Hamming-windowed sinc LPF | fc = 4 kHz, group delay = 1 sample @ 8 kHz |

The TX/RX round-trip is mathematically exact in loopback (nearest-neighbour × box-
average = identity).  For real soundcard input the FIR provides >40 dB stopband
attenuation above 4 kHz, preventing speech/noise aliases from contaminating the
OFDM band (312–2531 Hz).

---

## Status

Work in progress. The full codec stack (TX + RX + simulation) is implemented and tested.
Image encode/decode, Hamlib integration, and the web interface are not yet wired up.

---

## License

MIT
