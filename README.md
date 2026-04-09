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
     LDPC (1/2…5/6)        inner FEC — adaptive rate per link SNR, z=210 (n=2520)
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

#### Inner — LDPC (z = 210, n = 2520)

| Rate | k (info bits) | Code rate |
|------|--------------|-----------|
| 1/2  | 1 260        | 0.500     |
| 2/3  | 1 680        | 0.667     |
| 3/4  | 1 890        | 0.750     |
| 5/6  | 2 100        | 0.833     |

QC-LDPC with quasi-cyclic shift z=210, scaled min-sum BP decoder (50 iterations, α=0.75).
Failed LDPC blocks are forwarded to the RS layer as erasures.

#### Outer — Reed–Solomon, 3 selectable levels

| Level | RS params   | Parity (2t) | Erasure capacity | Throughput |
|-------|-------------|-------------|------------------|------------|
| L0    | RS(255,239) | 16 bytes    | 16 erasures      | high       |
| L1    | RS(255,191) | 64 bytes    | 64 erasures      | balanced (default) |
| L2    | RS(255,127) | 128 bytes   | 128 erasures     | strong FEC |

RS codewords are **M-way byte-interleaved**: a single failed LDPC block spreads its
`⌈k/8⌉` affected bytes across M RS codewords, keeping the per-codeword erasure count
within the RS budget.  M is derived automatically for every rate/level pair.

### Modulations

| Mode | bps | Min SNR | Use case |
|------|-----|---------|----------|
| BPSK | 1 | ~4 dB | Very weak signals |
| QPSK | 2 | ~7 dB | Marginal links |
| 16-QAM | 4 | ~12 dB | Normal links |
| 32-QAM | 5 | ~15 dB | Good links |
| 64-QAM | 6 | ~21 dB | Excellent links |

### Packets per super-frame (RS level L1)

| Modulation | LDPC rate | Pkts/frame | Payload/frame |
|------------|-----------|-----------|---------------|
| BPSK       | R1/2      | 21        | ~4 KB         |
| QPSK       | R2/3      | 56        | ~10 KB        |
| 16-QAM     | R3/4      | 112       | ~20 KB        |
| 64-QAM     | R3/4      | 164       | ~30 KB        |
| 64-QAM     | R5/6      | 205       | ~38 KB        |

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
    --snr-min 18 --snr-max 26 --snr-step 1 --ppm 10 --runs 30 --resync

 SNR(dB)   ch.BER%   LDPC_ok%  LDPC_fail%   RS_ok%  RS_fail%   RS_margin%  CRC32_ok%
------------------------------------------------------------------------------------
    18.0      0.51       77.6        22.4     69.5      30.5          0.7        0.0
    19.0      0.29       89.1        10.9     88.4      11.6          1.9        0.0
    20.0      0.14       95.7         4.3     97.0       3.0          4.1       13.3
    21.0      0.06       98.5         1.5     99.4       0.6          5.8       83.3
    22.0      0.03       99.3         0.7     99.8       0.2         12.4       83.3
    23.0      0.01       99.7         0.3    100.0       0.0         43.8       96.7
    24.0      0.01       99.8         0.2    100.0       0.0         40.7      100.0
    25.0      0.00       99.9         0.1    100.0       0.0         78.1      100.0
    26.0      0.00      100.0         0.0    100.0       0.0         94.8      100.0
```

---

## Status

Work in progress. The full codec stack (TX + RX + simulation) is implemented and tested.
Audio I/O, image encode/decode, Hamlib integration, and the web interface are not yet wired up.

---

## License

MIT
