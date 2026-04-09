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
     RS(255,191)          outer FEC — corrects burst errors
           │
           ▼
     LDPC (1/2…5/6)       inner FEC — adaptive rate per link SNR
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
| Mode header | BPSK + CRC-16/CCITT, 32-bit diversity repeat — encodes modulation, LDPC rate, packet count/offset |
| Data symbols | BPSK / QPSK / 16-QAM / 32-QAM / 64-QAM — 63 data + 9 pilot subcarriers per symbol |
| Re-sync ZC | Optional, inserted every 12 data symbols — maintains timing over long frames |
| EOT | CRC-32/IEEE-802.3 of the payload in the first 32 data subcarriers (BPSK) |

Large payloads are split across multiple super-frames. The mode header carries a 13-bit
`packet_offset` field so the receiver can reassemble them in order.

### FEC

| Layer | Codec | Parameters |
|---|---|---|
| Outer | Reed–Solomon | RS(255,191) over GF(2⁸), t=32 errors / 64 erasures |
| Inner | LDPC (QC, scaled min-sum) | n=252, rates 1/2 / 2/3 / 3/4 / 5/6 |

Failed LDPC blocks are passed to RS as erasures, maximising correction budget utilisation.

### Modulations

| Mode | bps | Min SNR | Use case |
|------|-----|---------|----------|
| BPSK | 1 | ~4 dB | Very weak signals |
| QPSK | 2 | ~7 dB | Marginal links |
| 16-QAM | 4 | ~12 dB | Normal links |
| 32-QAM | 5 | ~15 dB | Good links |
| 64-QAM | 6 | ~21 dB | Excellent links |

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
│   └── rs.rs                     RS(255,191) encoder/decoder over GF(2⁸)
└── ofdm/
    ├── params.rs                 system constants
    ├── zc.rs                     Zadoff–Chu preamble
    ├── rx/
    │   ├── sync.rs               ZC correlator, CFO estimator, channel estimator, resync
    │   ├── mode_detect.rs        mode-header encoder/decoder
    │   ├── equalizer.rs          pilot-based EMA channel equalizer with drift compensation
    │   ├── demapper.rs           soft LLR demapper (BPSK/QPSK/16/32/64-QAM)
    │   └── frame.rs              FrameReceiver — full RX pipeline, FrameMetrics
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
  --payload-size N   Payload bytes                  [default: 191]
  --resync           Enable re-sync ZC symbols
```

Example — 64-QAM R3/4, 100 KB payload, ±10 ppm clock offset, resync enabled:

```
$ cargo run --release --bin simtest -- \
    --mod 64qam --rate 3/4 --payload-size 100000 \
    --snr-min 18 --snr-max 30 --snr-step 1 --ppm 10 --runs 50 --resync

 SNR(dB)   ch.BER%   LDPC_ok%  LDPC_fail%   RS_ok%  RS_fail%   RS_margin%  CRC32_ok%
------------------------------------------------------------------------------------
    18.0      0.34       98.7         1.3     99.5       0.5         20.8       46.0
    19.0      0.16       99.6         0.4    100.0       0.0         42.5       88.0
    20.0      0.10       99.8         0.2     99.9       0.1         53.1       88.0
    21.0      0.04      100.0         0.0    100.0       0.0         75.5      100.0
    22.0      0.02      100.0         0.0    100.0       0.0         93.1      100.0
    23.0      0.01      100.0         0.0    100.0       0.0         96.2      100.0
    24.0      0.00      100.0         0.0    100.0       0.0         99.2      100.0
    25.0      0.00      100.0         0.0    100.0       0.0        100.0      100.0
```

---

## Status

Work in progress. The full codec stack (TX + RX + simulation) is implemented and tested.
Audio I/O, image encode/decode, Hamlib integration, and the web interface are not yet wired up.

---

## License

MIT
