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
| Mode header | BPSK + CRC-16/CCITT, 32-bit diversity repeat — encodes modulation, LDPC rate, packet count |
| Data symbols | BPSK / QPSK / 16-QAM — 63 data + 9 pilot subcarriers per symbol |
| Re-sync ZC | Optional, inserted every 12 data symbols for long frames |
| EOT | CRC-32/IEEE-802.3 of the payload in the first 32 data subcarriers (BPSK) |

### FEC

| Layer | Codec | Parameters |
|---|---|---|
| Outer | Reed–Solomon | RS(255,191) over GF(2⁸), t=32 errors / 64 erasures |
| Inner | LDPC (QC, scaled min-sum) | n=252, rates 1/2 / 2/3 / 3/4 / 5/6 |

Failed LDPC blocks are passed to RS as erasures, maximising correction budget utilisation.

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
    │   ├── sync.rs               ZC correlator, CFO estimator, channel estimator
    │   ├── mode_detect.rs        mode-header encoder/decoder
    │   ├── equalizer.rs          pilot-based EMA channel equalizer
    │   ├── demapper.rs           soft LLR demapper (BPSK/QPSK/16-QAM)
    │   └── frame.rs              FrameReceiver — full RX pipeline, FrameMetrics
    └── tx/
        ├── mod.rs                OFDM modulator, constellation mapper
        └── frame.rs              build_frame — complete TX assembler
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
  --mod MOD          Modulation: bpsk qpsk 16qam    [default: bpsk]
  --payload-size N   Payload bytes                  [default: 191]
  --resync           Enable re-sync ZC symbols
```

Example (QPSK, rate 3/4, with all impairments):

```
$ cargo run --release --bin simtest -- --mod qpsk --rate 3/4 --snr-min 2 --snr-max 12

 SNR(dB)   ch.BER%   LDPC_ok%  LDPC_fail%   RS_ok%  RS_fail%   RS_margin%  CRC32_ok%
------------------------------------------------------------------------------------
     2.0      3.61       48.5        51.5     33.3      66.7         76.7       33.3
     4.0      1.40       83.6        16.4     80.0      20.0         83.6       80.0
     6.0      0.55       95.5         4.5     95.0       5.0         97.5       95.0
     8.0      0.10       99.5         0.5    100.0       0.0         98.1      100.0
    10.0      0.01      100.0         0.0    100.0       0.0        100.0      100.0
    12.0      0.00      100.0         0.0    100.0       0.0        100.0      100.0
```

---

## Status

Work in progress. The full codec stack (TX + RX + simulation) is implemented and tested.
Audio I/O, image encode/decode, Hamlib integration, and the web interface are not yet wired up.

---

## License

MIT
