# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RustPIC sends digital images over narrow-band FM voice repeaters (2.5 kHz audio BW), simplex
point-to-multipoint, no ARQ. TX/RX are WAV-based; CAT/PTT/image encode are not yet wired.

## Common commands

```bash
cargo build --release
cargo test                                # all unit tests
cargo test --lib ofdm::params             # run a single module's tests
cargo test param_consistency              # run a single test by name

# SNR sweep over the AWGN simulator
cargo run --release --bin simtest -- --mod 64qam --rate 3/4 --rs-level 1 --resync

# Loopback at 48 kHz native
cargo run --release --bin sim48 -- ...

# Encode a payload file to a WAV
cargo run --release --bin tx -- --input <file> --output out.wav --callsign HB9TOB \
    --mod bpsk --rate 1/2

# Decode every transmission found in a WAV
cargo run --release --bin rx -- --input out.wav [--outdir .]
```

Debug tracing: several stages honour `FRAME_DEBUG=1` and `SYNC_DEBUG=1` environment
variables (gated `eprintln!` in `src/ofdm/rx/{frame,sync}.rs` and `src/bin/rx.rs`).

## Architecture (the big picture)

RustPIC is a **DRM Robustness Mode B** OFDM modem running natively at 48 kHz — no
resampling. The README describes an older 8 kHz / FFT=256 design; **`src/ofdm/params.rs` is
authoritative**, not the README. Current constants:

- `FFT_SIZE=1024`, `CP_LEN=256`, `SYMBOL_LEN=1280`, `SAMPLE_RATE=48000`
- `NUM_CARRIERS=42` on bins 12..53 (≈ 562–2484 Hz) — the low edge stays
  **above the NBFM audio HPF corner** (~300 Hz) so pilot 0 is in the
  channel's linear-delay region, not in the HPF phase-non-linear zone
- `PILOT_SPACING=8` → 6 pilots per symbol at active-carrier indices 0,8,16,24,32,40
- Zadoff–Chu preamble, `ZC_ROOT=25`, `ZC_LEN=42`
- Re-sync ZC every `RESYNC_PERIOD=12` data symbols

### Pipeline

```
payload → RS(255,k) outer (L0/L1/L2) → M-way byte interleave →
LDPC 802.11n QC z=81 n=1944 (R1/2 … R5/6) → G3RUH scrambler (17-bit LFSR, seed 0x1FFFF) →
BPSK/QPSK/16/32/64-QAM mapping → 47-carrier OFDM with scattered BPSK pilots →
ZC preamble + mode header + data + resync ZC + EOT → 48 kHz PCM WAV
```

Reverse on RX. The RX scans a full WAV for preambles (multiple transmissions per file are
decoded in one pass); a failed LDPC block is passed to RS as a full erasure, and the M-way
byte interleaver guarantees that one failed LDPC block per RS group stays within the RS
correction budget.

### Super-frame layout

`ZC#1 | ZC#2 | mode header (BPSK+CRC16, 32-bit diversity) | data×N [| resync ZC | data×…] | EOT (CRC-32 in first 32 data carriers)`

The mode header encodes modulation, LDPC rate, RS level, and a 12-bit `packet_offset` so
large payloads split across multiple super-frames reassemble in order.

### Crate layout (non-obvious)

- `src/ofdm/params.rs` — **single source of truth** for all OFDM constants. Editing
  any constant here propagates everywhere; the `param_consistency` test enforces
  invariants (CP/FFT ≥ 1/4, first carrier ≥ 300 Hz, LAST_BIN < Nyquist).
- `src/ofdm/rx/sync.rs` — time-domain normalized ZC cross-correlator with two-stage
  thresholds (primary ~0.35, confirm ~0.20) + CFO estimator + channel estimator.
  **Subtle invariant** (see `memory/feedback_resync_floor.md`): re-sync ZC channel
  estimation must always use `floor(true_pos)`; when fractional offset is negative,
  use `found_int - 1`.
- `src/ofdm/rx/frame.rs` — `FrameReceiver` drives the full RX state machine, including
  M-interleaved RS groups and the LLR path into LDPC.
- `src/ofdm/rx/demapper.rs` — max-log LLR demapper. **LLRs are clipped to ±20** before
  LDPC; the min-sum decoder diverges with unclipped values when pilot SNR estimates are
  optimistic.
- `src/ofdm/rx/equalizer.rs` — pilot-based EMA channel estimator with drift compensation.
  1D only (frequency, within a symbol) — there is no 2D time-frequency Wiener filter.
- `src/fec/ldpc.rs` — IEEE 802.11n QC-LDPC (z=81, n=1944), scaled min-sum BP, α=0.75, 50
  iterations. `src/fec/rs.rs` holds `RsLevel` L0/L1/L2 and the M-interleave helper
  (`M` is derived so a single failed LDPC block per RS group is always correctable).

### Binaries

All four binaries share `src/ofdm` and `src/fec`:

| Binary       | Purpose                                              |
|--------------|------------------------------------------------------|
| `tx`         | file → WAV encoder (beacon + OFDM super-frames)      |
| `rx`         | WAV → file decoder, multi-transmission scan          |
| `simtest`    | AWGN SNR sweep over the modem + FEC stack            |
| `sim48`      | 48 kHz native loopback                               |

### Beacon & audio pipeline

`tx` prepends a ~612 ms beacon per transmission: 360 ms of 1 kHz VOX tone, then a ZC
preamble + 5 BPSK symbols carrying `"DE <CALL> <mode> <filename>"`. Re-sync ZC symbols
are **on by default** (`--no-resync` to disable) and are essential for long BPSK/QPSK
frames where independent soundcard clock drift accumulates.

A 51-tap Kaiser FIR handles any non-48 kHz I/O; the native OFDM chain is already at
48 kHz so loopback through `sim48` is mathematically exact.

## Constraints and gotchas

- Simplex point-to-multipoint. **No ARQ, no retransmit.** Every design tradeoff favors
  open-loop robustness (RS + interleave + resync) over throughput.
- Radios of interest use G3E marine FM with 75 µs pre-emphasis. This boosts high OFDM
  carriers up to ~21 dB and drives FM deviation peaks into clipping. A disabled utility
  `preemphasis_gain()` exists in `params.rs` for future compensation.
- The README is partially **stale** (mentions 8 kHz / FFT=256 / 72 carriers and a 288-
  sample ZC). Treat `src/ofdm/params.rs` as authoritative when they disagree.
- Status: codec stack (TX/RX/sim) is implemented and tested. Image encode/decode, Hamlib,
  web UI are not wired.
