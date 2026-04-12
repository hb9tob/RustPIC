//! FIR Hilbert transform — converts a real signal to its analytic (complex)
//! representation, enabling proper complex baseband processing.
//!
//! Inspired by QSSTV's `drmHilbertFilter` (153-tap Nuttall-windowed FIR).
//! Works in streaming mode: processes sample-by-sample through a ring buffer,
//! ready for both WAV files and future real-time soundcard I/O.
//!
//! The analytic signal z[n] = x[n−d] + j·H{x}[n] has only positive-frequency
//! content, so the FFT of each OFDM symbol yields the correct pilot phases
//! without conjugate-spectrum folding.  This also enables CFO estimation and
//! correction in the complex domain — essential for SSB where TX/RX oscillator
//! offsets must be tracked.

use num_complex::Complex32;
use std::f32::consts::PI;

use crate::ofdm::params::SAMPLE_RATE;

/// Number of Hilbert FIR taps (matches QSSTV's `DRMHILBERTTAPS`).
const HILBERT_TAPS: usize = 153;

/// FIR Hilbert filter that converts real samples to complex analytic signal.
///
/// Output: `z[n] = x[n − delay] + j · hilbert{x}[n]`
///
/// The group delay is `(HILBERT_TAPS − 1) / 2 = 76` samples at 48 kHz.
pub struct HilbertFilter {
    coefs: [f32; HILBERT_TAPS],
    buffer: [f32; HILBERT_TAPS],
    pos: usize,
    delay: usize,
}

impl HilbertFilter {
    /// Creates a new Hilbert filter with Nuttall-windowed coefficients
    /// (same window as QSSTV: a₀=0.3635819, a₁=0.4891775, a₂=0.1365995,
    /// a₃=0.0106411).  The gain is normalised so that |Q| ≈ |I| in the
    /// passband (562–2484 Hz).
    pub fn new() -> Self {
        let n = HILBERT_TAPS;
        let m = (n - 1) / 2; // center = group delay = 76

        // ── Compute raw Hilbert FIR with Nuttall window ──────────────────
        let mut coefs = [0.0_f32; HILBERT_TAPS];
        for i in 0..n {
            let k = i as i32 - m as i32;
            // Ideal Hilbert: h[k] = 2/(πk) for k odd, 0 for k even/zero
            let h = if k == 0 || k % 2 == 0 {
                0.0
            } else {
                2.0 / (PI * k as f32)
            };

            // Nuttall window (QSSTV coefficients from MatlibSigProToolbox.cpp)
            let t = 2.0 * PI * i as f32 / (n - 1) as f32;
            let w = 0.3635819 - 0.4891775 * t.cos()
                   + 0.1365995 * (2.0 * t).cos()
                   - 0.0106411 * (3.0 * t).cos();

            coefs[i] = h * w;
        }

        // ── Normalise gain at band centre (~1500 Hz) ─────────────────────
        let omega = 2.0 * PI * 1500.0 / SAMPLE_RATE as f32;
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (i, &c) in coefs.iter().enumerate() {
            re += c * (omega * i as f32).cos();
            im -= c * (omega * i as f32).sin();
        }
        let gain = (re * re + im * im).sqrt();
        if gain > 1e-6 {
            for c in &mut coefs {
                *c /= gain;
            }
        }

        Self {
            coefs,
            buffer: [0.0; HILBERT_TAPS],
            pos: 0,
            delay: m,
        }
    }

    /// Processes a block of real samples and returns complex analytic samples.
    ///
    /// Can be called repeatedly with successive blocks (streaming).
    pub fn process(&mut self, input: &[f32]) -> Vec<Complex32> {
        let n = HILBERT_TAPS;
        let mut output = Vec::with_capacity(input.len());

        for &x in input {
            // Push new sample into ring buffer
            self.buffer[self.pos] = x;

            // Q channel: FIR convolution (Hilbert transform)
            let mut q = 0.0_f32;
            for j in 0..n {
                let idx = (self.pos + n - j) % n;
                q += self.buffer[idx] * self.coefs[j];
            }

            // I channel: delayed input (group delay = 76 samples)
            let i_idx = (self.pos + n - self.delay) % n;
            let i_val = self.buffer[i_idx];

            output.push(Complex32::new(i_val, q));

            self.pos = (self.pos + 1) % n;
        }

        output
    }

    /// Group delay in samples (76 at 153 taps).
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Reset internal state (ring buffer) for a new signal.
    pub fn reset(&mut self) {
        self.buffer = [0.0; HILBERT_TAPS];
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure cosine through the Hilbert filter should produce a complex
    /// exponential (I=cos, Q=sin) in steady state.
    #[test]
    fn cosine_becomes_analytic() {
        let mut hf = HilbertFilter::new();
        let freq = 1500.0_f32;
        let n_samples = 2000;

        // Generate cosine
        let input: Vec<f32> = (0..n_samples)
            .map(|n| (2.0 * PI * freq * n as f32 / SAMPLE_RATE as f32).cos())
            .collect();

        let output = hf.process(&input);

        // Check steady-state (skip transient = 2× delay)
        let skip = 2 * hf.delay();
        let d = hf.delay();
        for n in skip..n_samples {
            let expected_phase = 2.0 * PI * freq * (n - d) as f32 / SAMPLE_RATE as f32;
            let expected_i = expected_phase.cos();
            let expected_q = expected_phase.sin();
            let got = output[n];
            let err_i = (got.re - expected_i).abs();
            let err_q = (got.im - expected_q).abs();
            assert!(err_i < 0.05, "I error at n={n}: {err_i:.4}");
            assert!(err_q < 0.05, "Q error at n={n}: {err_q:.4}");
        }
    }

    /// Verify that the analytic signal has ~0 energy at negative frequencies.
    #[test]
    fn no_negative_frequencies() {
        use rustfft::FftPlanner;

        let mut hf = HilbertFilter::new();
        let freq = 1000.0_f32;
        let n = 4096;

        let input: Vec<f32> = (0..n)
            .map(|k| (2.0 * PI * freq * k as f32 / SAMPLE_RATE as f32).cos())
            .collect();

        let output = hf.process(&input);

        // FFT the analytic signal (skip transient)
        let skip = 2 * hf.delay();
        let fft_len = n - skip;
        let mut buf: Vec<num_complex::Complex<f32>> = output[skip..]
            .iter()
            .copied()
            .collect();
        buf.truncate(fft_len);

        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(fft_len).process(&mut buf);

        // Positive freqs should have energy, negative should be near zero
        let pos_energy: f32 = buf[1..fft_len / 2].iter().map(|c| c.norm_sqr()).sum();
        let neg_energy: f32 = buf[fft_len / 2 + 1..].iter().map(|c| c.norm_sqr()).sum();

        let ratio = neg_energy / (pos_energy + 1e-12);
        assert!(ratio < 0.01, "Negative freq energy ratio: {ratio:.4} (should be <0.01)");
    }
}
