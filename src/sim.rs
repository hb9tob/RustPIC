//! Channel simulation: AWGN noise, clock-offset resampling, guard noise.
//!
//! # Model
//!
//! Impairments are applied in this order:
//!
//! 1. **Guard noise** — prepend a random number of AWGN samples (0…`guard_samples_max`)
//!    before the frame.  Simulates the receiver starting to process audio before the
//!    frame begins.
//!
//! 2. **AWGN** — add complex Gaussian noise at the configured SNR.  Noise power is
//!    computed from the mean signal power of the TX sample buffer.
//!
//! 3. **Clock offset** — linearly resample the combined buffer at rate
//!    `1 + ppm/1_000_000` where `ppm` is drawn uniformly from
//!    `[−clock_ppm_max, +clock_ppm_max]`.  This simulates a receiver ADC running at
//!    a slightly different frequency than the transmitter DAC.

use num_complex::Complex32;
use rand::Rng;

// ── SimChannel ────────────────────────────────────────────────────────────────

/// Configurable impairment channel.
pub struct SimChannel {
    /// Signal-to-noise ratio in dB.
    pub snr_db: f32,
    /// Maximum absolute clock frequency offset in parts-per-million.
    /// Set to 0.0 to disable resampling.
    pub clock_ppm_max: f32,
    /// Maximum guard samples prepended before the frame.
    /// Set to 0 to disable guard noise.
    pub guard_samples_max: usize,
}

impl SimChannel {
    /// Apply channel impairments to a clean TX sample stream.
    ///
    /// Returns the impaired receive buffer (longer than `tx` by `guard_len` samples,
    /// possibly shorter after clock-offset resampling).
    pub fn apply<R: Rng>(&self, tx: &[Complex32], rng: &mut R) -> Vec<Complex32> {
        // ── Noise sigma ───────────────────────────────────────────────────────
        let signal_power = if tx.is_empty() {
            1.0f32
        } else {
            tx.iter().map(|s| s.norm_sqr()).sum::<f32>() / tx.len() as f32
        };
        // For complex AWGN: total noise power = sigma_I² + sigma_Q² = 2·sigma²
        // SNR = signal_power / (2·sigma²)  →  sigma = sqrt(signal_power / (2·10^(SNR/10)))
        let noise_sigma = f32::sqrt(signal_power / (2.0 * 10f32.powf(self.snr_db / 10.0)));

        // ── 1. Guard noise ────────────────────────────────────────────────────
        let guard_len = if self.guard_samples_max > 0 {
            rng.gen_range(0..=self.guard_samples_max)
        } else {
            0
        };
        let total_len = guard_len + tx.len();
        let mut out: Vec<Complex32> = Vec::with_capacity(total_len);

        for _ in 0..guard_len {
            out.push(awgn_sample(rng, noise_sigma));
        }

        // ── 2. Signal + AWGN ──────────────────────────────────────────────────
        for &s in tx {
            out.push(s + awgn_sample(rng, noise_sigma));
        }

        // ── 3. Clock-offset resampling ────────────────────────────────────────
        if self.clock_ppm_max > 0.0 {
            let ppm: f32 = rng.gen_range(-self.clock_ppm_max..=self.clock_ppm_max);
            out = linear_resample(&out, 1.0 + ppm / 1_000_000.0);
        }

        out
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// One complex AWGN sample: N(0, sigma²) on each of I and Q.
#[inline]
fn awgn_sample<R: Rng>(rng: &mut R, sigma: f32) -> Complex32 {
    Complex32::new(box_muller(rng) * sigma, box_muller(rng) * sigma)
}

/// Box–Muller transform → single standard-normal sample.
#[inline]
fn box_muller<R: Rng>(rng: &mut R) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Linear-interpolation resampler.
///
/// With `rate > 1.0` the output is shorter (receiver clock faster than TX),
/// with `rate < 1.0` the output is longer (receiver clock slower).
///
/// `output[i] = lerp(input, i × rate)`
fn linear_resample(samples: &[Complex32], rate: f32) -> Vec<Complex32> {
    if (rate - 1.0).abs() < 1e-9 {
        return samples.to_vec();
    }
    let n_in  = samples.len();
    let n_out = (n_in as f32 / rate).ceil() as usize;
    (0..n_out).map(|i| {
        let pos  = i as f32 * rate;
        let idx  = pos as usize;
        let frac = pos - idx as f32;
        if idx + 1 >= n_in {
            samples.last().copied().unwrap_or(Complex32::new(0.0, 0.0))
        } else {
            samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        }
    }).collect()
}
