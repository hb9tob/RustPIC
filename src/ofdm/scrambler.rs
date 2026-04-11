//! G3RUH PRBS scrambler / descrambler.
//!
//! Implements the G3RUH 17-stage LFSR with polynomial **x¹⁷ + x¹² + 1**,
//! the de-facto standard for digital radio (9600 Bd packet, AX.25 G3RUH FSK).
//!
//! **Why:** Zero-bit padding at the end of a frame produces long runs of +1.0
//! BPSK symbols (all 72 subcarriers in phase), creating PAPR spikes up to
//! 18.5 dB that saturate FM modulators.  The scrambler randomises the bit
//! stream so peak power is bounded regardless of payload content.
//!
//! **Synchronisation:** Both TX and RX reset the LFSR to [`G3RUH_SEED`] at
//! the very start of each super-frame's coded-bit stream — no extra overhead.
//!
//! **Application point:**
//! - TX: XOR each coded bit (after LDPC encode) with `prbs.next_bit()`.
//! - RX: For each LDPC block, negate LLRs where the PRBS bit is 1 (`descramble_llrs`).

/// G3RUH 17-bit LFSR scrambler (polynomial x¹⁷ + x¹² + 1).
///
/// Both TX and RX must be initialised to the same seed and advanced in
/// lock-step through the frame's coded-bit stream.
pub struct G3ruhScrambler {
    state: u32,
}

/// Fixed initial LFSR state — all ones (standard for G3RUH).
const G3RUH_SEED: u32 = 0x0001_FFFF; // 17 bits, all 1

impl Default for G3ruhScrambler {
    fn default() -> Self { Self::new() }
}

impl G3ruhScrambler {
    /// Creates a new G3RUH scrambler in the initial state.
    pub fn new() -> Self {
        Self { state: G3RUH_SEED }
    }

    /// Returns the next PRBS bit (0 or 1) and advances the LFSR.
    ///
    /// Feedback: `out = state[16] XOR state[11]`  (taps at degree 17 and 12).
    /// State update: shift left by 1, insert `out` at bit 0.
    #[inline(always)]
    pub fn next_bit(&mut self) -> u8 {
        let out = ((self.state >> 16) ^ (self.state >> 11)) & 1;
        self.state = ((self.state << 1) | out) & 0x0001_FFFF;
        out as u8
    }

    /// Scrambles a slice of hard bits in-place (XOR each bit with PRBS).
    pub fn scramble_bits(&mut self, bits: &mut [u8]) {
        for b in bits.iter_mut() {
            *b ^= self.next_bit();
        }
    }

    /// Descrambles LLRs in-place: negate each LLR where the PRBS bit is 1.
    ///
    /// Equivalent to hard-decision XOR but operates in the soft-decision domain.
    pub fn descramble_llrs(&mut self, llrs: &mut [f32]) {
        for l in llrs.iter_mut() {
            if self.next_bit() == 1 {
                *l = -*l;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scrambled sequence descrambled by a second generator (same seed)
    /// must recover the original.
    #[test]
    fn scramble_roundtrip() {
        let original: Vec<u8> = (0..256usize).map(|i| (i & 1) as u8).collect();
        let mut tx = G3ruhScrambler::new();
        let mut bits = original.clone();
        tx.scramble_bits(&mut bits);

        // Output must differ from input (very unlikely to be identical)
        assert_ne!(bits, original);

        // Descramble with fresh generator at same seed
        let mut rx = G3ruhScrambler::new();
        rx.scramble_bits(&mut bits);
        assert_eq!(bits, original);
    }

    /// PRBS has the expected period for the 17-bit polynomial (2¹⁷ − 1 = 131071).
    #[test]
    fn prbs_period() {
        let mut gen = G3ruhScrambler::new();
        let first_bit = gen.next_bit();
        // Advance 131070 more steps (total 131071)
        for _ in 0..131070 {
            gen.next_bit();
        }
        // The next bit should match the first one (sequence is periodic)
        assert_eq!(gen.next_bit(), first_bit);
    }

    /// LLR descrambling sign-flips exactly at PRBS=1 positions.
    #[test]
    fn llr_descramble_matches_bits() {
        let mut ref_gen = G3ruhScrambler::new();
        let prbs_bits: Vec<u8> = (0..64).map(|_| ref_gen.next_bit()).collect();

        let original_llrs: Vec<f32> = (0..64)
            .map(|i| if i % 3 == 0 { 1.5_f32 } else { -0.7_f32 })
            .collect();
        let mut llrs = original_llrs.clone();

        let mut rx = G3ruhScrambler::new();
        rx.descramble_llrs(&mut llrs);

        for (i, (&bit, (&orig, &desc))) in prbs_bits.iter()
            .zip(original_llrs.iter().zip(llrs.iter()))
            .enumerate()
        {
            if bit == 0 {
                assert_eq!(desc, orig,  "pos {i}: prbs=0, LLR must be unchanged");
            } else {
                assert_eq!(desc, -orig, "pos {i}: prbs=1, LLR must be negated");
            }
        }
    }
}
