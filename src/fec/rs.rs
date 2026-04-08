//! Reed-Solomon RS(255, 191) codec over GF(2⁸).
//!
//! # Parameters
//!
//! | Parameter | Value | Meaning                                   |
//! |-----------|-------|-------------------------------------------|
//! | n         | 255   | Codeword length (symbols)                 |
//! | k         | 191   | Information symbols                       |
//! | 2t        | 64    | Parity symbols  (t = 32 error correction) |
//! | Field     | GF(2⁸)| Primitive poly x⁸+x⁴+x³+x²+1 (0x11D)  |
//! | Roots     | α¹…α⁶⁴| Generator polynomial roots              |
//!
//! # Codeword layout
//!
//! `[ parity₀ … parity₆₃ | info₀ … info₁₉₀ ]`
//!
//! The codeword polynomial is `c(x) = r(x) + m(x)·x^{2t}` where
//! `r(x) = m(x)·x^{2t} mod g(x)`.  With this layout,
//! `c(αʲ) = 0` for j = 1…64.
//!
//! # Error/erasure correction budget
//!
//! `2·errors + erasures ≤ 64`
//!
//! # Polynomial convention
//!
//! `poly[i]` = coefficient of `xⁱ` (index 0 = constant term).

// ── GF(2⁸) arithmetic ────────────────────────────────────────────────────────

/// Pre-computed GF(2⁸) lookup tables.
/// Primitive polynomial: x⁸+x⁴+x³+x²+1 = 0x11D.
pub struct Gf256 {
    exp: [u8; 512], // exp[i] = αⁱ  (extended for wrap-around arithmetic)
    log: [u8; 256], // log[αⁱ] = i  (log[0] undefined, stored as 0)
}

impl Gf256 {
    pub fn new() -> Self {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x = 1u16;
        for i in 0..255u16 {
            exp[i as usize] = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 { x ^= 0x011D; }
        }
        for i in 255..512usize { exp[i] = exp[i - 255]; }
        Self { exp, log }
    }

    /// αⁿ  (wraps mod 255 via the 512-element exp table).
    #[inline] pub fn alpha_pow(&self, n: usize) -> u8 { self.exp[n % 255] }

    #[inline] pub fn add(&self, a: u8, b: u8) -> u8 { a ^ b }

    #[inline] pub fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 { return 0; }
        self.exp[self.log[a as usize] as usize + self.log[b as usize] as usize]
    }

    /// a / b.  Panics if b = 0.
    #[inline] pub fn div(&self, a: u8, b: u8) -> u8 {
        if a == 0 { return 0; }
        debug_assert!(b != 0, "GF div by zero");
        self.exp[(self.log[a as usize] as usize + 255
                  - self.log[b as usize] as usize) % 255]
    }

    #[inline] pub fn inv(&self, a: u8) -> u8 { self.div(1, a) }

    // ── Polynomials ───────────────────────────────────────────────────────────

    /// Horner evaluation: `Σ poly[i]·xⁱ`.
    pub fn poly_eval(&self, poly: &[u8], x: u8) -> u8 {
        let mut acc = 0u8;
        for &c in poly.iter().rev() { acc = self.add(c, self.mul(acc, x)); }
        acc
    }

    pub fn poly_mul(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        if a.is_empty() || b.is_empty() { return vec![]; }
        let mut out = vec![0u8; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() { out[i + j] ^= self.mul(ai, bj); }
        }
        out
    }

    pub fn poly_add(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        let len = a.len().max(b.len());
        (0..len).map(|i| a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)).collect()
    }

    /// Formal derivative over GF(2ᵐ): only odd-index terms survive.
    /// `f'(x) = f₁ + f₃x² + f₅x⁴ + …`
    pub fn poly_deriv(&self, poly: &[u8]) -> Vec<u8> {
        if poly.len() <= 1 { return vec![0]; }
        let deg = poly.len() - 1;
        let mut out = vec![0u8; deg];
        for i in (1..=deg).step_by(2) { out[i - 1] = poly[i]; }
        out
    }
}

impl Default for Gf256 { fn default() -> Self { Self::new() } }

// ── RS codec ──────────────────────────────────────────────────────────────────

pub const RS_N:  usize = 255;
pub const RS_K:  usize = 191;
pub const RS_2T: usize = 64;
pub const RS_T:  usize = 32;

/// Statistics returned by a successful RS decode.
#[derive(Debug, Clone, PartialEq)]
pub struct RsDecodeStats {
    /// Number of random errors corrected (not counting erasures).
    pub errors_corrected: usize,
    /// Number of erasure positions consumed.
    pub erasures_used: usize,
}

/// RS(255, 191) codec.
pub struct RsCodec {
    gf:  Gf256,
    gen: Vec<u8>, // generator polynomial, degree 64, gen[64] = 1
}

impl RsCodec {
    pub fn new() -> Self {
        let gf  = Gf256::new();
        let gen = build_generator(&gf);
        Self { gf, gen }
    }

    // ── Encoder ───────────────────────────────────────────────────────────────

    /// Encodes `data` (191 bytes) into a 255-byte systematic codeword.
    ///
    /// Layout: `[ parity(0..64) | info(64..255) ]`
    ///
    /// `c(x) = r(x) + m(x)·x^{2t}` where `r = m(x)·x^{2t} mod g(x)`.
    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        assert_eq!(data.len(), RS_K);
        let gf  = &self.gf;
        let gen = &self.gen; // gen[0..=RS_2T], gen[RS_2T] = 1

        // Build dividend: m(x)·x^{2t}
        // rem[0..RS_2T] = 0 (will become parity)
        // rem[RS_2T..RS_N] = data
        let mut rem = vec![0u8; RS_N];
        rem[RS_2T..].copy_from_slice(data);

        // Polynomial long division by g(x) from the highest degree down.
        // After each step, rem[i] = 0 for all i ≥ RS_2T.
        for i in (RS_2T..RS_N).rev() {
            if rem[i] == 0 { continue; }
            let c = rem[i]; // leading coefficient (gen is monic: gen[RS_2T] = 1)
            for j in 0..RS_2T {
                rem[i - RS_2T + j] ^= gf.mul(c, gen[j]);
            }
            rem[i] = 0;
        }

        // Codeword = [parity, info]
        let mut cw = rem[..RS_2T].to_vec();
        cw.extend_from_slice(data);
        cw
    }

    // ── Syndromes ─────────────────────────────────────────────────────────────

    /// `S[j] = c(α^{j+1})` for j = 0…63.  All-zero ↔ no detectable error.
    pub fn syndromes(&self, received: &[u8]) -> Vec<u8> {
        assert_eq!(received.len(), RS_N);
        (1..=RS_2T)
            .map(|j| self.gf.poly_eval(received, self.gf.alpha_pow(j)))
            .collect()
    }

    // ── Decoder ───────────────────────────────────────────────────────────────

    /// Decodes a received codeword.
    ///
    /// * `received`  — 255 bytes.
    /// * `erasures`  — byte positions (0…254) that are known to be erased.
    ///
    /// Returns 191 decoded information bytes plus correction statistics, or an [`RsError`].
    pub fn decode(&self, received: &[u8], erasures: &[usize])
        -> Result<(Vec<u8>, RsDecodeStats), RsError>
    {
        assert_eq!(received.len(), RS_N);
        let f = erasures.len();
        if f > RS_2T {
            return Err(RsError::TooManyErasures { count: f, max: RS_2T });
        }

        let gf = &self.gf;

        // ── Syndromes ─────────────────────────────────────────────────────────
        let synd = self.syndromes(received);
        if synd.iter().all(|&s| s == 0) && f == 0 {
            return Ok((received[RS_2T..].to_vec(),
                       RsDecodeStats { errors_corrected: 0, erasures_used: 0 }));
        }

        // ── Erasure locator Γ(x) = Π (1 − α^ρᵢ · x) ─────────────────────────
        let mut gamma = vec![1u8];
        for &pos in erasures {
            let factor = vec![1u8, gf.alpha_pow(pos)];
            gamma = gf.poly_mul(&gamma, &factor);
        }

        // ── Modified syndromes T(x) = S(x)·Γ(x) mod x^{2t} ──────────────────
        // s_poly[i] = S[i]  (coefficient of xⁱ in the syndrome polynomial)
        let s_poly: Vec<u8> = synd.clone();
        let t_full = gf.poly_mul(&s_poly, &gamma);
        let t_poly: Vec<u8> = t_full.iter().take(RS_2T).copied().collect();

        // ── BM on the part of T beyond the erasure locator ────────────────────
        let sigma = berlekamp_massey(gf, &t_poly[f.min(RS_2T)..]);

        let n_err = sigma.len().saturating_sub(1); // random errors (excl. erasures)
        if 2 * n_err + f > RS_2T {
            return Err(RsError::TooManyErrors { errors: n_err, erasures: f });
        }

        // ── Combined locator λ = σ · Γ ────────────────────────────────────────
        let lambda = gf.poly_mul(&sigma, &gamma);

        // ── Chien search — roots of λ(x) ─────────────────────────────────────
        let error_positions = chien_search(gf, &lambda);
        if error_positions.len() != lambda.len().saturating_sub(1) {
            return Err(RsError::DecodingFailed("Chien: degree mismatch"));
        }

        // ── Error evaluator Ω(x) = S(x)·λ(x) mod x^{2t} ─────────────────────
        let omega_full = gf.poly_mul(&s_poly, &lambda);
        let omega: Vec<u8> = omega_full.iter().take(RS_2T).copied().collect();

        // ── Forney — error/erasure magnitudes ─────────────────────────────────
        // For generator roots at α¹…α^{2t} (b = 1):
        //   eⱼ = Ω(Xⱼ⁻¹) / λ'(Xⱼ⁻¹)          (no extra Xⱼ factor)
        let lambda_d = gf.poly_deriv(&lambda);
        let mut corrected = received.to_vec();

        for &pos in &error_positions {
            if pos >= RS_N {
                return Err(RsError::DecodingFailed("error position out of range"));
            }
            let x_inv     = if pos == 0 { 1u8 } else { gf.alpha_pow(255 - pos) };
            let omega_val = gf.poly_eval(&omega, x_inv);
            let lam_d_val = gf.poly_eval(&lambda_d, x_inv);
            if lam_d_val == 0 {
                return Err(RsError::DecodingFailed("Forney: zero derivative"));
            }
            corrected[pos] ^= gf.div(omega_val, lam_d_val);
        }

        // ── Verify ────────────────────────────────────────────────────────────
        if self.syndromes(&corrected).iter().any(|&s| s != 0) {
            return Err(RsError::DecodingFailed("syndromes non-zero after correction"));
        }

        Ok((corrected[RS_2T..].to_vec(),
            RsDecodeStats { errors_corrected: n_err, erasures_used: f }))
    }
}

impl Default for RsCodec { fn default() -> Self { Self::new() } }

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RsError {
    TooManyErasures { count: usize, max: usize },
    TooManyErrors   { errors: usize, erasures: usize },
    DecodingFailed(&'static str),
}
impl std::fmt::Display for RsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyErasures { count, max } =>
                write!(f, "too many erasures: {count} > {max}"),
            Self::TooManyErrors { errors, erasures } =>
                write!(f, "2·{errors}+{erasures} > 64"),
            Self::DecodingFailed(m) =>
                write!(f, "RS decode failed: {m}"),
        }
    }
}
impl std::error::Error for RsError {}

// ── Generator polynomial ──────────────────────────────────────────────────────

/// g(x) = Π_{i=1}^{64} (x + αⁱ)  (monic, degree 64).
fn build_generator(gf: &Gf256) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 1..=RS_2T {
        let factor = vec![gf.alpha_pow(i), 1u8]; // αⁱ + x
        g = gf.poly_mul(&g, &factor);
    }
    g
}

// ── Berlekamp-Massey ──────────────────────────────────────────────────────────

/// Finds the minimal error-locator polynomial from a syndrome slice.
fn berlekamp_massey(gf: &Gf256, syndromes: &[u8]) -> Vec<u8> {
    let two_t = syndromes.len();
    let mut lam  = vec![0u8; two_t + 1]; lam[0] = 1;
    let mut b    = vec![0u8; two_t + 1]; b[0]   = 1;
    let mut l      = 0usize;
    let mut m      = 1usize;
    let mut b_lead = 1u8;

    for n in 0..two_t {
        // Discrepancy
        let mut delta = syndromes[n];
        for i in 1..=l {
            if n >= i { delta ^= gf.mul(lam[i], syndromes[n - i]); }
        }

        if delta == 0 {
            m += 1;
        } else if 2 * l <= n {
            let lam_old = lam.clone();
            let coeff   = gf.div(delta, b_lead);
            for i in m..=two_t { lam[i] ^= gf.mul(coeff, b[i - m]); }
            l = n + 1 - l;
            b = lam_old; b_lead = delta; m = 1;
        } else {
            let coeff = gf.div(delta, b_lead);
            for i in m..=two_t { lam[i] ^= gf.mul(coeff, b[i - m]); }
            m += 1;
        }
    }
    lam[..=l].to_vec()
}

// ── Chien search ─────────────────────────────────────────────────────────────

/// Returns positions p ∈ [0, 254] where λ(α^{-p}) = 0.
fn chien_search(gf: &Gf256, lam: &[u8]) -> Vec<usize> {
    (0..RS_N).filter(|&pos| {
        let x_inv = if pos == 0 { 1u8 } else { gf.alpha_pow(255 - pos) };
        gf.poly_eval(lam, x_inv) == 0
    }).collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GF(256) ───────────────────────────────────────────────────────────────

    #[test]
    fn gf_mul_identity()     { let gf = Gf256::new(); for a in 1u8..=255 { assert_eq!(gf.mul(a,1), a); } }
    #[test]
    fn gf_mul_zero()         { let gf = Gf256::new(); for a in 0u8..=255 { assert_eq!(gf.mul(a,0), 0); } }
    #[test]
    fn gf_mul_inverse()      { let gf = Gf256::new(); for a in 1u8..=255 { assert_eq!(gf.mul(a, gf.inv(a)), 1); } }
    #[test]
    fn gf_pow_order()        { let gf = Gf256::new(); assert_eq!(gf.alpha_pow(255), gf.alpha_pow(0)); assert_eq!(gf.alpha_pow(0), 1); }
    #[test]
    fn gf_mul_commutative()  {
        let gf = Gf256::new();
        for a in [3u8,17,55,127,200] { for b in [2u8,9,33,100,250] { assert_eq!(gf.mul(a,b), gf.mul(b,a)); } }
    }

    // ── Generator ─────────────────────────────────────────────────────────────

    #[test]
    fn generator_degree() {
        let gf = Gf256::new(); assert_eq!(build_generator(&gf).len(), RS_2T + 1);
    }
    #[test]
    fn generator_roots() {
        let gf  = Gf256::new();
        let gen = build_generator(&gf);
        for i in 1..=RS_2T { assert_eq!(gf.poly_eval(&gen, gf.alpha_pow(i)), 0, "g(α^{i}) ≠ 0"); }
    }

    // ── Encode / syndromes ─────────────────────────────────────────────────────

    #[test]
    fn encode_length() {
        let rs = RsCodec::new();
        assert_eq!(rs.encode(&vec![0xABu8; RS_K]).len(), RS_N);
    }
    #[test]
    fn zero_syndromes_on_codeword() {
        let rs = RsCodec::new();
        let cw = rs.encode(&vec![0x42u8; RS_K]);
        assert!(rs.syndromes(&cw).iter().all(|&s| s == 0));
    }
    #[test]
    fn zero_syndromes_sequential() {
        let rs   = RsCodec::new();
        let data: Vec<u8> = (0..RS_K as u8).collect();
        let cw   = rs.encode(&data);
        assert!(rs.syndromes(&cw).iter().all(|&s| s == 0),
            "valid codeword must have zero syndromes");
    }

    // ── Decode ────────────────────────────────────────────────────────────────

    #[test]
    fn decode_no_errors() {
        let rs   = RsCodec::new();
        let data: Vec<u8> = (0..RS_K as u8).collect();
        let cw   = rs.encode(&data);
        let (dec, stats) = rs.decode(&cw, &[]).expect("no-error decode");
        assert_eq!(dec, data);
        assert_eq!(stats.errors_corrected, 0);
        assert_eq!(stats.erasures_used, 0);
    }

    #[test]
    fn decode_one_error() {
        let rs   = RsCodec::new();
        let data: Vec<u8> = (0..RS_K as u8).collect();
        let mut cw = rs.encode(&data);
        cw[5] ^= 0xFF; // error at parity position 5
        let (dec, stats) = rs.decode(&cw, &[]).expect("single error");
        assert_eq!(dec, data);
        assert_eq!(stats.errors_corrected, 1);
    }

    #[test]
    fn decode_t_errors() {
        let rs   = RsCodec::new();
        let data = vec![0x55u8; RS_K];
        let mut cw = rs.encode(&data);
        // 32 errors at even parity positions 0,2,4,...,62
        for i in 0..RS_T { cw[i * 2] ^= 0xA5; }
        let (dec, stats) = rs.decode(&cw, &[]).expect("t=32 errors");
        assert_eq!(dec, data);
        assert_eq!(stats.errors_corrected, RS_T);
    }

    #[test]
    fn decode_erasures_only() {
        let rs   = RsCodec::new();
        let data = vec![0xCCu8; RS_K];
        let cw   = rs.encode(&data);
        // Erase all 64 parity bytes (positions 0..64)
        let erase: Vec<usize> = (0..RS_2T).collect();
        let mut rx = cw.clone();
        for &p in &erase { rx[p] = 0; }
        let (dec, stats) = rs.decode(&rx, &erase).expect("64 erasures (all parity)");
        assert_eq!(dec, data);
        assert_eq!(stats.errors_corrected, 0);
        assert_eq!(stats.erasures_used, RS_2T);
    }

    #[test]
    fn decode_mixed_errors_and_erasures() {
        let rs   = RsCodec::new();
        let data: Vec<u8> = (0..RS_K as u8).collect();
        let mut cw = rs.encode(&data);
        // 10 errors at parity positions 0..9, 40 erasures at parity positions 20..60
        // budget: 2*10 + 40 = 60 ≤ 64 ✓
        let erase: Vec<usize> = (20..60).collect();
        for &p in &erase { cw[p] = 0; }
        for i in 0..10usize { cw[i] ^= 0x1F; } // errors at 0..9 (disjoint from erasures)
        let (dec, stats) = rs.decode(&cw, &erase).expect("10 errors + 40 erasures");
        assert_eq!(dec, data);
        assert_eq!(stats.errors_corrected, 10);
        assert_eq!(stats.erasures_used, 40);
    }

    #[test]
    fn too_many_erasures_returns_error() {
        let rs  = RsCodec::new();
        let cw  = rs.encode(&vec![0u8; RS_K]);
        assert!(matches!(
            rs.decode(&cw, &(0..65).collect::<Vec<_>>()),
            Err(RsError::TooManyErasures { .. })
        ));
    }
}
