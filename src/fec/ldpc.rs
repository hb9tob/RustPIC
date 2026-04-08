//! LDPC belief propagation decoder (min-sum).
//!
//! # Code structure
//!
//! An LDPC code is defined by a sparse binary parity-check matrix H (m × n).
//! The code has length n, dimension k ≥ n − m, and rate ≈ k/n.
//! [`LdpcCode`] stores H as two adjacency lists:
//!
//! * `check_to_var[c]` — variable-node indices connected to check c
//! * `var_to_check[v]` — check-node indices connected to variable v
//!
//! # Decoder algorithm: scaled min-sum
//!
//! The scaled min-sum approximates belief propagation with lower complexity:
//!
//! **Check-node update** (for check c, variable v):
//! ```text
//! r_{c→v} = α · (Π_{v'≠v} sign(q_{v'→c})) · min_{v'≠v} |q_{v'→c}|
//! ```
//!
//! **Variable-node update**:
//! ```text
//! q_{v→c} = L_v + Σ_{c'≠c} r_{c'→v}
//! ```
//!
//! **Decision**:
//! ```text
//! d_v = L_v + Σ_c r_{c→v} ;  bit_v = 1 if d_v < 0
//! ```
//!
//! # LLR convention
//!
//! Inputs are log-likelihood ratios: `L_v = log P(b=0|y) / P(b=1|y)`.
//! Positive LLR → bit 0, negative LLR → bit 1.
//! This matches the convention used in [`crate::ofdm::rx::demapper`].
//!
//! # Production codes
//!
//! The 4 code rates (1/2, 2/3, 3/4, 5/6) use the codes returned by
//! [`LdpcCode::for_rate`].  These are small quasi-cyclic codes generated
//! deterministically — replace the base matrices with optimised codes
//! (DVB-S2, Wi-Fi, custom) for production deployment.

use crate::ofdm::rx::mode_detect::LdpcRate;

// ── LdpcCode ──────────────────────────────────────────────────────────────────

/// Sparse parity-check matrix representation of an LDPC code.
#[derive(Debug, Clone)]
pub struct LdpcCode {
    /// Code length (number of variable nodes).
    pub n: usize,
    /// Number of parity checks (rows of H).
    pub m: usize,
    /// Number of information bits  k = n − rank(H).
    pub k: usize,
    /// `check_to_var[c]` = list of variable indices in check c's neighbourhood.
    pub check_to_var: Vec<Vec<usize>>,
    /// `var_to_check[v]` = list of check indices in variable v's neighbourhood.
    pub var_to_check: Vec<Vec<usize>>,
    /// `var_pos_in_check[v][j]` = position of v inside `check_to_var[c]` where
    /// c = `var_to_check[v][j]`.  Pre-computed to avoid linear search in the
    /// inner decoder loop.
    var_pos_in_check: Vec<Vec<usize>>,

    // ── Systematic form (pre-computed once in new()) ──────────────────────────
    /// Codeword column indices that carry the **information bits**.
    /// Length = k.  Extracted from the BP-decoded codeword by the RX.
    pub info_cols: Vec<usize>,
    /// Codeword column indices for the **parity bits** (RREF pivot columns).
    /// Length = m.
    pub parity_cols: Vec<usize>,
    /// Generator sub-matrix: `parity[i] = XOR of info[j]` for every j where
    /// `parity_gen[i][j] = 1`.  Dimensions m × k (one `u8` per bit for clarity).
    parity_gen: Vec<Vec<u8>>,
}

impl LdpcCode {
    /// Constructs an `LdpcCode` from a list of check-node adjacency lists.
    ///
    /// Internally performs Gauss–Jordan elimination over GF(2) to compute the
    /// systematic form (info/parity column split + parity generation matrix).
    pub fn new(n: usize, k: usize, check_to_var: Vec<Vec<usize>>) -> Self {
        let m = check_to_var.len();

        // ── Decoder adjacency structures ──────────────────────────────────────
        let mut var_to_check: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (c, vars) in check_to_var.iter().enumerate() {
            for &v in vars { var_to_check[v].push(c); }
        }
        let mut var_pos_in_check: Vec<Vec<usize>> = vec![Vec::new(); n];
        for v in 0..n {
            for &c in &var_to_check[v] {
                let pos = check_to_var[c].iter().position(|&x| x == v).unwrap();
                var_pos_in_check[v].push(pos);
            }
        }

        // ── Systematic form via RREF of H ─────────────────────────────────────
        let (info_cols, parity_cols, parity_gen) =
            compute_systematic_form(&check_to_var, n, m);
        debug_assert_eq!(info_cols.len(),   k, "H rank ≠ m; expected k={k}");
        debug_assert_eq!(parity_cols.len(), m);

        Self {
            n, m, k,
            check_to_var, var_to_check, var_pos_in_check,
            info_cols, parity_cols, parity_gen,
        }
    }

    /// Returns a pre-built code for the given LDPC rate.
    ///
    /// Block lengths are small (N = 504) — suitable for testing and low-latency
    /// applications.  For better performance substitute optimised code matrices.
    ///
    /// Codes are generated with a deterministic quasi-cyclic construction
    /// (circulant size z = 21).
    pub fn for_rate(rate: LdpcRate) -> Self {
        match rate {
            LdpcRate::R1_2 => qc_ldpc(21, PROTO_R12),
            LdpcRate::R2_3 => qc_ldpc(21, PROTO_R23),
            LdpcRate::R3_4 => qc_ldpc(21, PROTO_R34),
            LdpcRate::R5_6 => qc_ldpc(21, PROTO_R56),
        }
    }

    /// Returns `true` if the bit vector `bits` is a valid codeword
    /// (i.e. H·bits = 0 over GF(2)).
    pub fn is_codeword(&self, bits: &[u8]) -> bool {
        self.check_to_var.iter().all(|nbrs| {
            nbrs.iter().map(|&v| bits[v]).fold(0u8, |a, b| a ^ b) == 0
        })
    }

    /// Systematic encoder: given `k` info bits, returns an `n`-bit codeword.
    ///
    /// Info bits are placed at the positions given by `self.info_cols`;
    /// parity bits are computed at `self.parity_cols` using the pre-computed
    /// generator matrix `parity_gen`:
    ///
    /// ```text
    /// codeword[parity_cols[i]] = XOR of info_bits[j]
    ///                            for j in 0..k where parity_gen[i][j] = 1
    /// ```
    ///
    /// The RX reconstructs the info bits by extracting `decoded_bits[info_cols]`.
    pub fn encode(&self, info_bits: &[u8]) -> Vec<u8> {
        assert_eq!(info_bits.len(), self.k);

        let mut codeword = vec![0u8; self.n];

        // Place info bits at their codeword positions
        for (j, &col) in self.info_cols.iter().enumerate() {
            codeword[col] = info_bits[j];
        }

        // Compute each parity bit: p_i = parity_gen[i] · info (mod 2)
        for (i, &pcol) in self.parity_cols.iter().enumerate() {
            let p: u8 = self.parity_gen[i].iter()
                .zip(info_bits.iter())
                .map(|(&g, &b)| g & b)
                .fold(0u8, |a, x| a ^ x);
            codeword[pcol] = p;
        }

        codeword
    }
}

// ── Systematic form computation ───────────────────────────────────────────────

/// Computes the systematic form of H via Gauss–Jordan elimination over GF(2).
///
/// Returns `(info_cols, parity_cols, parity_gen)`:
/// * `info_cols`  — non-pivot column indices (length k = n − rank).
/// * `parity_cols`— pivot column indices (length m = rank(H)).
/// * `parity_gen` — m × k binary matrix: `parity[i] = parity_gen[i] · info`.
fn compute_systematic_form(
    check_to_var: &[Vec<usize>],
    n: usize,
    m: usize,
) -> (Vec<usize>, Vec<usize>, Vec<Vec<u8>>) {
    // Build H as m rows of n bits (u8 per bit for simplicity at n=252)
    let mut h: Vec<Vec<u8>> = vec![vec![0u8; n]; m];
    for (c, vars) in check_to_var.iter().enumerate() {
        for &v in vars { h[c][v] = 1; }
    }

    // ── Gauss–Jordan elimination ──────────────────────────────────────────────
    let mut pivot_row = 0usize;
    let mut parity_cols: Vec<usize> = Vec::with_capacity(m);

    'col: for col in 0..n {
        // Find a row >= pivot_row with a 1 in this column
        for r in pivot_row..m {
            if h[r][col] == 1 {
                h.swap(pivot_row, r);
                parity_cols.push(col);

                // Eliminate this column from every other row (full RREF)
                for r2 in 0..m {
                    if r2 != pivot_row && h[r2][col] == 1 {
                        let prow = h[pivot_row].clone();
                        for c2 in 0..n { h[r2][c2] ^= prow[c2]; }
                    }
                }
                pivot_row += 1;
                if pivot_row == m { break 'col; }
                continue 'col;
            }
        }
    }

    // ── Info and parity column sets ───────────────────────────────────────────
    let parity_set: std::collections::HashSet<usize> =
        parity_cols.iter().copied().collect();
    let info_cols: Vec<usize> = (0..n).filter(|c| !parity_set.contains(c)).collect();

    // ── Parity generation matrix from RREF ───────────────────────────────────
    // After RREF, row i has pivot in parity_cols[i].  The remaining non-zero
    // entries in row i are exactly the info columns contributing to parity_cols[i].
    let parity_gen: Vec<Vec<u8>> = (0..pivot_row)
        .map(|i| info_cols.iter().map(|&c| h[i][c]).collect())
        .collect();

    (info_cols, parity_cols, parity_gen)
}

// ── LdpcDecoder ───────────────────────────────────────────────────────────────

/// Scaled min-sum LDPC decoder.
pub struct LdpcDecoder<'a> {
    code:     &'a LdpcCode,
    /// Maximum number of BP iterations.
    max_iter: usize,
    /// Min-sum scaling factor α ∈ (0, 1].  Typical value: 0.75.
    scaling:  f32,
}

/// Decoding result.
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// Hard-decision bits (length = `code.n`).
    pub bits:      Vec<u8>,
    /// `true` if the decoder found a codeword (all parity checks satisfied).
    pub converged: bool,
    /// Number of iterations executed.
    pub iterations: u32,
}

impl<'a> LdpcDecoder<'a> {
    /// Creates a decoder for `code`.
    ///
    /// * `max_iter` — iteration limit (typical: 50 … 100).
    /// * `scaling`  — min-sum correction factor (typical: 0.75).
    pub fn new(code: &'a LdpcCode, max_iter: usize, scaling: f32) -> Self {
        Self { code, max_iter, scaling }
    }

    /// Decodes `llr` (length = `code.n`, LLR values from the demapper).
    ///
    /// Returns decoded bits, convergence flag, and iteration count.
    pub fn decode(&self, llr: &[f32]) -> DecodeResult {
        assert_eq!(llr.len(), self.code.n);
        let code    = self.code;
        let n       = code.n;
        let m       = code.m;
        let scaling = self.scaling;

        // ── Message arrays ────────────────────────────────────────────────────
        // r[c][i]: check-to-variable message from check c to its i-th variable
        // q[c][i]: variable-to-check message from variable check_to_var[c][i] to c
        let mut r: Vec<Vec<f32>> = code.check_to_var.iter()
            .map(|nbrs| vec![0.0f32; nbrs.len()])
            .collect();
        let mut q: Vec<Vec<f32>> = code.check_to_var.iter()
            .map(|nbrs| nbrs.iter().map(|&v| llr[v]).collect())
            .collect();

        let mut bits   = vec![0u8; n];
        let mut converged = false;

        for iter in 0..self.max_iter {
            // ── Check-node update (min-sum) ───────────────────────────────────
            for c in 0..m {
                let q_c = &q[c];
                let deg = q_c.len();
                if deg == 0 { continue; }

                // Find min and second-min of |q|, and the product of signs
                let mut min1 = f32::INFINITY;
                let mut min2 = f32::INFINITY;
                let mut min1_idx  = 0usize;
                let mut sign_prod: i32 = 1;

                for (i, &qi) in q_c.iter().enumerate() {
                    let aq = qi.abs();
                    if aq < min1 { min2 = min1; min1 = aq; min1_idx = i; }
                    else if aq < min2 { min2 = aq; }
                    if qi < 0.0 { sign_prod = -sign_prod; }
                }

                for (i, &qi) in q_c.iter().enumerate() {
                    let min_excl = if i == min1_idx { min2 } else { min1 };
                    // sign_excl_i = sign_prod * sign(q_i)   [since sign²=1]
                    let sign = if (qi < 0.0) == (sign_prod < 0) { 1.0f32 } else { -1.0 };
                    r[c][i] = scaling * sign * min_excl;
                }
            }

            // ── Variable-node update + tentative decision ─────────────────────
            // Accumulate total incoming LLR for each variable, then subtract
            // the self-contribution when sending to each check.
            let total: Vec<f32> = (0..n).map(|v| {
                let sum_r: f32 = code.var_to_check[v].iter()
                    .zip(code.var_pos_in_check[v].iter())
                    .map(|(&c, &pos)| r[c][pos])
                    .sum();
                llr[v] + sum_r
            }).collect();

            for v in 0..n {
                for (j, (&c, &pos)) in code.var_to_check[v].iter()
                    .zip(code.var_pos_in_check[v].iter())
                    .enumerate()
                {
                    q[c][pos] = total[v] - r[c][pos];
                    let _ = j;
                }
                bits[v] = u8::from(total[v] < 0.0);
            }

            // ── Parity-check convergence test ─────────────────────────────────
            if code.is_codeword(&bits) {
                converged = true;
                return DecodeResult { bits, converged, iterations: (iter + 1) as u32 };
            }
        }

        DecodeResult { bits, converged, iterations: self.max_iter as u32 }
    }
}

// ── Quasi-cyclic code construction ───────────────────────────────────────────

/// Protograph matrix entry: (check_group, var_group, shift).
/// −1 as shift means the circulant sub-matrix is all-zero.
#[allow(dead_code)]
type ProtoEntry = (usize, usize, i32);

/// Builds an [`LdpcCode`] from a protograph + lift factor.
///
/// The protograph is given as a list of non-negative entries
/// `(check_group, var_group, circulant_shift)`.
/// The lift factor `z` is the circulant size.
/// The resulting code has `n = n_b * z` bits and `m = m_b * z` checks.
fn qc_ldpc(z: usize, proto: &[(usize, usize, i32)]) -> LdpcCode {
    // Infer dimensions
    let m_b = proto.iter().map(|&(c, _, _)| c).max().unwrap_or(0) + 1;
    let n_b = proto.iter().map(|&(_, v, _)| v).max().unwrap_or(0) + 1;
    let n   = n_b * z;
    let m   = m_b * z;
    let k   = n - m; // assuming full-rank H

    let mut ctv: Vec<Vec<usize>> = vec![Vec::new(); m];
    for &(cb, vb, shift) in proto {
        if shift < 0 { continue; }
        let s = shift as usize;
        for i in 0..z {
            let c_abs = cb * z + i;
            let v_abs = vb * z + (i + s) % z;
            ctv[c_abs].push(v_abs);
        }
    }
    LdpcCode::new(n, k, ctv)
}

// ── Protograph matrices for the 4 rates (circulant size z = 21) ───────────────
//
// Each entry: (check_group, var_group, circulant_shift).
// These are simple irregular protographs — not rate-optimised.
// Replace with DVB-S2 or custom designed matrices for production.

/// Rate 1/2 protograph: m_b=6, n_b=12, code (252, 126).
const PROTO_R12: &[(usize, usize, i32)] = &[
    (0,0,0),(0,1,3),(0,5,7),(0,6,0),
    (1,0,11),(1,2,1),(1,3,8),(1,7,0),
    (2,1,5),(2,2,14),(2,4,2),(2,8,0),
    (3,2,9),(3,3,6),(3,5,13),(3,9,0),
    (4,3,4),(4,4,10),(4,6,15),(4,10,0),
    (5,0,7),(5,4,12),(5,5,3),(5,11,0),
];

/// Rate 2/3 protograph: m_b=4, n_b=12, code (252, 168).
const PROTO_R23: &[(usize, usize, i32)] = &[
    (0,0,0),(0,1,5),(0,4,2),(0,7,9),(0,10,0),
    (1,1,3),(1,2,11),(1,5,6),(1,8,14),(1,11,0),
    (2,0,8),(2,3,1),(2,6,13),(2,9,4),(2,10,0),(2,11,7),
    (3,2,6),(3,4,15),(3,7,0),(3,8,11),(3,9,3),
];

/// Rate 3/4 protograph: m_b=3, n_b=12, code (252, 189).
const PROTO_R34: &[(usize, usize, i32)] = &[
    (0,0,0),(0,2,7),(0,4,1),(0,6,14),(0,8,5),(0,10,12),
    (1,1,3),(1,3,10),(1,5,8),(1,7,2),(1,9,15),(1,11,0),
    (2,0,11),(2,1,6),(2,3,4),(2,5,13),(2,7,9),(2,9,0),(2,11,2),
];

/// Rate 5/6 protograph: m_b=2, n_b=12, code (252, 210).
///
/// The parity columns (vg10, vg11) are lower-triangular to guarantee a
/// full-rank H.  The info side (vg0..9) uses distinct shifts in each check
/// group to avoid GF(2) cancellations that would reduce the matrix rank.
const PROTO_R56: &[(usize, usize, i32)] = &[
    // info connections (var_groups 0..9)
    (0,0,0),(0,1,4),(0,2,9),(0,3,14),(0,4,2),(0,5,7),(0,6,12),(0,7,1),(0,8,6),(0,9,11),
    (1,0,5),(1,1,10),(1,2,15),(1,3,3),(1,4,8),(1,5,13),(1,6,1),(1,7,6),(1,8,11),(1,9,2),
    // parity connections — lower triangular (vg10..11)
    (0,10,0),
    (1,10,7),(1,11,0),
];

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::rx::mode_detect::LdpcRate;

    /// All-zeros is always a valid codeword; the decoder on noiseless input
    /// (all LLR = +large) must return all-zeros in one iteration.
    #[test]
    fn allzero_codeword_every_rate() {
        for &rate in &[LdpcRate::R1_2, LdpcRate::R2_3, LdpcRate::R3_4, LdpcRate::R5_6] {
            let code = LdpcCode::for_rate(rate);
            assert!(code.n > 0,    "{rate:?}: n=0");
            assert!(code.k > 0,    "{rate:?}: k=0");
            assert!(code.k < code.n, "{rate:?}: k >= n");
            assert!(code.is_codeword(&vec![0u8; code.n]),
                "{rate:?}: all-zeros not a codeword");

            let llr  = vec![10.0f32; code.n]; // confident 0s
            let dec  = LdpcDecoder::new(&code, 50, 0.75);
            let res  = dec.decode(&llr);
            assert!(res.converged, "{rate:?}: did not converge on noiseless input");
            assert!(res.bits.iter().all(|&b| b == 0),
                "{rate:?}: non-zero bit in all-zeros decode");
        }
    }

    /// Flip one bit (set one LLR very negative) and verify the decoder
    /// recovers it within 50 iterations for the rate-1/2 code.
    #[test]
    fn single_bit_error_correction() {
        let code = LdpcCode::for_rate(LdpcRate::R1_2);
        let n    = code.n;
        let dec  = LdpcDecoder::new(&code, 100, 0.75);

        // Flip variable-node 0 (set LLR strongly negative = bit-1 decision)
        let mut llr = vec![5.0f32; n];
        llr[0] = -5.0; // one error

        let res = dec.decode(&llr);
        // With a single error the decoder should converge on this rate-1/2 code
        assert!(res.converged, "single-error: no convergence (iter={})", res.iterations);
        assert_eq!(res.bits[0], 0, "bit 0 not corrected");
        assert!(res.bits.iter().all(|&b| b == 0), "spurious bit errors");
    }

    /// Erasures (LLR = 0) should not cause panics and the decoder should
    /// make a tentative decision (bit undetermined, but parity satisfied if possible).
    #[test]
    fn erasures_no_panic() {
        let code = LdpcCode::for_rate(LdpcRate::R1_2);
        let n    = code.n;
        // Mark the first 5 variable nodes as erasures (LLR = 0)
        let mut llr = vec![5.0f32; n];
        for i in 0..5 { llr[i] = 0.0; }
        let dec = LdpcDecoder::new(&code, 50, 0.75);
        let res = dec.decode(&llr); // must not panic
        assert_eq!(res.bits.len(), n);
    }

    /// Code dimensions must satisfy n = n_b * z, m = m_b * z, k = n - m.
    #[test]
    fn code_dimensions() {
        let z = 21usize;
        let expected = [
            (LdpcRate::R1_2, 12*z, 6*z),
            (LdpcRate::R2_3, 12*z, 4*z),
            (LdpcRate::R3_4, 12*z, 3*z),
            (LdpcRate::R5_6, 12*z, 2*z),
        ];
        for (rate, exp_n, exp_m) in expected {
            let code = LdpcCode::for_rate(rate);
            assert_eq!(code.n, exp_n, "{rate:?} n");
            assert_eq!(code.m, exp_m, "{rate:?} m");
            assert_eq!(code.k, exp_n - exp_m, "{rate:?} k");
        }
    }

    /// The parity check matrix is consistent: var_to_check is the transpose
    /// of check_to_var.
    #[test]
    fn adjacency_consistency() {
        let code = LdpcCode::for_rate(LdpcRate::R1_2);
        for (c, vars) in code.check_to_var.iter().enumerate() {
            for &v in vars {
                assert!(
                    code.var_to_check[v].contains(&c),
                    "var_to_check[{v}] missing check {c}"
                );
            }
        }
        for (v, checks) in code.var_to_check.iter().enumerate() {
            for &c in checks {
                assert!(
                    code.check_to_var[c].contains(&v),
                    "check_to_var[{c}] missing var {v}"
                );
            }
        }
    }
}
