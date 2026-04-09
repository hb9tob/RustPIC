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
    /// Codes use circulant size z = 210 (10× the prototype) yielding
    /// n = 2520 for all rates.  The same quasi-cyclic protograph matrices as
    /// the z=21 prototype are reused (shifts wrap mod z).
    pub fn for_rate(rate: LdpcRate) -> Self {
        match rate {
            LdpcRate::R1_2 => qc_ldpc(81, WIFI_R12),
            LdpcRate::R2_3 => qc_ldpc(81, WIFI_R23),
            LdpcRate::R3_4 => qc_ldpc(81, WIFI_R34),
            LdpcRate::R5_6 => qc_ldpc(81, WIFI_R56),
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
///
/// H is stored as packed u64 rows (ceil(n/64) words per row) for fast XOR
/// operations — critical for n = 2520, m = 1260 (z = 210 codes).
fn compute_systematic_form(
    check_to_var: &[Vec<usize>],
    n: usize,
    m: usize,
) -> (Vec<usize>, Vec<usize>, Vec<Vec<u8>>) {
    let words = n.div_ceil(64);

    // Build H as m rows of `words` packed u64 values.
    let mut h: Vec<Vec<u64>> = vec![vec![0u64; words]; m];
    for (c, vars) in check_to_var.iter().enumerate() {
        for &v in vars {
            h[c][v / 64] |= 1u64 << (v % 64);
        }
    }

    // ── Gauss–Jordan elimination (packed rows) ────────────────────────────────
    let mut pivot_row  = 0usize;
    let mut parity_cols: Vec<usize> = Vec::with_capacity(m);
    let mut prow_buf = vec![0u64; words]; // scratch — avoids per-pivot allocation

    'col: for col in 0..n {
        let word = col / 64;
        let mask = 1u64 << (col % 64);

        // Find a row with a 1 in column `col`
        for r in pivot_row..m {
            if h[r][word] & mask != 0 {
                h.swap(pivot_row, r);
                parity_cols.push(col);

                // Copy pivot row into scratch buffer
                prow_buf.copy_from_slice(&h[pivot_row]);

                // Eliminate column from all other rows (full RREF)
                for r2 in 0..m {
                    if r2 != pivot_row && h[r2][word] & mask != 0 {
                        for w in 0..words { h[r2][w] ^= prow_buf[w]; }
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

    // ── Parity generation matrix ──────────────────────────────────────────────
    // After RREF, row i has pivot in parity_cols[i]; non-zero info entries give
    // the parity formula.
    let parity_gen: Vec<Vec<u8>> = (0..pivot_row)
        .map(|i| {
            info_cols.iter().map(|&c| {
                u8::from(h[i][c / 64] & (1u64 << (c % 64)) != 0)
            }).collect()
        })
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

// ── IEEE 802.11n LDPC base matrices (z = 81, n = 1944) ───────────────────────
//
// Source: IEEE 802.11n-2009 Annex R, Tables R.5/R.10/R.15/R.20.
// Verified against two independent open-source implementations:
//   simgunz/802.11n-ldpc  and  tavildar/LDPC (WiFiLDPC.h).
//
// Entry format: (check_group, var_group, circulant_shift).
// Shift = -1 means zero sub-matrix (omitted here).
// The circulant size is z = 81; n_b = 24 variable groups for all rates.
// The last (n_b - k_b) variable groups form the dual-diagonal parity section.

/// Rate 1/2 — H_B 12×24, n=1944, k=972.  (IEEE 802.11n Table R.20, z=81)
const WIFI_R12: &[(usize, usize, i32)] = &[
    // row 0
    (0, 0,57),(0, 4,50),(0, 6,11),(0, 8,50),(0,10,79),(0,12, 1),(0,13, 0),
    // row 1
    (1, 0, 3),(1, 2,28),(1, 4, 0),(1, 8,55),(1, 9, 7),(1,13, 0),(1,14, 0),
    // row 2
    (2, 0,30),(2, 4,24),(2, 5,37),(2, 8,56),(2, 9,14),(2,14, 0),(2,15, 0),
    // row 3
    (3, 0,62),(3, 1,53),(3, 4,53),(3, 7, 3),(3, 8,35),(3,15, 0),(3,16, 0),
    // row 4
    (4, 0,40),(4, 3,20),(4, 4,66),(4, 7,22),(4, 8,28),(4,16, 0),(4,17, 0),
    // row 5
    (5, 0, 0),(5, 4, 8),(5, 6,42),(5, 8,50),(5,11, 8),(5,17, 0),(5,18, 0),
    // row 6
    (6, 0,69),(6, 1,79),(6, 2,79),(6, 6,56),(6, 8,52),(6,12, 0),(6,18, 0),(6,19, 0),
    // row 7
    (7, 0,65),(7, 4,38),(7, 5,57),(7, 8,72),(7,10,27),(7,19, 0),(7,20, 0),
    // row 8
    (8, 0,64),(8, 4,14),(8, 5,52),(8, 8,30),(8,11,32),(8,20, 0),(8,21, 0),
    // row 9
    (9, 1,45),(9, 3,70),(9, 4, 0),(9, 8,77),(9, 9, 9),(9,21, 0),(9,22, 0),
    // row 10
    (10, 0, 2),(10, 1,56),(10, 3,57),(10, 4,35),(10,10,12),(10,22, 0),(10,23, 0),
    // row 11
    (11, 0,24),(11, 2,61),(11, 4,60),(11, 7,27),(11, 8,51),(11,11,16),(11,12, 1),(11,23, 0),
];

/// Rate 2/3 — H_B 8×24, n=1944, k=1296.  (IEEE 802.11n Table R.15, z=81)
const WIFI_R23: &[(usize, usize, i32)] = &[
    // row 0
    (0, 0,61),(0, 1,75),(0, 2, 4),(0, 3,63),(0, 4,56),(0,11, 8),
    (0,13, 2),(0,14,17),(0,15,25),(0,16, 1),(0,17, 0),
    // row 1
    (1, 0,56),(1, 1,74),(1, 2,77),(1, 3,20),(1, 7,64),(1, 8,24),
    (1, 9, 4),(1,10,67),(1,12, 7),(1,17, 0),(1,18, 0),
    // row 2
    (2, 0,28),(2, 1,21),(2, 2,68),(2, 3,10),(2, 4, 7),(2, 5,14),
    (2, 6,65),(2,10,23),(2,14,75),(2,18, 0),(2,19, 0),
    // row 3
    (3, 0,48),(3, 1,38),(3, 2,43),(3, 3,78),(3, 4,76),(3, 9, 5),
    (3,10,36),(3,12,15),(3,13,72),(3,19, 0),(3,20, 0),
    // row 4
    (4, 0,40),(4, 1, 2),(4, 2,53),(4, 3,25),(4, 5,52),(4, 6,62),
    (4, 8,20),(4,11,44),(4,16, 0),(4,20, 0),(4,21, 0),
    // row 5
    (5, 0,69),(5, 1,23),(5, 2,64),(5, 3,10),(5, 4,22),(5, 6,21),
    (5,12,68),(5,13,23),(5,14,29),(5,21, 0),(5,22, 0),
    // row 6
    (6, 0,12),(6, 1, 0),(6, 2,68),(6, 3,20),(6, 4,55),(6, 5,61),
    (6, 7,40),(6,11,52),(6,15,44),(6,22, 0),(6,23, 0),
    // row 7
    (7, 0,58),(7, 1, 8),(7, 2,34),(7, 3,64),(7, 4,78),(7, 7,11),
    (7, 8,78),(7, 9,24),(7,15,58),(7,16, 1),(7,23, 0),
];

/// Rate 3/4 — H_B 6×24, n=1944, k=1458.  (IEEE 802.11n Table R.10, z=81)
const WIFI_R34: &[(usize, usize, i32)] = &[
    // row 0
    (0, 0,48),(0, 1,29),(0, 2,28),(0, 3,39),(0, 4, 9),(0, 5,61),
    (0, 9,63),(0,10,45),(0,11,80),(0,15,37),(0,16,32),(0,17,22),(0,18, 1),(0,19, 0),
    // row 1
    (1, 0, 4),(1, 1,49),(1, 2,42),(1, 3,48),(1, 4,11),(1, 5,30),
    (1, 9,49),(1,10,17),(1,11,41),(1,12,37),(1,13,15),(1,15,54),(1,19, 0),(1,20, 0),
    // row 2
    (2, 0,35),(2, 1,76),(2, 2,78),(2, 3,51),(2, 4,37),(2, 5,35),
    (2, 6,21),(2, 8,17),(2, 9,64),(2,13,59),(2,14, 7),(2,17,32),(2,20, 0),(2,21, 0),
    // row 3
    (3, 0, 9),(3, 1,65),(3, 2,44),(3, 3, 9),(3, 4,54),(3, 5,56),
    (3, 6,73),(3, 7,34),(3, 8,42),(3,12,35),(3,16,46),(3,17,39),(3,18, 0),(3,21, 0),(3,22, 0),
    // row 4
    (4, 0, 3),(4, 1,62),(4, 2, 7),(4, 3,80),(4, 4,68),(4, 5,26),
    (4, 7,80),(4, 8,55),(4,10,36),(4,12,26),(4,14, 9),(4,16,72),(4,22, 0),(4,23, 0),
    // row 5
    (5, 0,26),(5, 1,75),(5, 2,33),(5, 3,21),(5, 4,69),(5, 5,59),
    (5, 6, 3),(5, 7,38),(5,11,35),(5,13,62),(5,14,36),(5,15,26),(5,18, 1),(5,23, 0),
];

/// Rate 5/6 — H_B 4×24, n=1944, k=1620.  (IEEE 802.11n Table R.5, z=81)
const WIFI_R56: &[(usize, usize, i32)] = &[
    // row 0
    (0, 0,13),(0, 1,48),(0, 2,80),(0, 3,66),(0, 4, 4),(0, 5,74),
    (0, 6, 7),(0, 7,30),(0, 8,76),(0, 9,52),(0,10,37),(0,11,60),
    (0,13,49),(0,14,73),(0,15,31),(0,16,74),(0,17,73),(0,18,23),(0,20, 1),(0,21, 0),
    // row 1
    (1, 0,69),(1, 1,63),(1, 2,74),(1, 3,56),(1, 4,64),(1, 5,77),
    (1, 6,57),(1, 7,65),(1, 8, 6),(1, 9,16),(1,10,51),(1,12,64),
    (1,14,68),(1,15, 9),(1,16,48),(1,17,62),(1,18,54),(1,19,27),(1,21, 0),(1,22, 0),
    // row 2
    (2, 0,51),(2, 1,15),(2, 2, 0),(2, 3,80),(2, 4,24),(2, 5,25),
    (2, 6,42),(2, 7,54),(2, 8,44),(2, 9,71),(2,10,71),(2,11, 9),
    (2,12,67),(2,13,35),(2,15,58),(2,17,29),(2,19,53),(2,20, 0),(2,22, 0),(2,23, 0),
    // row 3
    (3, 0,16),(3, 1,29),(3, 2,36),(3, 3,41),(3, 4,44),(3, 5,56),
    (3, 6,59),(3, 7,37),(3, 8,50),(3, 9,24),(3,11,65),(3,12, 4),
    (3,13,65),(3,14,52),(3,16, 4),(3,18,73),(3,19,52),(3,20, 1),(3,23, 0),
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
        // 802.11n codes: z=81, n_b=24 variable groups for all rates.
        let z = 81usize;
        let expected = [
            (LdpcRate::R1_2, 24*z, 12*z),
            (LdpcRate::R2_3, 24*z,  8*z),
            (LdpcRate::R3_4, 24*z,  6*z),
            (LdpcRate::R5_6, 24*z,  4*z),
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
