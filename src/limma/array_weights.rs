//! limma's `arrayWeights` and `duplicateCorrelation`.
//!
//! Two estimators `voomLmFit` reaches for once an experiment stops being a
//! collection of independent, equally good samples. [`array_weights`] gives one
//! precision weight per sample, so a bad array is down-weighted everywhere
//! rather than dropped. [`duplicate_correlation`] gives the consensus
//! intra-block correlation that turns a repeated-measures design into a
//! generalised least squares fit.
//!
//! ### The model behind the weights
//!
//! Sample `j` is assumed to have variance `sigma^2 / w_j` with
//! `log(1 / w_j) = z_j' gamma`, where `z_j` is a row of the variance design
//! (`contr.sum(n_samples)` by default, so the log weights sum to zero). Both
//! methods solve the same REML score equations for `gamma`, and differ only in
//! how they sweep the genes:
//!
//! * [`ArrayWeightMethod::GeneByGene`] takes one Fisher scoring step per gene,
//!   updating the weights as it goes. One pass, no convergence test, and it
//!   copes with missing values and gene-level weights.
//! * [`ArrayWeightMethod::Reml`] accumulates the score and information over all
//!   genes and iterates the whole sweep to convergence. Sharper, but it wants a
//!   complete matrix.
//!
//! `prior_n` is a ridge on the information: it is the number of notional genes
//! that saw every weight equal to one, and it stops a small experiment from
//! chasing noise into extreme weights.
//!
//! ### The correlation
//!
//! [`duplicate_correlation`] fits a two-component mixed model per gene through
//! `statmod::mixedModel2Fit` and takes the trimmed mean of the Fisher
//! z-transformed correlations. The per-gene fit is not a moment estimator: it
//! projects the residual space out of the fixed effects, takes an SVD of the
//! projected block design, and fits a gamma GLM of the squared rotated
//! residuals against the squared singular values. That gamma GLM is the REML
//! likelihood for the two variance components, and its Levenberg damping is
//! reproduced here step for step, because the answer depends on where it stops.
//!
//! edgePython replaces all of that with a one-way ANOVA moment estimator. See
//! the note on [`duplicate_correlation`] and `UPSTREAM_DEVIATIONS.md`.
//!
//! ### No scalar optimiser
//!
//! Nothing here goes through [`crate::numeric::optimise`], and that is not an
//! oversight. limma reaches for `optimize` or `uniroot` in the empirical Bayes
//! fits, but neither of these two routines does: `arrayWeights` is Fisher
//! scoring on the REML score equations with a closed-form step, and
//! `duplicateCorrelation` bottoms out in `statmod::glmgam.fit`, a Levenberg-
//! damped Fisher scoring of its own. Substituting a general optimiser for either
//! would land somewhere else, because both stop on their own iteration rule
//! rather than at a minimum an optimiser would find.
//!
//! The dense linear algebra is likewise local: a LINPACK `dqrdc2` Householder QR
//! and a partial-pivot Gaussian elimination, both written out here so the rank
//! rule and the pivot order are R's rather than faer's. faer is used for exactly
//! one thing, the SVD inside the mixed model.
//!
//! ### Numeric policy
//!
//! `f64` only, and deliberately not generic. Everything here is a difference of
//! sums of squares fed into a log, on matrices whose size is the sample count
//! rather than the gene count, so there is no memory to save and real precision
//! to lose.
//!
//! ### References
//!
//! Ritchie, Diyagama, Neilson, van Laar, Dobrovic, Holloway and Smyth, BMC
//! Bioinformatics, 2006 (array weights)
//!
//! Smyth, Michaud and Scott, Bioinformatics, 2005 (duplicate correlation)

use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::numeric::stats::trimmed_mean;
use crate::utils::recycled::Recycled;

///////////////
// Constants //
///////////////

/// Relative tolerance below which a column counts as linearly dependent.
///
/// R's `lm.fit` and `qr` both default to `tol = 1e-7`, and both pass it to
/// LINPACK's `dqrdc2`, which drops a column once the norm of its part orthogonal
/// to the columns already accepted falls below `tol` times its original norm.
/// The rank this produces is what `arrayWeights` reduces the design to, so it
/// has to be R's rule rather than the SVD rule [`crate::utils::design::matrix_rank`]
/// uses.
const LM_RANK_TOL: f64 = 1e-7;

/// Residual variances below this are treated as structurally zero.
///
/// limma skips such genes rather than dividing by them. A gene whose weighted
/// residual sum of squares is this small is either constant or perfectly fitted,
/// and carries no information about the sample variances either way.
const MIN_RESIDUAL_VAR: f64 = 1e-15;

/// Residual degrees of freedom a gene needs before it can contribute.
///
/// One residual degree of freedom gives a variance estimate with no leverage on
/// how that variance splits across samples, so limma requires two.
const MIN_RESIDUAL_DF: usize = 2;

/// Largest intra-block correlation `duplicateCorrelation` will report.
///
/// The Fisher z-transform is applied after clipping, and `atanh(1)` is infinite,
/// so the clip is what keeps the trimmed mean finite.
const RHO_MAX: f64 = 0.99;

/// Slack added to the theoretical lower bound on the correlation.
///
/// A block of size `m` cannot have an intra-block correlation below
/// `1 / (1 - m)`, since the block covariance matrix would stop being positive
/// definite. limma clips to that bound plus this, again to keep `atanh` finite.
const RHO_MIN_SLACK: f64 = 0.01;

/// Below this, the block factor counts as already spanned by the design.
///
/// limma projects the block indicators onto the residual space of the design and
/// gives up if nothing survives, because there would be no within-block
/// replication left to estimate a correlation from.
const BLOCK_ABSORBED_TOL: f64 = 1e-8;

/// Squared singular values above this count as non-zero.
///
/// `mixedModel2Fit` only bothers with the gamma GLM when the projected block
/// design has at least two of them; with fewer, the linear fit it starts from is
/// already the answer.
const SINGULAR_VALUE_TOL: f64 = 1e-15;

/// Convergence tolerance of `statmod::glmgam.fit` on the score-step product.
const GLMGAM_TOL: f64 = 1e-6;

/// Iteration budget `mixedModel2Fit` gives `glmgam.fit`.
///
/// limma calls `mixedModel2Fit(..., maxit = 20)`. `glmgam.fit` tests
/// `iter > maxit` at the bottom of the loop, so this permits 21 scoring steps.
const GLMGAM_MAX_ITER: usize = 20;

/// Ratio by which the gamma variance is floored inside `glmgam.fit`.
///
/// `v <- pmax(mu^2, max(mu^2) / 1e3)`, which stops a near-zero fitted value from
/// taking over the weighting.
const GLMGAM_VARIANCE_FLOOR: f64 = 1e3;

/// Damping ratio at which `glmgam.fit` declares the step hopeless and stops.
const GLMGAM_MAX_DAMPING: f64 = 1e15;

/// Factor by which Levenberg damping is relaxed after a step that worked.
const GLMGAM_DAMPING_RELAX: f64 = 10.0;

/// Relative size below which a deviance counts as converged regardless.
const GLMGAM_DEVIANCE_TOL: f64 = 1e-15;

/// Below this, an observation and its fitted value both count as zero.
///
/// `statmod`'s gamma deviance drops such terms rather than evaluating
/// `log(0 / 0)`. Numerically the same threshold as [`MIN_RESIDUAL_VAR`], but a
/// different quantity, so it gets its own name.
const GLMGAM_ZERO_TOL: f64 = 1e-15;

///////////////////
// Householder QR //
///////////////////

/// A LINPACK `dqrdc2` Householder QR: the factorisation R's `lm.fit` uses.
///
/// This is not faer's QR, and the difference is deliberate. `dqrdc2` does not
/// pivot for stability; it walks the columns left to right and moves a column
/// aside only when the part of it orthogonal to what came before has lost a
/// factor [`LM_RANK_TOL`] of its length. The surviving columns therefore keep
/// their original order, which is what makes `QR$pivot[1:rank]` in
/// `arrayWeights` a stable, order-preserving column subset rather than a
/// magnitude-sorted one.
///
/// Columns rejected here stay physically in place rather than being cycled to
/// the end as LINPACK does. Nothing downstream reads them, and leaving them
/// alone keeps [`LinpackQr::pivot`] ascending.
struct LinpackQr {
    /// Column-major `n * p` working store. For each accepted step `l` sitting in
    /// original column `j`, row `l` holds `R[l, l]` and rows `l + 1..n` hold the
    /// tail of that step's Householder vector.
    qr: Vec<f64>,
    /// Leading element of each accepted step's Householder vector, LINPACK's
    /// `qraux`. Kept out of `qr` because `R`'s diagonal needs that slot.
    qraux: Vec<f64>,
    /// Original index of each accepted column, ascending. Its length is the rank.
    pivot: Vec<usize>,
    /// Number of rows.
    n: usize,
}

impl LinpackQr {
    /// Factorises a column-major matrix.
    ///
    /// ### Params
    ///
    /// * `a` - Column-major `n * p` values, consumed as the working store
    /// * `n` - Number of rows
    /// * `p` - Number of columns
    ///
    /// ### Returns
    ///
    /// The factorisation. A zero column is always rejected, since its original
    /// norm is replaced by one before the comparison, exactly as `dqrdc2` does.
    fn new(mut a: Vec<f64>, n: usize, p: usize) -> Self {
        let original: Vec<f64> = (0..p)
            .map(|j| {
                let norm = euclidean_norm(&a[j * n..(j + 1) * n]);
                if norm == 0.0 { 1.0 } else { norm }
            })
            .collect();

        let mut qraux = Vec::with_capacity(p.min(n));
        let mut pivot = Vec::with_capacity(p.min(n));

        for j in 0..p {
            let l = pivot.len();
            if l >= n {
                break;
            }
            let remaining = euclidean_norm(&a[j * n + l..(j + 1) * n]);
            if remaining < original[j] * LM_RANK_TOL {
                continue;
            }

            // LINPACK's reflector: normalise by the signed norm, then bump the
            // leading element by one so that `v[0]` doubles as the scale.
            let mut nrmxl = remaining;
            if a[j * n + l] != 0.0 {
                nrmxl = nrmxl.copysign(a[j * n + l]);
            }
            for i in l..n {
                a[j * n + i] /= nrmxl;
            }
            a[j * n + l] += 1.0;
            let v0 = a[j * n + l];

            for c in (j + 1)..p {
                let mut t = 0.0;
                for i in l..n {
                    t += a[j * n + i] * a[c * n + i];
                }
                let t = -t / v0;
                for i in l..n {
                    a[c * n + i] += t * a[j * n + i];
                }
            }

            qraux.push(v0);
            a[j * n + l] = -nrmxl;
            pivot.push(j);
        }

        LinpackQr {
            qr: a,
            qraux,
            pivot,
            n,
        }
    }

    /// Numerical rank, in R's sense.
    ///
    /// ### Returns
    ///
    /// The number of columns that survived the [`LM_RANK_TOL`] test.
    fn rank(&self) -> usize {
        self.pivot.len()
    }

    /// Applies `Q'` in place, giving R's `effects`.
    ///
    /// ### Params
    ///
    /// * `y` - Vector of length `n`, overwritten with `Q' y`
    fn qty(&self, y: &mut [f64]) {
        for l in 0..self.rank() {
            self.reflect(l, y);
        }
    }

    /// Applies `Q` in place.
    ///
    /// Same reflectors as [`LinpackQr::qty`], applied in the opposite order:
    /// `Q = H_0 ... H_{r-1}` and `Q' = H_{r-1} ... H_0`, each reflector being its
    /// own inverse.
    ///
    /// ### Params
    ///
    /// * `y` - Vector of length `n`, overwritten with `Q y`
    fn qy(&self, y: &mut [f64]) {
        for l in (0..self.rank()).rev() {
            self.reflect(l, y);
        }
    }

    /// Applies one Householder reflector in place.
    ///
    /// ### Params
    ///
    /// * `l` - Index of the accepted step whose reflector to apply
    /// * `y` - Vector of length `n`, overwritten
    fn reflect(&self, l: usize, y: &mut [f64]) {
        let v = &self.qr[self.pivot[l] * self.n..(self.pivot[l] + 1) * self.n];
        let mut t = self.qraux[l] * y[l];
        for (vi, yi) in v[l + 1..].iter().zip(y[l + 1..].iter()) {
            t += vi * yi;
        }
        let t = -t / self.qraux[l];
        y[l] += t * self.qraux[l];
        for (vi, yi) in v[l + 1..].iter().zip(y[l + 1..].iter_mut()) {
            *yi += t * vi;
        }
    }

    /// The leading columns of `Q`, R's `qr.qy(qr, diag(1, n, k))`.
    ///
    /// ### Params
    ///
    /// * `k` - Number of columns wanted
    ///
    /// ### Returns
    ///
    /// Column-major `n * k`.
    fn leading_q(&self, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.n * k];
        for c in 0..k {
            out[c * self.n + c] = 1.0;
            let (_, column) = out.split_at_mut(c * self.n);
            self.qy(&mut column[..self.n]);
        }
        out
    }

    /// Leverages of the rows, R's `hat(qr)`.
    ///
    /// ### Returns
    ///
    /// The hat matrix diagonal, one value per row.
    fn hat(&self) -> Vec<f64> {
        let rank = self.rank();
        let q = self.leading_q(rank);
        (0..self.n)
            .map(|i| (0..rank).map(|c| q[c * self.n + i].powi(2)).sum())
            .collect()
    }

    /// Back-substitutes `R b = effects` over the accepted columns.
    ///
    /// ### Params
    ///
    /// * `effects` - `Q' y`, of which only the leading `rank` entries are read
    ///
    /// ### Returns
    ///
    /// The coefficients of the accepted columns, in [`LinpackQr::pivot`] order.
    fn solve_r(&self, effects: &[f64]) -> Vec<f64> {
        let rank = self.rank();
        let mut b = vec![0.0; rank];
        for l in (0..rank).rev() {
            let mut acc = effects[l];
            for (&column, coefficient) in self.pivot[l + 1..rank].iter().zip(b[l + 1..rank].iter())
            {
                acc -= self.qr[column * self.n + l] * coefficient;
            }
            b[l] = acc / self.qr[self.pivot[l] * self.n + l];
        }
        b
    }
}

/////////////////////
// Small dense help //
/////////////////////

/// Euclidean norm of a slice.
///
/// ### Params
///
/// * `x` - Values
///
/// ### Returns
///
/// `sqrt(sum(x^2))`.
fn euclidean_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Solves a small dense system by Gaussian elimination with partial pivoting.
///
/// This is LAPACK's `dgesv` and therefore R's `solve`. The systems here are
/// `ngam` by `ngam` with `ngam` one less than the sample count, so tens of rows
/// at most; a factorisation library would cost more in ceremony than it saves.
///
/// ### Params
///
/// * `a` - Column-major `n * n` coefficient matrix, copied before use
/// * `b` - Right hand side of length `n`
/// * `n` - Order of the system
///
/// ### Returns
///
/// The solution, or [`EdgeErrors::SolveFailed`] if a pivot vanishes or the
/// result is not finite.
fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, EdgeErrors> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();

    for k in 0..n {
        let (pivot_row, pivot) = (k..n).fold((k, 0.0), |(best, mag), i| {
            let v = m[k * n + i].abs();
            if v > mag { (i, v) } else { (best, mag) }
        });
        if pivot == 0.0 || !pivot.is_finite() {
            return Err(EdgeErrors::SolveFailed(format!(
                "the array weight information matrix is singular at pivot {k}"
            )));
        }
        if pivot_row != k {
            for c in 0..n {
                m.swap(c * n + k, c * n + pivot_row);
            }
            x.swap(k, pivot_row);
        }
        for i in (k + 1)..n {
            let factor = m[k * n + i] / m[k * n + k];
            if factor == 0.0 {
                continue;
            }
            for c in (k + 1)..n {
                m[c * n + i] -= factor * m[c * n + k];
            }
            x[i] -= factor * x[k];
        }
    }

    for k in (0..n).rev() {
        let mut acc = x[k];
        for c in (k + 1)..n {
            acc -= m[c * n + k] * x[c];
        }
        x[k] = acc / m[k * n + k];
    }

    if x.iter().any(|v| !v.is_finite()) {
        return Err(EdgeErrors::SolveFailed(
            "the array weight update was not finite".to_string(),
        ));
    }
    Ok(x)
}

/// Transposes a row-major matrix into column-major order.
///
/// ### Params
///
/// * `x` - Row-major `n_rows * n_cols` values
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
///
/// ### Returns
///
/// The same matrix, column-major.
fn to_column_major(x: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_rows * n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            out[j * n_rows + i] = x[i * n_cols + j];
        }
    }
    out
}

/// R's `contr.sum(n)`.
///
/// ### Params
///
/// * `n` - Number of levels
///
/// ### Returns
///
/// Column-major `n * (n - 1)`: an identity block over the first `n - 1` rows and
/// a row of `-1` underneath. Constraining the log weights against this basis is
/// what makes them sum to zero.
fn contr_sum(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * (n - 1)];
    for k in 0..(n - 1) {
        out[k * n + k] = 1.0;
        out[k * n + (n - 1)] = -1.0;
    }
    out
}

//////////////////
// Public types //
//////////////////

/// How [`array_weights`] sweeps the genes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayWeightMethod {
    /// One Fisher scoring step per gene, weights updated as the sweep proceeds.
    ///
    /// limma's `.arrayWeightsGeneByGene`. The only method that accepts missing
    /// values or gene-level weights without further ado, and the one limma's
    /// `method = "auto"` selects whenever either is present.
    GeneByGene,
    /// Score and information pooled over all genes, iterated to convergence.
    ///
    /// limma's `.arrayWeightsREML`, or `.arrayWeightsPrWtsREML` when gene-level
    /// weights are supplied. `method = "auto"` selects this for a complete,
    /// unweighted matrix, which is the common case.
    Reml,
}

/// Knobs for [`array_weights`].
#[derive(Clone, Debug)]
pub struct ArrayWeightParams {
    /// Which sweep to run. See [`ArrayWeightMethod`].
    pub method: ArrayWeightMethod,
    /// Prior number of genes pulling the weights towards one.
    ///
    /// Enters as a ridge `prior_n * Z2' Z2` on the information and, for the REML
    /// sweep, a matching `prior_n * (w - 1)` on the score. limma's default is 10.
    pub prior_n: f64,
    /// Iteration budget for the REML sweep. Ignored by the gene-by-gene sweep,
    /// which is a single pass by construction.
    pub max_iter: usize,
    /// Convergence tolerance for the REML sweep, on the score-step product
    /// scaled by the number of variance coefficients and the effective gene
    /// count. limma's `arrayWeights` default is `1e-5`.
    pub tol: f64,
    /// Variance design, row-major `n_samples * n_var_coef`.
    ///
    /// `None` uses `contr.sum(n_samples)`, one free weight per sample under a
    /// sum-to-zero constraint on the logs. Supplying a coarser basis, such as
    /// group indicators, pools samples into shared weights.
    pub var_design: Option<Vec<f64>>,
    /// Number of columns in `var_design`. Ignored when `var_design` is `None`.
    pub n_var_coef: usize,
}

impl Default for ArrayWeightParams {
    /// limma's defaults, with the method fixed at [`ArrayWeightMethod::Reml`].
    ///
    /// limma's own default is `method = "auto"`, which resolves to `"reml"` for
    /// a complete matrix with no gene-level weights and `"genebygene"`
    /// otherwise. There is no `Auto` variant here, so the default matches the
    /// branch a bare `arrayWeights(y, design)` call takes; pass
    /// [`ArrayWeightMethod::GeneByGene`] alongside gene-level weights or missing
    /// values to reproduce the other branch.
    fn default() -> Self {
        ArrayWeightParams {
            method: ArrayWeightMethod::Reml,
            prior_n: 10.0,
            max_iter: 50,
            tol: 1e-5,
            var_design: None,
            n_var_coef: 0,
        }
    }
}

impl ArrayWeightParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `method` - Which sweep to run
    /// * `prior_n` - Prior gene count shrinking the weights towards one, limma's 10
    /// * `max_iter` - REML iteration budget, limma's 50
    /// * `tol` - REML convergence tolerance, limma's 1e-5
    /// * `var_design` - Optional variance design, row-major `n_samples * n_var_coef`
    /// * `n_var_coef` - Number of columns in `var_design`
    ///
    /// ### Returns
    ///
    /// The parameters. Validation happens in [`array_weights`], which knows the
    /// sample count these have to agree with.
    pub fn new(
        method: ArrayWeightMethod,
        prior_n: f64,
        max_iter: usize,
        tol: f64,
        var_design: Option<Vec<f64>>,
        n_var_coef: usize,
    ) -> Self {
        ArrayWeightParams {
            method,
            prior_n,
            max_iter,
            tol,
            var_design,
            n_var_coef,
        }
    }
}

////////////////////
// Shared plumbing //
////////////////////

/// Checks the shape of an expression matrix and its design.
///
/// ### Params
///
/// * `y` - Row-major `n_genes * n_samples` expression values
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major `n_samples * n_coef` design
/// * `n_coef` - Number of design columns
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::EmptyCounts`], [`EdgeErrors::MustBePositive`] or
/// [`EdgeErrors::LengthMismatch`].
fn validate_shapes(
    y: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
) -> Result<(), EdgeErrors> {
    if n_genes == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts { n_genes, n_samples });
    }
    if n_coef == 0 {
        return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
    }
    if y.len() != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "y",
            expected: n_genes * n_samples,
            got: y.len(),
        });
    }
    if design.len() != n_samples * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: design.len(),
        });
    }
    Ok(())
}

/// Drops the linearly dependent columns of a design.
///
/// limma runs `qr(design)` and keeps `design[, QR$pivot[1:QR$rank]]`, so a rank
/// deficient design silently loses columns rather than erroring.
///
/// ### Params
///
/// * `design` - Row-major `n_samples * n_coef`
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of columns
///
/// ### Returns
///
/// The retained columns as a column-major `n_samples * rank` matrix, and the
/// rank.
fn reduce_design(design: &[f64], n_samples: usize, n_coef: usize) -> (Vec<f64>, usize) {
    let column_major = to_column_major(design, n_samples, n_coef);
    let qr = LinpackQr::new(column_major.clone(), n_samples, n_coef);
    let rank = qr.rank();
    if rank == n_coef {
        return (column_major, n_coef);
    }
    let mut out = Vec::with_capacity(n_samples * rank);
    for &j in &qr.pivot {
        out.extend_from_slice(&column_major[j * n_samples..(j + 1) * n_samples]);
    }
    (out, rank)
}

/// Builds the variance design `Z2`.
///
/// With no user basis this is `contr.sum(n_samples)`. With one, limma centres
/// every column and then drops the linearly dependent ones, which is how a
/// user-supplied intercept column disappears: centring turns it into zeros, and
/// a zero column always fails the [`LM_RANK_TOL`] test.
///
/// ### Params
///
/// * `n_samples` - Number of samples
/// * `var_design` - Optional row-major `n_samples * n_var_coef` basis
/// * `n_var_coef` - Number of columns in `var_design`
///
/// ### Returns
///
/// Column-major `n_samples * ngam` and `ngam`, or [`EdgeErrors`] on a shape
/// mismatch.
fn prepare_var_design(
    n_samples: usize,
    var_design: Option<&[f64]>,
    n_var_coef: usize,
) -> Result<(Vec<f64>, usize), EdgeErrors> {
    let Some(vd) = var_design else {
        return Ok((contr_sum(n_samples), n_samples - 1));
    };
    if n_var_coef == 0 {
        return Err(EdgeErrors::MustBePositive("n_var_coef".to_string()));
    }
    if vd.len() != n_samples * n_var_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "var_design",
            expected: n_samples * n_var_coef,
            got: vd.len(),
        });
    }

    let mut z2 = to_column_major(vd, n_samples, n_var_coef);
    for j in 0..n_var_coef {
        let column = &mut z2[j * n_samples..(j + 1) * n_samples];
        let mean = column.iter().sum::<f64>() / n_samples as f64;
        for v in column.iter_mut() {
            *v -= mean;
        }
    }

    let qr = LinpackQr::new(z2.clone(), n_samples, n_var_coef);
    let ngam = qr.rank();
    if ngam == n_var_coef {
        return Ok((z2, ngam));
    }
    let mut out = Vec::with_capacity(n_samples * ngam);
    for &j in &qr.pivot {
        out.extend_from_slice(&z2[j * n_samples..(j + 1) * n_samples]);
    }
    Ok((out, ngam))
}

/// One weighted least squares fit, R's `lm.wfit`.
struct WeightedFit {
    /// Unweighted residuals `y - X b`, one per observation.
    residuals: Vec<f64>,
    /// Leverages of the weighted design, R's `hat(fit$qr)`.
    hat: Vec<f64>,
    /// Numerical rank of the weighted design.
    rank: usize,
    /// Weighted residual sum of squares over `nobs - rank`, R's
    /// `mean(fit$effects[-(1:rank)]^2)`.
    s2: f64,
}

/// Fits `y ~ X` by weighted least squares.
///
/// ### Params
///
/// * `x` - Column-major `n * p` design
/// * `y` - Response of length `n`
/// * `w` - Weights of length `n`, all non-negative
/// * `n` - Number of observations
/// * `p` - Number of columns
///
/// ### Returns
///
/// The residuals, leverages, rank and residual variance. `s2` is `NaN` when the
/// fit has no residual degrees of freedom, matching R's `mean(numeric(0))`.
fn weighted_fit(x: &[f64], y: &[f64], w: &[f64], n: usize, p: usize) -> WeightedFit {
    let root_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let mut xw = vec![0.0; n * p];
    for j in 0..p {
        for i in 0..n {
            xw[j * n + i] = root_w[i] * x[j * n + i];
        }
    }
    let qr = LinpackQr::new(xw, n, p);
    let rank = qr.rank();

    let mut effects: Vec<f64> = (0..n).map(|i| root_w[i] * y[i]).collect();
    qr.qty(&mut effects);
    let coef = qr.solve_r(&effects);

    let mut residuals = y.to_vec();
    for (m, &j) in qr.pivot.iter().enumerate() {
        for i in 0..n {
            residuals[i] -= x[j * n + i] * coef[m];
        }
    }

    let s2 = if n > rank {
        effects[rank..].iter().map(|v| v * v).sum::<f64>() / (n - rank) as f64
    } else {
        f64::NAN
    };

    WeightedFit {
        residuals,
        hat: qr.hat(),
        rank,
        s2,
    }
}

/// Accumulates the score contribution `info[-1, -1] - info[-1, 1] info[1, -1] / info[1, 1]`.
///
/// `Z = [1 | Z2]`, so the leading row and column of `info` are the intercept's,
/// and sweeping them out is what leaves an information matrix for `gamma` alone.
/// Forming `info` as an explicit `(1 + ngam)^2` matrix and then slicing it would
/// be clearer but allocates once per gene, so the pieces are kept separate.
///
/// ### Params
///
/// * `info2` - Column-major `ngam * ngam` accumulator, updated in place
/// * `corner` - `info[1, 1]`
/// * `edge` - `info[1, -1]`, length `ngam`
/// * `block` - `info[-1, -1]`, column-major `ngam * ngam`
/// * `ngam` - Number of variance coefficients
///
/// ### Returns
///
/// `true` when the contribution was added, `false` when `corner` was
/// non-positive or not a number.
///
/// limma has no such guard: it divides by `info[1, 1]` unconditionally. The
/// guard is defensive rather than a deviation, since `corner` is `nobs - rank`
/// in the gene-by-gene sweep and at least two by the admission rules, and the
/// REML sweep's `n - 2 rank + sum(Q2' 1)^2` is positive for any design that got
/// this far. Nothing in the test suite reaches it; a `false` here would mean the
/// alternative was `NaN` weights.
fn sweep_intercept(
    info2: &mut [f64],
    corner: f64,
    edge: &[f64],
    block: &[f64],
    ngam: usize,
) -> bool {
    if corner.is_nan() || corner <= 0.0 {
        return false;
    }
    for b in 0..ngam {
        for a in 0..ngam {
            info2[b * ngam + a] += block[b * ngam + a] - edge[a] * edge[b] / corner;
        }
    }
    true
}

///////////////////
// Array weights //
///////////////////

/// Per-sample quality weights, limma's `arrayWeights`.
///
/// Estimates one precision weight per sample under the model that sample `j`
/// has residual variance `sigma_g^2 / w_j`, shared across genes. A weight below
/// one marks an array that is noisier than the rest; the weights are normalised
/// so that their logs sum to zero over whatever variance design is in force.
///
/// The estimator is REML on the variance components, solved either one gene at a
/// time or by pooling every gene and iterating. See [`ArrayWeightMethod`]. Both
/// return early with all-ones weights when there is nothing to estimate from:
/// fewer than two genes, or fewer than two residual degrees of freedom in the
/// design.
///
/// Zero weights are handled as limma does, by turning the corresponding
/// expression values into missing ones and the weights into ones, which forces
/// the gene-by-gene sweep to skip those observations entirely.
///
/// ### Params
///
/// * `y` - Row-major `n_genes * n_samples` log-expression values. Non-finite
///   entries are treated as missing.
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major `n_samples * n_coef` design matrix
/// * `n_coef` - Number of design columns. A rank deficient design is reduced to
///   its independent columns rather than rejected.
/// * `weights` - Optional gene by sample observation weights, non-negative
/// * `params` - See [`ArrayWeightParams`]; `None` takes the defaults
///
/// ### Returns
///
/// One weight per sample, or [`EdgeErrors`] on a shape mismatch, an invalid
/// weight, an out-of-domain parameter, or a singular information matrix.
///
/// ### Where edgePython disagrees with limma
///
/// `edgepython/voom_lmfit.py:834` diverges in five places, all followed here to
/// limma rather than to the Python:
///
/// * It has no equivalent of `.arrayWeightsPrWtsREML`. With `method = "reml"`
///   and gene-level weights, its REML branch takes no `weights` argument at all,
///   so the weights are silently dropped. Masked by its `auto` rule, which sends
///   weighted input to the gene-by-gene sweep, but not by an explicit `"reml"`.
/// * It reduces both the design and the variance design with a fully pivoted QR
///   (`scipy.linalg.qr(pivoting = True)`), which reorders columns by magnitude.
///   R's `qr` moves only negligible columns, and only to the end, so a rank
///   deficient design keeps a different subset in a different order.
/// * The gene-by-gene sweep solves the information system with
///   `pinv(info2, rcond = 1e-12)` where limma uses `solve`. A pseudo-inverse
///   returns a minimum-norm answer where limma would stop.
/// * It clamps the working weights up to machine epsilon and the leverages into
///   `[0, 1]`. limma does neither.
/// * Its gene-by-gene missing-value branch admits a gene on `sum(good) > p`
///   alone, where limma additionally requires two residual degrees of freedom.
///
/// ### References
///
/// Ritchie, Diyagama, Neilson, van Laar, Dobrovic, Holloway and Smyth, BMC
/// Bioinformatics 7:261, 2006
pub fn array_weights(
    y: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    weights: Option<&Recycled<f64>>,
    params: Option<ArrayWeightParams>,
) -> Result<Vec<f64>, EdgeErrors> {
    validate_shapes(y, n_genes, n_samples, design, n_coef)?;
    let params = params.unwrap_or_default();

    if !params.prior_n.is_finite() || params.prior_n < 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior_n must be finite and non-negative; got {}.",
            params.prior_n
        )));
    }
    if !params.tol.is_finite() || params.tol <= 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "tol must be finite and positive; got {}.",
            params.tol
        )));
    }
    if params.max_iter == 0 {
        return Err(EdgeErrors::MustBePositive("max_iter".to_string()));
    }

    let ones = vec![1.0; n_samples];
    if n_genes < 2 || n_samples < 2 {
        return Ok(ones);
    }

    let (design_cm, rank) = reduce_design(design, n_samples, n_coef);
    if n_samples < rank + MIN_RESIDUAL_DF {
        return Ok(ones);
    }

    // Zero weights become missing observations, exactly as limma does, so the
    // expression matrix has to be materialised before the sweep starts.
    let mut expression = y.to_vec();
    let mut weight_matrix: Option<Vec<f64>> = None;
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
        let mut dense = w.expand(n_genes, n_samples);
        if dense.iter().any(|v| !v.is_finite()) {
            return Err(EdgeErrors::InvalidArgument(
                "arrayWeights does not accept missing or infinite weights.".to_string(),
            ));
        }
        if dense.iter().any(|v| *v < 0.0) {
            return Err(EdgeErrors::InvalidArgument(
                "arrayWeights does not accept negative weights.".to_string(),
            ));
        }
        for (e, wv) in expression.iter_mut().zip(dense.iter_mut()) {
            if *wv == 0.0 {
                *e = f64::NAN;
                *wv = 1.0;
            }
        }
        weight_matrix = Some(dense);
    }

    let (z2, ngam) =
        prepare_var_design(n_samples, params.var_design.as_deref(), params.n_var_coef)?;
    if ngam == 0 {
        return Ok(ones);
    }

    match params.method {
        ArrayWeightMethod::GeneByGene => array_weights_gene_by_gene(
            &expression,
            n_genes,
            n_samples,
            &design_cm,
            rank,
            weight_matrix.as_deref(),
            &z2,
            ngam,
            params.prior_n,
        ),
        ArrayWeightMethod::Reml => {
            // limma drops rows that are wholly or partly missing before the REML
            // sweep, since it has no per-observation branch.
            let complete: Vec<usize> = (0..n_genes)
                .filter(|g| {
                    expression[g * n_samples..(g + 1) * n_samples]
                        .iter()
                        .all(|v| v.is_finite())
                })
                .collect();
            if complete.len() < 2 {
                return Ok(ones);
            }
            let kept: Vec<f64> = complete
                .iter()
                .flat_map(|&g| {
                    expression[g * n_samples..(g + 1) * n_samples]
                        .iter()
                        .copied()
                })
                .collect();

            match weight_matrix {
                None => array_weights_reml(
                    &kept,
                    complete.len(),
                    n_samples,
                    &design_cm,
                    rank,
                    &z2,
                    ngam,
                    params.prior_n,
                    params.max_iter,
                    params.tol,
                ),
                Some(w) => {
                    let kept_w: Vec<f64> = complete
                        .iter()
                        .flat_map(|&g| w[g * n_samples..(g + 1) * n_samples].iter().copied())
                        .collect();
                    array_weights_prior_reml(
                        &kept,
                        &kept_w,
                        complete.len(),
                        n_samples,
                        &design_cm,
                        rank,
                        &z2,
                        ngam,
                        params.prior_n,
                        params.max_iter,
                        params.tol,
                    )
                }
            }
        }
    }
}

/// limma's `.arrayWeightsGeneByGene`.
///
/// One Fisher scoring step per gene, with the weights refreshed after every
/// step, so the sweep is inherently sequential: gene `i + 1` is fitted under the
/// weights gene `i` produced. That rules out a parallel gene axis, which is why
/// this is the one estimator in the module that runs on a single thread.
///
/// ### Params
///
/// * `expression` - Row-major `n_genes * n_samples`, non-finite meaning missing
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Column-major `n_samples * p`, already reduced to full rank
/// * `p` - Number of design columns
/// * `weights` - Optional row-major `n_genes * n_samples` observation weights
/// * `z2` - Column-major `n_samples * ngam` variance design
/// * `ngam` - Number of variance coefficients
/// * `prior_n` - Prior gene count
///
/// ### Returns
///
/// One weight per sample.
#[allow(clippy::too_many_arguments)]
fn array_weights_gene_by_gene(
    expression: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    p: usize,
    weights: Option<&[f64]>,
    z2: &[f64],
    ngam: usize,
    prior_n: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    let mut gam = vec![0.0; ngam];
    let mut aw = vec![1.0; n_samples];

    // info2 starts at the prior ridge `prior_n * Z2' Z2`.
    let mut info2 = vec![0.0; ngam * ngam];
    for b in 0..ngam {
        for a in 0..ngam {
            let dot: f64 = (0..n_samples)
                .map(|i| z2[a * n_samples + i] * z2[b * n_samples + i])
                .sum();
            info2[b * ngam + a] = prior_n * dot;
        }
    }

    let mut edge = vec![0.0; ngam];
    let mut block = vec![0.0; ngam * ngam];

    for gene in 0..n_genes {
        let row = &expression[gene * n_samples..(gene + 1) * n_samples];
        let w: Vec<f64> = match weights {
            None => aw.clone(),
            Some(m) => (0..n_samples)
                .map(|j| aw[j] * m[gene * n_samples + j])
                .collect(),
        };

        let mut d = vec![0.0; n_samples];
        let mut h1 = vec![0.0; n_samples];
        let s2;

        if row.iter().all(|v| v.is_finite()) {
            // No residual degrees of freedom check on this branch: limma has
            // none either, and the caller has already guaranteed at least two.
            let fit = weighted_fit(design, row, &w, n_samples, p);
            for j in 0..n_samples {
                d[j] = w[j] * fit.residuals[j] * fit.residuals[j];
                h1[j] = 1.0 - fit.hat[j];
            }
            s2 = fit.s2;
        } else {
            let observed: Vec<usize> = (0..n_samples).filter(|&j| row[j].is_finite()).collect();
            let n_obs = observed.len();
            if n_obs <= MIN_RESIDUAL_DF {
                continue;
            }
            let mut sub_x = vec![0.0; n_obs * p];
            for c in 0..p {
                for (k, &j) in observed.iter().enumerate() {
                    sub_x[c * n_obs + k] = design[c * n_samples + j];
                }
            }
            let sub_y: Vec<f64> = observed.iter().map(|&j| row[j]).collect();
            let sub_w: Vec<f64> = observed.iter().map(|&j| w[j]).collect();
            let fit = weighted_fit(&sub_x, &sub_y, &sub_w, n_obs, p);
            if n_obs < fit.rank + MIN_RESIDUAL_DF {
                continue;
            }
            for (k, &j) in observed.iter().enumerate() {
                d[j] = sub_w[k] * fit.residuals[k] * fit.residuals[k];
                h1[j] = 1.0 - fit.hat[k];
            }
            s2 = fit.s2;
        }

        if !s2.is_finite() || s2 < MIN_RESIDUAL_VAR {
            continue;
        }

        // info = Z' diag(h1) Z with Z = [1 | Z2], kept in its three blocks.
        let corner: f64 = h1.iter().sum();
        for a in 0..ngam {
            edge[a] = (0..n_samples).map(|i| h1[i] * z2[a * n_samples + i]).sum();
        }
        for b in 0..ngam {
            for a in 0..=b {
                let v: f64 = (0..n_samples)
                    .map(|i| h1[i] * z2[a * n_samples + i] * z2[b * n_samples + i])
                    .sum();
                block[b * ngam + a] = v;
                block[a * ngam + b] = v;
            }
        }
        if !sweep_intercept(&mut info2, corner, &edge, &block, ngam) {
            continue;
        }

        let score: Vec<f64> = (0..ngam)
            .map(|a| {
                (0..n_samples)
                    .map(|i| z2[a * n_samples + i] * (d[i] / s2 - h1[i]))
                    .sum()
            })
            .collect();
        let step = solve_linear(&info2, &score, ngam)?;
        for a in 0..ngam {
            gam[a] += step[a];
        }
        for (j, w) in aw.iter_mut().enumerate() {
            let eta: f64 = (0..ngam).map(|a| z2[a * n_samples + j] * gam[a]).sum();
            *w = (-eta).exp();
        }
    }

    Ok(aw)
}

/// Builds limma's `Q2` and the leverages from a fitted QR.
///
/// `Q2` holds every product `Q[, i] * Q[, j]` for `i <= j` over the leading `p`
/// columns of `Q`, with the off-diagonal blocks scaled by `sqrt(2)`. That gives
/// `Q2' Q2` the second moments of the residual projection, which is the second
/// derivative term in the REML information for the log weights.
///
/// ### Params
///
/// * `qr` - Factorisation of the weighted design
/// * `n_samples` - Number of samples
/// * `p` - Number of design columns
///
/// ### Returns
///
/// Column-major `n_samples * (p * (p + 1) / 2)` and the leverages, which are the
/// row sums of the first `p` columns of `Q2`.
fn residual_moments(qr: &LinpackQr, n_samples: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
    let p2 = p * (p + 1) / 2;
    let q = qr.leading_q(p);
    let mut q2 = vec![0.0; n_samples * p2];
    let mut j0 = 0;
    for k in 0..p {
        for c in 0..(p - k) {
            for i in 0..n_samples {
                q2[(j0 + c) * n_samples + i] = q[c * n_samples + i] * q[(c + k) * n_samples + i];
            }
        }
        j0 += p - k;
    }
    let root_two = 2.0_f64.sqrt();
    for v in q2[p * n_samples..].iter_mut() {
        *v *= root_two;
    }
    let hat: Vec<f64> = (0..n_samples)
        .map(|i| (0..p).map(|c| q2[c * n_samples + i]).sum())
        .collect();
    (q2, hat)
}

/// The REML information for `gamma`, `crossprod(Z, (1 - 2h) Z) + crossprod(crossprod(Q2, Z))`.
///
/// ### Params
///
/// * `q2` - Column-major `n_samples * p2` second-moment matrix
/// * `p2` - Number of columns in `q2`
/// * `hat` - Leverages, one per sample
/// * `z2` - Column-major `n_samples * ngam` variance design
/// * `ngam` - Number of variance coefficients
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// `info[1, 1]`, `info[1, -1]` and `info[-1, -1]` of the `(1 + ngam)` square
/// information, ready for [`sweep_intercept`].
fn reml_information(
    q2: &[f64],
    p2: usize,
    hat: &[f64],
    z2: &[f64],
    ngam: usize,
    n_samples: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let u: Vec<f64> = hat.iter().map(|h| 1.0 - 2.0 * h).collect();

    // g = Q2' Z, of shape p2 by (1 + ngam), stored column-major.
    let m = 1 + ngam;
    let mut g = vec![0.0; p2 * m];
    for r in 0..p2 {
        g[r] = (0..n_samples).map(|i| q2[r * n_samples + i]).sum();
        for c in 0..ngam {
            g[(1 + c) * p2 + r] = (0..n_samples)
                .map(|i| q2[r * n_samples + i] * z2[c * n_samples + i])
                .sum();
        }
    }

    let corner = u.iter().sum::<f64>() + (0..p2).map(|r| g[r] * g[r]).sum::<f64>();
    let edge: Vec<f64> = (0..ngam)
        .map(|a| {
            let direct: f64 = (0..n_samples).map(|i| u[i] * z2[a * n_samples + i]).sum();
            let moment: f64 = (0..p2).map(|r| g[r] * g[(1 + a) * p2 + r]).sum();
            direct + moment
        })
        .collect();
    let mut block = vec![0.0; ngam * ngam];
    for b in 0..ngam {
        for a in 0..=b {
            let direct: f64 = (0..n_samples)
                .map(|i| u[i] * z2[a * n_samples + i] * z2[b * n_samples + i])
                .sum();
            let moment: f64 = (0..p2)
                .map(|r| g[(1 + a) * p2 + r] * g[(1 + b) * p2 + r])
                .sum();
            block[b * ngam + a] = direct + moment;
            block[a * ngam + b] = direct + moment;
        }
    }
    (corner, edge, block)
}

/// Turns the current `gamma` into weights, `exp(-Z2 gamma)`.
///
/// ### Params
///
/// * `z2` - Column-major `n_samples * ngam` variance design
/// * `gam` - Current variance coefficients
/// * `n_samples` - Number of samples
/// * `ngam` - Number of variance coefficients
///
/// ### Returns
///
/// One weight per sample.
fn weights_from_gamma(z2: &[f64], gam: &[f64], n_samples: usize, ngam: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|j| {
            let eta: f64 = (0..ngam).map(|a| z2[a * n_samples + j] * gam[a]).sum();
            (-eta).exp()
        })
        .collect()
}

/// limma's `.arrayWeightsREML`, the unweighted pooled sweep.
///
/// Every gene shares the same weighted design, so the QR is factorised once per
/// iteration and reused across genes; only the projection of each gene's
/// response changes. Genes are chunked and reduced in a fixed order so the
/// answer does not depend on how rayon happened to split the work.
///
/// ### Params
///
/// * `y` - Row-major `n_genes * n_samples`, complete
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Column-major `n_samples * p`, full rank
/// * `p` - Number of design columns
/// * `z2` - Column-major `n_samples * ngam` variance design
/// * `ngam` - Number of variance coefficients
/// * `prior_n` - Prior gene count
/// * `max_iter` - Iteration budget
/// * `tol` - Convergence tolerance
///
/// ### Returns
///
/// One weight per sample.
#[allow(clippy::too_many_arguments)]
fn array_weights_reml(
    y: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    p: usize,
    z2: &[f64],
    ngam: usize,
    prior_n: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    // limma drops genes with no residual variation once, from the unweighted
    // fit, before the iteration starts.
    let unit = vec![1.0; n_samples];
    let (_, first_s2, _) = reml_gene_pass(y, n_genes, n_samples, design, p, &unit);
    let usable: Vec<usize> = (0..n_genes)
        .filter(|&g| first_s2[g] >= MIN_RESIDUAL_VAR)
        .collect();
    let (kept, n_kept): (Vec<f64>, usize) = if usable.len() == n_genes {
        (y.to_vec(), n_genes)
    } else {
        if usable.len() < 2 {
            return Ok(vec![1.0; n_samples]);
        }
        (
            usable
                .iter()
                .flat_map(|&g| y[g * n_samples..(g + 1) * n_samples].iter().copied())
                .collect(),
            usable.len(),
        )
    };

    let p2 = p * (p + 1) / 2;
    let ridge: Vec<f64> = {
        let mut m = vec![0.0; ngam * ngam];
        for b in 0..ngam {
            for a in 0..ngam {
                m[b * ngam + a] = prior_n
                    * (0..n_samples)
                        .map(|i| z2[a * n_samples + i] * z2[b * n_samples + i])
                        .sum::<f64>();
            }
        }
        m
    };

    let mut gam = vec![0.0; ngam];
    let mut w: Vec<f64> = vec![1.0; n_samples];
    let mut last_convergence = f64::INFINITY;
    let genes = n_kept as f64;

    for _ in 0..max_iter {
        let (mean_scaled, _, qr) = reml_gene_pass(&kept, n_kept, n_samples, design, p, &w);
        let (q2, hat) = residual_moments(&qr, n_samples, p);
        let (corner, edge, block) = reml_information(&q2, p2, &hat, z2, ngam, n_samples);
        if corner.is_nan() || corner <= 0.0 {
            break;
        }

        let mut info2 = vec![0.0; ngam * ngam];
        sweep_intercept(&mut info2, corner, &edge, &block, ngam);
        for (slot, prior) in info2.iter_mut().zip(ridge.iter()) {
            *slot = genes * *slot + prior;
        }

        let z: Vec<f64> = (0..n_samples)
            .map(|j| genes * (mean_scaled[j] - (1.0 - hat[j])) + prior_n * (w[j] - 1.0))
            .collect();
        let score: Vec<f64> = (0..ngam)
            .map(|a| (0..n_samples).map(|i| z2[a * n_samples + i] * z[i]).sum())
            .collect();
        let step = solve_linear(&info2, &score, ngam)?;

        let convergence = score
            .iter()
            .zip(step.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / ngam as f64
            / (genes + prior_n);
        // limma stops rather than cycling once the criterion stops improving.
        if !convergence.is_finite() || convergence >= last_convergence {
            break;
        }
        last_convergence = convergence;

        for a in 0..ngam {
            gam[a] += step[a];
        }
        w = weights_from_gamma(z2, &gam, n_samples, ngam);
        if convergence < tol {
            break;
        }
    }

    Ok(w)
}

/// One pooled pass over the genes at fixed sample weights.
///
/// Returns the two quantities the REML update needs: the mean over genes of
/// `w_j r_gj^2 / s2_g` at each sample, and the per-gene residual variances.
///
/// The QR of the weighted design is shared, so this is `O(n_genes * n_samples * p)`
/// with no factorisation per gene. Genes are split into fixed chunks whose
/// partial sums are added back in index order, which keeps the result
/// bit-reproducible across runs.
///
/// ### Params
///
/// * `y` - Row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Column-major `n_samples * p`
/// * `p` - Number of design columns
/// * `w` - Current sample weights
///
/// ### Returns
///
/// The per-sample mean of `w_j r^2 / s2`, the per-gene `s2`, and the shared
/// factorisation of the weighted design, which the caller needs for the REML
/// information and would otherwise have to redo.
fn reml_gene_pass(
    y: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    p: usize,
    w: &[f64],
) -> (Vec<f64>, Vec<f64>, LinpackQr) {
    let root_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let mut xw = vec![0.0; n_samples * p];
    for j in 0..p {
        for i in 0..n_samples {
            xw[j * n_samples + i] = root_w[i] * design[j * n_samples + i];
        }
    }
    let qr = LinpackQr::new(xw, n_samples, p);
    let rank = qr.rank();
    let df = (n_samples - rank) as f64;

    let chunk = n_genes.div_ceil(rayon::current_num_threads().max(1)).max(1);
    let partials: Vec<(Vec<f64>, Vec<f64>)> = y
        .par_chunks(chunk * n_samples)
        .map(|slab| {
            let mut acc = vec![0.0; n_samples];
            let mut s2 = Vec::with_capacity(slab.len() / n_samples);
            for row in slab.chunks_exact(n_samples) {
                let mut effects: Vec<f64> = (0..n_samples).map(|i| root_w[i] * row[i]).collect();
                qr.qty(&mut effects);
                let coef = qr.solve_r(&effects);
                let variance = effects[rank..].iter().map(|v| v * v).sum::<f64>() / df;
                s2.push(variance);
                let mut residual = row.to_vec();
                for (m, &j) in qr.pivot.iter().enumerate() {
                    for i in 0..n_samples {
                        residual[i] -= design[j * n_samples + i] * coef[m];
                    }
                }
                for i in 0..n_samples {
                    acc[i] += w[i] * residual[i] * residual[i] / variance;
                }
            }
            (acc, s2)
        })
        .collect();

    let mut total = vec![0.0; n_samples];
    let mut s2 = Vec::with_capacity(n_genes);
    for (acc, part) in partials {
        for (t, a) in total.iter_mut().zip(acc.iter()) {
            *t += a;
        }
        s2.extend(part);
    }
    for t in total.iter_mut() {
        *t /= n_genes as f64;
    }
    (total, s2, qr)
}

/// limma's `.arrayWeightsPrWtsREML`, the pooled sweep with gene-level weights.
///
/// Every gene now has its own weighted design, so the QR cannot be shared and
/// each iteration refactorises once per gene. The score and information are
/// accumulated by chunk and summed back in index order for reproducibility.
///
/// Unlike [`array_weights_reml`], this one does not test the convergence
/// criterion for improvement, only for finiteness and for falling below `tol`.
/// That is limma's behaviour, not an oversight here.
///
/// ### Params
///
/// * `y` - Row-major `n_genes * n_samples`, complete
/// * `weights` - Row-major `n_genes * n_samples` observation weights
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Column-major `n_samples * p`, full rank
/// * `p` - Number of design columns
/// * `z2` - Column-major `n_samples * ngam` variance design
/// * `ngam` - Number of variance coefficients
/// * `prior_n` - Prior gene count
/// * `max_iter` - Iteration budget
/// * `tol` - Convergence tolerance
///
/// ### Returns
///
/// One weight per sample.
#[allow(clippy::too_many_arguments)]
fn array_weights_prior_reml(
    y: &[f64],
    weights: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    p: usize,
    z2: &[f64],
    ngam: usize,
    prior_n: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    let p2 = p * (p + 1) / 2;
    let ridge: Vec<f64> = {
        let mut m = vec![0.0; ngam * ngam];
        for b in 0..ngam {
            for a in 0..ngam {
                m[b * ngam + a] = prior_n
                    * (0..n_samples)
                        .map(|i| z2[a * n_samples + i] * z2[b * n_samples + i])
                        .sum::<f64>();
            }
        }
        m
    };

    let mut gam = vec![0.0; ngam];
    let mut w = vec![1.0; n_samples];
    let chunk = n_genes.div_ceil(rayon::current_num_threads().max(1)).max(1);

    for _ in 0..max_iter {
        let partials: Vec<(Vec<f64>, Vec<f64>)> = y
            .par_chunks(chunk * n_samples)
            .zip(weights.par_chunks(chunk * n_samples))
            .map(|(slab, wslab)| {
                let mut info2 = vec![0.0; ngam * ngam];
                let mut z = vec![0.0; n_samples];
                for (row, wrow) in slab
                    .chunks_exact(n_samples)
                    .zip(wslab.chunks_exact(n_samples))
                {
                    let full_w: Vec<f64> = (0..n_samples).map(|j| w[j] * wrow[j]).collect();
                    let root_w: Vec<f64> = full_w.iter().map(|v| v.sqrt()).collect();
                    let mut xw = vec![0.0; n_samples * p];
                    for j in 0..p {
                        for i in 0..n_samples {
                            xw[j * n_samples + i] = root_w[i] * design[j * n_samples + i];
                        }
                    }
                    let qr = LinpackQr::new(xw, n_samples, p);
                    let rank = qr.rank();
                    let mut effects: Vec<f64> =
                        (0..n_samples).map(|i| root_w[i] * row[i]).collect();
                    qr.qty(&mut effects);
                    let coef = qr.solve_r(&effects);
                    let s2 = effects[rank..].iter().map(|v| v * v).sum::<f64>()
                        / (n_samples - rank) as f64;

                    let (q2, hat) = residual_moments(&qr, n_samples, p);
                    let (corner, edge, block) =
                        reml_information(&q2, p2, &hat, z2, ngam, n_samples);
                    sweep_intercept(&mut info2, corner, &edge, &block, ngam);

                    if s2 > MIN_RESIDUAL_VAR {
                        let mut residual = row.to_vec();
                        for (m, &j) in qr.pivot.iter().enumerate() {
                            for i in 0..n_samples {
                                residual[i] -= design[j * n_samples + i] * coef[m];
                            }
                        }
                        for i in 0..n_samples {
                            z[i] += full_w[i] * residual[i] * residual[i] / s2 - (1.0 - hat[i]);
                        }
                    }
                }
                (info2, z)
            })
            .collect();

        let mut info2 = ridge.clone();
        let mut z: Vec<f64> = w.iter().map(|v| prior_n * (v - 1.0)).collect();
        for (part_info, part_z) in partials {
            for (slot, v) in info2.iter_mut().zip(part_info.iter()) {
                *slot += v;
            }
            for (slot, v) in z.iter_mut().zip(part_z.iter()) {
                *slot += v;
            }
        }
        let scale = n_genes as f64 + prior_n;
        for slot in info2.iter_mut() {
            *slot /= scale;
        }
        for slot in z.iter_mut() {
            *slot /= scale;
        }

        let score: Vec<f64> = (0..ngam)
            .map(|a| (0..n_samples).map(|i| z2[a * n_samples + i] * z[i]).sum())
            .collect();
        let step = solve_linear(&info2, &score, ngam)?;
        for a in 0..ngam {
            gam[a] += step[a];
        }
        w = weights_from_gamma(z2, &gam, n_samples, ngam);

        let convergence = score
            .iter()
            .zip(step.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / scale
            / ngam as f64;
        if !convergence.is_finite() || convergence < tol {
            break;
        }
    }

    Ok(w)
}

//////////////////////////
// Duplicate correlation //
//////////////////////////

/// Consensus intra-block correlation, limma's `duplicateCorrelation`.
///
/// Fits `y_g = X beta_g + Z u_g + e_g` for each gene, with `u_g` a random block
/// effect, and reports `tanh` of the trimmed mean of the Fisher z-transformed
/// per-gene correlations `sigma_block^2 / (sigma_block^2 + sigma_e^2)`.
///
/// Two situations short-circuit to a correlation of exactly zero, as in limma:
/// every block has a single member, so there is no within-block replication; or
/// the block factor already lies in the column space of the design, so it has
/// been absorbed into the fixed effects.
///
/// A gene contributes only when it has more than `n_coef + 2` observations, more
/// than one block, and fewer blocks than `n_obs - 1`. Genes failing any of those
/// are dropped from the trimmed mean rather than counted as zero.
///
/// ### edgePython disagrees with limma here
///
/// `edgepython/voom_lmfit.py:898` replaces the per-gene mixed model with a
/// one-way ANOVA moment estimator, `(MS_between - MS_within) / (MS_between +
/// (n0 - 1) MS_within)`. That is the classical intraclass correlation, not the
/// REML estimate limma computes, and the two agree only for balanced blocks with
/// no fixed effects beyond an intercept. Alongside that it:
///
/// * clips the per-gene correlations to `[-0.99, 0.99]` rather than to limma's
///   block-size-dependent lower bound `1 / (1 - max_block_size) + 0.01`;
/// * drops limma's `nblocks < n_obs - 1` and `n_obs > n_coef + 2` admission
///   rules on its vectorised path, so genes limma refuses still contribute;
/// * has no check for a block factor already spanned by the design, and none for
///   blocks that are all of size one, both of which limma answers with an exact
///   zero.
///
/// This implementation follows limma throughout.
///
/// ### Params
///
/// * `y` - Row-major `n_genes * n_samples` log-expression values
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major `n_samples * n_coef` design matrix
/// * `n_coef` - Number of design columns
/// * `block` - Block label per sample; only equality of labels matters
/// * `weights` - Optional gene by sample observation weights. Non-positive or
///   non-finite weights mark the observation missing, as limma does.
/// * `trim` - Fraction trimmed from each tail of the z-transformed
///   correlations, in `[0, 0.5)`. limma's default is 0.15.
///
/// ### Returns
///
/// The consensus correlation, or [`EdgeErrors`] on a shape mismatch, an
/// out-of-range `trim`, or when no gene yielded a correlation at all.
///
/// ### References
///
/// Smyth, Michaud and Scott, Bioinformatics 21:2067-2075, 2005
#[allow(clippy::too_many_arguments)]
pub fn duplicate_correlation(
    y: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    block: &[usize],
    weights: Option<&Recycled<f64>>,
    trim: f64,
) -> Result<f64, EdgeErrors> {
    validate_shapes(y, n_genes, n_samples, design, n_coef)?;
    if block.len() != n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "block",
            expected: n_samples,
            got: block.len(),
        });
    }
    if !(0.0..0.5).contains(&trim) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "trim must lie in [0, 0.5); got {trim}."
        )));
    }

    let mut levels: Vec<usize> = block.to_vec();
    levels.sort_unstable();
    levels.dedup();
    let max_block = levels
        .iter()
        .map(|lv| block.iter().filter(|b| *b == lv).count())
        .max()
        .unwrap_or(0);
    if max_block <= 1 {
        return Ok(0.0);
    }

    let design_cm = to_column_major(design, n_samples, n_coef);
    let qr = LinpackQr::new(design_cm.clone(), n_samples, n_coef);
    let rank = qr.rank();

    // Treatment-coded block indicators: if none of them survives the projection
    // onto the design's residual space, the blocks are already in the model.
    let mut absorbed = true;
    for lv in levels.iter().skip(1) {
        let mut column: Vec<f64> = block
            .iter()
            .map(|b| if b == lv { 1.0 } else { 0.0 })
            .collect();
        qr.qty(&mut column);
        if column[rank..].iter().any(|v| v.abs() >= BLOCK_ABSORBED_TOL) {
            absorbed = false;
            break;
        }
    }
    if absorbed {
        return Ok(0.0);
    }

    let weight_matrix = match weights {
        None => None,
        Some(w) => {
            w.validate(n_genes, n_samples)?;
            Some(w.expand(n_genes, n_samples))
        }
    };

    let rho_min = 1.0 / (1.0 - max_block as f64) + RHO_MIN_SLACK;
    let correlations: Vec<Option<f64>> = (0..n_genes)
        .into_par_iter()
        .map(|gene| {
            let row = &y[gene * n_samples..(gene + 1) * n_samples];
            let wrow = weight_matrix
                .as_ref()
                .map(|m| &m[gene * n_samples..(gene + 1) * n_samples]);
            let observed: Vec<usize> = (0..n_samples)
                .filter(|&j| {
                    row[j].is_finite() && wrow.is_none_or(|w| w[j].is_finite() && w[j] > 0.0)
                })
                .collect();
            let n_obs = observed.len();
            if n_obs <= n_coef + 2 {
                return None;
            }
            let mut present: Vec<usize> = observed.iter().map(|&j| block[j]).collect();
            present.sort_unstable();
            present.dedup();
            let n_blocks = present.len();
            if n_blocks <= 1 || n_blocks >= n_obs - 1 {
                return None;
            }

            let mut x = vec![0.0; n_obs * n_coef];
            for c in 0..n_coef {
                for (k, &j) in observed.iter().enumerate() {
                    x[c * n_obs + k] = design_cm[c * n_samples + j];
                }
            }
            let response: Vec<f64> = observed.iter().map(|&j| row[j]).collect();
            let z: Vec<f64> = present
                .iter()
                .flat_map(|lv| {
                    observed
                        .iter()
                        .map(move |&j| if block[j] == *lv { 1.0 } else { 0.0 })
                })
                .collect();
            let w = wrow.map(|w| observed.iter().map(|&j| w[j]).collect::<Vec<f64>>());

            let varcomp =
                mixed_model_varcomp(&response, &x, &z, w.as_deref(), n_obs, n_coef, n_blocks)?;
            let total = varcomp[0] + varcomp[1];
            let rho = varcomp[1] / total;
            rho.is_finite().then_some(rho)
        })
        .collect();

    let z: Vec<f64> = correlations
        .into_iter()
        .flatten()
        .map(|rho| rho.clamp(rho_min, RHO_MAX).atanh())
        .collect();
    if z.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "no gene yielded an intra-block correlation; check the block structure and design."
                .to_string(),
        ));
    }
    Ok(trimmed_mean(&z, trim)?.tanh())
}

/// `statmod::mixedModel2Fit(..., only.varcomp = TRUE)`.
///
/// Projects the response and the block design onto the residual space of the
/// fixed effects, rotates by the SVD of the projected block design, and fits a
/// gamma GLM of the squared rotated residuals against the squared singular
/// values. The two coefficients of that GLM are the residual and block variance
/// components.
///
/// `w` scales the response and the fixed effects but deliberately not the block
/// design, which is what `statmod` does.
///
/// The left singular vectors beyond the rank of the projected block design are
/// an arbitrary orthonormal completion, and faer's differs from LAPACK's. It
/// does not matter: all of those rotated residuals share a squared singular
/// value of zero, and both the linear start and the gamma GLM's score equations
/// see rows with equal covariates only through their sum, which the completion
/// preserves.
///
/// ### Params
///
/// * `y` - Response of length `n_obs`
/// * `x` - Column-major `n_obs * n_coef` fixed effects
/// * `z` - Column-major `n_obs * n_blocks` block indicators
/// * `w` - Optional observation weights of length `n_obs`
/// * `n_obs` - Number of observations
/// * `n_coef` - Number of fixed effects
/// * `n_blocks` - Number of blocks
///
/// ### Returns
///
/// The residual and block variance components, or `None` when the projection
/// leaves no residual space or the fit is degenerate.
fn mixed_model_varcomp(
    y: &[f64],
    x: &[f64],
    z: &[f64],
    w: Option<&[f64]>,
    n_obs: usize,
    n_coef: usize,
    n_blocks: usize,
) -> Option<[f64; 2]> {
    let mut xw = x.to_vec();
    let mut yw = y.to_vec();
    if let Some(w) = w {
        for i in 0..n_obs {
            let root = w[i].sqrt();
            yw[i] *= root;
            for c in 0..n_coef {
                xw[c * n_obs + i] *= root;
            }
        }
    }

    let qr = LinpackQr::new(xw, n_obs, n_coef);
    let rank = qr.rank();
    let mq = n_obs - rank;
    if mq == 0 {
        return None;
    }

    // Q' Z restricted to the residual rows, column-major mq by n_blocks.
    let mut qtz = vec![0.0; mq * n_blocks];
    for c in 0..n_blocks {
        let mut column = z[c * n_obs..(c + 1) * n_obs].to_vec();
        qr.qty(&mut column);
        qtz[c * mq..(c + 1) * mq].copy_from_slice(&column[rank..]);
    }
    qr.qty(&mut yw);
    let effects = &yw[rank..];

    let svd = faer::MatRef::from_column_major_slice(&qtz, mq, n_blocks)
        .svd()
        .ok()?;
    let singular = svd.S().column_vector();
    let u = svd.U();

    let mut d = vec![0.0; mq];
    for i in 0..singular.nrows().min(mq) {
        d[i] = singular[i] * singular[i];
    }
    let dy: Vec<f64> = (0..mq)
        .map(|c| {
            let projected: f64 = (0..mq).map(|i| u[(i, c)] * effects[i]).sum();
            projected * projected
        })
        .collect();

    // Start from the ordinary least squares fit of dy on (1, d).
    let mut dx = Vec::with_capacity(mq * 2);
    dx.extend(std::iter::repeat_n(1.0, mq));
    dx.extend_from_slice(&d);
    let dqr = LinpackQr::new(dx.clone(), mq, 2);
    if dqr.rank() < 2 {
        return None;
    }
    let mut dy_effects = dy.clone();
    dqr.qty(&mut dy_effects);
    let start = dqr.solve_r(&dy_effects);
    let fitted: Vec<f64> = (0..mq).map(|i| start[0] + start[1] * d[i]).collect();

    let non_zero = d.iter().filter(|v| v.abs() > SINGULAR_VALUE_TOL).count();
    let mean_d = d.iter().sum::<f64>() / mq as f64;
    let var_d = d.iter().map(|v| (v - mean_d).powi(2)).sum::<f64>() / (mq as f64 - 1.0);
    if !(mq > 2 && non_zero > 1 && var_d > SINGULAR_VALUE_TOL) {
        return Some([start[0], start[1]]);
    }

    let coef_start = if fitted.iter().all(|v| *v >= 0.0) {
        [start[0], start[1]]
    } else {
        [dy.iter().sum::<f64>() / mq as f64, 0.0]
    };
    glmgam_fit(&dx, &dy, coef_start, mq)
}

/// `statmod::glmgam.fit` with a supplied starting value.
///
/// A gamma GLM with an identity link, fitted by Levenberg-damped Fisher scoring
/// on the deviance. The damping schedule is the whole point of reproducing this
/// verbatim: the objective is nearly flat near the optimum, so where the
/// iteration stops decides the answer well before the arithmetic does.
///
/// ### Params
///
/// * `x` - Column-major `n * 2` design, an intercept and the squared singular values
/// * `y` - Squared rotated residuals, non-negative
/// * `start` - Starting coefficients, from the linear fit
/// * `n` - Number of observations
///
/// ### Returns
///
/// The fitted coefficients, or `None` if the damping ran away or the result was
/// not finite.
fn glmgam_fit(x: &[f64], y: &[f64], start: [f64; 2], n: usize) -> Option<[f64; 2]> {
    let max_y = y.iter().fold(0.0_f64, |acc, v| acc.max(*v));
    if max_y == 0.0 {
        return Some([0.0, 0.0]);
    }

    let fitted = |beta: [f64; 2]| -> Vec<f64> {
        (0..n)
            .map(|i| x[i] * beta[0] + x[n + i] * beta[1])
            .collect()
    };

    let mut beta = start;
    let mut mu = fitted(beta);
    if mu.iter().any(|v| *v < 0.0) {
        return None;
    }
    let mut deviance = gamma_deviance(y, &mu);

    let mut lambda = 0.0;
    let mut max_information;
    let mut iter = 0;
    loop {
        iter += 1;
        let v: Vec<f64> = mu.iter().map(|m| m * m).collect();
        let floor = v.iter().fold(0.0_f64, |acc, s| acc.max(*s)) / GLMGAM_VARIANCE_FLOOR;
        let v: Vec<f64> = v.into_iter().map(|s| s.max(floor)).collect();

        // Fisher information X' V^-1 X, symmetric two by two.
        let mut xvx = [0.0_f64; 4];
        for i in 0..n {
            let (a, b) = (x[i], x[n + i]);
            xvx[0] += a * a / v[i];
            xvx[1] += a * b / v[i];
            xvx[3] += b * b / v[i];
        }
        xvx[2] = xvx[1];
        max_information = xvx[0].max(xvx[3]);
        if iter == 1 {
            lambda = ((xvx[0] + xvx[3]) / 2.0).abs() / 2.0;
        }

        let mut score = [0.0_f64; 2];
        for i in 0..n {
            let r = (y[i] - mu[i]) / v[i];
            score[0] += x[i] * r;
            score[1] += x[n + i] * r;
        }

        let old_beta = beta;
        let old_deviance = deviance;
        let mut level = 0;
        let mut step;
        let mut overdamped = false;
        loop {
            level += 1;
            let a = xvx[0] + lambda;
            let b = xvx[1];
            let c = xvx[3] + lambda;
            let det = a * c - b * b;
            if det.is_nan() || det <= 0.0 {
                return None;
            }
            step = [
                (c * score[0] - b * score[1]) / det,
                (a * score[1] - b * score[0]) / det,
            ];
            beta = [old_beta[0] + step[0], old_beta[1] + step[1]];
            mu = fitted(beta);
            deviance = gamma_deviance(y, &mu);
            let peak = mu.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));
            if deviance <= old_deviance || deviance / peak < GLMGAM_DEVIANCE_TOL {
                break;
            }
            if lambda / max_information > GLMGAM_MAX_DAMPING {
                beta = old_beta;
                overdamped = true;
                break;
            }
            lambda *= 2.0;
        }
        if overdamped || lambda / max_information > GLMGAM_MAX_DAMPING {
            break;
        }
        if level == 1 {
            lambda /= GLMGAM_DAMPING_RELAX;
        }
        let peak = mu.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));
        if score[0] * step[0] + score[1] * step[1] < GLMGAM_TOL
            || deviance / peak < GLMGAM_DEVIANCE_TOL
        {
            break;
        }
        if iter > GLMGAM_MAX_ITER {
            break;
        }
    }

    beta.iter().all(|v| v.is_finite()).then_some(beta)
}

/// Gamma deviance with `statmod`'s treatment of joint zeros.
///
/// ### Params
///
/// * `y` - Observed values, non-negative
/// * `mu` - Fitted values
///
/// ### Returns
///
/// `2 * sum((y - mu) / mu - log(y / mu))` over the terms where `y` and `mu` are
/// not both negligible, `Inf` if any fitted value is negative.
fn gamma_deviance(y: &[f64], mu: &[f64]) -> f64 {
    if mu.iter().any(|v| *v < 0.0) {
        return f64::INFINITY;
    }
    let kept: Vec<usize> = (0..y.len())
        .filter(|&i| !(y[i] < GLMGAM_ZERO_TOL && mu[i] < GLMGAM_ZERO_TOL))
        .collect();
    if kept.is_empty() {
        return 0.0;
    }
    2.0 * kept
        .iter()
        .map(|&i| (y[i] - mu[i]) / mu[i] - (y[i] / mu[i]).ln())
        .sum::<f64>()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Every reference value in this module comes from limma 3.66 on R 4.5,
    /// with `statmod` 1.5 underneath `duplicateCorrelation`. The inputs are
    /// embedded alongside the outputs so nothing depends on reproducing an RNG
    /// stream:
    ///
    /// ```r
    /// set.seed(1); y <- matrix(rnorm(200), 20, 10)
    /// cat(sprintf("%.17g", as.vector(t(y))), sep = ",\n")
    /// ```
    ///
    /// Row-major, 20 genes by 10 samples.
    ///
    /// ### Agreement achieved
    ///
    /// [`array_weights`] reproduces limma to better than `1e-16` relative on
    /// every fixture here, both methods, weighted and not, which is the printing
    /// precision of the references and therefore as close as they can pin it.
    /// The assertions sit at `1e-14` for headroom on other platforms.
    ///
    /// [`duplicate_correlation`] reaches `5e-15` at worst, and the binding
    /// constraint is `glmgam.fit`'s own `tol = 1e-6` on the score-step product
    /// rather than the arithmetic: both implementations stop at the same iterate
    /// of a nearly flat objective, and the residual difference is the rounding
    /// accumulated in the SVD ahead of it. The assertions sit at `1e-13`.
    const Y: [f64; 200] = [
        -0.6264538107423324,
        0.9189773716082182,
        -0.1645235962535869,
        2.401617760504777,
        -0.568668732818502,
        -0.620366677224124,
        -0.5059574621142573,
        -1.914359425680012,
        0.42510037737244794,
        -1.231323421558044,
        0.18364332422208227,
        0.782136300731067,
        -0.2533616801365075,
        -0.039240002733169244,
        -0.13517861512383206,
        0.04211587314423524,
        1.3430388251704113,
        1.1765833120185623,
        -0.2386471009130328,
        0.9838955700533794,
        -0.8356286124100472,
        0.0745649833651906,
        0.6969633754047375,
        0.6897393624507766,
        1.1780869965732044,
        -0.9109216485524455,
        -0.21457940854686872,
        -1.6649724362120033,
        1.0584830487090204,
        0.21992480366065134,
        1.5952808021377916,
        -1.9893516958633728,
        0.5566631986736573,
        0.028002158780666062,
        -1.523566800429762,
        0.15802877240407498,
        -0.17955653004338712,
        -0.46353040147238567,
        0.886422651374936,
        -1.467250029092243,
        0.3295077718153605,
        0.6198257478947102,
        -0.6887556945495199,
        -0.7432732088824053,
        0.5939461876284216,
        -0.6545846439188178,
        -0.10019074121356196,
        -1.1159201050428456,
        -0.619243048231147,
        0.5210227426481385,
        -0.8204683841180155,
        -0.056128739529000764,
        -0.7074951569621196,
        0.18879229951434284,
        0.3329503712135183,
        1.7672872693726465,
        0.7126663070514053,
        -0.7508190011934479,
        2.2061024645404674,
        -0.15875460471601593,
        0.4874290524284853,
        -0.1557955067053293,
        0.3645819621368303,
        -1.8049586288910378,
        1.0630998372763627,
        0.7167074760172057,
        -0.07356440412632632,
        2.087166545628347,
        -0.25502703014101524,
        1.4645873119697974,
        0.7383247051292172,
        -1.4707523838992744,
        0.7685329245154158,
        1.4655548615628862,
        -0.30418392363430063,
        0.9101742294952271,
        -0.03763417146704791,
        0.017395619693251662,
        -1.4244946502128084,
        -0.7660819996046648,
        0.5757813516534921,
        -0.4781500551086204,
        -0.11234621215022805,
        0.1532533382118977,
        0.37001880991628816,
        0.3841853578263446,
        -0.6816604787556568,
        -1.2863005304343256,
        -0.14439960195421936,
        -0.43021175392854644,
        -0.305388387156356,
        0.4179415601997024,
        0.881107726454215,
        2.1726116703621523,
        0.26709879077223103,
        1.6821760805194184,
        -0.32427027224631916,
        -1.6406055344185784,
        0.20753833923234472,
        -0.9261094973774372,
        1.5117811684508478,
        1.3586795515290442,
        0.398105880367068,
        0.4755095288996625,
        -0.5425200309916505,
        -0.6357364539489768,
        0.06016044043451516,
        0.4501871012726556,
        2.3079783990593614,
        -0.17710396143654025,
        0.38984323641143104,
        -0.10278772734299552,
        -0.6120263932507712,
        -0.7099464309218142,
        1.2078678059831724,
        -0.461644730360566,
        -0.5888944862596638,
        -0.018559832714637976,
        0.10580236789371146,
        0.4020117794863379,
        -0.6212405805418039,
        0.38767161155936924,
        0.34111969142442483,
        0.6107263534890547,
        1.1604026156949516,
        1.4322822385416627,
        0.5314961926325724,
        -0.3180683745438444,
        0.4569988054234134,
        -0.7317481731196062,
        -2.2146998871775,
        -0.05380504058290512,
        -1.1293630960807926,
        -0.9340976316442515,
        0.7002136495149983,
        -0.6506963533103668,
        -1.5183940817867871,
        -0.9293621474537023,
        -0.077152935356531,
        0.8303731679816739,
        1.1249309181431082,
        -1.3770595568286068,
        1.4330237017010372,
        -1.2536334002391023,
        1.5868334545408456,
        -0.20738074360196543,
        0.3065578607897656,
        -1.4874603101414847,
        -0.3340008423665445,
        -1.208082786304465,
        -0.04493360901523085,
        -0.41499456329967976,
        1.98039989850586,
        0.29144623551746296,
        0.558486425565304,
        -0.3928079294419839,
        -1.536449823537586,
        -1.0751922966156808,
        -0.034726028311276184,
        -1.0479844128077418,
        -0.016190263098946084,
        -0.3942899537103493,
        -0.36722147646650916,
        -0.4432918732184329,
        -1.2765922084580363,
        -0.3199928685485067,
        -0.30097612683661074,
        1.000028803713914,
        0.787639605630162,
        1.441157706844281,
        0.9438362106852992,
        -0.05931339671118567,
        -1.0441346263165303,
        0.0011053516316241311,
        -0.5732654142368862,
        -0.27911330297655895,
        -0.5282799044450062,
        -0.6212666947968234,
        2.075245008652285,
        -1.0158474653046492,
        0.8212211950980886,
        1.1000253719838828,
        0.5697196274424129,
        0.07434132415166406,
        -1.2246126148983558,
        0.494188331267827,
        -0.6520947806809989,
        -1.3844268473844912,
        1.0273924387637678,
        0.4119747123175149,
        0.5939013212175087,
        0.7631757484575442,
        -0.13505460388082438,
        -0.589520946188072,
        -0.4734006364393116,
        -0.17733048226960638,
        -0.05689677784739257,
        1.8692906224235806,
        1.2079083983867038,
        -0.38107605110891957,
    ];

    /// Observation weights, row-major 20 by 10:
    ///
    /// ```r
    /// set.seed(42); W <- matrix(runif(200, 0.2, 3), 20, 10)
    /// cat(sprintf("%.17g", as.vector(t(W))), sep = ",\n")
    /// ```
    const W: [f64; 200] = [
        2.761456921789795,
        2.731287884432822,
        1.26276587350294,
        2.091700368653983,
        1.8284912070259451,
        1.9534869646653532,
        1.1988215981982648,
        0.6156504987739027,
        2.8175042705610394,
        2.770557968318462,
        2.823811157234013,
        0.5883884696289897,
        1.4201604379341006,
        2.9518881541676816,
        0.6421345829032361,
        0.8080415550619363,
        1.349778352305293,
        0.42474050642922523,
        1.7413834362290799,
        2.615363375749439,
        1.001190697401762,
        2.9688968409784136,
        0.3048068920150399,
        2.3267239491455256,
        1.2052792564034462,
        0.8063884707167744,
        1.8057325170375407,
        1.4993947437033057,
        1.8849454591982067,
        1.0875306658446788,
        2.525253352988511,
        2.8506710511632263,
        2.9259117585606873,
        1.7861675874330103,
        2.007769259531051,
        1.2890460802242159,
        1.85109924999997,
        2.382230852078646,
        0.751584566757083,
        0.9259296127595007,
        1.9968874529004097,
        0.430825162678957,
        1.4089034968987106,
        2.579131211992353,
        2.372305415384471,
        2.838875937554985,
        2.2150404185988006,
        2.2538782867603,
        1.6986625095829366,
        2.2783460654318333,
        1.6534686575643718,
        1.6397929961793123,
        2.881214470602572,
        0.7305270191282034,
        1.778211156371981,
        2.895302438549697,
        1.3059245266020298,
        2.488245244324207,
        0.7027560699731111,
        2.292611127067357,
        2.2624472809955476,
        1.292569707892835,
        2.685713735409081,
        0.959602521173656,
        0.8543695161119104,
        2.2715947818011046,
        2.773771001584828,
        0.6764549475163221,
        1.4652821843512356,
        2.7701312952674924,
        0.5770664722658694,
        2.736066766548902,
        1.991940554510802,
        2.518843758665025,
        0.451945445779711,
        2.253088536020368,
        2.8951968220062554,
        2.84521691147238,
        1.08774938499555,
        2.4209353857673706,
        2.0395784131251276,
        1.4515149586834013,
        2.918706509005278,
        2.1409734970889986,
        0.4397137817926705,
        1.7001316119916736,
        0.8538658714853227,
        1.0221467558294535,
        0.5252890771254897,
        0.5733229311183095,
        2.174181395303458,
        2.5408119279891253,
        1.932746980525553,
        0.8735252710059285,
        1.054611434508115,
        0.20636430503800512,
        2.228593279980123,
        0.6174017468467354,
        0.7210860389284789,
        1.0056993060745298,
        1.4816769734956323,
        2.2652677297592163,
        1.1335961915552615,
        0.32036862885579465,
        2.0687942410819233,
        1.905024867132306,
        2.7301766703836616,
        2.214260055590421,
        2.243244270607829,
        0.7450932026840746,
        2.213514304626733,
        2.4709543955512343,
        1.1708950949832797,
        0.5933414635248482,
        0.20066891042515636,
        2.5430443664081395,
        1.8897274373099207,
        1.1074406669475139,
        1.3532417993992567,
        2.3955062714405355,
        2.8170822920277714,
        1.286703191883862,
        1.3157591519877314,
        0.805879162158817,
        0.7839958794414997,
        2.3042631755582987,
        1.9682204369455576,
        2.3806665958836675,
        1.359339108876884,
        0.5608420528471469,
        0.9152007081545889,
        2.1184752423316238,
        2.3971397719345986,
        1.5423159797675907,
        2.812495556566864,
        1.467648403160274,
        2.8246803790330883,
        1.3044348050840198,
        1.5448683612048626,
        0.5614499938674271,
        1.4944199031218885,
        0.21105534862726927,
        0.30902217514812946,
        0.7527489583939314,
        2.7918052961118516,
        1.7002119827084243,
        2.581351701915264,
        2.1000600301660595,
        1.3969845036976039,
        0.4023087115958333,
        2.8320406637154516,
        2.532165024708956,
        2.2966270812787113,
        2.214196345489472,
        2.255464042816311,
        1.7046547468751667,
        1.8234985173679887,
        2.3723101196810603,
        0.5821730083785951,
        0.3487625537440181,
        2.9390339994803067,
        0.22053561126813293,
        2.096375124529004,
        0.22207726845517756,
        1.1326015535742044,
        0.20386636201292277,
        2.499930986948311,
        0.7260333232581615,
        2.5091023378074166,
        1.68924842197448,
        0.5289646126329899,
        0.7814451239071786,
        0.6795401250943541,
        1.25137190092355,
        1.6421773235313593,
        1.1958646707236766,
        0.5184121054597199,
        0.2814402929507196,
        1.8584518791176379,
        0.514463076647371,
        1.5299918283708394,
        2.73848394183442,
        0.9310462987050414,
        1.6403415832668542,
        2.283129009697586,
        1.9139726524241267,
        2.3406217242591083,
        0.5799986317753791,
        2.424311537388712,
        2.2809256152249873,
        1.7689316894859075,
        1.9129802016541362,
        1.640356217045337,
        0.20439755180850627,
        1.933645872119814,
        2.5210379655472934,
        1.9461176801472901,
        2.104459698777646,
        2.3532907919958235,
        2.2476833363994957,
    ];

    /// A three-coefficient design, row-major 10 by 3:
    ///
    /// ```r
    /// set.seed(7); X3 <- cbind(1, rep(0:1, each = 5), round(rnorm(10), 6))
    /// ```
    const X3: [f64; 30] = [
        1.0, 0.0, 2.287247, 1.0, 0.0, -1.196772, 1.0, 0.0, -0.694293, 1.0, 0.0, -0.412293, 1.0,
        0.0, -0.970673, 1.0, 1.0, -0.94728, 1.0, 1.0, 0.748139, 1.0, 1.0, -0.116955, 1.0, 1.0,
        0.152658, 1.0, 1.0, 2.189978,
    ];

    const N_GENES: usize = 20;
    const N_SAMPLES: usize = 10;

    /// `cbind(1, rep(0:1, each = 5))`, row-major.
    fn two_group_design() -> Vec<f64> {
        let mut d = Vec::with_capacity(20);
        for i in 0..10 {
            d.push(1.0);
            d.push(if i < 5 { 0.0 } else { 1.0 });
        }
        d
    }

    /// `cbind(1, rep(0:1, each = 5), rep(1:0, each = 5))`, row-major. Rank two.
    fn rank_deficient_design() -> Vec<f64> {
        let mut d = Vec::with_capacity(30);
        for i in 0..10 {
            d.push(1.0);
            d.push(if i < 5 { 0.0 } else { 1.0 });
            d.push(if i < 5 { 1.0 } else { 0.0 });
        }
        d
    }

    fn assert_weights(got: &[f64], want: &[f64], tolerance: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, max_relative = tolerance);
        }
    }

    ///////////////////////////
    // arrayWeights, no weights //
    ///////////////////////////

    /// `Rscript -e 'suppressMessages(library(limma)); set.seed(1); y <- matrix(rnorm(200), 20, 10); X <- cbind(1, rep(0:1, each=5)); cat(arrayWeights(y, X), "\n")'`
    ///
    /// `method = "auto"` resolves to `"reml"` here, so this is also the default
    /// path of [`ArrayWeightParams`].
    #[test]
    fn test_array_weights_reml_matches_limma() {
        let design = two_group_design();
        let got = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, None).unwrap();
        let want = [
            0.9742805411895777,
            1.026017651831166,
            1.3455650214699084,
            1.0480584447646801,
            0.8463342237363761,
            0.9747809021137841,
            1.4807872631702028,
            0.8161127881463421,
            0.683967423080732,
            1.0402644635138032,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /// `arrayWeights(y, X, method = "genebygene")` on the same fixture.
    #[test]
    fn test_array_weights_gene_by_gene_matches_limma() {
        let design = two_group_design();
        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            ..Default::default()
        };
        let got = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(params)).unwrap();
        let want = [
            0.9855865468275133,
            1.0354449545106663,
            1.3167194637260566,
            0.9892096743664166,
            0.883308082435258,
            0.9875617156255206,
            1.4399727557165392,
            0.8087646791684824,
            0.7206377138806224,
            1.027606206031394,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /// `arrayWeights(y, X, method = "genebygene", prior.n = 5)`. Halving the
    /// prior widens the spread of the weights, which is the point of the knob.
    #[test]
    fn test_array_weights_prior_n_loosens_the_shrinkage() {
        let design = two_group_design();
        let params = ArrayWeightParams::new(ArrayWeightMethod::GeneByGene, 5.0, 50, 1e-5, None, 0);
        let got = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(params)).unwrap();
        let want = [
            0.9753130283019934,
            1.0345246343437244,
            1.4463951214443538,
            0.9581734676776122,
            0.86147110226916,
            0.982408539384646,
            1.6499951468701075,
            0.7624393728516808,
            0.6570573074282369,
            1.0222567198392511,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /// `arrayWeights(y, method = "reml")` and `method = "genebygene"`, that is
    /// an intercept-only design.
    #[test]
    fn test_array_weights_intercept_only_design() {
        let design = vec![1.0; N_SAMPLES];
        let reml = array_weights(&Y, N_GENES, N_SAMPLES, &design, 1, None, None).unwrap();
        let want_reml = [
            1.0039401405346158,
            1.2457559559217304,
            1.2269924234759986,
            0.9319055327134608,
            0.8102086600806064,
            1.2349234755723884,
            1.4595139406006172,
            0.6290074491230409,
            0.8344850279179324,
            0.9122753178175574,
        ];
        assert_weights(&reml, &want_reml, 1e-14);

        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            ..Default::default()
        };
        let gbg = array_weights(&Y, N_GENES, N_SAMPLES, &design, 1, None, Some(params)).unwrap();
        let want_gbg = [
            1.0332474602716566,
            1.2339092930392186,
            1.206630141064309,
            0.8772804188552625,
            0.8376944311823541,
            1.218771000735954,
            1.4282811697295135,
            0.6395457423258586,
            0.8434756288804333,
            0.9419637959843594,
        ];
        assert_weights(&gbg, &want_gbg, 1e-14);
    }

    /// `arrayWeights(y, X3, method = "reml")` and `"genebygene"` on a design
    /// with an intercept, a group indicator and a continuous covariate. Three
    /// coefficients means `Q2` has six columns rather than three, which is the
    /// part of the REML information a two-column design never exercises.
    #[test]
    fn test_array_weights_three_coefficient_design() {
        let reml = array_weights(&Y, N_GENES, N_SAMPLES, &X3, 3, None, None).unwrap();
        let want_reml = [
            1.1123858654314625,
            0.8725880455354021,
            1.3674933744743083,
            1.1067965225608778,
            0.8492159040121154,
            1.1449733538385882,
            1.414815203220406,
            0.6507714947593773,
            0.7456578248308494,
            1.0196730581383637,
        ];
        assert_weights(&reml, &want_reml, 1e-14);

        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            ..Default::default()
        };
        let gbg = array_weights(&Y, N_GENES, N_SAMPLES, &X3, 3, None, Some(params)).unwrap();
        let want_gbg = [
            1.0789676466626188,
            0.9176081655063308,
            1.3301594166920214,
            1.039568426333845,
            0.8938386978754715,
            1.1276155513517505,
            1.388971000002176,
            0.673140420315832,
            0.7647038723639742,
            1.0135966976839954,
        ];
        assert_weights(&gbg, &want_gbg, 1e-14);
    }

    /// A design whose third column is `1 - ` its second is reduced to the first
    /// two, so the answer has to equal the two-column fit exactly. limma does
    /// the same through `QR$pivot[1:QR$rank]`.
    #[test]
    fn test_array_weights_reduces_a_rank_deficient_design() {
        let got = array_weights(
            &Y,
            N_GENES,
            N_SAMPLES,
            &rank_deficient_design(),
            3,
            None,
            None,
        )
        .unwrap();
        let want = [
            0.9742805411895777,
            1.026017651831166,
            1.3455650214699084,
            1.0480584447646801,
            0.8463342237363761,
            0.9747809021137841,
            1.4807872631702028,
            0.8161127881463421,
            0.683967423080732,
            1.0402644635138032,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    //////////////////////////////
    // arrayWeights, with weights //
    //////////////////////////////

    /// `arrayWeights(y, X, weights = W, method = "genebygene")`. This is also
    /// what `method = "auto"` picks once weights are present, and the branch
    /// `voomLmFit` always takes.
    #[test]
    fn test_array_weights_gene_by_gene_with_observation_weights() {
        let design = two_group_design();
        let weights = Recycled::full(W.to_vec(), N_GENES, N_SAMPLES).unwrap();
        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            ..Default::default()
        };
        let got = array_weights(
            &Y,
            N_GENES,
            N_SAMPLES,
            &design,
            2,
            Some(&weights),
            Some(params),
        )
        .unwrap();
        let want = [
            0.9165352875614898,
            1.0605424640617098,
            1.3409356509781636,
            0.9427478239354696,
            0.996662010628232,
            1.0484128636731012,
            1.1908686059553828,
            0.8437343085959492,
            0.7664520368017649,
            1.0113104325677869,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /// `arrayWeights(y, X, weights = W, method = "reml")`, which limma routes to
    /// `.arrayWeightsPrWtsREML` rather than `.arrayWeightsREML`.
    #[test]
    fn test_array_weights_reml_with_observation_weights() {
        let design = two_group_design();
        let weights = Recycled::full(W.to_vec(), N_GENES, N_SAMPLES).unwrap();
        let got = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, Some(&weights), None).unwrap();
        let want = [
            0.9209037806194705,
            1.0645265066532537,
            1.37274621213818,
            0.9947452835222503,
            0.9441035850402303,
            1.040449837341745,
            1.1976947015007404,
            0.8374281704069466,
            0.7330195053976797,
            1.0343728257277731,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /// Both weighted branches on the three-coefficient design.
    #[test]
    fn test_array_weights_weighted_three_coefficient_design() {
        let weights = Recycled::full(W.to_vec(), N_GENES, N_SAMPLES).unwrap();
        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            ..Default::default()
        };
        let gbg =
            array_weights(&Y, N_GENES, N_SAMPLES, &X3, 3, Some(&weights), Some(params)).unwrap();
        let want_gbg = [
            1.1997764476593915,
            0.9528873823812497,
            1.3052954876962655,
            0.9659096989828025,
            0.9271573637545184,
            1.1130783426736774,
            1.078354693852502,
            0.7064691660204454,
            0.8130233115307034,
            1.0853647189535636,
        ];
        assert_weights(&gbg, &want_gbg, 1e-14);

        let reml = array_weights(&Y, N_GENES, N_SAMPLES, &X3, 3, Some(&weights), None).unwrap();
        let want_reml = [
            1.2370489318901068,
            0.9124220797886592,
            1.3534944344493567,
            1.0292176614054709,
            0.8823440119643678,
            1.1423166038779065,
            1.077558416900025,
            0.6755922802424658,
            0.7926814918065541,
            1.093465231695838,
        ];
        assert_weights(&reml, &want_reml, 1e-14);
    }

    /// A zero weight becomes a missing observation and a weight of one, which is
    /// what forces the gene-by-gene sweep down its subsetting branch:
    ///
    /// ```r
    /// W0 <- W; W0[1, 1] <- 0; W0[3, 7] <- 0
    /// arrayWeights(y, X, weights = W0, method = "genebygene")
    /// ```
    #[test]
    fn test_array_weights_zero_weights_become_missing() {
        let design = two_group_design();
        let mut w = W.to_vec();
        w[0] = 0.0;
        w[2 * N_SAMPLES + 6] = 0.0;
        let weights = Recycled::full(w, N_GENES, N_SAMPLES).unwrap();
        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            ..Default::default()
        };
        let got = array_weights(
            &Y,
            N_GENES,
            N_SAMPLES,
            &design,
            2,
            Some(&weights),
            Some(params),
        )
        .unwrap();
        let want = [
            0.9468801432504194,
            1.0719407282350122,
            1.3187893121618048,
            0.987724517154794,
            0.9497467768369868,
            1.047094852664723,
            1.1467989187355798,
            0.8571863121305534,
            0.7667221536949717,
            1.0090869707631613,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /// Missing expression values, reached without any weights at all:
    ///
    /// ```r
    /// yNA <- y; yNA[2, 3] <- NA; yNA[5, 9] <- NA
    /// arrayWeights(yNA, X, method = "genebygene")
    /// ```
    #[test]
    fn test_array_weights_handles_missing_values() {
        let design = two_group_design();
        let mut y = Y.to_vec();
        y[N_SAMPLES + 2] = f64::NAN;
        y[4 * N_SAMPLES + 8] = f64::NAN;
        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            ..Default::default()
        };
        let got = array_weights(&y, N_GENES, N_SAMPLES, &design, 2, None, Some(params)).unwrap();
        let want = [
            0.9844172815741963,
            1.0605350684234676,
            1.3055898280414249,
            0.9877067956763591,
            0.8794361790533194,
            0.9842652252777853,
            1.4500921121935295,
            0.8113169367939767,
            0.6976973359698356,
            1.0454211225192833,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /// A non-finite value also pushes the REML sweep to drop that gene, as
    /// limma does before calling `.arrayWeightsREML`. Dropping gene 1 leaves 19
    /// genes, so the answer moves; the assertion is that it stays finite,
    /// normalised and different from the complete-matrix fit.
    #[test]
    fn test_array_weights_reml_drops_incomplete_genes() {
        let design = two_group_design();
        let mut y = Y.to_vec();
        y[3] = f64::INFINITY;
        let got = array_weights(&y, N_GENES, N_SAMPLES, &design, 2, None, None).unwrap();
        assert!(got.iter().all(|v| v.is_finite() && *v > 0.0));
        let full = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, None).unwrap();
        assert!(
            got.iter()
                .zip(full.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    ///////////////////////////
    // arrayWeights, var.design //
    ///////////////////////////

    /// A two-level variance design pools the samples into two weights:
    ///
    /// ```r
    /// VD <- cbind(rep(c(1, -1), each = 5))
    /// arrayWeights(y, X, var.design = VD, method = "reml")
    /// arrayWeights(y, X, var.design = VD, method = "genebygene")
    /// ```
    #[test]
    fn test_array_weights_with_a_var_design() {
        let design = two_group_design();
        let var_design: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { -1.0 }).collect();

        let params = ArrayWeightParams {
            var_design: Some(var_design.clone()),
            n_var_coef: 1,
            ..Default::default()
        };
        let reml = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(params)).unwrap();
        let want_reml = [
            1.0380239930063324,
            1.0380239930063324,
            1.0380239930063324,
            1.0380239930063324,
            1.0380239930063324,
            0.9633688688676578,
            0.9633688688676578,
            0.9633688688676578,
            0.9633688688676578,
            0.9633688688676578,
        ];
        assert_weights(&reml, &want_reml, 1e-14);

        let params = ArrayWeightParams {
            method: ArrayWeightMethod::GeneByGene,
            var_design: Some(var_design),
            n_var_coef: 1,
            ..Default::default()
        };
        let gbg = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(params)).unwrap();
        let want_gbg = [
            1.035442258871988,
            1.035442258871988,
            1.035442258871988,
            1.035442258871988,
            1.035442258871988,
            0.9657708978281427,
            0.9657708978281427,
            0.9657708978281427,
            0.9657708978281427,
            0.9657708978281427,
        ];
        assert_weights(&gbg, &want_gbg, 1e-14);
    }

    /// Centring kills an intercept column in the variance design and the rank
    /// reduction then drops it, so `cbind(1, g)` must give the same answer as
    /// `cbind(g)`:
    ///
    /// ```r
    /// arrayWeights(y, X, var.design = cbind(1, rep(c(1, -1), each = 5)), method = "reml")
    /// ```
    #[test]
    fn test_var_design_intercept_column_is_dropped() {
        let design = two_group_design();
        let mut var_design = Vec::with_capacity(20);
        for i in 0..10 {
            var_design.push(1.0);
            var_design.push(if i < 5 { 1.0 } else { -1.0 });
        }
        let params = ArrayWeightParams {
            var_design: Some(var_design),
            n_var_coef: 2,
            ..Default::default()
        };
        let got = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(params)).unwrap();
        let want = [
            1.0380239930063324,
            1.0380239930063324,
            1.0380239930063324,
            1.0380239930063324,
            1.0380239930063324,
            0.9633688688676578,
            0.9633688688676578,
            0.9633688688676578,
            0.9633688688676578,
            0.9633688688676578,
        ];
        assert_weights(&got, &want, 1e-14);
    }

    /////////////////////////////
    // arrayWeights, early exits //
    /////////////////////////////

    #[test]
    fn test_array_weights_returns_ones_for_a_single_gene() {
        let design = two_group_design();
        let got = array_weights(&Y[..N_SAMPLES], 1, N_SAMPLES, &design, 2, None, None).unwrap();
        assert_eq!(got, vec![1.0; N_SAMPLES]);
    }

    /// Three samples against a two-column design leaves one residual degree of
    /// freedom, which limma refuses to work with.
    #[test]
    fn test_array_weights_returns_ones_without_residual_df() {
        let y: Vec<f64> = (0..12).map(|i| i as f64 * 0.37).collect();
        let design = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let got = array_weights(&y, 4, 3, &design, 2, None, None).unwrap();
        assert_eq!(got, vec![1.0; 3]);
    }

    ///////////////////////////
    // arrayWeights, errors   //
    ///////////////////////////

    #[test]
    fn test_array_weights_rejects_empty_input() {
        let err = array_weights(&[], 0, 4, &[1.0; 4], 1, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
        let err = array_weights(&[], 4, 0, &[1.0; 4], 1, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    #[test]
    fn test_array_weights_rejects_a_shape_mismatch() {
        let err = array_weights(&[1.0; 7], 2, 4, &[1.0; 4], 1, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { name: "y", .. }));
        let err = array_weights(&[1.0; 8], 2, 4, &[1.0; 3], 1, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "design", .. }
        ));
    }

    #[test]
    fn test_array_weights_rejects_a_zero_coefficient_design() {
        let err = array_weights(&[1.0; 8], 2, 4, &[], 0, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }

    #[test]
    fn test_array_weights_rejects_bad_weights() {
        let design = two_group_design();
        let mut w = W.to_vec();
        w[3] = -1.0;
        let weights = Recycled::full(w, N_GENES, N_SAMPLES).unwrap();
        let err =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, Some(&weights), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));

        let mut w = W.to_vec();
        w[3] = f64::INFINITY;
        let weights = Recycled::full(w, N_GENES, N_SAMPLES).unwrap();
        let err =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, Some(&weights), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));

        let weights = Recycled::by_sample(vec![1.0; 3]);
        let err =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, Some(&weights), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    #[test]
    fn test_array_weights_rejects_bad_parameters() {
        let design = two_group_design();
        let bad_prior = ArrayWeightParams {
            prior_n: -1.0,
            ..Default::default()
        };
        let err =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(bad_prior)).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));

        let bad_tol = ArrayWeightParams {
            tol: 0.0,
            ..Default::default()
        };
        let err =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(bad_tol)).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));

        let bad_iter = ArrayWeightParams {
            max_iter: 0,
            ..Default::default()
        };
        let err =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(bad_iter)).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }

    #[test]
    fn test_array_weights_rejects_a_bad_var_design() {
        let design = two_group_design();
        let short = ArrayWeightParams {
            var_design: Some(vec![1.0; 7]),
            n_var_coef: 1,
            ..Default::default()
        };
        let err = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(short)).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "var_design",
                ..
            }
        ));

        let zero_cols = ArrayWeightParams {
            var_design: Some(vec![1.0; 10]),
            n_var_coef: 0,
            ..Default::default()
        };
        let err =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(zero_cols)).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }

    /// A variance design that centres away to nothing leaves no coefficients to
    /// estimate, so the weights stay at one.
    #[test]
    fn test_array_weights_constant_var_design_gives_unit_weights() {
        let design = two_group_design();
        let params = ArrayWeightParams {
            var_design: Some(vec![1.0; 10]),
            n_var_coef: 1,
            ..Default::default()
        };
        let got = array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, None, Some(params)).unwrap();
        assert_eq!(got, vec![1.0; N_SAMPLES]);
    }

    /// The parallel reductions are chunked in index order, so repeated calls
    /// have to agree bit for bit.
    #[test]
    fn test_array_weights_is_reproducible() {
        let design = two_group_design();
        let weights = Recycled::full(W.to_vec(), N_GENES, N_SAMPLES).unwrap();
        let first =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, Some(&weights), None).unwrap();
        let second =
            array_weights(&Y, N_GENES, N_SAMPLES, &design, 2, Some(&weights), None).unwrap();
        assert_eq!(first, second);
    }

    ////////////////////////////
    // duplicateCorrelation    //
    ////////////////////////////

    /// `Rscript -e 'suppressMessages(library(limma)); set.seed(1); y <- matrix(rnorm(200), 20, 10); X <- cbind(1, rep(0:1, each=5)); b <- rep(1:5, 2); cat(duplicateCorrelation(y, X, block=b)$consensus.correlation, "\n")'`
    #[test]
    fn test_duplicate_correlation_balanced_blocks() {
        let design = two_group_design();
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let got =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.15).unwrap();
        assert_relative_eq!(got, 0.07936899121798541, max_relative = 1e-14);
    }

    /// The same fixture with the trimming turned off, `trim = 0`.
    #[test]
    fn test_duplicate_correlation_untrimmed() {
        let design = two_group_design();
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let got =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.0).unwrap();
        assert_relative_eq!(got, 0.05971830681411544, max_relative = 1e-14);
    }

    /// `block = c(1,1,1,2,2,3,3,3,4,4)`, so the blocks are of size 3, 2, 3 and
    /// 2. The lower clip becomes `1 / (1 - 3) + 0.01`, not `-0.99`.
    #[test]
    fn test_duplicate_correlation_unbalanced_blocks() {
        let design = two_group_design();
        let block = vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 3];
        let trimmed =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.15).unwrap();
        assert_relative_eq!(trimmed, -0.15040825061556953, max_relative = 1e-14);

        let untrimmed =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.0).unwrap();
        assert_relative_eq!(untrimmed, -0.09752275604037172, max_relative = 1e-14);
    }

    /// `duplicateCorrelation(y, X, block = b, weights = W)`, balanced and
    /// unbalanced. `mixedModel2Fit` scales the response and the fixed effects by
    /// `sqrt(w)` but leaves the block design alone, which is the detail this
    /// pins.
    #[test]
    fn test_duplicate_correlation_with_weights() {
        let design = two_group_design();
        let weights = Recycled::full(W.to_vec(), N_GENES, N_SAMPLES).unwrap();

        let balanced: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let got = duplicate_correlation(
            &Y,
            N_GENES,
            N_SAMPLES,
            &design,
            2,
            &balanced,
            Some(&weights),
            0.15,
        )
        .unwrap();
        assert_relative_eq!(got, 0.04173698661489739, max_relative = 1e-14);

        let unbalanced = vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 3];
        let got = duplicate_correlation(
            &Y,
            N_GENES,
            N_SAMPLES,
            &design,
            2,
            &unbalanced,
            Some(&weights),
            0.15,
        )
        .unwrap();
        assert_relative_eq!(got, -0.1610960088669458, max_relative = 1e-14);
    }

    /// `duplicateCorrelation(y, X3, block = b)`, three fixed effects.
    #[test]
    fn test_duplicate_correlation_three_coefficient_design() {
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let got =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &X3, 3, &block, None, 0.15).unwrap();
        assert_relative_eq!(got, 0.15590493686350534, max_relative = 1e-14);
    }

    /// `duplicateCorrelation(y, matrix(1, 10, 1), block = b)`.
    #[test]
    fn test_duplicate_correlation_intercept_only() {
        let design = vec![1.0; N_SAMPLES];
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let got =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 1, &block, None, 0.15).unwrap();
        assert_relative_eq!(got, 0.031669365881093696, max_relative = 1e-14);
    }

    /// Every per-gene correlation, on the Fisher scale, against
    /// `duplicateCorrelation(y, X, block = b)$atanh.correlations`. The consensus
    /// is a trimmed mean, so it can hide a gene that is wrong in the tail;
    /// this cannot.
    #[test]
    fn test_duplicate_correlation_per_gene_values() {
        let design = two_group_design();
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let want = [
            0.908548407046912,
            0.21099075616381313,
            0.3440500133251655,
            0.4185794031987016,
            0.9957853699803254,
            0.0565836402590664,
            0.9173034946434048,
            -0.19437110982329026,
            0.787175958405252,
            -0.2506510886181334,
            -0.20227495727455905,
            0.2062792827285404,
            -1.2213369921806791,
            0.3954884812282991,
            -0.7169799798471564,
            -0.05498686558164328,
            -0.8010392798825152,
            0.23348428595384552,
            -0.4464403717163369,
            -0.3903994527358152,
        ];
        // One gene at a time reproduces the per-gene value, since a single gene
        // makes the trimmed mean the identity.
        for (gene, expected) in want.iter().enumerate() {
            let row = &Y[gene * N_SAMPLES..(gene + 1) * N_SAMPLES];
            let got =
                duplicate_correlation(row, 1, N_SAMPLES, &design, 2, &block, None, 0.0).unwrap();
            assert_relative_eq!(got.atanh(), expected, max_relative = 1e-13);
        }
    }

    /// Blocks of size one carry no within-block replication, so limma warns and
    /// returns exactly zero.
    #[test]
    fn test_duplicate_correlation_singleton_blocks_are_zero() {
        let design = two_group_design();
        let block: Vec<usize> = (0..10).collect();
        let got =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.15).unwrap();
        assert_eq!(got, 0.0);
    }

    /// A block factor that is already a column of the design has been absorbed
    /// into the fixed effects, so again limma returns exactly zero.
    #[test]
    fn test_duplicate_correlation_absorbed_block_is_zero() {
        let design = two_group_design();
        let block: Vec<usize> = (0..10).map(|i| usize::from(i >= 5)).collect();
        let got =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.15).unwrap();
        assert_eq!(got, 0.0);
    }

    #[test]
    fn test_duplicate_correlation_rejects_a_bad_block() {
        let design = two_group_design();
        let err = duplicate_correlation(
            &Y,
            N_GENES,
            N_SAMPLES,
            &design,
            2,
            &[0, 0, 1, 1],
            None,
            0.15,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "block", .. }
        ));
    }

    #[test]
    fn test_duplicate_correlation_rejects_a_bad_trim() {
        let design = two_group_design();
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        for trim in [-0.1, 0.5, f64::NAN] {
            let err = duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, trim)
                .unwrap_err();
            assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
        }
    }

    #[test]
    fn test_duplicate_correlation_rejects_a_shape_mismatch() {
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let err = duplicate_correlation(
            &Y,
            N_GENES,
            N_SAMPLES,
            &two_group_design()[..18],
            2,
            &block,
            None,
            0.15,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "design", .. }
        ));
        let err = duplicate_correlation(&[], 0, 10, &two_group_design(), 2, &block, None, 0.15)
            .unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    /// Every gene fails the admission rules, so there is nothing to average and
    /// a number would be a lie.
    #[test]
    fn test_duplicate_correlation_errors_when_no_gene_qualifies() {
        // Four samples against a two-column design: `n_obs > n_coef + 2` fails.
        let y = vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.9, 0.1, -0.2];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let block = vec![0, 1, 0, 1];
        let err = duplicate_correlation(&y, 2, 4, &design, 2, &block, None, 0.15).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_duplicate_correlation_is_reproducible() {
        let design = two_group_design();
        let block: Vec<usize> = (0..10).map(|i| i % 5).collect();
        let first =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.15).unwrap();
        let second =
            duplicate_correlation(&Y, N_GENES, N_SAMPLES, &design, 2, &block, None, 0.15).unwrap();
        assert_eq!(first, second);
    }

    /////////////////////////
    // Internal machinery   //
    /////////////////////////

    /// The QR has to reproduce R's, so its leverages must match
    /// `hat(qr(X))` and its rank must match `qr(X)$rank`.
    #[test]
    fn test_linpack_qr_leverages_and_rank() {
        // Column-major two-group design: each group of five gives a leverage of
        // 1/5, and the leverages sum to the rank.
        let mut a = vec![1.0; 10];
        a.extend((0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }));
        let qr = LinpackQr::new(a, 10, 2);
        assert_eq!(qr.rank(), 2);
        let hat = qr.hat();
        for h in &hat {
            assert_relative_eq!(h, &0.2, max_relative = 1e-14);
        }
        assert_relative_eq!(hat.iter().sum::<f64>(), 2.0, max_relative = 1e-14);
    }

    /// A duplicated column is dropped, and the survivors keep their original
    /// order rather than being sorted by magnitude.
    #[test]
    fn test_linpack_qr_drops_dependent_columns_in_place() {
        let mut a = vec![1.0; 6];
        a.extend([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        a.extend([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let qr = LinpackQr::new(a, 6, 3);
        assert_eq!(qr.rank(), 2);
        assert_eq!(qr.pivot, vec![0, 1]);
    }

    /// `Q' Q = I` on the leading columns, which is what every downstream
    /// projection assumes.
    #[test]
    fn test_linpack_qr_leading_q_is_orthonormal() {
        let a = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 0.3, -1.2, 0.7, 2.1, -0.4, 1.7, 0.2, -0.9, 0.5, 1.1,
        ];
        let qr = LinpackQr::new(a, 5, 3);
        let q = qr.leading_q(3);
        for a in 0..3 {
            for b in 0..3 {
                let dot: f64 = (0..5).map(|i| q[a * 5 + i] * q[b * 5 + i]).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert_relative_eq!(dot, want, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_solve_linear_round_trips() {
        // Column-major [[4, 1], [1, 3]].
        let a = [4.0, 1.0, 1.0, 3.0];
        let x = solve_linear(&a, &[1.0, 2.0], 2).unwrap();
        assert_relative_eq!(x[0], 1.0 / 11.0, max_relative = 1e-14);
        assert_relative_eq!(x[1], 7.0 / 11.0, max_relative = 1e-14);
    }

    #[test]
    fn test_solve_linear_rejects_a_singular_system() {
        let a = [1.0, 2.0, 2.0, 4.0];
        assert!(matches!(
            solve_linear(&a, &[1.0, 1.0], 2),
            Err(EdgeErrors::SolveFailed(_))
        ));
    }

    /// `contr.sum(4)` in R, column-major.
    #[test]
    fn test_contr_sum_matches_r() {
        let z = contr_sum(4);
        assert_eq!(
            z,
            vec![
                1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0
            ]
        );
    }
}
