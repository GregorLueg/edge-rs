//! The crate-level error enum.
//!
//! One `thiserror` enum for all of `edge-rs`. Variants are grouped by subsystem
//! with section comments; add new ones to the matching section rather than the
//! bottom, so the file stays navigable as the port grows.
//!
//! Anything a caller could plausibly hit returns `Err` rather than panicking.
//! Panics are reserved for broken invariants that the type system cannot state.

use thiserror::Error;

/// Every failure mode in the crate.
#[derive(Debug, Error)]
pub enum EdgeErrors {
    // -- arguments --
    /// A parameter fell outside its documented domain.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// A count, size or index that must be non-zero was zero.
    #[error("Parameter '{0}' must be a positive non-zero integer.")]
    MustBePositive(String),

    /// A named method string did not match any known variant.
    #[error("Unknown {kind} method '{name}'. Expected one of: {expected}.")]
    UnknownMethod {
        /// What kind of method was being selected, e.g. "normalisation"
        kind: &'static str,
        /// The string the caller supplied
        name: String,
        /// Comma-separated list of accepted values
        expected: &'static str,
    },

    // -- dimensions --
    /// Two operands disagreed on shape.
    #[error("Expected shape {expected:?}; got {got:?}.")]
    ShapeMismatch {
        /// Shape that was required
        expected: (usize, usize),
        /// Shape that was supplied
        got: (usize, usize),
    },

    /// A vector was not the length its consumer required.
    #[error("'{name}' has length {got}, expected {expected}.")]
    LengthMismatch {
        /// Name of the offending argument
        name: &'static str,
        /// Length that was required
        expected: usize,
        /// Length that was supplied
        got: usize,
    },

    /// The counts matrix had no genes or no samples.
    ///
    /// Every downstream estimator divides by one of these, so an empty input is
    /// an error rather than a trivially empty result.
    #[error("Counts matrix is empty: {n_genes} genes by {n_samples} samples.")]
    EmptyCounts {
        /// Number of genes (rows)
        n_genes: usize,
        /// Number of samples (columns)
        n_samples: usize,
    },

    // -- design matrices --
    /// The design matrix is rank deficient, so coefficients are not identifiable.
    #[error("Design matrix is not full rank: {n_cols} columns but rank {rank}.")]
    DesignNotFullRank {
        /// Number of columns in the design
        n_cols: usize,
        /// Numerical rank actually found
        rank: usize,
    },

    /// A contrast or coefficient index pointed outside the design.
    #[error("Coefficient index {index} is out of range for a design with {n_coef} coefficients.")]
    CoefOutOfRange {
        /// Index that was requested
        index: usize,
        /// Number of coefficients available
        n_coef: usize,
    },

    // -- linear algebra --
    /// A Cholesky factorisation failed, meaning the matrix was not positive definite.
    #[error("Cholesky factorisation failed: {0}")]
    CholeskyFailed(String),

    /// A linear solve produced a non-finite result.
    #[error("Linear solve failed: {0}")]
    SolveFailed(String),

    // -- numeric support --
    /// A scalar optimiser or root finder did not converge inside its iteration budget.
    #[error(
        "{routine} failed to converge after {iterations} iterations (last change {last_delta:e})."
    )]
    NoConvergence {
        /// Name of the routine that gave up
        routine: &'static str,
        /// Iterations actually performed
        iterations: usize,
        /// Magnitude of the final update
        last_delta: f64,
    },

    /// A bracketing root finder was handed an interval that does not bracket a root.
    #[error(
        "Interval [{lower}, {upper}] does not bracket a root: f(lower) = {f_lower:e}, f(upper) = {f_upper:e}."
    )]
    RootNotBracketed {
        /// Lower end of the interval
        lower: f64,
        /// Upper end of the interval
        upper: f64,
        /// Function value at `lower`
        f_lower: f64,
        /// Function value at `upper`
        f_upper: f64,
    },

    /// A bounded optimisation was given a box with a lower bound above its upper bound.
    #[error("Invalid bounds for parameter {index}: lower {lower} exceeds upper {upper}.")]
    InvalidBounds {
        /// Index of the offending parameter
        index: usize,
        /// Lower bound supplied
        lower: f64,
        /// Upper bound supplied
        upper: f64,
    },

    /// An interpolation was asked for a point outside the knot range, where the
    /// spline is not defined.
    #[error("Interpolation point {x} lies outside the knot range [{lower}, {upper}].")]
    OutsideKnotRange {
        /// Requested abscissa
        x: f64,
        /// Smallest knot
        lower: f64,
        /// Largest knot
        upper: f64,
    },

    // -- sparse --
    /// A compressed sparse structure violated its own invariants.
    #[error("Malformed sparse matrix: {0}")]
    MalformedSparse(String),

    // -- normalisation --
    /// Every gene was filtered out before a normalisation factor could be computed.
    #[error("No genes survived filtering for normalisation; all {n_genes} were dropped.")]
    NoGenesAfterFiltering {
        /// Number of genes present before filtering
        n_genes: usize,
    },

    /// The reference column for TMM could not be chosen or was out of range.
    #[error("Invalid TMM reference column: {0}")]
    InvalidReferenceColumn(String),

    // -- GLM --
    /// The negative binomial GLM did not converge for the given gene.
    #[error("GLM fit failed to converge for gene {gene} after {iterations} iterations.")]
    GlmNoConvergence {
        /// Index of the gene that failed
        gene: usize,
        /// Iterations performed
        iterations: usize,
    },

    /// A dispersion outside `[0, inf)` was supplied to a negative binomial routine.
    #[error("Dispersion must be non-negative and finite; got {0}.")]
    InvalidDispersion(f64),

    // -- single cell / NEBULA --
    /// Cells were not grouped contiguously by subject, which every inner loop assumes.
    #[error(
        "Cells are not contiguous by subject: subject {subject} appears in more than one block."
    )]
    SubjectsNotContiguous {
        /// Identifier of the subject found in multiple blocks
        subject: usize,
    },

    /// The design matrix handed to NEBULA has no intercept column.
    ///
    /// Centring and the starting values both key off the intercept, so its
    /// absence is a hard error rather than something to work around.
    #[error("NEBULA requires an intercept column in the design matrix; none was found.")]
    MissingIntercept,

    /// Fewer subjects than the mixed model can identify a random effect from.
    #[error("NEBULA needs at least {required} subjects; got {got}.")]
    TooFewSubjects {
        /// Minimum number of subjects needed
        required: usize,
        /// Number of subjects supplied
        got: usize,
    },
}
