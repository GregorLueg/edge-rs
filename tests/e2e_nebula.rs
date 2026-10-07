//! End-to-end parity for the single-cell chain: `nebula`, then the Wald test on
//! its output, then the dispersion shrinkage.
//!
//! Two realistic datasets, either side of the thirty-cells-per-subject threshold
//! that decides between NEBULA-LN and NEBULA-HL:
//!
//! * `sc`, 300 genes over 1005 cells and 15 subjects (67 cells per subject).
//!   `method = "LN"` survives and all three of nebula's sub-algorithms appear.
//! * `sc_small`, 120 genes over 300 cells and 15 subjects (20 per subject).
//!   `method = "LN"` is silently downgraded and every gene takes the HL path.
//!
//! Five small sets (thirty genes each) target corners of the GPU kernel. All but
//! `sc_blocks` run HL, which sends every gene through stage two and so onto the
//! device:
//!
//! * `sc_k2`: two subjects of 23 and 41 cells. Every gene pins `sigma^2` on its
//!   lower bound, in R as here.
//! * `sc_blocks`: eight subjects of 5, 31, 32, 33, 127, 128, 129 and 160 cells,
//!   either side of a 32-lane plane and of the four-plane unroll. LN, with all
//!   three sub-algorithms, plus planted genes: one silent in a subject, two on
//!   the third-order Laplace path, one at `mincp` and one filtered below it.
//! * `sc_intercept`: intercept only, no offset.
//! * `sc_wide`: eight columns (the kernel's cap), 40 subjects of 12 cells.
//! * `sc_high`: means of `1e3` to `1e4` on offsets around `5e3`.
//!
//! Three more have a categorical cell type in place of the continuous
//! cell-level column, so the CPU kernels sum the zero counts from tables:
//!
//! * `sc_cat_ln`: LN over eight subjects of 60 to 400 cells, offsets over two
//!   orders of magnitude, the planted genes.
//! * `sc_cat_hl`: HL, one subject fifteen times the smallest, `1 / size` of 3
//!   to 10, and a rare type in three subjects whose groups are too small to
//!   table, so tabled and swept cells mix.
//! * `sc_cat_sparse`: HL at means low enough for the third-order Laplace
//!   correction.
//!
//! The in-crate golden is eight genes at 25 cells per subject: it reaches HL
//! only, and LN only through a relabelling trick on the same 150 cells.
//!
//! Tolerances follow `src/sc/nebula.rs`: nebula stops its optimiser on a profile
//! likelihood that is discontinuous at the stopping tolerance, so two of its own
//! runs from different starting points disagree by as much. The values here are
//! one to two decades looser than that module's, measured on these fixtures.

mod common;

use std::sync::OnceLock;

use common::{Tol, assert_close, assert_close_scalar, assert_eq_usize};

use edge_rs::sc::nebula::{CONV_OUTER_FAILED, NebulaFit, NebulaMethod, NebulaParams, nebula};
use edge_rs::sc::shrink::{sc_residual_df, shrink_sc_dispersion};
use edge_rs::sc::test::{ScTested, glm_sc_test, packed_len};

////////////////
// Tolerances //
////////////////

/// Coefficients on the pure paths. Needs `7.4e-6`, against `1.0e-6` in the
/// in-crate eight-gene golden.
const TOL_COEF: Tol = Tol::new(3e-5, 1e-9);

/// Standard errors. Needs `2.3e-5`.
const TOL_SE: Tol = Tol::rel(1e-4);

/// Covariance entries, which are standard errors squared and so carry twice the
/// relative error. Needs `2.3e-4`.
const TOL_COV: Tol = Tol::new(1e-3, 1e-12);

/// Subject-level overdispersion. The profile likelihood is flatter in this
/// direction than in the cell-level one. Needs `3.8e-5`.
const TOL_SUBJECT: Tol = Tol::rel(2e-4);

/// Cell-level overdispersion. Needs `2.2e-5`.
const TOL_CELL: Tol = Tol::rel(1e-4);

/// Wald p-values. The normal tail falls like `exp(-z^2/2)`, so the standard
/// error's relative error is amplified by roughly `z^2`: at `p = 1e-16`, `z = 8.2`
/// and a `2e-5` error in the SE arrives as `1.4e-3`. Needs `1.5e-3`.
const TOL_P_VALUE: Tol = Tol::new(1e-2, 1e-300);

// The LN+HL path is gated apart because the two optimisers pick different basins.
//
// nebula picks one of three sub-paths per gene. Worst relative disagreement on
// the coefficients, 1005-cell set:
//
//   NBGMM (LN)      11 genes    3.0e-6
//   NBGMM (HL)       6 genes    3.8e-7      (118 genes at 7.4e-6 on sc_small)
//   NBGMM (LN+HL)  281 genes    5.9e-1
//
// The LN+HL refit itself is faithful: its objective agrees with
// `nebula:::pql_gamma_ll` at identical `(sigma, phi)` to `6e-10` absolute,
// `2e-12` relative, on three affected genes over an eight-point grid.
//
// The divergence is in stage one, the joint optimisation over
// `[beta, sigma, phi]`. R runs nlopt's `LD_LBFGS`, this crate L-BFGS-B, and on
// 37 of the 281 genes they finish in different places. LN+HL then holds stage
// one's `phi` fixed and refits only `sigma`, so the choice propagates into both
// variance components and, through the `-sigma/2` intercept shift, into the
// coefficients.
//
// R's endpoint is the lower bound on `sigma`, where the marginal likelihood is
// still strictly decreasing: lowering the bound from `1e-4` to `1e-8` pulls 13
// of the 19 genes on it straight down. It is a KKT point of the box constraint,
// reached by sliding onto the wall. This crate stops at the genuine interior
// stationary point; a second basin sits around `0.03` to `0.15`, behind a ridge
// near `0.01`.
//
// Both are converged, but R's lower NLL on 27 of the 37 (this crate's 9) is not
// an endorsement: sigma on the bound means no subject variance, i.e. a plain NB
// GLM, in an approximation LN+HL exists to distrust. Against nebula's own
// `method = "HL"` (both variance components on the profile likelihood), this
// crate is closer on 29 of the 37 for `sigma`, 29 for `phi`, 32 for the SEs and
// 32 for the p-values; worst-case `sigma` error over all 281 genes is 15.8% here
// against 84.4% for R. Multi-start taking the lowest NLL is counterproductive:
// it lands on the bound on 30 of the 37, moves away from HL, and costs +81%
// runtime.
//
// R also disagrees with itself: `opt = "trust"` moves the subject
// overdispersion by more than 1% on 78 of the 281 genes, twice the 37 where this
// crate differs from `opt = "lbfgs"`, and on those 37 it lands on this crate's
// answer (22 agree to `1e-6`, 32 to `1e-4`; none agree with `lbfgs` to `1e-6`).
// Matching the fixture would mean reproducing nlopt's LD_LBFGS trajectory,
// including the overshoot clipped onto the `sigma` bound that carries it over
// the ridge.
//
// `gene_id` 299 is not stage one: both optimisers agree on `[beta, sigma, phi]`
// to `3e-7`, and R's `nlminb` inside the refit returns the lower bound `1e-4`
// with `convergence == 0` where its own `pql_gamma_ll` is still falling.
//
// The two constants below are gated on their absolute legs. The comparator
// normalises by `max(|a|, |b|)`, so a relative difference cannot exceed 2 and
// any `max_relative` at or above that can never fail.

/// Coefficients on the LN+HL path. Needs `0` beyond a `1e-2` absolute floor; the
/// worst absolute disagreement across all 281 genes is `4.4e-3` in log fold
/// change. The `5.9e-1` relative figure is a coefficient of magnitude `8.6e-4`.
const LN_HL_COEF: Tol = Tol::new(1e-5, 1e-2);

/// Subject-level overdispersion on the LN+HL path. Needs `7.9e-2` beyond a
/// `1e-3` absolute floor.
const LN_HL_SUBJECT: Tol = Tol::new(2.5e-1, 1e-3);

/// Shrunk dispersions from `shrink_sc_dispersion`, against limma's `squeezeVar`
/// on the same inputs. Needs `7.7e-14`.
///
/// The tightest tolerance in the file: the test feeds the crate R's own cell
/// overdispersions, so only the empirical Bayes step is compared.
/// `shrink_sc_dispersion` has no upstream of its own (edgePython's invention), so
/// limma applied directly is the reference.
const TOL_SHRINK: Tol = Tol::rel(1e-12);

/// Prior degrees of freedom from that shrinkage. Needs `8.3e-13`.
const TOL_SHRINK_DF: Tol = Tol::rel(1e-11);

/// Every tolerance the R-parity check applies to one NEBULA fit.
#[derive(Clone, Copy)]
struct NebulaTols {
    /// Coefficients on the pure paths.
    coef: Tol,
    /// Standard errors on the pure paths.
    se: Tol,
    /// Covariance entries on the pure paths.
    cov: Tol,
    /// Subject-level overdispersion on the pure paths.
    subject: Tol,
    /// Cell-level overdispersion on the pure paths.
    cell: Tol,
    /// Coefficients on the mixed LN-then-HL path.
    ln_hl_coef: Tol,
    /// Subject-level overdispersion on the mixed path.
    ln_hl_subject: Tol,
    /// Wald p-values on the pure paths.
    p_value: Tol,
}

/// The CPU path's gates.
const CPU_TOLS: NebulaTols = NebulaTols {
    coef: TOL_COEF,
    se: TOL_SE,
    cov: TOL_COV,
    subject: TOL_SUBJECT,
    cell: TOL_CELL,
    ln_hl_coef: LN_HL_COEF,
    ln_hl_subject: LN_HL_SUBJECT,
    p_value: TOL_P_VALUE,
};

/// A path's gates for one dataset: `base`, loosened only where the dataset
/// measurably needs it. Every loosening is an absolute floor, so the gate can
/// still fail.
///
/// * `sc_blocks`: one pure-LN gene, R's `gene_id` 29, where stage one puts
///   `phi` on its upper bound of 1000 and R stops at 414. Both call the gene
///   near-Poisson. Needs `3.2e-5` on the coefficients, `1.4e-3` absolute on the
///   cell overdispersion, and a p-value `1.2e-2` off at `p = 6e-238`, which the
///   `1e-12` floor absorbs.
/// * `sc_cat_ln`: the same planted near-Poisson gene, `gene_id` 29, where R
///   stops at `phi = 322`. Needs `5.6e-4` on the coefficients and on
///   `sigma^2`, and `2.1e-3` absolute on the cell overdispersion.
/// * `sc_high`: means of `1e3` to `1e4`. Needs `1.7e-4` on the coefficients,
///   `4.4e-4` on `sigma^2` and `2.2e-3` on a covariance entry of `2.7e-7`.
///
/// The `sc_blocks` and `sc_cat_ln` gene is not settled as a port fault or an R
/// one.
///
/// ### Params
///
/// * `tag` - Dataset prefix
/// * `base` - The path's own gates
///
/// ### Returns
///
/// The gates to apply.
fn tols(tag: &str, base: NebulaTols) -> NebulaTols {
    match tag {
        "sc_blocks" => NebulaTols {
            coef: Tol::new(base.coef.max_relative.max(1e-4), base.coef.epsilon),
            cell: Tol::new(base.cell.max_relative, 3e-3),
            p_value: Tol::new(base.p_value.max_relative, 1e-12),
            ..base
        },
        "sc_cat_ln" => NebulaTols {
            coef: Tol::new(base.coef.max_relative.max(1e-3), base.coef.epsilon),
            subject: Tol::new(base.subject.max_relative.max(1e-3), base.subject.epsilon),
            cell: Tol::new(base.cell.max_relative, 3e-3),
            ..base
        },
        "sc_high" => NebulaTols {
            coef: Tol::new(base.coef.max_relative.max(5e-4), base.coef.epsilon),
            subject: Tol::new(base.subject.max_relative.max(5e-4), base.subject.epsilon),
            cov: Tol::new(base.cov.max_relative.max(3e-3), base.cov.epsilon),
            ..base
        },
        _ => base,
    }
}

/////////////
// Loading //
/////////////

/// One single-cell dataset.
struct Dataset {
    /// Fixture prefix.
    tag: &'static str,
    /// The method R was asked for.
    method: NebulaMethod,
    /// Whether LN is expected to survive the cells-per-subject check (read only
    /// when `method` is LN).
    expect_ln: bool,
    /// Whether R was given the offsets in the meta file.
    has_offset: bool,
    /// Coefficient suffixes of the fixture's `logFC_`, `se_` and `p_` columns,
    /// one per design column.
    names: &'static [&'static str],
    /// Whether the design is its own `{tag}_design.csv` rather than built as
    /// `[1, grp, cov2]` from the meta file.
    design_file: bool,
}

/// Coefficient suffixes of the two realistic datasets.
const NAMES_SC: &[&str] = &["int", "grp", "cov2"];

/// Coefficient suffixes of the edge datasets (numbered by R).
const NAMES_EDGE: [&str; 8] = ["1", "2", "3", "4", "5", "6", "7", "8"];

/// Every single-cell dataset. `sc` is first; single-set tests read it by index.
const DATASETS: [Dataset; 10] = [
    Dataset {
        tag: "sc",
        method: NebulaMethod::Ln,
        expect_ln: true,
        has_offset: true,
        names: NAMES_SC,
        design_file: false,
    },
    Dataset {
        tag: "sc_small",
        method: NebulaMethod::Ln,
        expect_ln: false,
        has_offset: true,
        names: NAMES_SC,
        design_file: false,
    },
    Dataset {
        tag: "sc_k2",
        method: NebulaMethod::Hl,
        expect_ln: true,
        has_offset: true,
        names: NAMES_EDGE.split_at(3).0,
        design_file: true,
    },
    Dataset {
        tag: "sc_blocks",
        method: NebulaMethod::Ln,
        expect_ln: true,
        has_offset: true,
        names: NAMES_EDGE.split_at(3).0,
        design_file: true,
    },
    Dataset {
        tag: "sc_intercept",
        method: NebulaMethod::Hl,
        expect_ln: true,
        has_offset: false,
        names: NAMES_EDGE.split_at(1).0,
        design_file: true,
    },
    Dataset {
        tag: "sc_wide",
        method: NebulaMethod::Hl,
        expect_ln: false,
        has_offset: true,
        names: &NAMES_EDGE,
        design_file: true,
    },
    Dataset {
        tag: "sc_high",
        method: NebulaMethod::Hl,
        expect_ln: true,
        has_offset: true,
        names: NAMES_EDGE.split_at(3).0,
        design_file: true,
    },
    Dataset {
        tag: "sc_cat_ln",
        method: NebulaMethod::Ln,
        expect_ln: true,
        has_offset: true,
        names: NAMES_EDGE.split_at(4).0,
        design_file: true,
    },
    Dataset {
        tag: "sc_cat_hl",
        method: NebulaMethod::Hl,
        expect_ln: true,
        has_offset: true,
        names: NAMES_EDGE.split_at(4).0,
        design_file: true,
    },
    Dataset {
        tag: "sc_cat_sparse",
        method: NebulaMethod::Hl,
        expect_ln: true,
        has_offset: true,
        names: NAMES_EDGE.split_at(3).0,
        design_file: true,
    },
];

/// Counts, subject labels, design and offsets for one dataset.
struct Loaded {
    /// Counts, row-major `n_genes * n_cells`.
    counts: Vec<f64>,
    /// Number of genes before filtering.
    n_genes: usize,
    /// Number of cells.
    n_cells: usize,
    /// Subject index per cell, zero-based and contiguous.
    subject: Vec<usize>,
    /// Design, row-major `n_cells * n_coef`, intercept first.
    design: Vec<f64>,
    /// Design columns.
    n_coef: usize,
    /// Per-cell offset on the linear scale, or `None` where R was given none.
    offset: Option<Vec<f64>>,
    /// Number of subjects.
    n_subjects: usize,
}

/// Loads one single-cell dataset.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// The counts and the per-cell covariates.
fn load(d: &Dataset) -> Loaded {
    let counts_t = common::table(&format!("{}_counts.csv", d.tag));
    let n_genes = counts_t.n_rows();
    let n_cells = counts_t.n_cols();
    let counts = counts_t.row_major_counts();

    let meta = common::table(&format!("{}_meta.csv", d.tag));
    assert_eq!(
        meta.n_rows(),
        n_cells,
        "{}: meta rows must match cells",
        d.tag
    );

    // R labels subjects from one; the crate wants them zero-based, and requires
    // them already grouped into contiguous runs.
    let subject: Vec<usize> = meta.column_usize("subject").iter().map(|s| s - 1).collect();
    let n_subjects = subject.iter().max().expect("at least one cell") + 1;

    let n_coef = d.names.len();
    let design = if d.design_file {
        let (design, rows, cols) = common::matrix(&format!("{}_design.csv", d.tag));
        assert_eq!((rows, cols), (n_cells, n_coef), "{}: design shape", d.tag);
        design
    } else {
        let grp = meta.column("grp");
        let cov2 = meta.column("cov2");
        (0..n_cells).flat_map(|c| [1.0, grp[c], cov2[c]]).collect()
    };

    Loaded {
        counts,
        n_genes,
        n_cells,
        subject,
        design,
        n_coef,
        offset: d.has_offset.then(|| meta.column("offset").to_vec()),
        n_subjects,
    }
}

/// The CPU fit of one dataset at its fixture's parameters, computed once per test
/// binary.
///
/// ### Params
///
/// * `i` - Position in [`DATASETS`]
///
/// ### Returns
///
/// The fit.
fn cpu_fit(i: usize) -> &'static NebulaFit {
    static FITS: [OnceLock<NebulaFit>; DATASETS.len()] = [const { OnceLock::new() }; DATASETS.len()];
    FITS[i].get_or_init(|| {
        let d = &DATASETS[i];
        let l = load(d);
        nebula(
            &l.counts,
            l.n_genes,
            l.n_cells,
            &l.subject,
            &l.design,
            l.n_coef,
            l.offset.as_deref(),
            Some(golden_params(d)),
        )
        .expect("nebula failed")
    })
}

/// The parameter set a dataset's golden was generated with.
///
/// `nebula(..., model = "NBGMM", method, covariance = TRUE, ncore = 1)` at
/// every other default.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// The knobs.
fn golden_params(d: &Dataset) -> NebulaParams {
    NebulaParams {
        method: d.method,
        ..NebulaParams::default()
    }
}

/// Reorders one gene's R covariance row into the crate's packing.
///
/// R returns `lower.tri(diag = TRUE)` column-major, so three coefficients read
/// `V11, V21, V31, V22, V32, V33`. The crate packs the upper triangle
/// column-major, entry `(i, j)` with `i <= j` at `j * (j + 1) / 2 + i`:
/// `V11, V12, V22, V13, V23, V33`. The two coincide for two coefficients and
/// diverge from three.
///
/// ### Params
///
/// * `row` - One gene's covariance entries as R wrote them
/// * `n` - Number of coefficients
///
/// ### Returns
///
/// The same values in the crate's order.
fn repack(row: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; packed_len(n)];
    let mut k = 0;
    for j in 0..n {
        for i in j..n {
            // R's `(i, j)` with `i >= j` is the crate's `(j, i)`.
            out[i * (i + 1) / 2 + j] = row[k];
            k += 1;
        }
    }
    out
}

/// Asserts a convergence code against the R package's.
///
/// nebula grades a fit by how often the last Newton step was halved, which at a
/// converged point is settled by rounding, so a gene R calls converged may come
/// back as `CONV_CRITICAL_POINT` here (same leniency as `src/sc/nebula.rs`).
///
/// Where R's outer optimiser failed (`-50`) and this crate converged, the crate
/// is not held to R's failure: `sc_blocks` gene 3, and `sc` gene 299 for R's
/// `nlminb`.
///
/// ### Params
///
/// * `got` - The crate's code
/// * `want` - The R package's code
/// * `gene` - Gene index, for the message
fn assert_convergence(got: i32, want: i32, gene: usize) {
    let equivalent = |c: i32| c == 1 || c == -10;
    if equivalent(want) || (want == CONV_OUTER_FAILED && equivalent(got)) {
        assert!(
            equivalent(got),
            "gene {gene}: convergence {got}, expected 1 or -10"
        );
    } else {
        assert_eq!(got, want, "gene {gene}: convergence");
    }
}

////////////
// nebula //
////////////

#[test]
fn test_nebula_matches_the_r_package() {
    let s = common::scalars();

    for (i, d) in DATASETS.iter().enumerate() {
        let l = load(d);

        let cells_per_subject = l.n_cells as f64 / l.n_subjects as f64;
        assert_eq!(
            l.n_subjects,
            s.get_usize(d.tag, "n_subjects"),
            "{}: subject count",
            d.tag
        );
        if d.method == NebulaMethod::Ln {
            assert_eq!(
                d.expect_ln,
                cells_per_subject >= 30.0,
                "{}: the dataset no longer sits on the intended side of the LN threshold",
                d.tag
            );
        }

        check_against_r(d, d.tag, cpu_fit(i), &s, &tols(d.tag, CPU_TOLS));
    }
}

/// Gates one NEBULA fit against the R package's output for its dataset.
///
/// Shared by the CPU and GPU tests, so both run the same checks and the report
/// shows each under its own label.
///
/// ### Params
///
/// * `d` - The dataset
/// * `label` - Prefix for the tolerance report
/// * `fit` - The fit to check
/// * `s` - The fixture scalars
/// * `tols` - The gates to apply
fn check_against_r(
    d: &Dataset,
    label: &str,
    fit: &NebulaFit,
    s: &common::Scalars,
    tols: &NebulaTols,
) {
    let tag = d.tag;
    let n_coef = d.names.len();
    {
        let want = common::table(&format!("{}_nebula.csv", tag));
        let want_cov = common::table(&format!("{}_covariance.csv", tag));

        // Gene filtering. R reports the surviving genes one-based.
        let want_index: Vec<usize> = want.column_usize("gene_id").iter().map(|g| g - 1).collect();
        assert_eq_usize(&fit.gene_index, &want_index, &format!("{label}/gene_index"));
        assert_eq!(
            fit.gene_index.len(),
            s.get_usize(tag, "n_genes_out"),
            "{}: surviving gene count",
            tag
        );
        assert_eq!(fit.n_coef, n_coef, "{}: coefficient count", tag);

        let n = fit.gene_index.len();
        let algorithm = want.column("algorithm");

        // The three sub-paths agree unequally well, so they are gated separately
        // (see the LN+HL note above).
        let pure: Vec<bool> = (0..n).map(|g| algorithm[g] as usize != 2).collect();
        let mixed: Vec<bool> = (0..n).map(|g| algorithm[g] as usize == 2).collect();

        let pick = |v: &[f64], mask: &[bool], stride: usize| -> Vec<f64> {
            v.chunks_exact(stride)
                .zip(mask)
                .filter(|(_, k)| **k)
                .flat_map(|(r, _)| r.iter().copied())
                .collect()
        };

        let mut want_coef = Vec::with_capacity(n * n_coef);
        let mut want_se = Vec::with_capacity(n * n_coef);
        for g in 0..n {
            for name in d.names {
                want_coef.push(want.column(&format!("logFC_{name}"))[g]);
                want_se.push(want.column(&format!("se_{name}"))[g]);
            }
        }

        // Pure LN and pure HL, where the port is right.
        assert_close(
            &pick(&fit.coefficients, &pure, n_coef),
            &pick(&want_coef, &pure, n_coef),
            tols.coef,
            &format!("{label}/coefficients_pure"),
        );
        assert_close(
            &pick(&fit.se, &pure, n_coef),
            &pick(&want_se, &pure, n_coef),
            tols.se,
            &format!("{label}/se_pure"),
        );
        assert_close(
            &pick(fit.subject_overdispersion.as_slice(), &pure, 1),
            &pick(want.column("Subject"), &pure, 1),
            tols.subject,
            &format!("{label}/subject_overdispersion_pure"),
        );
        assert_close(
            &pick(fit.cell_overdispersion.as_slice(), &pure, 1),
            &pick(want.column("Cell"), &pure, 1),
            tols.cell,
            &format!("{label}/cell_overdispersion_pure"),
        );

        // The mixed path, at the tolerance it currently earns. A recorded gap.
        if mixed.iter().any(|k| *k) {
            assert_close(
                &pick(&fit.coefficients, &mixed, n_coef),
                &pick(&want_coef, &mixed, n_coef),
                tols.ln_hl_coef,
                &format!("{label}/coefficients_ln_hl"),
            );
            assert_close(
                &pick(fit.subject_overdispersion.as_slice(), &mixed, 1),
                &pick(want.column("Subject"), &mixed, 1),
                tols.ln_hl_subject,
                &format!("{label}/subject_overdispersion_ln_hl"),
            );
        }

        let packed = packed_len(n_coef);
        let mut want_packed = Vec::with_capacity(n * packed);
        for g in 0..n {
            let row: Vec<f64> = (1..=packed)
                .map(|k| want_cov.column(&format!("cov_{k}"))[g])
                .collect();
            want_packed.extend(repack(&row, n_coef));
        }
        // This checks the packing, so only the pure genes (whose values agree)
        // are used.
        assert_close(
            &pick(&fit.covariance, &pure, packed),
            &pick(&want_packed, &pure, packed),
            tols.cov,
            &format!("{label}/covariance_pure"),
        );

        for g in 0..n {
            assert_convergence(fit.convergence[g], want.column("convergence")[g] as i32, g);
        }
    }
}

#[test]
fn test_sigma_at_bound_marks_the_collapsed_fits() {
    // The flag is the only signal that a mixed model has no random effect:
    // nebula's `check_conv` tests the upper bound on `sigma^2`, not the lower, so
    // these genes report convergence. Checked against R's `Subject` column: a
    // gene is pinned there exactly when it is pinned here.
    let s = common::scalars();
    let mut total = 0;
    for (i, d) in DATASETS.iter().enumerate() {
        let want = common::table(&format!("{}_nebula.csv", d.tag));
        let fit = cpu_fit(i);

        let floor = NebulaParams::default().min.0;
        let r_subject = want.column("Subject");
        let n = fit.gene_index.len();
        assert_eq!(fit.sigma_at_bound.len(), n);

        let mismatched: Vec<(usize, bool, f64, f64)> = (0..n)
            .filter(|&g| fit.sigma_at_bound[g] != (r_subject[g] <= floor))
            .map(|g| {
                (
                    want.column_usize("gene_id")[g],
                    fit.sigma_at_bound[g],
                    fit.subject_overdispersion[g],
                    r_subject[g],
                )
            })
            .collect();
        println!(
            "{}: {} mismatched: {:?}",
            d.tag,
            mismatched.len(),
            mismatched
        );

        let flagged = fit.sigma_at_bound.iter().filter(|b| **b).count();
        assert_eq!(
            flagged,
            s_pinned(d.tag, &s),
            "{}: number of genes with no fitted subject variance",
            d.tag
        );
        total += flagged;
    }
    // Guard against a vacuous test.
    assert!(total > 0, "nothing pinned, so this gates nothing");
}

/// Genes whose subject-level variance finishes on the lower bound, per dataset.
///
/// Recorded, not derived, so a change in how many collapse fails. R pins the same
/// 18 on `sc` (plus gene 299, where its `nlminb` fails) and the same 40 on
/// `sc_small`, a third of its genes: twenty cells per subject is too little
/// between-subject replication, so the model collapses to a plain NB GLM and
/// nebula reports every one as converged.
///
/// The edge datasets read R's `n_pinned` from the scalars.
///
/// ### Params
///
/// * `tag` - Dataset prefix
/// * `s` - The fixture scalars
///
/// ### Returns
///
/// The expected count.
fn s_pinned(tag: &str, s: &common::Scalars) -> usize {
    match tag {
        "sc" => 18,
        "sc_small" => 40,
        _ => s.get_usize(tag, "n_pinned"),
    }
}

#[test]
fn test_nebula_ln_and_hl_are_genuinely_different_paths() {
    // HL on the large dataset must differ from LN (around 50% on the group
    // coefficient), or the LN fixture gates nothing.
    let l = load(&DATASETS[0]);
    let ln = cpu_fit(0);

    let hl = nebula(
        &l.counts,
        l.n_genes,
        l.n_cells,
        &l.subject,
        &l.design,
        l.n_coef,
        l.offset.as_deref(),
        Some(NebulaParams {
            method: NebulaMethod::Hl,
            ..Default::default()
        }),
    )
    .expect("nebula failed");

    let worst = (0..ln.gene_index.len())
        .map(|g| {
            let a = ln.coefficients[g * 3 + 1];
            let b = hl.coefficients[g * 3 + 1];
            (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst > 1e-3,
        "LN and HL agreed to {worst:e} on the group coefficient, so the large \
         fixture is not exercising the LN path"
    );
}

///////////////
// Wald test //
///////////////

#[test]
fn test_glm_sc_test_reproduces_the_r_p_values() {
    for (i, d) in DATASETS.iter().enumerate() {
        let want = common::table(&format!("{}_nebula.csv", d.tag));
        let fit = cpu_fit(i);

        let n = fit.gene_index.len();
        let algorithm = want.column("algorithm");
        // Pure genes only: the LN+HL ones would re-report the fit gap.
        let pure: Vec<bool> = (0..n).map(|g| algorithm[g] as usize != 2).collect();
        let keep = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&pure)
                .filter(|(_, k)| **k)
                .map(|(x, _)| *x)
                .collect()
        };

        for (coef, name) in d.names.iter().enumerate() {
            let got = glm_sc_test(
                &fit.coefficients,
                &fit.covariance,
                n,
                fit.n_coef,
                &ScTested::Coef(coef),
            )
            .expect("glm_sc_test failed");

            assert_close(
                &keep(&got.p_value),
                &keep(want.column(&format!("p_{name}"))),
                tols(d.tag, CPU_TOLS).p_value,
                &format!("{}/wald_p_{name}", d.tag),
            );
            assert_close(
                &keep(&got.se),
                &keep(want.column(&format!("se_{name}"))),
                tols(d.tag, CPU_TOLS).se,
                &format!("{}/wald_se_{name}", d.tag),
            );
        }
    }
}

///////////////
// Shrinkage //
///////////////

#[test]
fn test_shrink_sc_dispersion_matches_limma_squeeze_var() {
    // `shrink_sc_dispersion` is edgePython's invention, so the reference is
    // limma's `squeezeVar` on the reciprocal cell overdispersions at the df from
    // `sc_residual_df`, through the converged-`uniroot` `squeezeVar`
    // (UPSTREAM_DEVIATIONS.md B2). This is the one fixture where that override
    // changes the answer.
    let s = common::scalars();
    let l = load(&DATASETS[0]);
    let want = common::table("sc_shrink.csv");
    let nebula_fixture = common::table("sc_nebula.csv");

    let df_residual = sc_residual_df(l.n_cells, 3, l.n_subjects);
    assert_close_scalar(
        df_residual,
        s.get("sc_shrink", "df_residual"),
        Tol::rel(0.0),
        "sc/residual_df",
    );

    // Inputs are R's: the shrinkage is a joint fit over every usable gene, so the
    // LN+HL divergence above would move every output.
    let dispersion: Vec<f64> = nebula_fixture.column("Cell").to_vec();
    let convergence: Vec<i32> = nebula_fixture
        .column("convergence")
        .iter()
        .map(|v| *v as i32)
        .collect();
    let covariate: Vec<f64> = nebula_fixture.column("logFC_int").to_vec();
    let n = dispersion.len();

    let shrunk = shrink_sc_dispersion(
        &dispersion,
        &convergence,
        Some(&covariate),
        df_residual,
        None,
    )
    .expect("shrink_sc_dispersion failed");

    // R writes only usable genes (converged, positive cell overdispersion).
    let gene_ids: Vec<usize> = nebula_fixture
        .column_usize("gene_id")
        .iter()
        .map(|g| g - 1)
        .collect();
    let position: std::collections::HashMap<usize, usize> =
        gene_ids.iter().enumerate().map(|(i, g)| (*g, i)).collect();
    let want_ids: Vec<usize> = want.column_usize("gene_id").iter().map(|g| g - 1).collect();
    assert_eq!(
        want_ids.len(),
        s.get_usize("sc_shrink", "n_usable"),
        "usable gene count"
    );
    assert!(
        want_ids.len() < n,
        "some genes must be excluded, or the filter is inert"
    );

    let mut phi_raw = Vec::with_capacity(want_ids.len());
    let mut phi_post = Vec::with_capacity(want_ids.len());
    let mut phi_prior = Vec::with_capacity(want_ids.len());
    let mut df_prior = Vec::with_capacity(want_ids.len());
    // `phi_raw`, `phi_post` and `phi_prior` are indexed by gene, `df_prior` by
    // *usable* gene (the robust fit returns one entry per gene it fitted).
    // Conflating them reads the right numbers off the wrong rows.
    assert_eq!(
        shrunk.df_prior.len(),
        want_ids.len(),
        "df_prior should carry one entry per usable gene"
    );
    for (k, id) in want_ids.iter().enumerate() {
        let i = position[id];
        phi_raw.push(shrunk.phi_raw[i].max(1e-8));
        phi_post.push(shrunk.phi_post[i]);
        phi_prior.push(shrunk.phi_prior[i % shrunk.phi_prior.len()]);
        df_prior.push(shrunk.df_prior[k]);
    }

    assert_close(
        &phi_raw,
        want.column("phi_raw"),
        TOL_CELL,
        "sc/shrink_phi_raw",
    );
    assert_close(
        &phi_post,
        want.column("var_post"),
        TOL_SHRINK,
        "sc/shrink_phi_post",
    );
    assert_close(
        &phi_prior,
        want.column("var_prior"),
        TOL_SHRINK,
        "sc/shrink_phi_prior",
    );
    assert_close(
        &df_prior,
        want.column("df_prior"),
        TOL_SHRINK_DF,
        "sc/shrink_df_prior",
    );

    assert_eq!(
        shrunk.df_residual, df_residual,
        "the result should carry the residual df it was given"
    );
}

/// The GPU path's gates, read off the NEEDS column of its own report.
///
/// Stage two's penalised fits run in `f32` on the device and finish in `f64` on
/// the host, so the variance components land within about `1e-4` of R. That is
/// the floor for any inner-fit trajectory other than R's own: the CPU restarted
/// from a point perturbed by `1e-4` needs the same. Worst measured needs across
/// both fixtures, against R:
///
/// * coefficients `2.4e-4`, standard errors `2.4e-4`, cell-level
///   overdispersion `1.5e-4`;
/// * covariance entries `4.8e-4` above an absolute `1e-6`, `1.6e-3` without it.
///   The floor is for entries like `cov(intercept, cov2) = 6.9e-7` on a gene
///   with diagonals `1.8e-2` and `6.9e-3` (a correlation of `6e-5`), which
///   needed `3.6e-1` before the finish took its log-determinant at the stepped
///   point;
/// * subject-level overdispersion within an absolute `1e-3` (the CPU's
///   mixed-path floor); `1.7e-3` relative without it. The floor is for genes R
///   and the CPU put just above the `1e-4` bound and the GPU may put on it; all
///   say the gene has no subject effect.
///
/// The mixed LN-then-HL path uses the CPU's gates, which the GPU meets equally
/// (`7.9e-2`).
///
/// The gates sit well above these needs on purpose: two builds differing only in
/// dead code have measured up to five times apart on the hardest gene, since the
/// `f32` code is whatever the shader compiler makes of it.
#[cfg(feature = "gpu-tests")]
const GPU_TOLS: NebulaTols = NebulaTols {
    coef: Tol::new(1e-2, 1e-9),
    se: Tol::rel(1e-2),
    cov: Tol::new(1.5e-2, 1e-6),
    subject: Tol::new(1.5e-2, 1e-3),
    cell: Tol::rel(1e-2),
    ln_hl_coef: LN_HL_COEF,
    ln_hl_subject: LN_HL_SUBJECT,
    p_value: TOL_P_VALUE,
};

/// The GPU path, against the same R goldens as the CPU path.
///
/// Stage two's penalised fits run on the device in `f32`; the optimum's value,
/// the search and stage three stay in `f64` on the host (`edge_rs::gpu::stage_two`).
#[cfg(feature = "gpu-tests")]
#[test]
fn test_gpu_nebula_matches_the_r_package() {
    use edge_rs::gpu::stage_two::nebula_sparse_gpu;
    use edge_rs::prelude::{CompressedSparse, SparseFormat};

    let client = common::gpu_client();
    let s = common::scalars();

    for d in &DATASETS {
        let l = load(d);
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        for g in 0..l.n_genes {
            for (c, &v) in l.counts[g * l.n_cells..(g + 1) * l.n_cells]
                .iter()
                .enumerate()
            {
                if v > 0.0 {
                    data.push(v);
                    indices.push(c as u32);
                }
            }
            indptr.push(data.len() as u32);
        }
        let sparse = CompressedSparse::from_parts(
            data,
            indices,
            indptr,
            SparseFormat::Csr,
            (l.n_genes, l.n_cells),
        )
        .expect("well-formed CSR");

        let fit = nebula_sparse_gpu(
            &sparse,
            &l.subject,
            &l.design,
            l.n_coef,
            l.offset.as_deref(),
            Some(golden_params(d)),
            &client,
        )
        .expect("gpu nebula failed");
        check_against_r(
            d,
            &format!("gpu/{}", d.tag),
            &fit,
            &s,
            &tols(d.tag, GPU_TOLS),
        );
    }
}
