//! Ranking and calling: edgeR's `topTags` and `decideTests`.
//!
//! Both take a finished test table, adjust the p-values and then sort it or
//! reduce it to a per-gene call. Sequential on purpose: it is one sort of the
//! table at the end of an analysis, with no gene axis worth fanning out over.
//!
//! Only Benjamini-Hochberg ([`crate::numeric::stats::p_adjust_bh`]) is offered;
//! edgeR accepts seven methods.
//!
//! ### Ties
//!
//! R's `order` is a stable radix sort for numeric input, so ties keep the
//! original gene order. Both sorts here are stable too. edgePython's
//! `np.argsort(-alfc)` is a quicksort and is not.

use crate::errors::EdgeErrors;
use crate::numeric::stats::p_adjust_bh;

///////////
// Types //
///////////

/// Which column [`top_tags`] ranks on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortBy {
    /// Ascending p-value, breaking ties on descending absolute fold change.
    /// edgeR's default.
    PValue,
    /// Descending absolute fold change.
    LogFc,
    /// Leave the genes in the order they arrived.
    None,
}

/// A ranked, filtered slice of a test table.
///
/// Every vector has the same length and is in sorted order; `index[k]` is the
/// original gene of row `k`.
#[derive(Clone, Debug)]
pub struct TopTags {
    /// Original gene index of each retained row, in the sorted order.
    pub index: Vec<usize>,
    /// Log2 fold change.
    pub log_fc: Vec<f64>,
    /// Average log2 counts per million.
    pub log_cpm: Vec<f64>,
    /// Test statistic. Empty when the caller supplied none (the exact test has
    /// none).
    pub statistic: Vec<f64>,
    /// Raw p-value.
    pub p_value: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value, computed over the *whole* table
    /// before any filtering, as in edgeR.
    pub fdr: Vec<f64>,
}

////////////////
// Validation //
////////////////

/// Checks that a companion column is one value per gene.
///
/// ### Params
///
/// * `name` - Argument name, used verbatim in the error message
/// * `values` - Column to check
/// * `n_genes` - Number of genes the table holds
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::LengthMismatch`].
fn check_length(name: &'static str, values: &[f64], n_genes: usize) -> Result<(), EdgeErrors> {
    if values.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name,
            expected: n_genes,
            got: values.len(),
        });
    }
    Ok(())
}

/// Checks that a cutoff is a usable probability.
///
/// ### Params
///
/// * `name` - Argument name, used verbatim in the error message
/// * `value` - Cutoff supplied by the caller
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`]. NaN is rejected, since it would
/// silently return nothing.
fn check_cutoff(name: &str, value: f64) -> Result<(), EdgeErrors> {
    if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "'{name}' must be a probability in [0, 1]; got {value}."
        )));
    }
    Ok(())
}

/////////////
// topTags //
/////////////

/// The top differentially expressed genes, edgeR's `topTags`.
///
/// Adjusts, sorts, filters and truncates, in that order. The adjustment covers
/// every gene, so `n` and `p_cutoff` do not change it, and the filter compares
/// the *adjusted* p-value against `p_cutoff`.
///
/// ### Params
///
/// * `log_fc` - Log2 fold change per gene
/// * `log_cpm` - Average log2 counts per million per gene
/// * `statistic` - Test statistic per gene, or `None` ([`TopTags::statistic`] is
///   then empty)
/// * `p_value` - Raw p-value per gene, same length as `log_fc`
/// * `n` - Maximum number of rows, at least one
/// * `sort_by` - Which column to rank on
/// * `p_cutoff` - Keep genes with adjusted p-value at most this; 1 keeps
///   everything, as edgeR's `p.value = 1`
///
/// ### Returns
///
/// The retained rows in sorted order (possibly empty). Errors are
/// [`EdgeErrors::EmptyCounts`] for an empty table,
/// [`EdgeErrors::MustBePositive`] for `n = 0`,
/// [`EdgeErrors::LengthMismatch`] if the columns disagree, and
/// [`EdgeErrors::InvalidArgument`] for a `p_cutoff` outside `[0, 1]`.
pub fn top_tags(
    log_fc: &[f64],
    log_cpm: &[f64],
    statistic: Option<&[f64]>,
    p_value: &[f64],
    n: usize,
    sort_by: SortBy,
    p_cutoff: f64,
) -> Result<TopTags, EdgeErrors> {
    let n_genes = log_fc.len();
    if n_genes == 0 {
        return Err(EdgeErrors::EmptyCounts {
            n_genes: 0,
            n_samples: 0,
        });
    }
    if n == 0 {
        return Err(EdgeErrors::MustBePositive("n".to_string()));
    }
    check_length("log_cpm", log_cpm, n_genes)?;
    check_length("p_value", p_value, n_genes)?;
    if let Some(s) = statistic {
        check_length("statistic", s, n_genes)?;
    }
    check_cutoff("p_cutoff", p_cutoff)?;

    let fdr = p_adjust_bh(p_value);

    let mut order: Vec<usize> = (0..n_genes).collect();
    match sort_by {
        SortBy::PValue => order.sort_by(|&a, &b| {
            p_value[a]
                .total_cmp(&p_value[b])
                .then_with(|| log_fc[b].abs().total_cmp(&log_fc[a].abs()))
        }),
        SortBy::LogFc => order.sort_by(|&a, &b| log_fc[b].abs().total_cmp(&log_fc[a].abs())),
        SortBy::None => {}
    }

    if p_cutoff < 1.0 {
        order.retain(|&g| fdr[g] <= p_cutoff);
    }
    order.truncate(n);

    Ok(TopTags {
        log_fc: order.iter().map(|&g| log_fc[g]).collect(),
        log_cpm: order.iter().map(|&g| log_cpm[g]).collect(),
        statistic: match statistic {
            Some(s) => order.iter().map(|&g| s[g]).collect(),
            None => Vec::new(),
        },
        p_value: order.iter().map(|&g| p_value[g]).collect(),
        fdr: order.iter().map(|&g| fdr[g]).collect(),
        index: order,
    })
}

/////////////////
// decideTests //
/////////////////

/// Calls each gene up, down or not significant, edgeR's `decideTests`.
///
/// A gene is called when its BH-adjusted p-value is *strictly* below `p_cutoff`
/// (edgeR's `p < p.value`). The sign comes from the fold change, and any gene
/// with absolute fold change below `lfc` is then reset to zero.
///
/// As in edgeR the `lfc` threshold is a filter on an already significant call,
/// not a joint test like `glmTreat`.
///
/// ### Params
///
/// * `p_value` - Raw p-value per gene
/// * `log_fc` - Log2 fold change per gene
/// * `p_cutoff` - Adjusted p-value threshold in `[0, 1]`; edgeR's default is 0.05
/// * `lfc` - Absolute log2 fold change threshold, non-negative; zero disables it
///
/// ### Returns
///
/// One of -1, 0 or 1 per gene, in the input order. Errors are
/// [`EdgeErrors::LengthMismatch`] if the columns disagree and
/// [`EdgeErrors::InvalidArgument`] for a `p_cutoff` outside `[0, 1]` or a
/// negative or non-finite `lfc`.
pub fn decide_tests(
    p_value: &[f64],
    log_fc: &[f64],
    p_cutoff: f64,
    lfc: f64,
) -> Result<Vec<i8>, EdgeErrors> {
    check_length("log_fc", log_fc, p_value.len())?;
    check_cutoff("p_cutoff", p_cutoff)?;
    if !lfc.is_finite() || lfc < 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "'lfc' must be non-negative and finite; got {lfc}."
        )));
    }

    let fdr = p_adjust_bh(p_value);
    Ok(fdr
        .iter()
        .zip(log_fc.iter())
        .map(|(q, fc)| {
            if *q >= p_cutoff || fc.abs() < lfc {
                0
            } else if *fc < 0.0 {
                -1
            } else {
                1
            }
        })
        .collect())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    // References are pasted verbatim from R's 17-digit output so they stay
    // checkable against the `Rscript` line above each test.
    #![allow(clippy::excessive_precision)]

    use super::*;
    use approx::assert_relative_eq;

    /// The six-gene exact test from `exact::tests`, the source of every fixture
    /// here.
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0,
    ///               0,0,0,7,9,8, 1200,1100,1300,2400,2500,2300, 5,7,6,5,6,7),
    ///             nrow = 6, byrow = TRUE)
    /// d <- DGEList(counts=y, group=c(1,1,1,2,2,2),
    ///              lib.size=c(1e6,1.2e6,0.9e6,1.1e6,1e6,1.3e6),
    ///              norm.factors=c(0.95,1.05,1.0,1.1,0.9,1.0))
    /// r <- exactTest(d, dispersion=c(0.05,0.1,0.2,0.15,0.08,0.12))
    /// ```
    const P_VALUE: [f64; 6] = [
        3.4349237412116053e-06,
        7.6299000470704081e-01,
        4.6743523769487572e-01,
        1.4972774802797630e-05,
        1.0337909917591625e-02,
        8.9951524706410912e-01,
    ];

    const LOG_FC: [f64; 6] = [
        1.76837964949810367,
        -0.13190914581416660,
        -0.89031602850510549,
        5.98706174338887909,
        0.86708062515865070,
        -0.12249104113230144,
    ];

    const LOG_CPM: [f64; 6] = [
        4.6828570480490503,
        5.6094058355452656,
        1.8369026364109835,
        2.4631896748603697,
        10.7199768972231873,
        2.8848509178259678,
    ];

    /// `cat(format(p.adjust(r$table$PValue, method="BH"), digits=17), sep=", ")`
    const FDR: [f64; 6] = [
        2.0609542447269630e-05,
        8.9951524706410912e-01,
        7.0115285654231352e-01,
        4.4918324408392889e-05,
        2.0675819835183250e-02,
        8.9951524706410912e-01,
    ];

    // -------- //
    // top_tags //
    // -------- //

    /// `topTags(r, n=Inf, sort.by="PValue")` gives rows 1, 4, 5, 3, 2, 6 with
    /// `cat(format(tt$table$FDR, digits=17), sep=", ")` as below.
    #[test]
    fn test_top_tags_sorts_by_p_value() {
        let got = top_tags(
            &LOG_FC,
            &LOG_CPM,
            None,
            &P_VALUE,
            usize::MAX,
            SortBy::PValue,
            1.0,
        )
        .unwrap();
        assert_eq!(got.index, vec![0, 3, 4, 2, 1, 5]);

        let want_fdr = [
            2.0609542447269630e-05,
            4.4918324408392889e-05,
            2.0675819835183250e-02,
            7.0115285654231352e-01,
            8.9951524706410912e-01,
            8.9951524706410912e-01,
        ];
        for (g, w) in got.fdr.iter().zip(want_fdr.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-12);
        }

        let want_fc = [
            1.76837964949810367,
            5.98706174338887909,
            0.86708062515865070,
            -0.89031602850510549,
            -0.13190914581416660,
            -0.12249104113230144,
        ];
        for (g, w) in got.log_fc.iter().zip(want_fc.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-12);
        }
        assert!(got.statistic.is_empty());
    }

    /// `topTags(r, n=Inf, sort.by="logFC")` gives rows 4, 1, 3, 5, 2, 6.
    #[test]
    fn test_top_tags_sorts_by_absolute_fold_change() {
        let got = top_tags(
            &LOG_FC,
            &LOG_CPM,
            None,
            &P_VALUE,
            usize::MAX,
            SortBy::LogFc,
            1.0,
        )
        .unwrap();
        assert_eq!(got.index, vec![3, 0, 2, 4, 1, 5]);

        let want_fdr = [
            4.4918324408392889e-05,
            2.0609542447269630e-05,
            7.0115285654231352e-01,
            2.0675819835183250e-02,
            8.9951524706410912e-01,
            8.9951524706410912e-01,
        ];
        for (g, w) in got.fdr.iter().zip(want_fdr.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-12);
        }
    }

    /// `topTags(r, n=Inf, sort.by="none")` leaves the table alone.
    #[test]
    fn test_top_tags_leaves_the_order_alone_when_asked() {
        let got = top_tags(
            &LOG_FC,
            &LOG_CPM,
            None,
            &P_VALUE,
            usize::MAX,
            SortBy::None,
            1.0,
        )
        .unwrap();
        assert_eq!(got.index, vec![0, 1, 2, 3, 4, 5]);
        for (g, w) in got.fdr.iter().zip(FDR.iter()) {
            assert_relative_eq!(g, w, max_relative = 1e-12);
        }
    }

    /// `topTags(r, n=Inf, p.value=0.05)` keeps rows 1, 4, 5. The cutoff is on
    /// the adjusted p-value, so gene 5 survives at an FDR of 0.0207 despite a
    /// raw p-value of 0.0103.
    #[test]
    fn test_top_tags_filters_on_the_adjusted_p_value() {
        let got = top_tags(
            &LOG_FC,
            &LOG_CPM,
            None,
            &P_VALUE,
            usize::MAX,
            SortBy::PValue,
            0.05,
        )
        .unwrap();
        assert_eq!(got.index, vec![0, 3, 4]);
        assert_eq!(got.p_value.len(), 3);
        assert!(got.fdr.iter().all(|q| *q <= 0.05));
    }

    /// `topTags(r, n=3)` truncates after sorting, giving rows 1, 4, 5.
    #[test]
    fn test_top_tags_truncates_to_n() {
        let got = top_tags(&LOG_FC, &LOG_CPM, None, &P_VALUE, 3, SortBy::PValue, 1.0).unwrap();
        assert_eq!(got.index, vec![0, 3, 4]);
    }

    /// A cutoff nothing passes gives an empty table, as edgeR returns
    /// `data.frame()`.
    #[test]
    fn test_top_tags_can_return_nothing() {
        let got = top_tags(
            &LOG_FC,
            &LOG_CPM,
            None,
            &P_VALUE,
            usize::MAX,
            SortBy::PValue,
            1e-9,
        )
        .unwrap();
        assert!(got.index.is_empty());
        assert!(got.log_fc.is_empty());
        assert!(got.fdr.is_empty());
    }

    /// A statistic column is carried through in the sorted order when supplied.
    #[test]
    fn test_top_tags_carries_a_statistic_column() {
        let stat = [10.0, 1.0, 2.0, 9.0, 5.0, 0.5];
        let got = top_tags(
            &LOG_FC,
            &LOG_CPM,
            Some(&stat),
            &P_VALUE,
            usize::MAX,
            SortBy::PValue,
            1.0,
        )
        .unwrap();
        assert_eq!(got.statistic, vec![10.0, 9.0, 5.0, 2.0, 1.0, 0.5]);
    }

    /// Ties in the p-value fall back to descending absolute fold change, and
    /// ties in both fall back to the original order.
    #[test]
    fn test_top_tags_breaks_ties_the_way_r_does() {
        let p = [0.1, 0.1, 0.1, 0.05];
        let fc = [1.0, -3.0, 1.0, 0.5];
        let got = top_tags(&fc, &fc, None, &p, usize::MAX, SortBy::PValue, 1.0).unwrap();
        assert_eq!(got.index, vec![3, 1, 0, 2]);

        // The two genes tied at |1.0| keep their input order.
        let got = top_tags(&fc, &fc, None, &p, usize::MAX, SortBy::LogFc, 1.0).unwrap();
        assert_eq!(got.index, vec![1, 0, 2, 3]);
    }

    #[test]
    fn test_top_tags_rejects_bad_input() {
        assert!(matches!(
            top_tags(&[], &[], None, &[], 5, SortBy::PValue, 1.0),
            Err(EdgeErrors::EmptyCounts { .. })
        ));
        assert!(matches!(
            top_tags(&LOG_FC, &LOG_CPM, None, &P_VALUE, 0, SortBy::PValue, 1.0),
            Err(EdgeErrors::MustBePositive(_))
        ));
        assert!(matches!(
            top_tags(
                &LOG_FC,
                &LOG_CPM[..3],
                None,
                &P_VALUE,
                5,
                SortBy::PValue,
                1.0
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "log_cpm",
                ..
            })
        ));
        assert!(matches!(
            top_tags(
                &LOG_FC,
                &LOG_CPM,
                None,
                &P_VALUE[..3],
                5,
                SortBy::PValue,
                1.0
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "p_value",
                ..
            })
        ));
        assert!(matches!(
            top_tags(
                &LOG_FC,
                &LOG_CPM,
                Some(&[1.0, 2.0]),
                &P_VALUE,
                5,
                SortBy::PValue,
                1.0
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "statistic",
                ..
            })
        ));
        assert!(matches!(
            top_tags(&LOG_FC, &LOG_CPM, None, &P_VALUE, 5, SortBy::PValue, 1.5),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            top_tags(
                &LOG_FC,
                &LOG_CPM,
                None,
                &P_VALUE,
                5,
                SortBy::PValue,
                f64::NAN
            ),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    // ------------ //
    // decide_tests //
    // ------------ //

    /// `cat(decideTests(r, p.value=0.05, lfc=0))` -> 1, 0, 0, 1, 1, 0
    #[test]
    fn test_decide_tests_matches_edger_without_a_fold_change_threshold() {
        let got = decide_tests(&P_VALUE, &LOG_FC, 0.05, 0.0).unwrap();
        assert_eq!(got, vec![1, 0, 0, 1, 1, 0]);
    }

    /// `cat(decideTests(r, p.value=0.05, lfc=1))` -> 1, 0, 0, 1, 0, 0. Gene 5 is
    /// significant but only moves 0.867 on the log2 scale, so it is dropped.
    #[test]
    fn test_decide_tests_applies_the_fold_change_threshold() {
        let got = decide_tests(&P_VALUE, &LOG_FC, 0.05, 1.0).unwrap();
        assert_eq!(got, vec![1, 0, 0, 1, 0, 0]);
    }

    /// `cat(decideTests(r, p.value=0.05, lfc=2))` -> 0, 0, 0, 1, 0, 0
    #[test]
    fn test_decide_tests_at_a_higher_fold_change_threshold() {
        let got = decide_tests(&P_VALUE, &LOG_FC, 0.05, 2.0).unwrap();
        assert_eq!(got, vec![0, 0, 0, 1, 0, 0]);
    }

    /// `cat(decideTests(r, p.value=1, lfc=0))` -> 1, -1, -1, 1, 1, -1. Everything
    /// is called, so this pins the sign of the fold change.
    #[test]
    fn test_decide_tests_calls_the_down_direction() {
        let got = decide_tests(&P_VALUE, &LOG_FC, 1.0, 0.0).unwrap();
        assert_eq!(got, vec![1, -1, -1, 1, 1, -1]);
    }

    /// The comparison is strict: an adjusted p-value equal to the cutoff is not
    /// called, as in edgeR's `p < p.value`.
    #[test]
    fn test_decide_tests_is_strict_at_the_cutoff() {
        // A single gene means BH leaves the p-value alone.
        assert_eq!(decide_tests(&[0.05], &[3.0], 0.05, 0.0).unwrap(), vec![0]);
        assert_eq!(decide_tests(&[0.05], &[3.0], 0.06, 0.0).unwrap(), vec![1]);
    }

    /// A cutoff of zero calls nothing, and one of one calls everything with a
    /// non-zero fold change.
    #[test]
    fn test_decide_tests_at_the_extremes_of_the_cutoff() {
        assert_eq!(
            decide_tests(&P_VALUE, &LOG_FC, 0.0, 0.0).unwrap(),
            vec![0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_decide_tests_rejects_bad_input() {
        assert!(matches!(
            decide_tests(&P_VALUE, &LOG_FC[..3], 0.05, 0.0),
            Err(EdgeErrors::LengthMismatch { name: "log_fc", .. })
        ));
        assert!(matches!(
            decide_tests(&P_VALUE, &LOG_FC, -0.1, 0.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            decide_tests(&P_VALUE, &LOG_FC, 1.1, 0.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            decide_tests(&P_VALUE, &LOG_FC, f64::NAN, 0.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            decide_tests(&P_VALUE, &LOG_FC, 0.05, -1.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            decide_tests(&P_VALUE, &LOG_FC, 0.05, f64::INFINITY),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    /// An empty table is not an error: nothing to call.
    #[test]
    fn test_decide_tests_on_an_empty_table() {
        assert!(decide_tests(&[], &[], 0.05, 0.0).unwrap().is_empty());
    }
}
