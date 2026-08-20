//! Fixture loading and comparison for the end-to-end parity tests.
//!
//! Every fixture under `tests/data/e2e` is written by `tests/r/generate_fixtures.R`
//! against edgeR 4.8.2, limma 3.66 and nebula 1.5.8. CI never runs the R; the
//! files are committed and read from here.
//!
//! ### Why the comparator is not `assert_relative_eq!` in a loop
//!
//! That is what the in-crate tests do, and at thirty genes it is the right
//! answer. Over a thousand it reports the first failing gene and nothing else,
//! which is the wrong end of the problem: what you want to know is the worst
//! disagreement, where it landed and how many genes are involved.
//! [`assert_close`] scans the whole vector and says all three, and takes care to
//! point at the worst *failing* gene rather than the worst gene, which are often
//! different once a tolerance has an absolute leg.
//!
//! The same scan doubles as the calibration tool. With `EDGE_RS_TOL_REPORT` set
//! it prints the worst observed error for every labelled quantity even when the
//! test passes:
//!
//! ```text
//! EDGE_RS_TOL_REPORT=1 cargo test --release --test 'e2e_*' -- --nocapture
//! ```
//!
//! Tolerances are set from that table rather than guessed, and each one carries
//! its measured worst case in the doc comment on the constant, following what
//! `src/sc/nebula.rs` already does.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

///////////////
// Tolerance //
///////////////

/// A comparison tolerance, split the way `approx` splits it.
///
/// A pair passes when it is within `epsilon` absolutely or within
/// `max_relative` relatively. The absolute leg matters for quantities that
/// legitimately reach zero, such as a log fold change on a gene with no signal,
/// where a relative test is meaningless.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    /// Largest acceptable relative difference.
    pub max_relative: f64,
    /// Largest acceptable absolute difference. Zero for a pure relative test.
    pub epsilon: f64,
}

impl Tol {
    /// A pure relative tolerance, with no absolute escape hatch.
    ///
    /// ### Params
    ///
    /// * `max_relative` - Largest acceptable relative difference
    ///
    /// ### Returns
    ///
    /// The tolerance.
    pub const fn rel(max_relative: f64) -> Self {
        Self { max_relative, epsilon: 0.0 }
    }

    /// A relative tolerance with an absolute floor.
    ///
    /// ### Params
    ///
    /// * `max_relative` - Largest acceptable relative difference
    /// * `epsilon` - Largest acceptable absolute difference
    ///
    /// ### Returns
    ///
    /// The tolerance.
    pub const fn new(max_relative: f64, epsilon: f64) -> Self {
        Self { max_relative, epsilon }
    }
}

//////////////////
// Comparisons  //
//////////////////

/// Relative difference between two values, on the `approx` convention.
///
/// Returns zero when both are exactly equal, including when both are infinite
/// with the same sign, and infinity when only one of them is non-finite.
///
/// ### Params
///
/// * `got` - Value under test
/// * `want` - Reference value
///
/// ### Returns
///
/// The relative difference, or infinity if the pair cannot be compared.
fn relative_difference(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    if got.is_nan() != want.is_nan() {
        return f64::INFINITY;
    }
    if got.is_nan() && want.is_nan() {
        return 0.0;
    }
    if !got.is_finite() || !want.is_finite() {
        return f64::INFINITY;
    }
    let scale = got.abs().max(want.abs());
    if scale == 0.0 { 0.0 } else { (got - want).abs() / scale }
}

/// One candidate for the worst disagreement in a comparison.
///
/// [`assert_close`] tracks two of these: the worst pair overall, for the
/// calibration report, and the worst pair that actually failed, for the panic.
#[derive(Clone, Copy, Debug)]
struct Worst {
    /// Index of the pair.
    index: usize,
    /// Relative difference there.
    relative: f64,
    /// Absolute difference there.
    absolute: f64,
    /// Value under test.
    got: f64,
    /// Reference value.
    want: f64,
}

/// Compares two vectors elementwise and panics naming the worst failure.
///
/// Two running maxima are kept, and they are not the same entry. The panic
/// message names the worst *failing* pair, because the largest relative
/// difference in the vector is frequently one the absolute leg of the tolerance
/// let through, and pointing at it sends the reader to a value that is fine. The
/// calibration report names the worst pair overall, since that is the number a
/// tolerance has to clear.
///
/// `NaN` against `NaN` passes, since R writes `NA` where the crate carries
/// `NaN` for a non-estimable coefficient. `NaN` against a finite value fails.
///
/// When `EDGE_RS_TOL_REPORT` is set the worst observed error is printed whether
/// or not the comparison passes, which is how the tolerance constants in these
/// tests were calibrated.
///
/// ### Params
///
/// * `got` - Values produced by the crate
/// * `want` - Reference values from R
/// * `tol` - Tolerance to apply
/// * `label` - Quantity name, used in the report and the panic message
///
/// ### Panics
///
/// If the lengths disagree, or any pair falls outside `tol`.
pub fn assert_close(got: &[f64], want: &[f64], tol: Tol, label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch, got {} want {}",
        got.len(),
        want.len()
    );

    // Two running maxima. `worst` is over everything and drives the calibration
    // report; `worst_failing` is over the failures only and drives the panic
    // message, because the largest relative difference is often a passing entry
    // that the absolute leg of the tolerance let through.
    let mut worst: Option<Worst> = None;
    let mut worst_failing: Option<Worst> = None;
    let mut n_failed = 0_usize;

    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let relative = relative_difference(g, w);
        let absolute = if g.is_finite() && w.is_finite() { (g - w).abs() } else { f64::INFINITY };
        let here = Worst { index: i, relative, absolute, got: g, want: w };

        if worst.is_none_or(|prev| relative > prev.relative) {
            worst = Some(here);
        }
        if absolute > tol.epsilon && relative > tol.max_relative {
            n_failed += 1;
            if worst_failing.is_none_or(|prev| relative > prev.relative) {
                worst_failing = Some(here);
            }
        }
    }

    let worst = match worst {
        Some(w) => w,
        None => return,
    };

    if report_enabled() {
        println!(
            "[tol] {label:<34} n={:<6} worst rel {:.3e} at {} (got {:.17e}, want {:.17e})",
            got.len(),
            worst.relative,
            worst.index,
            worst.got,
            worst.want
        );
    }

    let Some(bad) = worst_failing else { return };

    panic!(
        "{label}: {n_failed} of {} values outside tolerance \
         (max_relative {:.1e}, epsilon {:.1e}).\n  \
         worst failure at index {}: got {:.17e}, want {:.17e}, \
         relative {:.3e}, absolute {:.3e}",
        got.len(),
        tol.max_relative,
        tol.epsilon,
        bad.index,
        bad.got,
        bad.want,
        bad.relative,
        bad.absolute
    );
}

/// Compares two scalars, with the same reporting as [`assert_close`].
///
/// ### Params
///
/// * `got` - Value produced by the crate
/// * `want` - Reference value from R
/// * `tol` - Tolerance to apply
/// * `label` - Quantity name
///
/// ### Panics
///
/// If the pair falls outside `tol`.
pub fn assert_close_scalar(got: f64, want: f64, tol: Tol, label: &str) {
    assert_close(&[got], &[want], tol, label);
}

/// Asserts two integer-valued vectors are equal, reporting how many differ.
///
/// Used for the filter masks and the gene index vectors, where a tolerance
/// would be meaningless: a disagreement is a different gene set, not a drift.
///
/// ### Params
///
/// * `got` - Values produced by the crate
/// * `want` - Reference values from R
/// * `label` - Quantity name
///
/// ### Panics
///
/// If the lengths disagree or any pair differs.
pub fn assert_eq_usize(got: &[usize], want: &[usize], label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch, got {} want {}",
        got.len(),
        want.len()
    );
    let diffs: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    assert!(
        diffs.is_empty(),
        "{label}: {} of {} entries differ, first at index {} (got {}, want {})",
        diffs.len(),
        got.len(),
        diffs[0],
        got[diffs[0]],
        want[diffs[0]]
    );
}

/// Whether the calibration report is switched on.
fn report_enabled() -> bool {
    std::env::var_os("EDGE_RS_TOL_REPORT").is_some()
}

///////////
// Table //
///////////

/// A headed fixture, held column-major so a named column is a contiguous slice.
#[derive(Clone, Debug)]
pub struct Table {
    /// Column names, in file order.
    names: Vec<String>,
    /// One vector per column, each `n_rows` long.
    columns: Vec<Vec<f64>>,
    /// Number of data rows.
    n_rows: usize,
    /// File name, for error messages.
    source: String,
}

impl Table {
    /// Number of data rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Borrows one column by name.
    ///
    /// ### Params
    ///
    /// * `name` - Column header as written by the R script
    ///
    /// ### Returns
    ///
    /// The column, `n_rows` long.
    ///
    /// ### Panics
    ///
    /// If no column carries that name.
    pub fn column(&self, name: &str) -> &[f64] {
        match self.names.iter().position(|n| n == name) {
            Some(i) => &self.columns[i],
            None => panic!(
                "{}: no column {name:?}; available: {}",
                self.source,
                self.names.join(", ")
            ),
        }
    }

    /// Borrows one column and rounds it to `usize`.
    ///
    /// The fixtures carry integer quantities as doubles because everything else
    /// in the file is a double. Values must be non-negative whole numbers.
    ///
    /// ### Params
    ///
    /// * `name` - Column header
    ///
    /// ### Returns
    ///
    /// The column as indices.
    pub fn column_usize(&self, name: &str) -> Vec<usize> {
        self.column(name)
            .iter()
            .map(|v| {
                assert!(
                    v.is_finite() && *v >= 0.0 && v.fract() == 0.0,
                    "{}: column {name:?} holds {v}, which is not a non-negative whole number",
                    self.source
                );
                *v as usize
            })
            .collect()
    }

    /// Borrows one column as booleans, treating any non-zero as true.
    ///
    /// ### Params
    ///
    /// * `name` - Column header
    ///
    /// ### Returns
    ///
    /// The column as a mask.
    pub fn column_bool(&self, name: &str) -> Vec<bool> {
        self.column(name).iter().map(|v| *v != 0.0).collect()
    }

    /// Flattens the whole table row-major.
    ///
    /// This is the layout the crate uses for a gene-by-sample matrix, so a
    /// fixture written one gene per row comes back ready to compare.
    ///
    /// ### Returns
    ///
    /// The values row-major, `n_rows * n_cols`.
    pub fn row_major(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_rows * self.columns.len());
        for r in 0..self.n_rows {
            for c in &self.columns {
                out.push(c[r]);
            }
        }
        out
    }

    /// Flattens the whole table row-major and rounds to `f64` counts.
    ///
    /// ### Returns
    ///
    /// The values row-major, checked to be whole numbers.
    pub fn row_major_counts(&self) -> Vec<f64> {
        let v = self.row_major();
        for x in &v {
            assert!(
                x.is_finite() && x.fract() == 0.0,
                "{}: expected whole counts, found {x}",
                self.source
            );
        }
        v
    }
}

/////////////
// Loading //
/////////////

/// Resolves a fixture path under `tests/data/e2e`.
///
/// ### Params
///
/// * `name` - File name
///
/// ### Returns
///
/// The absolute path, independent of the working directory.
fn fixture_path(name: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "data", "e2e", name]
        .iter()
        .collect()
}

/// Parses one field, mapping R's spellings of the non-finite values.
///
/// R writes `NaN`, `Inf` and `-Inf`, all of which Rust accepts. The generator
/// rewrites `NA` to `NaN` on the way out, but the mapping is kept here too so a
/// hand-edited fixture cannot fail obscurely.
///
/// ### Params
///
/// * `field` - Raw field text
/// * `name` - File name, for the error message
///
/// ### Returns
///
/// The parsed value.
fn parse_field(field: &str, name: &str) -> f64 {
    let t = field.trim().trim_matches('"');
    match t {
        "NA" | "NaN" | "nan" => f64::NAN,
        "Inf" | "inf" => f64::INFINITY,
        "-Inf" | "-inf" => f64::NEG_INFINITY,
        _ => t
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("bad field {field:?} in {name}: {e}")),
    }
}

/// Reads a headed fixture.
///
/// ### Params
///
/// * `name` - File name inside `tests/data/e2e`
///
/// ### Returns
///
/// The table.
///
/// ### Panics
///
/// If the file is missing, ragged, or holds an unparseable field.
pub fn table(name: &str) -> Table {
    let path = fixture_path(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}. Run `Rscript tests/r/generate_fixtures.R` to create it.",
            path.display()
        )
    });

    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().unwrap_or_else(|| panic!("{name} has no header"));
    let names: Vec<String> = header
        .split(',')
        .map(|h| h.trim().trim_matches('"').to_string())
        .collect();

    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let mut n_rows = 0_usize;
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            names.len(),
            "ragged fixture {name}: row {n_rows} has {} fields, header has {}",
            fields.len(),
            names.len()
        );
        for (c, field) in fields.iter().enumerate() {
            columns[c].push(parse_field(field, name));
        }
        n_rows += 1;
    }

    Table { names, columns, n_rows, source: name.to_string() }
}

/// Reads a fixture and returns it flattened row-major with its shape.
///
/// ### Params
///
/// * `name` - File name inside `tests/data/e2e`
///
/// ### Returns
///
/// The values row-major, the row count and the column count.
pub fn matrix(name: &str) -> (Vec<f64>, usize, usize) {
    let t = table(name);
    let (r, c) = (t.n_rows(), t.n_cols());
    (t.row_major(), r, c)
}

/////////////
// Scalars //
/////////////

/// The `scenario,name,value` table, keyed by both fields.
#[derive(Clone, Debug)]
pub struct Scalars {
    /// Values keyed by `(scenario, name)`.
    values: HashMap<(String, String), f64>,
}

impl Scalars {
    /// Looks up one scalar.
    ///
    /// ### Params
    ///
    /// * `scenario` - Scenario tag, for example `"fac"`
    /// * `name` - Quantity name, for example `"common_dispersion"`
    ///
    /// ### Returns
    ///
    /// The value.
    ///
    /// ### Panics
    ///
    /// If the pair is not present.
    pub fn get(&self, scenario: &str, name: &str) -> f64 {
        *self
            .values
            .get(&(scenario.to_string(), name.to_string()))
            .unwrap_or_else(|| panic!("scalars.csv has no {scenario}/{name}"))
    }

    /// Looks up one scalar and rounds it to `usize`.
    ///
    /// ### Params
    ///
    /// * `scenario` - Scenario tag
    /// * `name` - Quantity name
    ///
    /// ### Returns
    ///
    /// The value as a count.
    pub fn get_usize(&self, scenario: &str, name: &str) -> usize {
        let v = self.get(scenario, name);
        assert!(
            v.is_finite() && v >= 0.0 && v.fract() == 0.0,
            "{scenario}/{name} is {v}, not a non-negative whole number"
        );
        v as usize
    }
}

/// Reads `scalars.csv`.
///
/// ### Returns
///
/// Every scalar the generator recorded.
pub fn scalars() -> Scalars {
    let name = "scalars.csv";
    let path = fixture_path(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut values = HashMap::new();
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let mut parts = line.splitn(3, ',');
        let scenario = parts.next().expect("scenario").trim().to_string();
        let key = parts.next().expect("name").trim().to_string();
        let raw = parts.next().expect("value");
        values.insert((scenario, key), parse_field(raw, name));
    }
    Scalars { values }
}
