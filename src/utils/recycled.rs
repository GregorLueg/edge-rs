//! Recycled matrices: the `compressedMatrix` of edgeR, done properly.
//!
//! Offsets, weights, dispersions and prior counts are all logically
//! `n_genes` by `n_samples`, and are almost never actually that big. A library
//! size offset varies by sample only. A tagwise dispersion varies by gene only.
//! A common dispersion is one number.

use crate::prelude::*;

//////////////
// Recycled //
//////////////

/// A logically `n_genes` by `n_samples` matrix stored in whatever form it
/// actually varies in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Recycled<T> {
    /// One value shared by every gene and every sample.
    Scalar(T),
    /// One value per gene, repeated across samples. Length `n_genes`.
    ByGene(Vec<T>),
    /// One value per sample, repeated across genes. Length `n_samples`.
    BySample(Vec<T>),
    /// A full matrix, row-major, `n_genes * n_samples` values.
    Full(Vec<T>),
}

/////////////////
// RecycledRow //
/////////////////

/// One gene's row of a [`Recycled`], borrowed rather than materialised.
///
/// The whole point of the type: a `Scalar` or `ByGene` row is a single value,
/// and pretending otherwise would allocate `n_samples` floats per gene per
/// iteration.
#[derive(Clone, Copy, Debug)]
pub enum RecycledRow<'a, T> {
    /// Every sample in this row carries the same value.
    Constant(T),
    /// The row varies by sample.
    Values(&'a [T]),
}

impl<T: Copy> RecycledRow<'_, T> {
    /// Value for one sample of this row.
    ///
    /// ### Params
    ///
    /// * `sample` - Sample index. Only bounds-checked in the `Values` case; a
    ///   `Constant` row is defined for every index.
    ///
    /// ### Returns
    ///
    /// The value at that sample.
    #[inline]
    pub fn get(&self, sample: usize) -> T {
        match self {
            RecycledRow::Constant(v) => *v,
            RecycledRow::Values(vals) => vals[sample],
        }
    }

    /// Whether this row is a single repeated value.
    ///
    /// ### Returns
    ///
    /// `true` for `Constant`. Hot loops branch on this once and then take a
    /// specialised path, rather than calling [`RecycledRow::get`] per sample.
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, RecycledRow::Constant(_))
    }

    /// Iterates the row across `n_samples` samples.
    ///
    /// ### Params
    ///
    /// * `n_samples` - Number of samples to yield. Ignored for `Values`, which
    ///   yields its own length.
    ///
    /// ### Returns
    ///
    /// An iterator over the row's values.
    #[inline]
    pub fn iter(&self, n_samples: usize) -> Box<dyn Iterator<Item = T> + '_> {
        match self {
            RecycledRow::Constant(v) => {
                let v = *v;
                Box::new(std::iter::repeat_n(v, n_samples))
            }
            RecycledRow::Values(vals) => Box::new(vals.iter().copied()),
        }
    }
}

impl<T: Copy> Recycled<T> {
    /// Builds a scalar-valued recycled matrix.
    ///
    /// ### Params
    ///
    /// * `value` - The single value
    ///
    /// ### Returns
    ///
    /// A [`Recycled::Scalar`].
    pub fn scalar(value: T) -> Self {
        Recycled::Scalar(value)
    }

    /// Builds a gene-varying recycled matrix.
    ///
    /// ### Params
    ///
    /// * `values` - One value per gene
    ///
    /// ### Returns
    ///
    /// A [`Recycled::ByGene`].
    pub fn by_gene(values: Vec<T>) -> Self {
        Recycled::ByGene(values)
    }

    /// Builds a sample-varying recycled matrix.
    ///
    /// ### Params
    ///
    /// * `values` - One value per sample
    ///
    /// ### Returns
    ///
    /// A [`Recycled::BySample`].
    pub fn by_sample(values: Vec<T>) -> Self {
        Recycled::BySample(values)
    }

    /// Builds a full recycled matrix from row-major values.
    ///
    /// ### Params
    ///
    /// * `values` - `n_genes * n_samples` values, row-major
    /// * `n_genes` - Number of genes
    /// * `n_samples` - Number of samples
    ///
    /// ### Returns
    ///
    /// A [`Recycled::Full`], or [`EdgeErrors::LengthMismatch`] if the length
    /// does not match the stated shape.
    pub fn full(values: Vec<T>, n_genes: usize, n_samples: usize) -> Result<Self, EdgeErrors> {
        if values.len() != n_genes * n_samples {
            return Err(EdgeErrors::LengthMismatch {
                name: "values",
                expected: n_genes * n_samples,
                got: values.len(),
            });
        }
        Ok(Recycled::Full(values))
    }

    /// Checks that this matrix is usable at the given shape.
    ///
    /// A `Scalar` fits any shape. The others must match on their varying axis.
    ///
    /// ### Params
    ///
    /// * `n_genes` - Number of genes the consumer expects
    /// * `n_samples` - Number of samples the consumer expects
    ///
    /// ### Returns
    ///
    /// `Ok(())` when compatible, otherwise [`EdgeErrors::LengthMismatch`].
    pub fn validate(&self, n_genes: usize, n_samples: usize) -> Result<(), EdgeErrors> {
        let (name, expected, got) = match self {
            Recycled::Scalar(_) => return Ok(()),
            Recycled::ByGene(v) => ("by_gene", n_genes, v.len()),
            Recycled::BySample(v) => ("by_sample", n_samples, v.len()),
            Recycled::Full(v) => ("full", n_genes * n_samples, v.len()),
        };
        if expected != got {
            return Err(EdgeErrors::LengthMismatch {
                name,
                expected,
                got,
            });
        }
        Ok(())
    }

    /// Borrows one gene's row.
    ///
    /// This is the accessor the inner loops use. It never allocates.
    ///
    /// ### Params
    ///
    /// * `gene` - Gene index
    /// * `n_samples` - Number of samples, needed to slice the `Full` case
    ///
    /// ### Returns
    ///
    /// The row as a [`RecycledRow`].
    #[inline]
    pub fn row(&self, gene: usize, n_samples: usize) -> RecycledRow<'_, T> {
        match self {
            Recycled::Scalar(v) => RecycledRow::Constant(*v),
            Recycled::ByGene(v) => RecycledRow::Constant(v[gene]),
            Recycled::BySample(v) => RecycledRow::Values(v),
            Recycled::Full(v) => {
                let start = gene * n_samples;
                RecycledRow::Values(&v[start..start + n_samples])
            }
        }
    }

    /// Single element lookup.
    ///
    /// Prefer [`Recycled::row`] in a loop over samples; this exists for the
    /// scattered accesses that do not fit that shape.
    ///
    /// ### Params
    ///
    /// * `gene` - Gene index
    /// * `sample` - Sample index
    /// * `n_samples` - Number of samples, needed to index the `Full` case
    ///
    /// ### Returns
    ///
    /// The value at that position.
    #[inline]
    pub fn get(&self, gene: usize, sample: usize, n_samples: usize) -> T {
        match self {
            Recycled::Scalar(v) => *v,
            Recycled::ByGene(v) => v[gene],
            Recycled::BySample(v) => v[sample],
            Recycled::Full(v) => v[gene * n_samples + sample],
        }
    }

    /// Materialises the full row-major matrix.
    ///
    /// Only for tests and for interop with code that genuinely needs a dense
    /// buffer. Calling this on the single-cell path defeats the purpose of the
    /// type.
    ///
    /// ### Params
    ///
    /// * `n_genes` - Number of genes
    /// * `n_samples` - Number of samples
    ///
    /// ### Returns
    ///
    /// A dense `n_genes * n_samples` vector, row-major.
    pub fn expand(&self, n_genes: usize, n_samples: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(n_genes * n_samples);
        for gene in 0..n_genes {
            match self.row(gene, n_samples) {
                RecycledRow::Constant(v) => out.extend(std::iter::repeat_n(v, n_samples)),
                RecycledRow::Values(vals) => out.extend_from_slice(vals),
            }
        }
        out
    }

    /// Number of stored values, as opposed to the logical element count.
    ///
    /// ### Returns
    ///
    /// How many `T` this actually holds.
    pub fn stored_len(&self) -> usize {
        match self {
            Recycled::Scalar(_) => 1,
            Recycled::ByGene(v) | Recycled::BySample(v) | Recycled::Full(v) => v.len(),
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    const N_GENES: usize = 3;
    const N_SAMPLES: usize = 4;

    #[test]
    fn test_scalar_expands_to_constant_matrix() {
        let r = Recycled::scalar(2.5_f64);
        assert_eq!(r.expand(N_GENES, N_SAMPLES), vec![2.5; N_GENES * N_SAMPLES]);
        assert_eq!(r.stored_len(), 1);
    }

    #[test]
    fn test_by_gene_repeats_across_samples() {
        let r = Recycled::by_gene(vec![1.0_f64, 2.0, 3.0]);
        let expanded = r.expand(N_GENES, N_SAMPLES);
        assert_eq!(
            expanded,
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
        assert!(r.row(1, N_SAMPLES).is_constant());
    }

    #[test]
    fn test_by_sample_repeats_across_genes() {
        let r = Recycled::by_sample(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let expanded = r.expand(N_GENES, N_SAMPLES);
        assert_eq!(
            expanded,
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert!(!r.row(0, N_SAMPLES).is_constant());
    }

    #[test]
    fn test_full_round_trips() {
        let values: Vec<f64> = (0..N_GENES * N_SAMPLES).map(|i| i as f64).collect();
        let r = Recycled::full(values.clone(), N_GENES, N_SAMPLES).unwrap();
        assert_eq!(r.expand(N_GENES, N_SAMPLES), values);
    }

    #[test]
    fn test_get_agrees_with_expand() {
        let cases = vec![
            Recycled::scalar(7.0_f64),
            Recycled::by_gene(vec![1.0, 2.0, 3.0]),
            Recycled::by_sample(vec![1.0, 2.0, 3.0, 4.0]),
            Recycled::full((0..12).map(|i| i as f64).collect(), N_GENES, N_SAMPLES).unwrap(),
        ];
        for r in cases {
            let dense = r.expand(N_GENES, N_SAMPLES);
            for g in 0..N_GENES {
                for s in 0..N_SAMPLES {
                    assert_eq!(r.get(g, s, N_SAMPLES), dense[g * N_SAMPLES + s]);
                }
            }
        }
    }

    #[test]
    fn test_row_iter_yields_n_samples() {
        let r = Recycled::by_gene(vec![5.0_f64, 6.0, 7.0]);
        let row: Vec<f64> = r.row(2, N_SAMPLES).iter(N_SAMPLES).collect();
        assert_eq!(row, vec![7.0; N_SAMPLES]);
    }

    #[test]
    fn test_full_rejects_wrong_length() {
        let err = Recycled::full(vec![0.0_f64; 5], N_GENES, N_SAMPLES).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                expected: 12,
                got: 5,
                ..
            }
        ));
    }

    #[test]
    fn test_validate_accepts_scalar_at_any_shape() {
        let r = Recycled::scalar(1.0_f64);
        assert!(r.validate(1000, 7).is_ok());
    }

    #[test]
    fn test_validate_rejects_mismatched_axis() {
        let r = Recycled::by_sample(vec![1.0_f64, 2.0]);
        assert!(r.validate(N_GENES, N_SAMPLES).is_err());
        assert!(r.validate(N_GENES, 2).is_ok());
    }
}
