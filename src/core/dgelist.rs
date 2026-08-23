//! The count container, edgeR's `DGEList`.
//!
//! Counts are dense and gene-major: `n_genes` rows of `n_samples` values,
//! row-major, so one gene is a contiguous slice and the rayon fan-out over genes
//! reads sequential memory. edgePython densifies every input on construction and
//! this matches that, deliberately. The single-cell path does not come through
//! here at all: NEBULA takes per-gene slices directly, which is what keeps this
//! crate free of a dependency on any particular sparse container.
//!
//! Counts carry the generic `T`. Everything derived from them, library sizes,
//! normalisation factors, offsets and dispersions, is `f64` per the crate
//! numeric policy.

use crate::core::expression::column_sums;
use crate::prelude::*;

////////////////////
// DispersionKind //
////////////////////

/// Which dispersion estimate a [`DgeList`] currently carries.
///
/// Ordered by how much structure it captures. [`DgeList::dispersion`] returns
/// the most complex one available, matching edgeR's `getDispersion`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispersionKind {
    /// One dispersion per gene, shrunk towards the trend.
    Tagwise,
    /// One dispersion per gene, read off the abundance trend.
    Trended,
    /// A single dispersion shared by every gene.
    Common,
}

/////////////
// DgeList //
/////////////

/// Counts plus the per-sample and per-gene quantities edgeR hangs off them.
#[derive(Clone, Debug)]
pub struct DgeList<T> {
    /// Counts, row-major, `n_genes * n_samples`.
    pub counts: Vec<T>,
    /// Number of genes, that is, rows.
    pub n_genes: usize,
    /// Number of samples, that is, columns.
    pub n_samples: usize,
    /// Library size per sample. Defaults to the column sums of `counts`.
    pub lib_size: Vec<f64>,
    /// Normalisation factor per sample. Defaults to 1.
    pub norm_factors: Vec<f64>,
    /// Experimental group per sample, if one was supplied.
    pub group: Option<Vec<usize>>,
    /// Average log counts per million per gene.
    pub ave_log_cpm: Option<Vec<f64>>,
    /// Single dispersion shared by every gene.
    pub common_dispersion: Option<f64>,
    /// Abundance-trended dispersion, one per gene.
    pub trended_dispersion: Option<Vec<f64>>,
    /// Shrunk per-gene dispersion.
    pub tagwise_dispersion: Option<Vec<f64>>,
    /// Explicit offsets. When absent, [`DgeList::offset`] derives them from the
    /// library sizes.
    pub offset: Option<Recycled<f64>>,
    /// Observation weights.
    pub weights: Option<Recycled<f64>>,
}

impl<T: EdgeFloat> DgeList<T> {
    /// Builds a container from a dense count matrix.
    ///
    /// Library sizes are the column sums and normalisation factors start at 1,
    /// as in `DGEList`.
    ///
    /// ### Params
    ///
    /// * `counts` - Row-major counts, `n_genes * n_samples`
    /// * `n_genes` - Number of genes
    /// * `n_samples` - Number of samples
    /// * `group` - Optional group label per sample
    ///
    /// ### Returns
    ///
    /// The container, or [`EdgeErrors`] if the counts do not match the stated
    /// shape, either dimension is zero, or a count is negative or non-finite.
    pub fn new(
        counts: Vec<T>,
        n_genes: usize,
        n_samples: usize,
        group: Option<Vec<usize>>,
    ) -> Result<Self, EdgeErrors> {
        if n_genes == 0 || n_samples == 0 {
            return Err(EdgeErrors::EmptyCounts { n_genes, n_samples });
        }
        if counts.len() != n_genes * n_samples {
            return Err(EdgeErrors::LengthMismatch {
                name: "counts",
                expected: n_genes * n_samples,
                got: counts.len(),
            });
        }
        if let Some(bad) = counts.iter().find(|v| !v.is_finite() || **v < T::zero()) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "counts must be non-negative and finite, found {bad}"
            )));
        }
        if let Some(g) = &group
            && g.len() != n_samples
        {
            return Err(EdgeErrors::LengthMismatch {
                name: "group",
                expected: n_samples,
                got: g.len(),
            });
        }

        let lib_size = column_sums(&counts, n_samples);

        Ok(Self {
            counts,
            n_genes,
            n_samples,
            lib_size,
            norm_factors: vec![1.0; n_samples],
            group,
            ave_log_cpm: None,
            common_dispersion: None,
            trended_dispersion: None,
            tagwise_dispersion: None,
            offset: None,
            weights: None,
        })
    }

    /// Borrows one gene's counts.
    ///
    /// ### Params
    ///
    /// * `gene` - Gene index
    ///
    /// ### Returns
    ///
    /// A contiguous slice of `n_samples` counts.
    #[inline]
    pub fn gene(&self, gene: usize) -> &[T] {
        let start = gene * self.n_samples;
        &self.counts[start..start + self.n_samples]
    }

    /// Iterates genes as contiguous slices.
    ///
    /// ### Returns
    ///
    /// An iterator yielding one `n_samples`-long slice per gene.
    #[inline]
    pub fn genes(&self) -> impl Iterator<Item = &[T]> {
        self.counts.chunks_exact(self.n_samples)
    }

    /// Effective library sizes, that is, `lib_size * norm_factors`.
    ///
    /// ### Returns
    ///
    /// One value per sample.
    pub fn norm_lib_sizes(&self) -> Vec<f64> {
        self.lib_size
            .iter()
            .zip(self.norm_factors.iter())
            .map(|(l, n)| l * n)
            .collect()
    }

    /// The offsets a GLM fit should use.
    ///
    /// Returns the explicit offsets when set, otherwise
    /// `log(lib_size * norm_factors)` as a per-sample recycled vector. Deriving
    /// them keeps the common case at `n_samples` values rather than
    /// `n_genes * n_samples`.
    ///
    /// ### Returns
    ///
    /// The offsets, or [`EdgeErrors::InvalidArgument`] if a derived library size
    /// is not positive and finite, since its logarithm would not be usable.
    pub fn offset(&self) -> Result<Recycled<f64>, EdgeErrors> {
        if let Some(offset) = &self.offset {
            return Ok(offset.clone());
        }
        let effective = self.norm_lib_sizes();
        if let Some(bad) = effective.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "library sizes must be positive and finite, found {bad}"
            )));
        }
        Ok(Recycled::by_sample(
            effective.iter().map(|v| v.ln()).collect(),
        ))
    }

    /// The most structured dispersion currently available.
    ///
    /// Prefers tagwise, then trended, then common, matching edgeR's
    /// `getDispersion`.
    ///
    /// ### Returns
    ///
    /// The dispersion with the kind it came from, or `None` if none has been
    /// estimated yet.
    pub fn dispersion(&self) -> Option<(Recycled<f64>, DispersionKind)> {
        if let Some(d) = &self.tagwise_dispersion {
            return Some((Recycled::by_gene(d.clone()), DispersionKind::Tagwise));
        }
        if let Some(d) = &self.trended_dispersion {
            return Some((Recycled::by_gene(d.clone()), DispersionKind::Trended));
        }
        self.common_dispersion
            .map(|d| (Recycled::scalar(d), DispersionKind::Common))
    }

    /// Keeps only the genes flagged in `keep`.
    ///
    /// Per-gene quantities are subset alongside the counts. Per-sample
    /// quantities, including library sizes, are left untouched: edgeR does not
    /// recompute them after filtering, so that the normalisation stays anchored
    /// to the full library.
    ///
    /// ### Params
    ///
    /// * `keep` - One flag per gene
    ///
    /// ### Returns
    ///
    /// A new container holding the retained genes, or [`EdgeErrors`] if `keep`
    /// is the wrong length or would drop every gene.
    pub fn subset_genes(&self, keep: &[bool]) -> Result<Self, EdgeErrors> {
        if keep.len() != self.n_genes {
            return Err(EdgeErrors::LengthMismatch {
                name: "keep",
                expected: self.n_genes,
                got: keep.len(),
            });
        }
        let retained = keep.iter().filter(|k| **k).count();
        if retained == 0 {
            return Err(EdgeErrors::NoGenesAfterFiltering {
                n_genes: self.n_genes,
            });
        }

        let mut counts = Vec::with_capacity(retained * self.n_samples);
        for (gene, flag) in keep.iter().enumerate() {
            if *flag {
                counts.extend_from_slice(self.gene(gene));
            }
        }

        let pick = |v: &Option<Vec<f64>>| -> Option<Vec<f64>> {
            v.as_ref().map(|values| {
                values
                    .iter()
                    .zip(keep.iter())
                    .filter(|(_, k)| **k)
                    .map(|(x, _)| *x)
                    .collect()
            })
        };

        // A full offset or weight matrix is per gene as well as per sample, so
        // it has to follow the subset. The recycled forms do not.
        let subset_recycled = |r: &Option<Recycled<f64>>| -> Option<Recycled<f64>> {
            r.as_ref().map(|rec| match rec {
                Recycled::Full(values) => {
                    let mut out = Vec::with_capacity(retained * self.n_samples);
                    for (gene, flag) in keep.iter().enumerate() {
                        if *flag {
                            let start = gene * self.n_samples;
                            out.extend_from_slice(&values[start..start + self.n_samples]);
                        }
                    }
                    Recycled::Full(out)
                }
                Recycled::ByGene(values) => Recycled::ByGene(
                    values
                        .iter()
                        .zip(keep.iter())
                        .filter(|(_, k)| **k)
                        .map(|(x, _)| *x)
                        .collect(),
                ),
                other => other.clone(),
            })
        };

        Ok(Self {
            counts,
            n_genes: retained,
            n_samples: self.n_samples,
            lib_size: self.lib_size.clone(),
            norm_factors: self.norm_factors.clone(),
            group: self.group.clone(),
            ave_log_cpm: pick(&self.ave_log_cpm),
            common_dispersion: self.common_dispersion,
            trended_dispersion: pick(&self.trended_dispersion),
            tagwise_dispersion: pick(&self.tagwise_dispersion),
            offset: subset_recycled(&self.offset),
            weights: subset_recycled(&self.weights),
        })
    }

    /// Drops genes that are zero in every sample.
    ///
    /// ### Returns
    ///
    /// A container without the all-zero genes, or [`EdgeErrors`] if that would
    /// leave nothing.
    pub fn remove_zero_genes(&self) -> Result<Self, EdgeErrors> {
        let keep: Vec<bool> = self
            .genes()
            .map(|row| row.iter().any(|v| *v > T::zero()))
            .collect();
        self.subset_genes(&keep)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 4 genes by 3 samples, including one all-zero gene.
    fn fixture() -> DgeList<f64> {
        let counts = vec![
            1.0, 2.0, 3.0, //
            0.0, 0.0, 0.0, //
            10.0, 20.0, 30.0, //
            5.0, 0.0, 1.0, //
        ];
        DgeList::new(counts, 4, 3, Some(vec![0, 0, 1])).unwrap()
    }

    #[test]
    fn test_library_sizes_are_the_column_sums() {
        let y = fixture();
        assert_eq!(y.lib_size, vec![16.0, 22.0, 34.0]);
        assert_eq!(y.norm_factors, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gene_borrows_a_contiguous_row() {
        let y = fixture();
        assert_eq!(y.gene(2), &[10.0, 20.0, 30.0]);
        assert_eq!(y.genes().count(), 4);
    }

    #[test]
    fn test_norm_lib_sizes_apply_the_factors() {
        let mut y = fixture();
        y.norm_factors = vec![0.5, 2.0, 1.0];
        assert_eq!(y.norm_lib_sizes(), vec![8.0, 44.0, 34.0]);
    }

    #[test]
    fn test_offset_defaults_to_log_effective_library_size() {
        let y = fixture();
        let offset = y.offset().unwrap();
        assert!(matches!(offset, Recycled::BySample(_)));
        let row = offset.expand(1, 3);
        assert_relative_eq!(row[0], 16.0_f64.ln(), max_relative = 1e-12);
        assert_relative_eq!(row[2], 34.0_f64.ln(), max_relative = 1e-12);
    }

    #[test]
    fn test_offset_rejects_a_zero_library() {
        let mut y = fixture();
        y.lib_size[0] = 0.0;
        assert!(y.offset().is_err());
    }

    #[test]
    fn test_dispersion_prefers_the_most_structured_estimate() {
        let mut y = fixture();
        assert!(y.dispersion().is_none());

        y.common_dispersion = Some(0.1);
        assert_eq!(y.dispersion().unwrap().1, DispersionKind::Common);

        y.trended_dispersion = Some(vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(y.dispersion().unwrap().1, DispersionKind::Trended);

        y.tagwise_dispersion = Some(vec![0.5, 0.6, 0.7, 0.8]);
        let (d, kind) = y.dispersion().unwrap();
        assert_eq!(kind, DispersionKind::Tagwise);
        assert_eq!(d.expand(4, 1), vec![0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn test_remove_zero_genes_drops_only_the_empty_row() {
        let y = fixture().remove_zero_genes().unwrap();
        assert_eq!(y.n_genes, 3);
        assert_eq!(y.gene(0), &[1.0, 2.0, 3.0]);
        assert_eq!(y.gene(1), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_subset_carries_per_gene_quantities_along() {
        let mut y = fixture();
        y.tagwise_dispersion = Some(vec![0.1, 0.2, 0.3, 0.4]);
        y.ave_log_cpm = Some(vec![1.0, 2.0, 3.0, 4.0]);

        let sub = y.subset_genes(&[true, false, true, false]).unwrap();
        assert_eq!(sub.n_genes, 2);
        assert_eq!(sub.tagwise_dispersion.unwrap(), vec![0.1, 0.3]);
        assert_eq!(sub.ave_log_cpm.unwrap(), vec![1.0, 3.0]);
        // Library sizes stay anchored to the unfiltered data.
        assert_eq!(sub.lib_size, vec![16.0, 22.0, 34.0]);
    }

    #[test]
    fn test_subset_rewrites_a_full_offset_matrix() {
        let mut y = fixture();
        y.offset = Some(Recycled::full((0..12).map(|i| i as f64).collect(), 4, 3).unwrap());
        let sub = y.subset_genes(&[false, true, false, true]).unwrap();
        assert_eq!(
            sub.offset.unwrap().expand(2, 3),
            vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
        );
    }

    #[test]
    fn test_subset_leaves_a_per_sample_offset_alone() {
        let mut y = fixture();
        y.offset = Some(Recycled::by_sample(vec![1.0, 2.0, 3.0]));
        let sub = y.subset_genes(&[true, false, true, false]).unwrap();
        assert!(matches!(sub.offset, Some(Recycled::BySample(_))));
    }

    #[test]
    fn test_rejects_dropping_every_gene() {
        let y = fixture();
        let err = y.subset_genes(&[false; 4]).unwrap_err();
        assert!(matches!(err, EdgeErrors::NoGenesAfterFiltering { .. }));
    }

    #[test]
    fn test_rejects_negative_counts() {
        let err = DgeList::new(vec![1.0, -2.0, 3.0, 4.0], 2, 2, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let err = DgeList::new(vec![1.0_f64; 5], 2, 3, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    #[test]
    fn test_rejects_empty_counts() {
        let err = DgeList::<f64>::new(vec![], 0, 3, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    #[test]
    fn test_rejects_a_group_of_the_wrong_length() {
        let err = DgeList::new(vec![1.0_f64; 6], 2, 3, Some(vec![0, 1])).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "group", .. }
        ));
    }
}
