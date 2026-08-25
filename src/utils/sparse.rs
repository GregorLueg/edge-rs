//! Compressed sparse storage for count matrices.
//!
//! Modelled on `CompressedSparseData` in `manifolds-rs` and
//! `CompressedSparseData2` in `bixverse-rs`, with `u32` indices as in the
//! latter: a single-cell count matrix has fewer than 4 billion entries along
//! either axis, and halving the index width halves the bandwidth of every scan.
//!
//! Two operations are worth keeping straight, because the sister crates blur
//! them under one name:
//!
//! * [`CompressedSparse::transpose`] represents the transposed matrix. It flips
//!   the format label and swaps the shape, and touches no data. Free.
//! * [`CompressedSparse::convert`] represents the same matrix in the other
//!   format. It actually moves data. Not free.
//!
//! That distinction is what makes the `bixverse-rs` boundary cheap. Its chunks
//! arrive as CSC over `(n_cells, n_genes)`, which is bit-for-bit a CSR over
//! `(n_genes, n_cells)`, the gene-major layout every algorithm here wants.
//! `transpose` is the whole adapter.

use crate::prelude::*;

//////////////////
// SparseFormat //
//////////////////

/// Which axis a compressed sparse matrix is compressed along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed sparse row: `indptr` walks rows, `indices` holds column
    /// indices.
    Csr,
    /// Compressed sparse column: `indptr` walks columns, `indices` holds row
    /// indices.
    Csc,
}

impl SparseFormat {
    /// The other format.
    ///
    /// ### Returns
    ///
    /// `Csc` for `Csr` and vice versa.
    #[inline]
    pub fn flipped(&self) -> Self {
        match self {
            SparseFormat::Csr => SparseFormat::Csc,
            SparseFormat::Csc => SparseFormat::Csr,
        }
    }
}

//////////////////////
// CompressedSparse //
//////////////////////

/// A sparse matrix in compressed row or column form.
///
/// Fields are public so callers can hand the buffers straight to a kernel, but
/// [`CompressedSparse::validate`] is the only thing that guarantees they are
/// consistent. Prefer the constructors.
#[derive(Clone, Debug)]
pub struct CompressedSparse<T> {
    /// Non-zero values, ordered by the compressed axis then by `indices`.
    pub data: Vec<T>,
    /// Index along the non-compressed axis for each value in `data`.
    pub indices: Vec<u32>,
    /// Offsets into `data`/`indices`, one per compressed slice plus a trailing total.
    pub indptr: Vec<u32>,
    /// Whether `indptr` walks rows or columns.
    pub format: SparseFormat,
    /// Logical shape as `(n_rows, n_cols)`, independent of `format`.
    pub shape: (usize, usize),
}

impl<T: Copy> CompressedSparse<T> {
    /// Assembles a matrix from its parts without copying.
    ///
    /// ### Params
    ///
    /// * `data` - Non-zero values
    /// * `indices` - Index along the non-compressed axis per value
    /// * `indptr` - Offsets per compressed slice, length `outer_len + 1`
    /// * `format` - Which axis `indptr` walks
    /// * `shape` - Logical `(n_rows, n_cols)`
    ///
    /// ### Returns
    ///
    /// The matrix, or [`EdgeErrors::MalformedSparse`] if the parts disagree.
    pub fn from_parts(
        data: Vec<T>,
        indices: Vec<u32>,
        indptr: Vec<u32>,
        format: SparseFormat,
        shape: (usize, usize),
    ) -> Result<Self, EdgeErrors> {
        let out = Self {
            data,
            indices,
            indptr,
            format,
            shape,
        };
        out.validate()?;
        Ok(out)
    }

    /// Checks the structural invariants.
    ///
    /// Verifies that `data` and `indices` agree in length, that `indptr` has
    /// the right length and is non-decreasing, that it ends at the number of
    /// non-zeros, and that every index is inside the non-compressed axis.
    ///
    /// ### Returns
    ///
    /// `Ok(())` when consistent, otherwise [`EdgeErrors::MalformedSparse`].
    pub fn validate(&self) -> Result<(), EdgeErrors> {
        if self.data.len() != self.indices.len() {
            return Err(EdgeErrors::MalformedSparse(format!(
                "data has {} values but indices has {}",
                self.data.len(),
                self.indices.len()
            )));
        }

        let expected_indptr = self.outer_len() + 1;
        if self.indptr.len() != expected_indptr {
            return Err(EdgeErrors::MalformedSparse(format!(
                "indptr has length {} but the {:?} layout of a {:?} matrix needs {}",
                self.indptr.len(),
                self.format,
                self.shape,
                expected_indptr
            )));
        }

        if self.indptr[0] != 0 {
            return Err(EdgeErrors::MalformedSparse(format!(
                "indptr must start at 0, starts at {}",
                self.indptr[0]
            )));
        }

        for w in self.indptr.windows(2) {
            if w[1] < w[0] {
                return Err(EdgeErrors::MalformedSparse(format!(
                    "indptr is not non-decreasing: {} follows {}",
                    w[1], w[0]
                )));
            }
        }

        let nnz = *self.indptr.last().unwrap_or(&0) as usize;
        if nnz != self.data.len() {
            return Err(EdgeErrors::MalformedSparse(format!(
                "indptr ends at {} but data has {} values",
                nnz,
                self.data.len()
            )));
        }

        let inner = self.inner_len();
        if let Some(&bad) = self.indices.iter().find(|&&i| i as usize >= inner) {
            return Err(EdgeErrors::MalformedSparse(format!(
                "index {bad} is out of range for a non-compressed axis of length {inner}"
            )));
        }

        Ok(())
    }

    /// Number of rows.
    ///
    /// ### Returns
    ///
    /// The first component of the shape.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Number of columns.
    ///
    /// ### Returns
    ///
    /// The second component of the shape.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Number of stored non-zeros.
    ///
    /// ### Returns
    ///
    /// The length of `data`.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Length of the compressed axis, that is, how many slices `indptr` walks.
    ///
    /// ### Returns
    ///
    /// Row count for CSR, column count for CSC.
    #[inline]
    pub fn outer_len(&self) -> usize {
        match self.format {
            SparseFormat::Csr => self.shape.0,
            SparseFormat::Csc => self.shape.1,
        }
    }

    /// Length of the non-compressed axis, that is, the range `indices` spans.
    ///
    /// ### Returns
    ///
    /// Column count for CSR, row count for CSC.
    #[inline]
    pub fn inner_len(&self) -> usize {
        match self.format {
            SparseFormat::Csr => self.shape.1,
            SparseFormat::Csc => self.shape.0,
        }
    }

    /// Borrows one slice along the compressed axis.
    ///
    /// This is the accessor the per-gene fan-out uses: for a gene-major matrix
    /// it hands back that gene's non-zero cells and counts with no copying.
    ///
    /// ### Params
    ///
    /// * `outer` - Index along the compressed axis
    ///
    /// ### Returns
    ///
    /// The indices and values of that slice, in `indices` order.
    #[inline]
    pub fn outer(&self, outer: usize) -> (&[u32], &[T]) {
        let start = self.indptr[outer] as usize;
        let end = self.indptr[outer + 1] as usize;
        (&self.indices[start..end], &self.data[start..end])
    }

    /// The transposed matrix, for free.
    ///
    /// The buffers are untouched: a CSR over `(m, n)` describes exactly the same
    /// bytes as a CSC over `(n, m)`. Only the label and the shape change.
    ///
    /// ### Returns
    ///
    /// A matrix representing the transpose, sharing this one's layout.
    pub fn transpose(self) -> Self {
        Self {
            data: self.data,
            indices: self.indices,
            indptr: self.indptr,
            format: self.format.flipped(),
            shape: (self.shape.1, self.shape.0),
        }
    }

    /// The same matrix stored in the other format.
    ///
    /// This is a real transposition of the storage and costs a pass over the
    /// non-zeros plus a counting sort. Sequential for now; the intended call
    /// site, ingest in `bixverse-rs`, does this once, not in a loop.
    ///
    /// ### Returns
    ///
    /// An equivalent matrix with the opposite `format`.
    pub fn convert(&self) -> Self
    where
        T: Default,
    {
        let outer_new = self.inner_len();
        let nnz = self.nnz();

        // Counting sort: how many entries land in each new compressed slice.
        let mut counts = vec![0u32; outer_new + 1];
        for &idx in &self.indices {
            counts[idx as usize + 1] += 1;
        }
        for i in 0..outer_new {
            counts[i + 1] += counts[i];
        }
        let indptr_new = counts.clone();

        let mut data_new = vec![T::default(); nnz];
        let mut indices_new = vec![0u32; nnz];
        let mut cursor = counts;

        for outer in 0..self.outer_len() {
            let start = self.indptr[outer] as usize;
            let end = self.indptr[outer + 1] as usize;
            for k in start..end {
                let inner = self.indices[k] as usize;
                let dst = cursor[inner] as usize;
                data_new[dst] = self.data[k];
                indices_new[dst] = outer as u32;
                cursor[inner] += 1;
            }
        }

        Self {
            data: data_new,
            indices: indices_new,
            indptr: indptr_new,
            format: self.format.flipped(),
            shape: self.shape,
        }
    }

    /// Builds a compressed matrix from a dense row-major buffer.
    ///
    /// Intended for tests and small inputs; nothing on the hot path densifies.
    ///
    /// ### Params
    ///
    /// * `dense` - `n_rows * n_cols` values, row-major
    /// * `n_rows` - Number of rows
    /// * `n_cols` - Number of columns
    /// * `format` - Storage format to build
    /// * `is_zero` - Predicate deciding which values to drop
    ///
    /// ### Returns
    ///
    /// The compressed matrix, or [`EdgeErrors::LengthMismatch`] if `dense` does
    /// not match the stated shape.
    pub fn from_dense<F>(
        dense: &[T],
        n_rows: usize,
        n_cols: usize,
        format: SparseFormat,
        is_zero: F,
    ) -> Result<Self, EdgeErrors>
    where
        F: Fn(&T) -> bool,
    {
        if dense.len() != n_rows * n_cols {
            return Err(EdgeErrors::LengthMismatch {
                name: "dense",
                expected: n_rows * n_cols,
                got: dense.len(),
            });
        }

        let (outer_len, inner_len) = match format {
            SparseFormat::Csr => (n_rows, n_cols),
            SparseFormat::Csc => (n_cols, n_rows),
        };

        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = Vec::with_capacity(outer_len + 1);
        indptr.push(0u32);

        for o in 0..outer_len {
            for i in 0..inner_len {
                let flat = match format {
                    SparseFormat::Csr => o * n_cols + i,
                    SparseFormat::Csc => i * n_cols + o,
                };
                let v = dense[flat];
                if !is_zero(&v) {
                    data.push(v);
                    indices.push(i as u32);
                }
            }
            indptr.push(data.len() as u32);
        }

        Self::from_parts(data, indices, indptr, format, (n_rows, n_cols))
    }

    /// Materialises a dense row-major buffer.
    ///
    /// Tests and interop only.
    ///
    /// ### Params
    ///
    /// * `zero` - Value to fill the structural zeros with
    ///
    /// ### Returns
    ///
    /// An `n_rows * n_cols` row-major vector.
    pub fn to_dense(&self, zero: T) -> Vec<T> {
        let mut out = vec![zero; self.nrows() * self.ncols()];
        for o in 0..self.outer_len() {
            let (idx, vals) = self.outer(o);
            for (&i, &v) in idx.iter().zip(vals.iter()) {
                let flat = match self.format {
                    SparseFormat::Csr => o * self.ncols() + i as usize,
                    SparseFormat::Csc => i as usize * self.ncols() + o,
                };
                out[flat] = v;
            }
        }
        out
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 by 4, five non-zeros, deliberately including an all-zero row.
    fn dense_fixture() -> (Vec<f64>, usize, usize) {
        let dense = vec![
            1.0, 0.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 4.0, 5.0, //
        ];
        (dense, 3, 4)
    }

    #[test]
    fn test_from_dense_csr_round_trips() {
        let (dense, r, c) = dense_fixture();
        let m =
            CompressedSparse::from_dense(&dense, r, c, SparseFormat::Csr, |v| *v == 0.0).unwrap();
        assert_eq!(m.nnz(), 5);
        assert_eq!(m.to_dense(0.0), dense);
    }

    #[test]
    fn test_from_dense_csc_round_trips() {
        let (dense, r, c) = dense_fixture();
        let m =
            CompressedSparse::from_dense(&dense, r, c, SparseFormat::Csc, |v| *v == 0.0).unwrap();
        assert_eq!(m.nnz(), 5);
        assert_eq!(m.to_dense(0.0), dense);
    }

    #[test]
    fn test_convert_preserves_the_matrix() {
        let (dense, r, c) = dense_fixture();
        let csr =
            CompressedSparse::from_dense(&dense, r, c, SparseFormat::Csr, |v| *v == 0.0).unwrap();
        let csc = csr.convert();
        assert_eq!(csc.format, SparseFormat::Csc);
        assert_eq!(csc.shape, (r, c));
        assert_eq!(csc.to_dense(0.0), dense);
        // Round trip.
        assert_eq!(csc.convert().to_dense(0.0), dense);
    }

    /// The free relabel: a CSR over (m, n) is a CSC over (n, m) with the same
    /// bytes. This is the bixverse-rs boundary, so pin it explicitly.
    #[test]
    fn test_transpose_is_a_relabel_not_a_copy() {
        let (dense, r, c) = dense_fixture();
        let csr =
            CompressedSparse::from_dense(&dense, r, c, SparseFormat::Csr, |v| *v == 0.0).unwrap();
        let data_before = csr.data.clone();
        let t = csr.transpose();

        assert_eq!(t.format, SparseFormat::Csc);
        assert_eq!(t.shape, (c, r));
        assert_eq!(t.data, data_before);

        // Transposing a 3x4 gives a 4x3 whose (i, j) is the original (j, i).
        let dense_t = t.to_dense(0.0);
        for i in 0..c {
            for j in 0..r {
                assert_eq!(dense_t[i * r + j], dense[j * c + i]);
            }
        }
    }

    #[test]
    fn test_outer_borrows_a_gene_slice() {
        let (dense, r, c) = dense_fixture();
        let csr =
            CompressedSparse::from_dense(&dense, r, c, SparseFormat::Csr, |v| *v == 0.0).unwrap();

        let (idx, vals) = csr.outer(0);
        assert_eq!(idx, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);

        // The empty row is a valid, empty slice rather than a special case.
        let (idx, vals) = csr.outer(1);
        assert!(idx.is_empty());
        assert!(vals.is_empty());

        let (idx, vals) = csr.outer(2);
        assert_eq!(idx, &[1, 2, 3]);
        assert_eq!(vals, &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_validate_rejects_out_of_range_index() {
        let err = CompressedSparse::from_parts(
            vec![1.0_f64],
            vec![99],
            vec![0, 1, 1, 1],
            SparseFormat::Csr,
            (3, 4),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::MalformedSparse(_)));
    }

    #[test]
    fn test_validate_rejects_bad_indptr_length() {
        let err = CompressedSparse::from_parts(
            vec![1.0_f64],
            vec![0],
            vec![0, 1],
            SparseFormat::Csr,
            (3, 4),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::MalformedSparse(_)));
    }

    #[test]
    fn test_convert_handles_empty_matrix() {
        let m: CompressedSparse<f64> =
            CompressedSparse::from_parts(vec![], vec![], vec![0; 4], SparseFormat::Csr, (3, 4))
                .unwrap();
        let c = m.convert();
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.indptr.len(), 5);
        assert_eq!(c.to_dense(0.0), vec![0.0; 12]);
    }
}
