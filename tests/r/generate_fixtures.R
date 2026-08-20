#!/usr/bin/env Rscript
#
# Reference fixtures for the end-to-end parity tests in tests/e2e_*.rs.
#
# Generated against R 4.5.1 with edgeR 4.8.2, limma 3.66.0, statmod 1.5.2 and
# nebula 1.5.8. CI never runs this: the outputs are committed and the Rust side
# reads them. Run it by hand after an upstream bump and inspect the diff.
#
#   Rscript tests/r/generate_fixtures.R
#
# Input matrices are written once and read back on every later run, never
# re-drawn. An R upgrade can therefore only move expected values, never the data
# underneath them, so a fixture diff always means something. Delete the input
# files to force a regeneration, and expect every expected value to move.
#
# Doubles go out at 17 significant digits, which uniquely determines an f64 and
# is parsed exactly by Rust. R's own parser is not correctly rounded on Apple
# Silicon, so an as.numeric(sprintf(...)) round-trip inside R appears to lose a
# ULP on some values. That is R reading, not R writing; Rust's parser is
# correctly rounded and recovers the value. Do not widen the format to "fix" it.
#
# R prints NA_real_ as "NA", which Rust cannot parse, so NA is rewritten to NaN
# on the way out. Inf is left alone: prior degrees of freedom can legitimately
# be infinite and Rust parses it.

suppressMessages({
  library(edgeR)
  library(limma)
})

RNGkind("Mersenne-Twister", "Inversion", "Rejection")

DATA_DIR <- file.path("tests", "data", "e2e")
dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)

##############
# IO helpers #
##############

# Formats doubles at full f64 precision, mapping R's unparseable NA to NaN.
fmt <- function(x) {
  s <- sprintf("%.17g", x)
  s[is.na(x) & !is.nan(x)] <- "NaN"
  s
}

# Writes a numeric matrix, headed, no row names. One file row per matrix row, so
# a gene-by-sample matrix lands gene-major row-major as the Rust expects.
write_num <- function(m, name, header = NULL) {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  if (is.null(header)) {
    header <- if (is.null(colnames(m))) paste0("V", seq_len(ncol(m))) else colnames(m)
  }
  stopifnot(length(header) == ncol(m))
  out <- matrix(fmt(m), nrow(m), ncol(m))
  con <- file(file.path(DATA_DIR, name), "wt")
  writeLines(paste(header, collapse = ","), con)
  writeLines(apply(out, 1, paste, collapse = ","), con)
  close(con)
  invisible(NULL)
}

# Writes an integer matrix, headed, no row names.
write_int <- function(m, name, header = NULL) {
  m <- as.matrix(m)
  storage.mode(m) <- "integer"
  if (is.null(header)) {
    header <- if (is.null(colnames(m))) paste0("V", seq_len(ncol(m))) else colnames(m)
  }
  stopifnot(length(header) == ncol(m))
  con <- file(file.path(DATA_DIR, name), "wt")
  writeLines(paste(header, collapse = ","), con)
  writeLines(apply(m, 1, paste, collapse = ","), con)
  close(con)
  invisible(NULL)
}

read_num <- function(name) {
  as.matrix(read.csv(file.path(DATA_DIR, name), check.names = FALSE))
}

read_int <- function(name) {
  m <- read_num(name)
  storage.mode(m) <- "integer"
  m
}

# Scalar results, flushed once at the end as scenario,name,value triples.
SCALARS <- new.env(parent = emptyenv())
SCALARS$rows <- character(0)

put <- function(scenario, name, value) {
  SCALARS$rows <- c(
    SCALARS$rows,
    sprintf("%s,%s,%s", scenario, name, fmt(as.numeric(value)))
  )
  invisible(NULL)
}

flush_scalars <- function() {
  con <- file(file.path(DATA_DIR, "scalars.csv"), "wt")
  writeLines("scenario,name,value", con)
  writeLines(SCALARS$rows, con)
  close(con)
}

########################
# Converged squeezeVar #
########################

# limma solves for df2 with uniroot(tol = 1e-8), a bracket width in the
# d / (1 + d) link worth around 1e-4 on a df2 near 100, so its answer sits a few
# parts in 1e8 from the root of its own objective. edge-rs converges properly
# (UPSTREAM_DEVIATIONS.md, Section B, entry 12), so the reference has to be
# limma-with-converged-uniroot rather than stock limma.
#
# Only the legacy branch of squeezeVar reaches fitFDistRobustly and its uniroot.
# The current pipeline goes to fitFDistUnequalDF1, which solves with optimize and
# is reproduced exactly by the crate. The override is therefore bit-identical
# wherever the branch is not taken, which is why it is applied unconditionally
# and the entry count recorded per scenario rather than assumed. On the bulk data
# the count is zero; the single-cell shrinkage is where it earns its keep.
#
# The S3 generic must be bypassed: calling estimateDisp() or glmQLFit() goes to
# the namespace copy and silently ignores the patched environment.

UNIROOT_CALLS <- new.env(parent = emptyenv())
UNIROOT_CALLS$n <- 0L

# The `tol` formal is present so that limma's own `tol = 1e-8` binds to it and is
# discarded, rather than colliding with the tightened one in `...`.
counting_uniroot <- function(tight) {
  function(f, interval, tol = 1e-8, ...) {
    UNIROOT_CALLS$n <- UNIROOT_CALLS$n + 1L
    stats::uniroot(f, interval, tol = tight, ...)
  }
}

shadow <- function(f, name, value) {
  e <- new.env(parent = environment(f))
  assign(name, value, envir = e)
  environment(f) <- e
  f
}

tightR <- shadow(limma:::fitFDistRobustly, "uniroot", counting_uniroot(1e-14))
tightS <- shadow(limma::squeezeVar, "fitFDistRobustly", tightR)
tightED <- shadow(edgeR:::estimateDisp.default, "squeezeVar", tightS)
tightQL <- shadow(edgeR:::glmQLFit.default, "squeezeVar", tightS)

uniroot_since <- function() {
  n <- UNIROOT_CALLS$n
  UNIROOT_CALLS$n <- 0L
  n
}

###################
# Synthetic data  #
###################

# Three bulk datasets, deliberately different in structure rather than just in
# seed. Between them they cover both GLM fit dispatches, both extremes of the
# dispersion range, balanced and unbalanced designs, and residual degrees of
# freedom from five to nine.

# Dataset A, "factorial". 1000 genes by 12 samples, two groups of six crossed
# with a two-level batch, three samples per cell. Abundances span sixteen log2
# units so the locfit trend and the lowess span have real range to work with.
# The additive design does not span the four cell means, so the one-way shortcut
# cannot fire and every gene goes through Levenberg.
#
# Rows 991 to 1000 are planted edge cases rather than hoped-for ones: the rows
# that exercise filterByExpr's boundary, the +/-1e8 coefficient clamp and
# voomLmFit's structural-zero masking. Roughly half survive filtering, which is
# the point of planting both kinds.
make_factorial <- function() {
  set.seed(20260820)
  ng <- 1000L
  ns <- 12L

  grp <- factor(rep(c("A", "B"), each = 6))
  bat <- factor(rep(rep(c("b1", "b2"), each = 3), 2))

  libsize <- round(exp(rnorm(ns, log(2e7), 0.35)))
  baseline <- 2^runif(ng, -2, 14)
  baseline <- baseline / sum(baseline)

  lfc <- rep(0, ng)
  lfc[1:60] <- rnorm(60, 2, 0.5)
  lfc[61:120] <- rnorm(60, -2, 0.5)

  bfc <- rep(0, ng)
  bfc[201:260] <- rnorm(60, 1, 0.3)

  disp <- 0.1 + 1 / (baseline * 1e6 + 5)

  mu <- outer(baseline, libsize) *
    2^(outer(lfc, as.numeric(grp == "B"))) *
    2^(outer(bfc, as.numeric(bat == "b2")))

  cnt <- matrix(rnbinom(ng * ns, mu = mu, size = 1 / disp), ng, ns)

  cnt[991, ] <- 0
  cnt[992, ] <- c(0, 0, 0, 0, 0, 0, 40, 55, 38, 61, 47, 52)
  cnt[993, ] <- c(120, 95, 110, 130, 88, 101, 0, 0, 0, 0, 0, 0)
  cnt[994, ] <- 0
  cnt[994, 3] <- 1
  cnt[995, ] <- 0
  cnt[995, 8] <- 5000
  cnt[996, ] <- round(runif(ns, 5e6, 9e6))
  cnt[997, ] <- 20L
  cnt[998, ] <- c(3, 4, 2, 5, 3, 4, 3, 2, 4, 3, 5, 2)
  cnt[999, ] <- c(1, 0, 2, 1, 0, 1, 900000, 1, 0, 2, 1, 0)
  cnt[1000, ] <- 1L

  storage.mode(cnt) <- "integer"
  colnames(cnt) <- sprintf("s%02d", seq_len(ns))
  list(counts = cnt, grp = grp, bat = bat)
}

# Dataset B, "unbalanced". 600 genes by 9 samples, three groups of 2, 3 and 4,
# plus a continuous covariate. Structurally the opposite of A in three ways that
# matter: the continuous column forces Levenberg for a different reason, four
# coefficients on nine samples leaves five residual degrees of freedom so the
# empirical Bayes prior dominates rather than nudges, and unequal group sizes put
# the exact test on its n1 != n2 path. Dispersion is higher and the abundance
# range narrower, so the trend has less to hold on to.
make_unbalanced <- function() {
  set.seed(20260824)
  ng <- 600L
  ns <- 9L

  grp <- factor(rep(c("g1", "g2", "g3"), times = c(2, 3, 4)))
  score <- round(scale(c(3.1, 7.4, 5.2, 9.8, 1.6, 6.3, 8.9, 2.7, 4.5))[, 1], 6)

  libsize <- round(exp(rnorm(ns, log(6e6), 0.5)))
  baseline <- 2^runif(ng, 0, 11)
  baseline <- baseline / sum(baseline)

  lfc2 <- rep(0, ng)
  lfc2[1:70] <- rnorm(70, 1.6, 0.6)
  lfc3 <- rep(0, ng)
  lfc3[51:120] <- rnorm(70, -1.8, 0.6)
  sfc <- rep(0, ng)
  sfc[301:370] <- rnorm(70, 0.7, 0.25)

  disp <- 0.2 + 0.4 * exp(-baseline * 1e6 / 200)

  mu <- outer(baseline, libsize) *
    2^(outer(lfc2, as.numeric(grp == "g2"))) *
    2^(outer(lfc3, as.numeric(grp == "g3"))) *
    2^(outer(sfc, score))

  cnt <- matrix(rnbinom(ng * ns, mu = mu, size = 1 / disp), ng, ns)

  cnt[596, ] <- 0
  cnt[597, ] <- c(0, 0, 30, 41, 26, 33, 45, 38, 29)
  cnt[598, ] <- 8L
  cnt[599, ] <- 0
  cnt[599, 5] <- 12L
  cnt[600, ] <- round(runif(ns, 2e6, 3e6))

  storage.mode(cnt) <- "integer"
  colnames(cnt) <- sprintf("s%02d", seq_len(ns))
  list(counts = cnt, grp = grp, score = score)
}

# Dataset C, "poisson". 300 genes by 8 samples, two groups of four, counts drawn
# essentially Poisson with means between roughly 0.5 and 40. The group-only
# design is a factor coding, so this is the only dataset that reaches the one-way
# fitter. It is also the only one where the quasi-likelihood Poisson bound does
# anything: the bound is a no-op unless the residual deviance falls below what
# Poisson noise alone would give.
make_poisson <- function() {
  set.seed(20260823)
  ng <- 300L
  ns <- 8L

  grp <- factor(rep(c("ctl", "trt"), each = 4))
  mu <- matrix(rep(runif(ng, 0.5, 40), ns), ng, ns)
  cnt <- matrix(rpois(ng * ns, mu), ng, ns)
  cnt[1:20, 5:8] <- cnt[1:20, 5:8] + rpois(80, 30)

  cnt[296, ] <- 0
  cnt[297, ] <- 1L
  cnt[298, ] <- c(0, 0, 0, 0, 4, 6, 3, 5)
  cnt[299, ] <- 0
  cnt[299, 2] <- 2L
  cnt[300, ] <- rpois(ns, 900)

  storage.mode(cnt) <- "integer"
  colnames(cnt) <- sprintf("s%d", seq_len(ns))
  list(counts = cnt, grp = grp)
}

#################
# Write inputs  #
#################

if (!file.exists(file.path(DATA_DIR, "fac_counts.csv"))) {
  a <- make_factorial()
  write_int(a$counts, "fac_counts.csv")
  d <- model.matrix(~ a$grp + a$bat)
  colnames(d) <- c("Int", "grpB", "batb2")
  write_num(d, "fac_design.csv")
  dg <- model.matrix(~ a$grp)
  colnames(dg) <- c("Int", "grpB")
  write_num(dg, "fac_design_grp.csv")
  write_int(
    cbind(grp = as.integer(a$grp) - 1L, bat = as.integer(a$bat) - 1L),
    "fac_samples.csv"
  )
  # Contrasts are column-major n_coef * n_contrasts on the Rust side, so the
  # transpose is written: reading these rows row-major yields that layout.
  con <- makeContrasts(grpB, grpB - 0.5 * batb2, levels = d)
  write_num(t(con), "fac_contrasts.csv", colnames(d))
  cat("wrote factorial inputs\n")
}

if (!file.exists(file.path(DATA_DIR, "unbal_counts.csv"))) {
  b <- make_unbalanced()
  write_int(b$counts, "unbal_counts.csv")
  d <- model.matrix(~ b$grp + b$score)
  colnames(d) <- c("Int", "g2", "g3", "score")
  write_num(d, "unbal_design.csv")
  dg <- model.matrix(~ b$grp)
  colnames(dg) <- c("Int", "g2", "g3")
  write_num(dg, "unbal_design_grp.csv")
  write_num(
    cbind(grp = as.integer(b$grp) - 1L, score = b$score),
    "unbal_samples.csv",
    c("grp", "score")
  )
  con <- makeContrasts(g3 - g2, levels = d)
  write_num(t(con), "unbal_contrasts.csv", colnames(d))
  cat("wrote unbalanced inputs\n")
}

if (!file.exists(file.path(DATA_DIR, "pois_counts.csv"))) {
  p <- make_poisson()
  write_int(p$counts, "pois_counts.csv")
  d <- model.matrix(~ p$grp)
  colnames(d) <- c("Int", "trt")
  write_num(d, "pois_design.csv")
  write_num(d, "pois_design_grp.csv")
  write_int(cbind(grp = as.integer(p$grp) - 1L), "pois_samples.csv")
  cat("wrote poisson inputs\n")
}

################
# Core pipeline #
################

# The chain every dataset goes through: filter, normalise, dispersions, GLM fit,
# likelihood ratio test, quasi-likelihood fit and F test, exact test, topTags.
# Everything is written under the dataset's tag so the Rust tests can run the
# same assertions three times over different data.
#
# `coef` is the design column under test, one-based here and one less on the Rust
# side. `pair` names the two groups for the exact test.
run_core <- function(tag, counts, des, des_grp, grp, coef, pair) {
  ns <- ncol(counts)
  y <- DGEList(counts = counts, group = grp)

  # -- filtering. The DGEList method multiplies the library sizes by the
  # normalisation factors, a no-op in the usual filter-then-normalise order but
  # not once the factors are set, so both orderings are pinned.
  keep_default <- filterByExpr(y)
  keep_group <- filterByExpr(y, group = grp)
  keep_design <- filterByExpr(y, design = des)
  keep_after_norm <- filterByExpr(calcNormFactors(y), design = des)
  write_int(
    cbind(
      default = as.integer(keep_default),
      group = as.integer(keep_group),
      design = as.integer(keep_design),
      after_norm = as.integer(keep_after_norm)
    ),
    paste0(tag, "_filter.csv")
  )
  put(tag, "filter_n_default", sum(keep_default))
  put(tag, "filter_n_group", sum(keep_group))
  put(tag, "filter_n_design", sum(keep_design))
  put(tag, "filter_n_after_norm", sum(keep_after_norm))

  # -- normalisation, on the unfiltered matrix so the sweep is independent of
  # the filter above.
  lib_all <- colSums(counts)
  methods <- c("TMM", "TMMwsp", "RLE", "upperquartile", "none")
  nf <- sapply(methods, function(m) calcNormFactors(counts, lib.size = lib_all, method = m))
  # refColumn is one-based in R and zero-based in Rust: column index 2 there.
  nf_ref <- calcNormFactors(counts, lib.size = lib_all, method = "TMM", refColumn = 3)
  write_num(
    cbind(lib_size = lib_all, nf, tmm_ref3 = nf_ref),
    paste0(tag, "_norm.csv"),
    c("lib_size", "tmm", "tmmwsp", "rle", "uq", "none", "tmm_ref3")
  )

  # -- the filtered, normalised object every later stage works from.
  yf <- calcNormFactors(y[keep_design, , keep.lib.sizes = FALSE])
  cf <- yf$counts
  ng <- nrow(cf)
  eff_lib <- yf$samples$lib.size * yf$samples$norm.factors
  off <- getOffset(yf)
  put(tag, "n_kept", ng)

  write_int(cf, paste0(tag, "_kept_counts.csv"))
  write_num(
    cbind(lib_size = yf$samples$lib.size, norm_factors = yf$samples$norm.factors, offset = off),
    paste0(tag, "_kept_samples.csv"),
    c("lib_size", "norm_factors", "offset")
  )

  # -- expression. aveLogCPM's dispersion argument defaults to 0.05 rather than
  # zero, so the second column has to move it or it would duplicate the first.
  write_num(cpm(cf, lib.size = eff_lib), paste0(tag, "_cpm.csv"),
            paste0("cpm", seq_len(ns)))
  write_num(cpm(cf, lib.size = eff_lib, log = TRUE, prior.count = 2),
            paste0(tag, "_logcpm.csv"), paste0("logcpm", seq_len(ns)))
  write_num(
    cbind(
      default = aveLogCPM(cf, lib.size = eff_lib),
      disp02 = aveLogCPM(cf, lib.size = eff_lib, dispersion = 0.2),
      with_offset = aveLogCPM(cf, offset = off)
    ),
    paste0(tag, "_avelogcpm.csv"),
    c("default", "disp02", "with_offset")
  )

  # -- dispersions.
  dd <- estimateDisp(yf, des)
  put(tag, "common_dispersion", dd$common.dispersion)
  put(tag, "span", dd$span)
  put(tag, "prior_df", dd$prior.df[1])
  put(tag, "prior_df_len", length(dd$prior.df))
  put(tag, "prior_n", dd$prior.n)
  write_num(
    cbind(
      ave_log_cpm = dd$AveLogCPM,
      trended = dd$trended.dispersion,
      tagwise = dd$tagwise.dispersion
    ),
    paste0(tag, "_disp.csv"),
    c("ave_log_cpm", "trended", "tagwise")
  )

  # -- GLM fit. edgeR reports coefficients on the natural log scale, shrunk by
  # the prior count, with the unshrunk pair kept alongside. Genes with a whole
  # empty group land on the +/-1e8 clamp, which is why they are worth planting.
  fit <- glmFit(dd, des)
  ncoef <- ncol(des)
  write_num(
    cbind(fit$coefficients, fit$unshrunk.coefficients, deviance = fit$deviance),
    paste0(tag, "_glmfit.csv"),
    c(paste0("coef", seq_len(ncoef)), paste0("unshrunk", seq_len(ncoef)), "deviance")
  )
  write_num(fit$fitted.values, paste0(tag, "_glmfitted.csv"), paste0("fit", seq_len(ns)))
  put(tag, "glm_df_residual", fit$df.residual[1])
  put(tag, "glm_method_oneway", as.numeric(identical(fit$method, "oneway")))

  # -- likelihood ratio test. The log p-value goes out alongside the raw one:
  # the planted zero-group genes can underflow to zero, where a relative
  # comparison is meaningless and the log scale is the only usable check.
  lrt <- glmLRT(fit, coef = coef)
  write_num(
    cbind(
      logFC = lrt$table$logFC,
      logCPM = lrt$table$logCPM,
      LR = lrt$table$LR,
      PValue = lrt$table$PValue,
      logPValue = pchisq(lrt$table$LR, df = lrt$df.test, lower.tail = FALSE, log.p = TRUE),
      FDR = p.adjust(lrt$table$PValue, "BH")
    ),
    paste0(tag, "_lrt.csv"),
    c("logFC", "logCPM", "LR", "PValue", "logPValue", "FDR")
  )
  put(tag, "lrt_df_test", lrt$df.test[1])

  # -- quasi-likelihood. The .default form is used rather than the DGEList one:
  # the DGEList method precomputes its own scalar dispersion from a hardcoded
  # top decile and bypasses top.proportion entirely, which is not what
  # glm_ql_fit(dispersion = None) does.
  invisible(uniroot_since())
  ql <- tightQL(
    y = cf, design = des, dispersion = NULL, offset = off,
    AveLogCPM = dd$AveLogCPM, legacy = FALSE
  )
  put(tag, "ql_uniroot_calls", uniroot_since())
  put(tag, "ql_average_ql_dispersion", ql$average.ql.dispersion)
  put(tag, "ql_top_proportion", ql$top.proportion)
  put(tag, "ql_dispersion", ql$dispersion[1])
  put(tag, "ql_df_prior_len", length(ql$df.prior))
  put(tag, "ql_s2_prior_len", length(ql$s2.prior))

  qlt <- glmQLFTest(ql, coef = coef)
  write_num(
    cbind(
      ql$coefficients,
      deviance = ql$deviance,
      s2_post = ql$s2.post,
      s2_prior = rep_len(ql$s2.prior, ng),
      df_prior = rep_len(ql$df.prior, ng),
      df_residual_adj = ql$df.residual.adj,
      deviance_adj = ql$deviance.adj,
      F = qlt$table$F,
      PValue = qlt$table$PValue,
      logPValue = pf(qlt$table$F, qlt$df.test, qlt$df.total,
                     lower.tail = FALSE, log.p = TRUE),
      FDR = p.adjust(qlt$table$PValue, "BH"),
      df_total = qlt$df.total
    ),
    paste0(tag, "_ql.csv"),
    c(paste0("coef", seq_len(ncoef)), "deviance", "s2_post", "s2_prior", "df_prior",
      "df_residual_adj", "deviance_adj", "F", "PValue", "logPValue", "FDR", "df_total")
  )
  put(tag, "qlf_df_test", qlt$df.test[1])

  # -- exact test, on the group-only dispersion as the classic workflow does.
  de <- estimateDisp(yf, des_grp)
  et <- exactTest(de, pair = pair)
  et_big5 <- exactTest(de, pair = pair, big.count = 5)
  # exactTest reuses the DGEList's stored AveLogCPM rather than recomputing one,
  # and estimateDisp computes that at its own first-pass common dispersion rather
  # than aveLogCPM's 0.05 default. The crate does the same when the field is set,
  # so the stored value has to travel with the fixture.
  write_num(
    cbind(
      tagwise = de$tagwise.dispersion,
      ave_log_cpm = de$AveLogCPM,
      logFC = et$table$logFC,
      logCPM = et$table$logCPM,
      PValue = et$table$PValue,
      FDR = p.adjust(et$table$PValue, "BH"),
      logFC_big5 = et_big5$table$logFC,
      PValue_big5 = et_big5$table$PValue
    ),
    paste0(tag, "_exact.csv"),
    c("tagwise", "ave_log_cpm", "logFC", "logCPM", "PValue", "FDR",
      "logFC_big5", "PValue_big5")
  )
  put(tag, "exact_common_dispersion", de$common.dispersion)

  # -- topTags. FDR is BH over the whole table, so it does not move with n, and
  # the index column is one-based here.
  for (v in c("PValue", "logFC", "none")) {
    tt <- topTags(lrt, n = 50, sort.by = v)
    write_num(
      cbind(
        index = match(rownames(tt$table), rownames(lrt$table)),
        logFC = tt$table$logFC,
        logCPM = tt$table$logCPM,
        LR = tt$table$LR,
        PValue = tt$table$PValue,
        FDR = tt$table$FDR
      ),
      paste0(tag, "_toptags_", v, ".csv"),
      c("index", "logFC", "logCPM", "LR", "PValue", "FDR")
    )
  }

  cat(sprintf(
    "%s: kept %d/%d  common %.6g  span %.6g  df %d  method %s\n",
    tag, ng, nrow(counts), dd$common.dispersion, dd$span, fit$df.residual[1], fit$method
  ))

  invisible(list(yf = yf, dd = dd, fit = fit, ql = ql, off = off, ng = ng))
}

fac_counts <- read_int("fac_counts.csv")
fac_samples <- read.csv(file.path(DATA_DIR, "fac_samples.csv"))
fac_grp <- factor(fac_samples$grp, levels = c(0, 1), labels = c("A", "B"))
fac_des <- read_num("fac_design.csv")
fac_des_grp <- read_num("fac_design_grp.csv")

unbal_counts <- read_int("unbal_counts.csv")
unbal_samples <- read.csv(file.path(DATA_DIR, "unbal_samples.csv"))
unbal_grp <- factor(unbal_samples$grp, levels = c(0, 1, 2), labels = c("g1", "g2", "g3"))
unbal_des <- read_num("unbal_design.csv")
unbal_des_grp <- read_num("unbal_design_grp.csv")

pois_counts <- read_int("pois_counts.csv")
pois_samples <- read.csv(file.path(DATA_DIR, "pois_samples.csv"))
pois_grp <- factor(pois_samples$grp, levels = c(0, 1), labels = c("ctl", "trt"))
pois_des <- read_num("pois_design.csv")

fac <- run_core("fac", fac_counts, fac_des, fac_des_grp, fac_grp, 2, c("A", "B"))
unbal <- run_core("unbal", unbal_counts, unbal_des, unbal_des_grp, unbal_grp, 2, c("g1", "g3"))
pois <- run_core("pois", pois_counts, pois_des, pois_des, pois_grp, 2, c("ctl", "trt"))

##########################################
# Factorial extras: sweeps and contrasts #
##########################################

# The trend method sweep only needs running once, and the factorial set is the
# one with enough abundance range for the smoothers to differ meaningfully.
trend_methods <- c("none", "movingave", "loess", "locfit", "locfit.mixed")
trend_fits <- lapply(trend_methods, function(tm) estimateDisp(fac$yf, fac_des, trend.method = tm))
names(trend_fits) <- trend_methods

for (tm in trend_methods) {
  d <- trend_fits[[tm]]
  put("fac_trend", paste0("common_", tm), d$common.dispersion)
  put("fac_trend", paste0("span_", tm), if (is.null(d$span)) NA_real_ else d$span)
}

# trend.method = "none" leaves no trended dispersion at all, so it contributes a
# tagwise column only.
write_num(
  cbind(
    do.call(cbind, lapply(trend_methods[-1], function(tm) trend_fits[[tm]]$trended.dispersion)),
    do.call(cbind, lapply(trend_methods, function(tm) trend_fits[[tm]]$tagwise.dispersion))
  ),
  "fac_trend.csv",
  c(paste0("trended_", trend_methods[-1]), paste0("tagwise_", trend_methods))
)

d_notag <- estimateDisp(fac$yf, fac_des, tagwise = FALSE)
d_pdf10 <- estimateDisp(fac$yf, fac_des, prior.df = 10)
put("fac_trend", "common_no_tagwise", d_notag$common.dispersion)
put("fac_trend", "prior_n_pdf10", d_pdf10$prior.n)
write_num(
  cbind(trended_no_tagwise = d_notag$trended.dispersion, tagwise_pdf10 = d_pdf10$tagwise.dispersion),
  "fac_trend_extra.csv",
  c("trended_no_tagwise", "tagwise_pdf10")
)

# The robust dispersion fit, through the converged-uniroot estimateDisp.
invisible(uniroot_since())
d_robust <- tightED(
  y = fac$yf$counts, design = fac_des, offset = fac$off,
  AveLogCPM = fac$dd$AveLogCPM, robust = TRUE
)
put("fac_trend", "uniroot_calls_robust_disp", uniroot_since())
put("fac_trend", "common_robust", d_robust$common.dispersion)
put("fac_trend", "prior_df_robust_len", length(d_robust$prior.df))
write_num(
  cbind(
    prior_df = rep_len(d_robust$prior.df, fac$ng),
    trended = d_robust$trended.dispersion,
    tagwise = d_robust$tagwise.dispersion
  ),
  "fac_disp_robust.csv",
  c("prior_df", "trended", "tagwise")
)

# Multi-coefficient and contrast tests. glmTreat reports no statistic column at
# all, so only the fold change, abundance and p-value are gated for it.
fac_con <- read_num("fac_contrasts.csv")
lrt_multi <- glmLRT(fac$fit, coef = 2:3)
lrt_con1 <- glmLRT(fac$fit, contrast = fac_con[1, ])
lrt_con2 <- glmLRT(fac$fit, contrast = fac_con[2, ])
put("fac_glm", "lrt_multi_df_test", lrt_multi$df.test[1])
put("fac_glm", "lrt_con1_df_test", lrt_con1$df.test[1])
put("fac_glm", "lrt_con2_df_test", lrt_con2$df.test[1])

write_num(
  cbind(
    multi_logFC = lrt_multi$table[[1]],
    multi_LR = lrt_multi$table$LR,
    multi_PValue = lrt_multi$table$PValue,
    con1_logFC = lrt_con1$table$logFC,
    con1_LR = lrt_con1$table$LR,
    con1_PValue = lrt_con1$table$PValue,
    con2_logFC = lrt_con2$table$logFC,
    con2_LR = lrt_con2$table$LR,
    con2_PValue = lrt_con2$table$PValue
  ),
  "fac_lrt_contrast.csv",
  c("multi_logFC", "multi_LR", "multi_PValue",
    "con1_logFC", "con1_LR", "con1_PValue",
    "con2_logFC", "con2_LR", "con2_PValue")
)

tr_int <- glmTreat(fac$fit, coef = 2, lfc = 1, null = "interval")
tr_wc <- glmTreat(fac$fit, coef = 2, lfc = 1, null = "worst.case")
tr_ql <- glmTreat(fac$ql, coef = 2, lfc = 1, null = "interval")
write_num(
  cbind(
    interval_logFC = tr_int$table$logFC,
    interval_unshrunk = tr_int$table$unshrunk.logFC,
    interval_PValue = tr_int$table$PValue,
    worstcase_PValue = tr_wc$table$PValue,
    ql_logFC = tr_ql$table$logFC,
    ql_PValue = tr_ql$table$PValue
  ),
  "fac_treat.csv",
  c("interval_logFC", "interval_unshrunk", "interval_PValue",
    "worstcase_PValue", "ql_logFC", "ql_PValue")
)

# The quasi-likelihood F test through a contrast rather than a coefficient.
qlt_con <- glmQLFTest(fac$ql, contrast = fac_con[2, ])
write_num(
  cbind(
    logFC = qlt_con$table$logFC,
    F = qlt_con$table$F,
    PValue = qlt_con$table$PValue
  ),
  "fac_qlf_contrast.csv",
  c("logFC", "F", "PValue")
)
cat("factorial extras written\n")

#############################################
# Poisson extras: legacy QL and the bound   #
#############################################

# The legacy pipeline is the only one that carries df.residual.zeros, and the
# Poisson bound is dead without it: glmQLFTest silently forces poisson.bound to
# FALSE when the slot is absent. Near-Poisson counts are what make the bound
# actually move a p-value, which is why this lives on dataset C.
#
# legacy = TRUE with dispersion = NULL errors out, so the trended dispersion has
# to be supplied explicitly.
invisible(uniroot_since())
pois_ql_legacy <- tightQL(
  y = pois$yf$counts, design = pois_des,
  dispersion = pois$dd$trended.dispersion, offset = pois$off,
  AveLogCPM = pois$dd$AveLogCPM, legacy = TRUE
)
put("pois_legacy", "uniroot_calls", uniroot_since())
put("pois_legacy", "df_prior_len", length(pois_ql_legacy$df.prior))

qlt_bound <- glmQLFTest(pois_ql_legacy, coef = 2, poisson.bound = TRUE)
qlt_nobound <- glmQLFTest(pois_ql_legacy, coef = 2, poisson.bound = FALSE)
put("pois_legacy", "n_bound_differs", sum(qlt_bound$table$PValue != qlt_nobound$table$PValue))

write_num(
  cbind(
    s2_post = pois_ql_legacy$s2.post,
    s2_prior = rep_len(pois_ql_legacy$s2.prior, pois$ng),
    df_prior = rep_len(pois_ql_legacy$df.prior, pois$ng),
    df_residual_zeros = pois_ql_legacy$df.residual.zeros,
    deviance = pois_ql_legacy$deviance,
    F_bound = qlt_bound$table$F,
    PValue_bound = qlt_bound$table$PValue,
    F_nobound = qlt_nobound$table$F,
    PValue_nobound = qlt_nobound$table$PValue,
    df_total = qlt_bound$df.total
  ),
  "pois_ql_legacy.csv",
  c("s2_post", "s2_prior", "df_prior", "df_residual_zeros", "deviance",
    "F_bound", "PValue_bound", "F_nobound", "PValue_nobound", "df_total")
)
cat(sprintf("poisson legacy: bound moves %d of %d p-values\n",
            sum(qlt_bound$table$PValue != qlt_nobound$table$PValue), pois$ng))

########
# voom #
########

# voomLmFit with block = NULL and sample.weights = FALSE, which is what the
# crate's voom_lmfit implements. Its defaults already agree with VoomParams:
# span 0.5, adaptive.span TRUE, and a prior count hardcoded at 0.5 upstream.
#
# normalize.method stays at "none" throughout. The crate's normalize_method is
# edgeR's calcNormFactors, scaling the library sizes, where limma's is
# normalizeBetweenArrays, shifting the log ratios. They coincide only here, so a
# non-default fixture would compare two different operations.
#
# save.plot = TRUE is what stores the fitted trend; without it voom.xy and
# voom.line are absent. MArrayLM carries no fitted.values either, so the fitted
# matrix is reconstructed from the coefficients.
run_voom <- function(tag, counts, des) {
  ns <- ncol(counts)
  ncoef <- ncol(des)
  v <- voomLmFit(counts, des, block = NULL, sample.weights = FALSE, save.plot = TRUE)

  write_num(v$EList$E, paste0(tag, "_voom_e.csv"), paste0("e", seq_len(ns)))
  write_num(v$EList$weights, paste0(tag, "_voom_weights.csv"), paste0("w", seq_len(ns)))
  write_num(
    cbind(v$coefficients, v$stdev.unscaled, sigma = v$sigma,
          df_residual = v$df.residual, amean = v$Amean),
    paste0(tag, "_voom_lmfit.csv"),
    c(paste0("coef", seq_len(ncoef)), paste0("stdev", seq_len(ncoef)),
      "sigma", "df_residual", "amean")
  )
  write_num(v$coefficients %*% t(des), paste0(tag, "_voom_fitted.csv"),
            paste0("fit", seq_len(ns)))
  write_num(cbind(x = v$voom.line$x, y = v$voom.line$y),
            paste0(tag, "_voom_trend.csv"), c("x", "y"))

  # squeezeVar on the resulting variances. The residual df are not constant here
  # (the structural-zero genes lose observations), so legacy = NULL auto-resolves
  # to FALSE and the fit goes through fitFDistUnequalDF1. The scalar-df call
  # forces the legacy branch so both are gated.
  invisible(uniroot_since())
  sq <- tightS(v$sigma^2, df = v$df.residual)
  sq_trend <- tightS(v$sigma^2, df = v$df.residual, covariate = v$Amean)
  sq_robust <- tightS(v$sigma^2, df = v$df.residual, robust = TRUE)
  df_one <- max(v$df.residual)
  sq_legacy <- tightS(v$sigma^2, df = df_one)
  put(paste0(tag, "_voom"), "squeeze_uniroot_calls", uniroot_since())

  ng <- nrow(counts)
  write_num(
    cbind(
      var_post = sq$var.post,
      var_prior = rep_len(sq$var.prior, ng),
      trend_var_post = sq_trend$var.post,
      trend_var_prior = rep_len(sq_trend$var.prior, ng),
      robust_var_post = sq_robust$var.post,
      robust_df_prior = rep_len(sq_robust$df.prior, ng),
      legacy_var_post = sq_legacy$var.post
    ),
    paste0(tag, "_voom_squeeze.csv"),
    c("var_post", "var_prior", "trend_var_post", "trend_var_prior",
      "robust_var_post", "robust_df_prior", "legacy_var_post")
  )

  put(paste0(tag, "_voom"), "df_prior", sq$df.prior[1])
  put(paste0(tag, "_voom"), "var_prior", sq$var.prior[1])
  put(paste0(tag, "_voom"), "trend_df_prior", sq_trend$df.prior[1])
  put(paste0(tag, "_voom"), "robust_df_prior_len", length(sq_robust$df.prior))
  put(paste0(tag, "_voom"), "legacy_df_prior", sq_legacy$df.prior[1])
  put(paste0(tag, "_voom"), "legacy_df_one", df_one)
  put(paste0(tag, "_voom"), "n_distinct_df", length(unique(v$df.residual)))

  cat(sprintf("%s voom: df.residual {%s}, trend %d points\n", tag,
              paste(sort(unique(v$df.residual)), collapse = ","),
              length(v$voom.line$x)))
}

run_voom("fac", fac$yf$counts, fac_des)
run_voom("unbal", unbal$yf$counts, unbal_des)

###############
# Single cell #
###############

# NEBULA silently downgrades method = "LN" to "HL" below thirty cells per
# subject (src/sc/nebula.rs:90, and `mfs <- nind/k` in nebula's own body). The
# existing eight-gene golden sits at 25 cells per subject and so only ever
# reaches HL. Two datasets are generated to pin both sides of that threshold at
# a realistic gene count: one at 67 cells per subject and one at 20.
#
# Counts are gene-major and cells must already be grouped contiguously by
# subject, which is what nebula and the Rust entry point both require.
make_sc <- function(seed, ng, nc, nsub, base_hi, off_mu) {
  set.seed(seed)
  stopifnot(nc %% nsub == 0)

  id <- rep(seq_len(nsub), each = nc / nsub)
  grp <- rep(0:1, length.out = nsub)[id]
  cov2 <- round(scale(rnorm(nc))[, 1], 8)

  base <- 2^runif(ng, -3, base_hi)
  lfc <- rep(0, ng)
  n_de <- floor(ng / 7.5)
  lfc[seq_len(n_de)] <- rnorm(n_de, 0.8, 0.2)
  lfc[(n_de + 1):(2 * n_de)] <- rnorm(n_de, -0.8, 0.2)

  re <- matrix(rnorm(ng * nsub, 0, 0.3), ng, nsub)
  sf <- round(exp(rnorm(nc, log(off_mu), 0.3)))

  eta <- log(base) %o% rep(1, nc) +
    outer(lfc, grp) +
    re[, id] +
    0.3 * outer(rep(1, ng), cov2)
  mu <- exp(eta) * rep(sf / off_mu, each = ng)
  phi <- 1 / runif(ng, 0.5, 3)

  cnt <- matrix(rnbinom(ng * nc, mu = as.vector(mu), size = rep(phi, nc)), ng, nc)

  # Planted: two genes below nebula's expression filter, one just above it, one
  # near-Poisson and very high, one expressed in only half the subjects.
  cnt[ng - 4, ] <- 0
  cnt[ng - 3, ] <- rbinom(nc, 1, 0.004)
  cnt[ng - 2, ] <- rbinom(nc, 1, 0.02) * 3
  cnt[ng - 1, ] <- rpois(nc, 300)
  cnt[ng, ] <- 0
  half <- id <= (nsub %/% 2)
  cnt[ng, half] <- rpois(sum(half), 8)

  storage.mode(cnt) <- "integer"
  list(counts = cnt, id = id, grp = grp, cov2 = cov2, sf = sf)
}

if (!file.exists(file.path(DATA_DIR, "sc_counts.csv"))) {
  s <- make_sc(20260821, 300L, 1005L, 15L, 5, 1000)
  write_int(s$counts, "sc_counts.csv", paste0("c", seq_len(ncol(s$counts))))
  write_num(
    cbind(subject = s$id, grp = s$grp, cov2 = s$cov2, offset = s$sf),
    "sc_meta.csv",
    c("subject", "grp", "cov2", "offset")
  )
  cat("wrote sc large inputs\n")
}

if (!file.exists(file.path(DATA_DIR, "sc_small_counts.csv"))) {
  s <- make_sc(20260822, 120L, 300L, 15L, 5, 800)
  write_int(s$counts, "sc_small_counts.csv", paste0("c", seq_len(ncol(s$counts))))
  write_num(
    cbind(subject = s$id, grp = s$grp, cov2 = s$cov2, offset = s$sf),
    "sc_small_meta.csv",
    c("subject", "grp", "cov2", "offset")
  )
  cat("wrote sc small inputs\n")
}

if (requireNamespace("nebula", quietly = TRUE)) {
  run_sc <- function(tag, counts_file, meta_file) {
    cnt <- read_int(counts_file)
    meta <- read.csv(file.path(DATA_DIR, meta_file))
    pred <- cbind(1, grp = meta$grp, cov2 = meta$cov2)
    nsub <- length(unique(meta$subject))
    cells_per <- nrow(meta) / nsub

    res <- nebula::nebula(
      cnt, meta$subject, pred = pred, offset = meta$offset,
      model = "NBGMM", method = "LN", covariance = TRUE,
      verbose = FALSE, ncore = 1
    )

    # The first thirteen columns are laid out exactly as the existing golden in
    # tests/data/nebula_expected.csv, which the Rust reader already knows. The
    # algorithm code is appended: nebula picks a sub-path per gene, and which one
    # it took explains most of how closely the port can be expected to agree.
    algo <- match(res$algorithm, c("NBGMM (LN)", "NBGMM (LN+HL)", "NBGMM (HL)"))
    write_num(
      cbind(as.matrix(res$summary[, 1:9]),
            gene_id = res$summary$gene_id,
            Subject = res$overdispersion$Subject,
            Cell = res$overdispersion$Cell,
            convergence = res$convergence,
            algorithm = algo),
      paste0(tag, "_nebula.csv"),
      c("logFC_int", "logFC_grp", "logFC_cov2",
        "se_int", "se_grp", "se_cov2",
        "p_int", "p_grp", "p_cov2",
        "gene_id", "Subject", "Cell", "convergence", "algorithm")
    )
    for (k in seq_along(c("ln", "lnhl", "hl"))) {
      put(tag, paste0("n_algo_", c("ln", "lnhl", "hl")[k]), sum(algo == k, na.rm = TRUE))
    }
    # R packs the covariance lower-triangular column-major; the Rust repacks it.
    write_num(as.matrix(res$covariance), paste0(tag, "_covariance.csv"),
              paste0("cov_", seq_len(ncol(res$covariance))))

    put(tag, "n_genes_in", nrow(cnt))
    put(tag, "n_genes_out", nrow(res$summary))
    put(tag, "n_subjects", nsub)
    put(tag, "cells_per_subject", cells_per)
    put(tag, "n_ln", sum(grepl("LN", res$algorithm)))
    put(tag, "n_hl_only", sum(res$algorithm == "NBGMM (HL)"))

    cat(sprintf("%s: %d/%d genes, %.1f cells/subject, algorithms {%s}\n",
                tag, nrow(res$summary), nrow(cnt), cells_per,
                paste(unique(res$algorithm), collapse = "; ")))
    res
  }

  sc_big <- run_sc("sc", "sc_counts.csv", "sc_meta.csv")
  invisible(run_sc("sc_small", "sc_small_counts.csv", "sc_small_meta.csv"))

  # shrink_sc_dispersion has no upstream of its own: its reference is limma's
  # squeezeVar applied to the reciprocal cell overdispersions, on the residual
  # degrees of freedom the crate's sc_residual_df computes. This is the one
  # fixture in the suite where the converged uniroot changes the answer.
  meta_big <- read.csv(file.path(DATA_DIR, "sc_meta.csv"))
  n_cells <- nrow(meta_big)
  n_coef <- 3L
  n_sub <- length(unique(meta_big$subject))
  df_res <- n_cells - n_coef - (n_sub - 1)

  ok <- sc_big$convergence == 1 & is.finite(1 / sc_big$overdispersion$Cell) &
    sc_big$overdispersion$Cell > 0
  phi <- pmax(1 / sc_big$overdispersion$Cell[ok], 1e-8)

  invisible(uniroot_since())
  sh <- tightS(phi, df = df_res, covariate = sc_big$summary$logFC_[ok], robust = TRUE)
  put("sc_shrink", "uniroot_calls", uniroot_since())
  put("sc_shrink", "df_residual", df_res)
  put("sc_shrink", "n_usable", sum(ok))

  write_num(
    cbind(
      gene_id = sc_big$summary$gene_id[ok],
      phi_raw = phi,
      var_post = sh$var.post,
      var_prior = rep_len(sh$var.prior, length(phi)),
      df_prior = rep_len(sh$df.prior, length(phi))
    ),
    "sc_shrink.csv",
    c("gene_id", "phi_raw", "var_post", "var_prior", "df_prior")
  )
  cat(sprintf("sc_shrink: %d usable, df %d, uniroot calls recorded\n", sum(ok), df_res))
} else {
  cat("nebula not installed, skipping single-cell fixtures\n")
}

flush_scalars()
cat("done\n")
