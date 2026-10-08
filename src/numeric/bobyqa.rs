//! Powell's BOBYQA, bound-constrained derivative-free minimisation, as NLopt
//! 2.7.1 runs it.
//!
//! nebula's stage two calls `nloptr::bobyqa`, which is NLopt's C translation of
//! Powell's Fortran (`src/algs/bobyqa/bobyqa.c`) behind NLopt's own set-up: the
//! default initial step of `nlopt_set_default_initial_step`, a rescaling that
//! makes those steps equal, `rhobeg` from the step and `rhoend = xtol_rel *
//! rhobeg`. This is a line-by-line port of all of it, so for the same objective
//! values it asks for the same points in the same order. The objective is
//! reverse-communicated: [`BobyqaStepper`] pauses wherever the C calls `calfun`
//! (`prelim`, the trust-region step and `rescue`), so the GPU path can batch
//! evaluations across genes.
//!
//! Arrays keep Powell's one-based indexing: one-dimensional state carries an
//! unused element zero, matrices go through index helpers, and the subroutines
//! that take zero-based slices subtract one. That keeps every line comparable
//! with the C.
//!
//! Only the stopping tests nloptr's defaults reach are kept: `xtol_rel` through
//! `rhoend` and `maxeval`. `ftol_rel = ftol_abs = 0` never fires in NLopt, and
//! `stopval = -inf` never either.
//!
//! ### References
//!
//! Powell, The BOBYQA algorithm for bound constrained optimization without
//! derivatives, DAMTP 2009/NA06, 2009

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::errors::EdgeErrors;

////////////
// Params //
////////////

/// Stopping knobs, nloptr's defaults.
#[derive(Clone, Copy, Debug)]
pub struct BobyqaParams {
    /// Relative tolerance on the variables, turned into `rhoend`.
    pub xtol_rel: f64,
    /// Evaluation budget.
    pub maxeval: usize,
}

impl Default for BobyqaParams {
    /// `xtol_rel = 1e-6` and `maxeval = 1000`, as `nloptr::nl.opts` sets them.
    fn default() -> Self {
        Self {
            xtol_rel: 1e-6,
            maxeval: 1000,
        }
    }
}

/// Why a search stopped, NLopt's codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BobyqaStatus {
    /// Still running.
    Running,
    /// The trust region shrank to `rhoend`: NLopt's `XTOL_REACHED` (4).
    XtolReached,
    /// The evaluation budget ran out: `MAXEVAL_REACHED` (5).
    MaxevalReached,
    /// Rounding errors stopped progress: `ROUNDOFF_LIMITED` (-4).
    RoundoffLimited,
    /// Plain `SUCCESS` (1), when the trust-region loop ends without one of the
    /// above.
    Success,
}

/// The finished search.
#[derive(Clone, Debug)]
pub struct BobyqaResult {
    /// Best point, on the caller's scale.
    pub x: Vec<f64>,
    /// Objective there.
    pub f: f64,
    /// Objective evaluations.
    pub evaluations: usize,
    /// Why the search stopped.
    pub status: BobyqaStatus,
}

///////////
// Label //
///////////

/// Where [`BobyqaStepper::advance`] resumes, Powell's statement labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    /// `prelim`, label 50: the next initial point.
    PrelimNext,
    /// `prelim`, after the value of that point.
    PrelimPost,
    /// `bobyqb` right after `prelim` returns.
    Start,
    /// Label 20: refresh `gopt` if `kopt` moved.
    L20,
    /// Label 60: trust-region step.
    L60,
    /// Label 90: shift `xbase` if `xopt` drifted.
    L90,
    /// Label 190: call `rescue`.
    L190,
    /// After `rescue` returns.
    L190Post,
    /// Label 210: alternative step by `altmov`.
    L210,
    /// Label 230: `vlag` and `beta` for the step.
    L230,
    /// Label 360: evaluate the objective at `xnew`.
    L360,
    /// After that value.
    L360Post,
    /// Label 650: is an interpolation point too far away?
    L650,
    /// Label 680: next `rho`.
    L680,
    /// Label 720: finish.
    L720,
    /// `rescue` up to label 260, which needs no values.
    RescueStart,
    /// `rescue`, label 260 loop: the next provisional point.
    RescueNext,
    /// `rescue`, after the value of that point.
    RescuePost,
    /// Waiting for a value; `tell` resumes at the stored label.
    Wait,
    /// Finished.
    Done,
}

/////////////
// Stepper //
/////////////

/// [`bobyqa`] as a reverse-communication state machine.
///
/// [`Self::ask`] gives the next point, [`Self::tell`] its value. The arithmetic
/// is NLopt's, so for identical values the sequence of points is NLopt's.
pub struct BobyqaStepper {
    /// Number of variables.
    n: usize,
    /// Interpolation points, `2n + 1`.
    npt: usize,
    /// `npt + n`.
    ndim: usize,
    /// NLopt's rescaling: the stepper works in `x / s`.
    s: Vec<f64>,
    /// Bounds on the rescaled variables, one-based.
    xl: Vec<f64>,
    /// See [`Self::xl`].
    xu: Vec<f64>,
    /// Final trust-region radius.
    rhoend: f64,
    /// Initial trust-region radius.
    rhobeg: f64,
    /// Evaluation budget.
    maxeval: usize,
    /// Evaluations so far, NLopt's `nevals`.
    nevals: usize,

    /// The point being evaluated, rescaled, one-based.
    x: Vec<f64>,
    /// The same point on the caller's scale, zero-based, for [`Self::ask`].
    pending: Vec<f64>,
    /// Where to resume once the pending value arrives.
    resume: Label,
    /// The current label.
    label: Label,
    /// Last value told.
    f: f64,
    /// Exit status.
    rc: BobyqaStatus,
    /// Best value at the end.
    minf: f64,
    /// Best point at the end, rescaled, one-based.
    xbest: Vec<f64>,

    // Powell's arrays: one-dimensional ones one-based, matrices column-major.
    /// Shift of origin.
    xbase: Vec<f64>,
    /// Interpolation points relative to `xbase`, `npt * n`.
    xpt: Vec<f64>,
    /// Values at the interpolation points.
    fval: Vec<f64>,
    /// Trust-region centre relative to `xbase`.
    xopt: Vec<f64>,
    /// Model gradient at `xopt`.
    gopt: Vec<f64>,
    /// Explicit second derivatives, packed.
    hq: Vec<f64>,
    /// Implicit second derivative parameters.
    pq: Vec<f64>,
    /// Last `n` columns of `H`, `ndim * n`.
    bmat: Vec<f64>,
    /// Factor of the leading `npt` block of `H`, `npt * (npt - n - 1)`.
    zmat: Vec<f64>,
    /// `xl - xbase`.
    sl: Vec<f64>,
    /// `xu - xbase`.
    su: Vec<f64>,
    /// Next point relative to `xbase`.
    xnew: Vec<f64>,
    /// Alternative to `xnew` from `altmov`.
    xalt: Vec<f64>,
    /// Trial step.
    d: Vec<f64>,
    /// Lagrange function values, `ndim`.
    vlag: Vec<f64>,
    /// Work space, one-based.
    w: Vec<f64>,

    // bobyqb's scalars, kept across pauses.
    /// Index of the best interpolation point.
    kopt: usize,
    /// Index of the point to replace.
    knew: usize,
    /// `kopt` at the last `gopt` refresh.
    kbase: usize,
    /// Trust-region iterations since the last alternative step, or -1.
    ntrits: i64,
    /// `nevals` at the last `rescue`.
    nresc: usize,
    /// `nevals` at the last improvement test.
    nfsav: usize,
    /// Consecutive iterations where the Frobenius model looked better.
    itest: i64,
    /// Recent model errors.
    diffa: f64,
    /// See [`Self::diffa`].
    diffb: f64,
    /// See [`Self::diffa`].
    diffc: f64,
    /// Lower bound on the trust-region radius.
    rho: f64,
    /// Trust-region radius.
    delta: f64,
    /// `|xopt|^2`.
    xoptsq: f64,
    /// Squared step length from `trsbox`.
    dsq: f64,
    /// Least curvature seen by `trsbox`.
    crvmin: f64,
    /// Step length.
    dnorm: f64,
    /// Bound on the alternative step.
    adelt: f64,
    /// Squared distance threshold at label 650.
    distsq: f64,
    /// From `altmov`.
    alpha: f64,
    /// From `altmov`.
    cauchy: f64,
    /// Updating parameter.
    beta: f64,
    /// Updating denominator.
    denom: f64,
    /// Predicted change of the model.
    vquad: f64,
    /// Actual over predicted reduction.
    ratio: f64,
    /// `fval[kopt]` before the step.
    fopt: f64,
    /// Value at the first point, `fsave` in the C.
    fsave: f64,

    // prelim's state.
    /// Points evaluated by `prelim` so far.
    nf: usize,
    /// `nf - 1`.
    nfm: usize,
    /// `nf - 1 - n`, may be negative.
    nfx: i64,
    /// First step along a coordinate.
    stepa: f64,
    /// Second step along a coordinate.
    stepb: f64,
    /// Value at the first point.
    fbeg: f64,
    /// Coordinate pair of an off-diagonal point.
    ipt: usize,
    /// See [`Self::ipt`].
    jpt: usize,

    // rescue's state.
    /// Point index in the label 260 loop.
    kpt: usize,
    /// Model value at the rescue point.
    rvquad: f64,
    /// `fval[kopt]` on entry to `rescue`.
    fbase: f64,
    /// Rescue coordinates of the provisional point.
    rip: usize,
    /// See [`Self::rip`].
    riq: usize,
    /// Rescue step along `rip`.
    xp: f64,
    /// Rescue step along `riq`.
    xq: f64,
}

impl BobyqaStepper {
    /// Sets up the search as `nlopt_optimize` and `bobyqa` do, and asks for the
    /// first point.
    ///
    /// ### Params
    ///
    /// * `x0` - Starting point, inside the box
    /// * `lower` - Lower bounds, finite
    /// * `upper` - Upper bounds, finite and above `lower`
    /// * `params` - Stopping knobs, or nloptr's defaults
    ///
    /// ### Returns
    ///
    /// The stepper, or [`EdgeErrors::InvalidArgument`] when the box is too
    /// narrow for the initial radius, as NLopt refuses it.
    pub fn new(
        x0: &[f64],
        lower: &[f64],
        upper: &[f64],
        params: Option<BobyqaParams>,
    ) -> Result<Self, EdgeErrors> {
        let params = params.unwrap_or_default();
        let n = x0.len();
        if n == 0 || lower.len() != n || upper.len() != n {
            return Err(EdgeErrors::InvalidArgument(
                "BOBYQA needs matching, non-empty x0 and bounds.".to_string(),
            ));
        }

        // nlopt_set_default_initial_step.
        let mut dx = vec![0.0; n];
        for i in 0..n {
            let (lb, ub, x) = (lower[i], upper[i], x0[i]);
            let mut step = f64::INFINITY;
            if ub.is_finite() && lb.is_finite() && (ub - lb) * 0.25 < step && ub > lb {
                step = (ub - lb) * 0.25;
            }
            if ub.is_finite() && ub - x < step && ub > x {
                step = (ub - x) * 0.75;
            }
            if lb.is_finite() && x - lb < step && x > lb {
                step = (x - lb) * 0.75;
            }
            if step.is_infinite() {
                if ub.is_finite() && (ub - x).abs() < step.abs() {
                    step = (ub - x) * 1.1;
                }
                if lb.is_finite() && (x - lb).abs() < step.abs() {
                    step = (x - lb) * 1.1;
                }
            }
            if step.is_infinite() || step == 0.0 || step.is_subnormal() {
                step = x;
            }
            if step.is_infinite() || step == 0.0 {
                step = 1.0;
            }
            dx[i] = step;
        }

        // nlopt_compute_rescaling: equal steps after rescaling by dx / dx[0].
        let mut s = vec![1.0; n];
        if n > 1 && (1..n).any(|i| dx[i] != dx[i - 1]) {
            for i in 1..n {
                s[i] = dx[i] / dx[0];
            }
        }
        if s.iter().any(|&v| v == 0.0 || !v.is_finite()) {
            return Err(EdgeErrors::InvalidArgument(
                "BOBYQA rescaling overflowed.".to_string(),
            ));
        }

        let one_based = |v: &[f64], scale: &[f64]| -> Vec<f64> {
            std::iter::once(0.0)
                .chain(v.iter().zip(scale).map(|(a, b)| a / b))
                .collect()
        };
        let mut x = one_based(x0, &s);
        let mut xl = one_based(lower, &s);
        let mut xu = one_based(upper, &s);
        for j in 1..=n {
            if xl[j] > xu[j] {
                std::mem::swap(&mut xl[j], &mut xu[j]);
            }
        }
        let rhobeg = (dx[0] / s[0]).abs();
        let rhoend = params.xtol_rel * rhobeg;

        let npt = 2 * n + 1;
        let np = n + 1;
        let ndim = npt + n;
        let mut sl = vec![0.0; n + 1];
        let mut su = vec![0.0; n + 1];
        for j in 1..=n {
            let temp = xu[j] - xl[j];
            if temp < rhobeg + rhobeg {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "insufficient space between the bounds: {} - {} < {}",
                    xu[j],
                    xl[j],
                    rhobeg + rhobeg
                )));
            }
            sl[j] = xl[j] - x[j];
            su[j] = xu[j] - x[j];
            if sl[j] >= -rhobeg {
                if sl[j] >= 0.0 {
                    x[j] = xl[j];
                    sl[j] = 0.0;
                    su[j] = temp;
                } else {
                    x[j] = xl[j] + rhobeg;
                    sl[j] = -rhobeg;
                    su[j] = (xu[j] - x[j]).max(rhobeg);
                }
            } else if su[j] <= rhobeg {
                if su[j] <= 0.0 {
                    x[j] = xu[j];
                    sl[j] = -temp;
                    su[j] = 0.0;
                } else {
                    x[j] = xu[j] - rhobeg;
                    sl[j] = (xl[j] - x[j]).min(-rhobeg);
                    su[j] = rhobeg;
                }
            }
        }

        let mut stepper = Self {
            n,
            npt,
            ndim,
            s,
            xl,
            xu,
            rhoend,
            rhobeg,
            maxeval: params.maxeval,
            nevals: 0,
            x,
            pending: vec![0.0; n],
            resume: Label::Done,
            label: Label::PrelimNext,
            f: 0.0,
            rc: BobyqaStatus::Running,
            minf: f64::INFINITY,
            xbest: vec![0.0; n + 1],
            xbase: vec![0.0; n + 1],
            xpt: vec![0.0; npt * n],
            fval: vec![0.0; npt + 1],
            xopt: vec![0.0; n + 1],
            gopt: vec![0.0; n + 1],
            hq: vec![0.0; n * np / 2 + 1],
            pq: vec![0.0; npt + 1],
            bmat: vec![0.0; ndim * n],
            zmat: vec![0.0; npt * (npt - np)],
            sl,
            su,
            xnew: vec![0.0; n + 1],
            xalt: vec![0.0; n + 1],
            d: vec![0.0; n + 1],
            vlag: vec![0.0; ndim + 1],
            w: vec![0.0; 3 * ndim + n + npt + 1],
            kopt: 1,
            knew: 0,
            kbase: 1,
            ntrits: 0,
            nresc: 0,
            nfsav: 0,
            itest: 0,
            diffa: 0.0,
            diffb: 0.0,
            diffc: 0.0,
            rho: rhobeg,
            delta: rhobeg,
            xoptsq: 0.0,
            dsq: 0.0,
            crvmin: 0.0,
            dnorm: 0.0,
            adelt: 0.0,
            distsq: 0.0,
            alpha: 0.0,
            cauchy: 0.0,
            beta: 0.0,
            denom: 0.0,
            vquad: 0.0,
            ratio: 0.0,
            fopt: 0.0,
            fsave: 0.0,
            nf: 0,
            nfm: 0,
            nfx: 0,
            stepa: 0.0,
            stepb: 0.0,
            fbeg: 0.0,
            ipt: 0,
            jpt: 0,
            kpt: 0,
            rvquad: 0.0,
            fbase: 0.0,
            rip: 0,
            riq: 0,
            xp: 0.0,
            xq: 0.0,
        };
        stepper.prelim_init();
        stepper.advance();
        Ok(stepper)
    }

    /// The point the search wants evaluated next.
    ///
    /// ### Returns
    ///
    /// The point on the caller's scale, or `None` once the search has finished.
    pub fn ask(&self) -> Option<&[f64]> {
        if self.label == Label::Wait {
            Some(&self.pending)
        } else {
            None
        }
    }

    /// Hands the stepper the value at the point [`Self::ask`] returned.
    ///
    /// ### Params
    ///
    /// * `value` - The objective there
    pub fn tell(&mut self, value: f64) {
        debug_assert_eq!(self.label, Label::Wait);
        self.f = value;
        self.label = self.resume;
        self.advance();
    }

    /// The result, once [`Self::ask`] returns `None`.
    ///
    /// ### Returns
    ///
    /// The best point on the caller's scale, its value, the evaluation count
    /// and the status.
    pub fn result(&self) -> BobyqaResult {
        BobyqaResult {
            x: (1..=self.n)
                .map(|j| self.xbest[j] * self.s[j - 1])
                .collect(),
            f: self.minf,
            evaluations: self.nevals,
            status: self.rc,
        }
    }

    /// Pauses for the objective at `self.x`, resuming at `next`.
    ///
    /// ### Params
    ///
    /// * `next` - Label to resume at
    fn request(&mut self, next: Label) {
        for j in 1..=self.n {
            self.pending[j - 1] = self.x[j] * self.s[j - 1];
        }
        self.nevals += 1;
        self.resume = next;
        self.label = Label::Wait;
    }

    /// Whether the evaluation budget is spent, `nlopt_stop_evals`.
    ///
    /// ### Returns
    ///
    /// `true` once `maxeval` evaluations have been made.
    fn stop_evals(&self) -> bool {
        self.maxeval > 0 && self.nevals >= self.maxeval
    }

    /// Index into `xpt`, one-based `(k, j)`.
    #[inline]
    fn ix(&self, k: usize, j: usize) -> usize {
        (k - 1) + (j - 1) * self.npt
    }

    /// Index into `bmat`, one-based `(k, j)`.
    #[inline]
    fn ib(&self, k: usize, j: usize) -> usize {
        (k - 1) + (j - 1) * self.ndim
    }

    /// Index into `zmat`, one-based `(k, j)`.
    #[inline]
    fn iz(&self, k: usize, j: usize) -> usize {
        (k - 1) + (j - 1) * self.npt
    }

    /// `prelim` up to its loop: `xbase` and the zeroed model.
    fn prelim_init(&mut self) {
        for j in 1..=self.n {
            self.xbase[j] = self.x[j];
        }
        self.xpt.fill(0.0);
        self.bmat.fill(0.0);
        self.hq.fill(0.0);
        self.pq.fill(0.0);
        self.zmat.fill(0.0);
        self.nf = 0;
    }

    /// Runs from the current label until the next pause or the end.
    fn advance(&mut self) {
        loop {
            self.label = match self.label {
                Label::PrelimNext => self.prelim_next(),
                Label::PrelimPost => self.prelim_post(),
                Label::Start => self.start(),
                Label::L20 => self.l20(),
                Label::L60 => self.l60(),
                Label::L90 => self.l90(),
                Label::L190 => {
                    self.nfsav = self.nevals;
                    self.kbase = self.kopt;
                    Label::RescueStart
                }
                Label::L190Post => self.l190_post(),
                Label::L210 => self.l210(),
                Label::L230 => self.l230(),
                Label::L360 => self.l360(),
                Label::L360Post => self.l360_post(),
                Label::L650 => self.l650(),
                Label::L680 => self.l680(),
                Label::L720 => {
                    self.l720();
                    Label::Done
                }
                Label::RescueStart => self.rescue_start(),
                Label::RescueNext => self.rescue_next(),
                Label::RescuePost => self.rescue_post(),
                Label::Wait | Label::Done => return,
            };
        }
    }

    ////////////
    // prelim //
    ////////////

    /// `prelim`, label 50: places the next initial point and asks for it.
    fn prelim_next(&mut self) -> Label {
        let n = self.n;
        let np = n + 1;
        self.nfm = self.nf;
        self.nfx = self.nf as i64 - n as i64;
        self.nf += 1;
        let nf = self.nf;
        let nfm = self.nfm;
        let rhobeg = self.rhobeg;
        if nfm <= 2 * n {
            if nfm >= 1 && nfm <= n {
                self.stepa = rhobeg;
                if self.su[nfm] == 0.0 {
                    self.stepa = -self.stepa;
                }
                let i = self.ix(nf, nfm);
                self.xpt[i] = self.stepa;
            } else if nfm > n {
                let nfx = self.nfx as usize;
                self.stepa = self.xpt[self.ix(nf - n, nfx)];
                self.stepb = -rhobeg;
                if self.sl[nfx] == 0.0 {
                    self.stepb = (2.0 * rhobeg).min(self.su[nfx]);
                }
                if self.su[nfx] == 0.0 {
                    self.stepb = (-2.0 * rhobeg).max(self.sl[nfx]);
                }
                let i = self.ix(nf, nfx);
                self.xpt[i] = self.stepb;
            }
        } else {
            let mut itemp = (nfm - np) / n;
            let mut jpt = nfm - itemp * n - n;
            let mut ipt = jpt + itemp;
            if ipt > n {
                itemp = jpt;
                jpt = ipt - n;
                ipt = itemp;
            }
            self.ipt = ipt;
            self.jpt = jpt;
            let a = self.xpt[self.ix(ipt + 1, ipt)];
            let i = self.ix(nf, ipt);
            self.xpt[i] = a;
            let b = self.xpt[self.ix(jpt + 1, jpt)];
            let i = self.ix(nf, jpt);
            self.xpt[i] = b;
        }

        for j in 1..=n {
            let p = self.xpt[self.ix(nf, j)];
            self.x[j] = self.xl[j].max(self.xbase[j] + p).min(self.xu[j]);
            if p == self.sl[j] {
                self.x[j] = self.xl[j];
            }
            if p == self.su[j] {
                self.x[j] = self.xu[j];
            }
        }
        self.request(Label::PrelimPost);
        Label::Wait
    }

    /// `prelim` after the value of point `nf`.
    fn prelim_post(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        let nf = self.nf;
        let nfm = self.nfm;
        let f = self.f;
        let rhosq = self.rhobeg * self.rhobeg;
        let recip = 1.0 / rhosq;
        self.fval[nf] = f;
        if nf == 1 {
            self.fbeg = f;
            self.kopt = 1;
        } else if f < self.fval[self.kopt] {
            self.kopt = nf;
        }

        if nf <= 2 * n + 1 {
            if nf >= 2 && nf <= n + 1 {
                self.gopt[nfm] = (f - self.fbeg) / self.stepa;
                if npt < nf + n {
                    let (a, b, c) = (self.ib(1, nfm), self.ib(nf, nfm), self.ib(npt + nfm, nfm));
                    self.bmat[a] = -1.0 / self.stepa;
                    self.bmat[b] = 1.0 / self.stepa;
                    self.bmat[c] = -0.5 * rhosq;
                }
            } else if nf >= n + 2 {
                let nfx = self.nfx as usize;
                let ih = nfx * (nfx + 1) / 2;
                let temp = (f - self.fbeg) / self.stepb;
                let diff = self.stepb - self.stepa;
                self.hq[ih] = 2.0 * (temp - self.gopt[nfx]) / diff;
                self.gopt[nfx] = (self.gopt[nfx] * self.stepb - temp * self.stepa) / diff;
                if self.stepa * self.stepb < 0.0 && f < self.fval[nf - n] {
                    self.fval[nf] = self.fval[nf - n];
                    self.fval[nf - n] = f;
                    if self.kopt == nf {
                        self.kopt = nf - n;
                    }
                    let (a, b) = (self.ix(nf - n, nfx), self.ix(nf, nfx));
                    self.xpt[a] = self.stepb;
                    self.xpt[b] = self.stepa;
                }
                let b1 = self.ib(1, nfx);
                let bnf = self.ib(nf, nfx);
                let bnn = self.ib(nf - n, nfx);
                self.bmat[b1] = -(self.stepa + self.stepb) / (self.stepa * self.stepb);
                self.bmat[bnf] = -0.5 / self.xpt[self.ix(nf - n, nfx)];
                self.bmat[bnn] = -self.bmat[b1] - self.bmat[bnf];
                let z1 = self.iz(1, nfx);
                let znf = self.iz(nf, nfx);
                let znn = self.iz(nf - n, nfx);
                self.zmat[z1] = 2.0f64.sqrt() / (self.stepa * self.stepb);
                self.zmat[znf] = 0.5f64.sqrt() / rhosq;
                self.zmat[znn] = -self.zmat[z1] - self.zmat[znf];
            }
        } else {
            let nfx = self.nfx as usize;
            let (ipt, jpt) = (self.ipt, self.jpt);
            let ih = ipt * (ipt - 1) / 2 + jpt;
            let idx = [
                self.iz(1, nfx),
                self.iz(nf, nfx),
                self.iz(ipt + 1, nfx),
                self.iz(jpt + 1, nfx),
            ];
            self.zmat[idx[0]] = recip;
            self.zmat[idx[1]] = recip;
            self.zmat[idx[2]] = -recip;
            self.zmat[idx[3]] = -recip;
            let temp = self.xpt[self.ix(nf, ipt)] * self.xpt[self.ix(nf, jpt)];
            self.hq[ih] = (self.fbeg - self.fval[ipt + 1] - self.fval[jpt + 1] + f) / temp;
        }
        if self.stop_evals() {
            self.rc = BobyqaStatus::MaxevalReached;
            return self.after_prelim();
        }
        if nf < npt {
            return Label::PrelimNext;
        }
        self.after_prelim()
    }

    /// `bobyqb` right after `prelim`: sets `xopt`, branching to the end when
    /// `prelim` stopped early.
    fn after_prelim(&mut self) -> Label {
        self.xoptsq = 0.0;
        for i in 1..=self.n {
            self.xopt[i] = self.xpt[self.ix(self.kopt, i)];
            self.xoptsq += self.xopt[i] * self.xopt[i];
        }
        self.fsave = self.fval[1];
        if self.rc != BobyqaStatus::Running {
            return Label::L720;
        }
        Label::Start
    }

    ////////////
    // bobyqb //
    ////////////

    /// The settings before the first iteration.
    fn start(&mut self) -> Label {
        self.kbase = 1;
        self.rho = self.rhobeg;
        self.delta = self.rho;
        self.nresc = self.nevals;
        self.ntrits = 0;
        self.diffa = 0.0;
        self.diffb = 0.0;
        self.itest = 0;
        self.nfsav = self.nevals;
        Label::L20
    }

    /// Label 20: moves `gopt` to `xopt` if `kopt` changed.
    fn l20(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        if self.kopt != self.kbase {
            let mut ih = 0;
            for j in 1..=n {
                for i in 1..=j {
                    ih += 1;
                    if i < j {
                        self.gopt[j] += self.hq[ih] * self.xopt[i];
                    }
                    self.gopt[i] += self.hq[ih] * self.xopt[j];
                }
            }
            if self.nevals > npt {
                for k in 1..=npt {
                    let mut temp = 0.0;
                    for j in 1..=n {
                        temp += self.xpt[self.ix(k, j)] * self.xopt[j];
                    }
                    temp *= self.pq[k];
                    for i in 1..=n {
                        self.gopt[i] += temp * self.xpt[self.ix(k, i)];
                    }
                }
            }
        }
        Label::L60
    }

    /// Label 60: the trust-region step, or the decision to shrink `rho`.
    fn l60(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        {
            let (gnew, rest) = self.w[1..].split_at_mut(n);
            let (xbdi, rest) = rest.split_at_mut(n);
            let (s, rest) = rest.split_at_mut(n);
            let (hs, rest) = rest.split_at_mut(n);
            let hred = &mut rest[..n];
            trsbox(
                n,
                npt,
                &self.xpt,
                &self.xopt[1..],
                &self.gopt[1..],
                &self.hq[1..],
                &self.pq[1..],
                &self.sl[1..],
                &self.su[1..],
                self.delta,
                &mut self.xnew[1..],
                &mut self.d[1..],
                gnew,
                xbdi,
                s,
                hs,
                hred,
                &mut self.dsq,
                &mut self.crvmin,
            );
        }
        self.dnorm = self.delta.min(self.dsq.sqrt());
        if self.dnorm < 0.5 * self.rho {
            self.ntrits = -1;
            self.distsq = (10.0 * self.rho) * (10.0 * self.rho);
            if self.nevals <= self.nfsav + 2 {
                return Label::L650;
            }
            let errbig = self.diffa.max(self.diffb).max(self.diffc);
            let frhosq = self.rho * 0.125 * self.rho;
            if self.crvmin > 0.0 && errbig > frhosq * self.crvmin {
                return Label::L650;
            }
            let bdtol = errbig / self.rho;
            for j in 1..=n {
                let mut bdtest = bdtol;
                if self.xnew[j] == self.sl[j] {
                    bdtest = self.w[j];
                }
                if self.xnew[j] == self.su[j] {
                    bdtest = -self.w[j];
                }
                if bdtest < bdtol {
                    let mut curv = self.hq[(j + j * j) / 2];
                    for k in 1..=npt {
                        let v = self.xpt[self.ix(k, j)];
                        curv += self.pq[k] * (v * v);
                    }
                    bdtest += 0.5 * curv * self.rho;
                    if bdtest < bdtol {
                        return Label::L650;
                    }
                }
            }
            return Label::L680;
        }
        self.ntrits += 1;
        Label::L90
    }

    /// Label 90: shifts `xbase` to `xopt` when the step is small against it.
    fn l90(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        let nptm = npt - n - 1;
        if self.dsq <= self.xoptsq * 0.001 {
            let fracsq = self.xoptsq * 0.25;
            let mut sumpq = 0.0;
            for k in 1..=npt {
                sumpq += self.pq[k];
                let mut sum = -0.5 * self.xoptsq;
                for i in 1..=n {
                    sum += self.xpt[self.ix(k, i)] * self.xopt[i];
                }
                self.w[npt + k] = sum;
                let temp = fracsq - 0.5 * sum;
                for i in 1..=n {
                    self.w[i] = self.bmat[self.ib(k, i)];
                    self.vlag[i] = sum * self.xpt[self.ix(k, i)] + temp * self.xopt[i];
                    let ip = npt + i;
                    for j in 1..=i {
                        let b = self.ib(ip, j);
                        self.bmat[b] =
                            self.bmat[b] + self.w[i] * self.vlag[j] + self.vlag[i] * self.w[j];
                    }
                }
            }

            for jj in 1..=nptm {
                let mut sumz = 0.0;
                let mut sumw = 0.0;
                for k in 1..=npt {
                    let z = self.zmat[self.iz(k, jj)];
                    sumz += z;
                    self.vlag[k] = self.w[npt + k] * z;
                    sumw += self.vlag[k];
                }
                for j in 1..=n {
                    let mut sum = (fracsq * sumz - 0.5 * sumw) * self.xopt[j];
                    for k in 1..=npt {
                        sum += self.vlag[k] * self.xpt[self.ix(k, j)];
                    }
                    self.w[j] = sum;
                    for k in 1..=npt {
                        let b = self.ib(k, j);
                        self.bmat[b] += sum * self.zmat[self.iz(k, jj)];
                    }
                }
                for i in 1..=n {
                    let ip = i + npt;
                    let temp = self.w[i];
                    for j in 1..=i {
                        let b = self.ib(ip, j);
                        self.bmat[b] += temp * self.w[j];
                    }
                }
            }

            let mut ih = 0;
            for j in 1..=n {
                self.w[j] = -0.5 * sumpq * self.xopt[j];
                for k in 1..=npt {
                    self.w[j] += self.pq[k] * self.xpt[self.ix(k, j)];
                    let x = self.ix(k, j);
                    self.xpt[x] -= self.xopt[j];
                }
                for i in 1..=j {
                    ih += 1;
                    self.hq[ih] = self.hq[ih] + self.w[i] * self.xopt[j] + self.xopt[i] * self.w[j];
                    let (dst, src) = (self.ib(npt + i, j), self.ib(npt + j, i));
                    self.bmat[dst] = self.bmat[src];
                }
            }
            for i in 1..=n {
                self.xbase[i] += self.xopt[i];
                self.xnew[i] -= self.xopt[i];
                self.sl[i] -= self.xopt[i];
                self.su[i] -= self.xopt[i];
                self.xopt[i] = 0.0;
            }
            self.xoptsq = 0.0;
        }
        if self.ntrits == 0 {
            Label::L210
        } else {
            Label::L230
        }
    }

    /// After `rescue`: the branch the C takes on its return.
    fn l190_post(&mut self) -> Label {
        let n = self.n;
        self.xoptsq = 0.0;
        if self.kopt != self.kbase {
            for i in 1..=n {
                self.xopt[i] = self.xpt[self.ix(self.kopt, i)];
                self.xoptsq += self.xopt[i] * self.xopt[i];
            }
        }
        if self.rc != BobyqaStatus::Running {
            return Label::L720;
        }
        self.nresc = self.nevals;
        if self.nfsav < self.nevals {
            self.nfsav = self.nevals;
            return Label::L20;
        }
        if self.ntrits > 0 {
            return Label::L60;
        }
        Label::L210
    }

    /// Label 210: the alternative step of `altmov`.
    fn l210(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        let ndim = self.ndim;
        {
            let (head, tail) = self.w.split_at_mut(ndim + 1);
            let (glag, hcol) = head[1..].split_at_mut(n);
            let hcol = &mut hcol[..npt];
            let wk = &mut tail[..2 * n];
            altmov(
                n,
                npt,
                &self.xpt,
                &self.xopt[1..],
                &self.bmat,
                &self.zmat,
                ndim,
                &self.sl[1..],
                &self.su[1..],
                self.kopt,
                self.knew,
                self.adelt,
                &mut self.xnew[1..],
                &mut self.xalt[1..],
                &mut self.alpha,
                &mut self.cauchy,
                glag,
                hcol,
                wk,
            );
        }
        for i in 1..=n {
            self.d[i] = self.xnew[i] - self.xopt[i];
        }
        Label::L230
    }

    /// Label 230: `vlag` and `beta` for the step, and the choice of `knew`.
    fn l230(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        let nptm = npt - n - 1;
        for k in 1..=npt {
            let mut suma = 0.0;
            let mut sumb = 0.0;
            let mut sum = 0.0;
            for j in 1..=n {
                suma += self.xpt[self.ix(k, j)] * self.d[j];
                sumb += self.xpt[self.ix(k, j)] * self.xopt[j];
                sum += self.bmat[self.ib(k, j)] * self.d[j];
            }
            self.w[k] = suma * (0.5 * suma + sumb);
            self.vlag[k] = sum;
            self.w[npt + k] = suma;
        }
        self.beta = 0.0;
        for jj in 1..=nptm {
            let mut sum = 0.0;
            for k in 1..=npt {
                sum += self.zmat[self.iz(k, jj)] * self.w[k];
            }
            self.beta -= sum * sum;
            for k in 1..=npt {
                self.vlag[k] += sum * self.zmat[self.iz(k, jj)];
            }
        }
        self.dsq = 0.0;
        let mut bsum = 0.0;
        let mut dx = 0.0;
        for j in 1..=n {
            self.dsq += self.d[j] * self.d[j];
            let mut sum = 0.0;
            for k in 1..=npt {
                sum += self.w[k] * self.bmat[self.ib(k, j)];
            }
            bsum += sum * self.d[j];
            let jp = npt + j;
            for i in 1..=n {
                sum += self.bmat[self.ib(jp, i)] * self.d[i];
            }
            self.vlag[jp] = sum;
            bsum += sum * self.d[j];
            dx += self.d[j] * self.xopt[j];
        }
        self.beta =
            dx * dx + self.dsq * (self.xoptsq + dx + dx + 0.5 * self.dsq) + self.beta - bsum;
        self.vlag[self.kopt] += 1.0;

        if self.ntrits == 0 {
            let vk = self.vlag[self.knew];
            self.denom = vk * vk + self.alpha * self.beta;
            if self.denom < self.cauchy && self.cauchy > 0.0 {
                for i in 1..=n {
                    self.xnew[i] = self.xalt[i];
                    self.d[i] = self.xnew[i] - self.xopt[i];
                }
                self.cauchy = 0.0;
                return Label::L230;
            }
            if self.denom <= 0.5 * (vk * vk) {
                if self.nevals > self.nresc {
                    return Label::L190;
                }
                self.rc = BobyqaStatus::RoundoffLimited;
                return Label::L720;
            }
        } else {
            let delsq = self.delta * self.delta;
            let mut scaden = 0.0;
            let mut biglsq = 0.0f64;
            self.knew = 0;
            for k in 1..=npt {
                if k == self.kopt {
                    continue;
                }
                let mut hdiag = 0.0;
                for jj in 1..=nptm {
                    let z = self.zmat[self.iz(k, jj)];
                    hdiag += z * z;
                }
                let den = self.beta * hdiag + self.vlag[k] * self.vlag[k];
                let mut distsq = 0.0;
                for j in 1..=n {
                    let v = self.xpt[self.ix(k, j)] - self.xopt[j];
                    distsq += v * v;
                }
                let r = distsq / delsq;
                let temp = 1.0f64.max(r * r);
                if temp * den > scaden {
                    scaden = temp * den;
                    self.knew = k;
                    self.denom = den;
                }
                biglsq = biglsq.max(temp * (self.vlag[k] * self.vlag[k]));
            }
            if scaden <= 0.5 * biglsq {
                if self.nevals > self.nresc {
                    return Label::L190;
                }
                self.rc = BobyqaStatus::RoundoffLimited;
                return Label::L720;
            }
        }
        Label::L360
    }

    /// Label 360: asks for the objective at `xbase + xnew`.
    fn l360(&mut self) -> Label {
        for i in 1..=self.n {
            self.x[i] = self.xl[i].max(self.xbase[i] + self.xnew[i]).min(self.xu[i]);
            if self.xnew[i] == self.sl[i] {
                self.x[i] = self.xl[i];
            }
            if self.xnew[i] == self.su[i] {
                self.x[i] = self.xu[i];
            }
        }
        if self.stop_evals() {
            self.rc = BobyqaStatus::MaxevalReached;
            return Label::L720;
        }
        self.request(Label::L360Post);
        Label::Wait
    }

    /// After the value at `xnew`: model error, radius, and the model update.
    fn l360_post(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        let nptm = npt - n - 1;
        let nh = n * (n + 1) / 2;
        let f = self.f;
        if self.ntrits == -1 {
            self.fsave = f;
            self.rc = BobyqaStatus::XtolReached;
            if self.fsave < self.fval[self.kopt] {
                self.minf = f;
                self.xbest.copy_from_slice(&self.x);
                return Label::Done;
            }
            return Label::L720;
        }

        self.fopt = self.fval[self.kopt];
        let fopt = self.fopt;
        let mut vquad = 0.0;
        let mut ih = 0;
        for j in 1..=n {
            vquad += self.d[j] * self.gopt[j];
            for i in 1..=j {
                ih += 1;
                let mut temp = self.d[i] * self.d[j];
                if i == j {
                    temp *= 0.5;
                }
                vquad += self.hq[ih] * temp;
            }
        }
        for k in 1..=npt {
            let v = self.w[npt + k];
            vquad += 0.5 * self.pq[k] * (v * v);
        }
        self.vquad = vquad;
        let diff = f - fopt - vquad;
        self.diffc = self.diffb;
        self.diffb = self.diffa;
        self.diffa = diff.abs();
        if self.dnorm > self.rho {
            self.nfsav = self.nevals;
        }

        if self.ntrits > 0 {
            if vquad >= 0.0 {
                self.rc = BobyqaStatus::RoundoffLimited;
                return Label::L720;
            }
            self.ratio = (f - fopt) / vquad;
            if self.ratio <= 0.1 {
                self.delta = (0.5 * self.delta).min(self.dnorm);
            } else if self.ratio <= 0.7 {
                self.delta = (0.5 * self.delta).max(self.dnorm);
            } else {
                self.delta = (0.5 * self.delta).max(self.dnorm + self.dnorm);
            }
            if self.delta <= self.rho * 1.5 {
                self.delta = self.rho;
            }

            if f < fopt {
                let ksav = self.knew;
                let densav = self.denom;
                let delsq = self.delta * self.delta;
                let mut scaden = 0.0;
                let mut biglsq = 0.0f64;
                self.knew = 0;
                for k in 1..=npt {
                    let mut hdiag = 0.0;
                    for jj in 1..=nptm {
                        let z = self.zmat[self.iz(k, jj)];
                        hdiag += z * z;
                    }
                    let den = self.beta * hdiag + self.vlag[k] * self.vlag[k];
                    let mut distsq = 0.0;
                    for j in 1..=n {
                        let v = self.xpt[self.ix(k, j)] - self.xnew[j];
                        distsq += v * v;
                    }
                    let r = distsq / delsq;
                    let temp = 1.0f64.max(r * r);
                    if temp * den > scaden {
                        scaden = temp * den;
                        self.knew = k;
                        self.denom = den;
                    }
                    biglsq = biglsq.max(temp * (self.vlag[k] * self.vlag[k]));
                }
                if scaden <= 0.5 * biglsq {
                    self.knew = ksav;
                    self.denom = densav;
                }
            }
        }

        let knew = self.knew;
        update(
            n,
            npt,
            &mut self.bmat,
            &mut self.zmat,
            self.ndim,
            &mut self.vlag[1..],
            self.beta,
            self.denom,
            knew,
            &mut self.w[1..],
        );
        let mut ih = 0;
        let pqold = self.pq[knew];
        self.pq[knew] = 0.0;
        for i in 1..=n {
            let temp = pqold * self.xpt[self.ix(knew, i)];
            for j in 1..=i {
                ih += 1;
                self.hq[ih] += temp * self.xpt[self.ix(knew, j)];
            }
        }
        for jj in 1..=nptm {
            let temp = diff * self.zmat[self.iz(knew, jj)];
            for k in 1..=npt {
                self.pq[k] += temp * self.zmat[self.iz(k, jj)];
            }
        }

        self.fval[knew] = f;
        for i in 1..=n {
            let x = self.ix(knew, i);
            self.xpt[x] = self.xnew[i];
            self.w[i] = self.bmat[self.ib(knew, i)];
        }
        for k in 1..=npt {
            let mut suma = 0.0;
            for jj in 1..=nptm {
                suma += self.zmat[self.iz(knew, jj)] * self.zmat[self.iz(k, jj)];
            }
            if suma.is_infinite() {
                self.rc = BobyqaStatus::RoundoffLimited;
                return Label::L720;
            }
            let mut sumb = 0.0;
            for j in 1..=n {
                sumb += self.xpt[self.ix(k, j)] * self.xopt[j];
            }
            let temp = suma * sumb;
            for i in 1..=n {
                self.w[i] += temp * self.xpt[self.ix(k, i)];
            }
        }
        for i in 1..=n {
            self.gopt[i] += diff * self.w[i];
        }

        if f < fopt {
            self.kopt = knew;
            self.xoptsq = 0.0;
            let mut ih = 0;
            for j in 1..=n {
                self.xopt[j] = self.xnew[j];
                self.xoptsq += self.xopt[j] * self.xopt[j];
                for i in 1..=j {
                    ih += 1;
                    if i < j {
                        self.gopt[j] += self.hq[ih] * self.d[i];
                    }
                    self.gopt[i] += self.hq[ih] * self.d[j];
                }
            }
            for k in 1..=npt {
                let mut temp = 0.0;
                for j in 1..=n {
                    temp += self.xpt[self.ix(k, j)] * self.d[j];
                }
                temp *= self.pq[k];
                for i in 1..=n {
                    self.gopt[i] += temp * self.xpt[self.ix(k, i)];
                }
            }
            // `nlopt_stop_ftol` with nloptr's zero tolerances never fires.
        }

        if self.ntrits > 0 {
            for k in 1..=npt {
                self.vlag[k] = self.fval[k] - self.fval[self.kopt];
                self.w[k] = 0.0;
            }
            for j in 1..=nptm {
                let mut sum = 0.0;
                for k in 1..=npt {
                    sum += self.zmat[self.iz(k, j)] * self.vlag[k];
                }
                for k in 1..=npt {
                    self.w[k] += sum * self.zmat[self.iz(k, j)];
                }
            }
            for k in 1..=npt {
                let mut sum = 0.0;
                for j in 1..=n {
                    sum += self.xpt[self.ix(k, j)] * self.xopt[j];
                }
                self.w[k + npt] = self.w[k];
                self.w[k] *= sum;
            }
            let mut gqsq = 0.0;
            let mut gisq = 0.0;
            for i in 1..=n {
                let mut sum = 0.0;
                for k in 1..=npt {
                    sum = sum
                        + self.bmat[self.ib(k, i)] * self.vlag[k]
                        + self.xpt[self.ix(k, i)] * self.w[k];
                }
                if self.xopt[i] == self.sl[i] {
                    let a = 0.0f64.min(self.gopt[i]);
                    gqsq += a * a;
                    let b = 0.0f64.min(sum);
                    gisq += b * b;
                } else if self.xopt[i] == self.su[i] {
                    let a = 0.0f64.max(self.gopt[i]);
                    gqsq += a * a;
                    let b = 0.0f64.max(sum);
                    gisq += b * b;
                } else {
                    gqsq += self.gopt[i] * self.gopt[i];
                    gisq += sum * sum;
                }
                self.vlag[npt + i] = sum;
            }

            self.itest += 1;
            if gqsq < 10.0 * gisq {
                self.itest = 0;
            }
            if self.itest >= 3 {
                for i in 1..=npt.max(nh) {
                    if i <= n {
                        self.gopt[i] = self.vlag[npt + i];
                    }
                    if i <= npt {
                        self.pq[i] = self.w[npt + i];
                    }
                    if i <= nh {
                        self.hq[i] = 0.0;
                    }
                    self.itest = 0;
                }
            }
        }

        if self.ntrits == 0 {
            return Label::L60;
        }
        if f <= fopt + 0.1 * vquad {
            return Label::L60;
        }
        let a = 2.0 * self.delta;
        let b = 10.0 * self.rho;
        self.distsq = (a * a).max(b * b);
        Label::L650
    }

    /// Label 650: replaces the farthest interpolation point, or decides the
    /// work at this `rho` is done.
    fn l650(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        self.knew = 0;
        for k in 1..=npt {
            let mut sum = 0.0;
            for j in 1..=n {
                let v = self.xpt[self.ix(k, j)] - self.xopt[j];
                sum += v * v;
            }
            if sum > self.distsq {
                self.knew = k;
                self.distsq = sum;
            }
        }

        if self.knew > 0 {
            let dist = self.distsq.sqrt();
            if self.ntrits == -1 {
                self.delta = (0.1 * self.delta).min(0.5 * dist);
                if self.delta <= self.rho * 1.5 {
                    self.delta = self.rho;
                }
            }
            self.ntrits = 0;
            self.adelt = (0.1 * dist).min(self.delta).max(self.rho);
            self.dsq = self.adelt * self.adelt;
            return Label::L90;
        }
        if self.ntrits == -1 {
            return Label::L680;
        }
        if self.ratio > 0.0 {
            return Label::L60;
        }
        if self.delta.max(self.dnorm) > self.rho {
            return Label::L60;
        }
        Label::L680
    }

    /// Label 680: the next `rho`, or the final Newton step, or the end.
    fn l680(&mut self) -> Label {
        if self.rho > self.rhoend {
            self.delta = 0.5 * self.rho;
            self.ratio = self.rho / self.rhoend;
            if self.ratio <= 16.0 {
                self.rho = self.rhoend;
            } else if self.ratio <= 250.0 {
                self.rho = self.ratio.sqrt() * self.rhoend;
            } else {
                self.rho *= 0.1;
            }
            self.delta = self.delta.max(self.rho);
            self.ntrits = 0;
            self.nfsav = self.nevals;
            return Label::L60;
        }
        if self.ntrits == -1 {
            return Label::L360;
        }
        Label::L720
    }

    /// Label 720: the best point, clamped onto the bounds it touches.
    fn l720(&mut self) {
        for i in 1..=self.n {
            self.x[i] = self.xl[i].max(self.xbase[i] + self.xopt[i]).min(self.xu[i]);
            if self.xopt[i] == self.sl[i] {
                self.x[i] = self.xl[i];
            }
            if self.xopt[i] == self.su[i] {
                self.x[i] = self.xu[i];
            }
        }
        self.minf = self.fval[self.kopt];
        self.xbest.copy_from_slice(&self.x);
        if self.rc == BobyqaStatus::Running {
            self.rc = BobyqaStatus::Success;
        }
    }

    ////////////
    // rescue //
    ////////////

    /// `ptsaux(r, j)` in bobyqb's work space, `r` 1 or 2.
    #[inline]
    fn pa(&self, r: usize, j: usize) -> usize {
        1 + (r - 1) + 2 * (j - 1)
    }

    /// `ptsid(k)` in bobyqb's work space.
    #[inline]
    fn pid(&self, k: usize) -> usize {
        2 * self.n + k
    }

    /// `rescue`'s own `w(i)` in bobyqb's work space.
    #[inline]
    fn rw(&self, i: usize) -> usize {
        self.ndim + self.n + i
    }

    /// The integer parts `(p, q)` of a provisional point's identifier.
    ///
    /// ### Params
    ///
    /// * `k` - Point index
    ///
    /// ### Returns
    ///
    /// `ip = (int) ptsid(k)` and `iq = (int) (np * ptsid(k) - ip * np)`.
    fn pid_parts(&self, k: usize) -> (usize, usize) {
        let np = (self.n + 1) as f64;
        let v = self.w[self.pid(k)];
        let ip = v as i64;
        let iq = (np * v - (ip as f64) * np) as i64;
        (ip.max(0) as usize, iq.max(0) as usize)
    }

    /// `rescue` up to label 260: new provisional points, `bmat` and `zmat`.
    fn rescue_start(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        let ndim = self.ndim;
        let np = n + 1;
        let nptm = npt - np;
        let sfrac = 0.5 / np as f64;

        let mut sumpq = 0.0;
        let mut winc = 0.0f64;
        for k in 1..=npt {
            let mut distsq = 0.0;
            for j in 1..=n {
                let x = self.ix(k, j);
                self.xpt[x] -= self.xopt[j];
                distsq += self.xpt[x] * self.xpt[x];
            }
            sumpq += self.pq[k];
            let r = self.rw(ndim + k);
            self.w[r] = distsq;
            winc = winc.max(distsq);
            for j in 1..=nptm {
                let z = self.iz(k, j);
                self.zmat[z] = 0.0;
            }
        }

        let mut ih = 0;
        for j in 1..=n {
            let rj = self.rw(j);
            self.w[rj] = 0.5 * sumpq * self.xopt[j];
            for k in 1..=npt {
                self.w[rj] += self.pq[k] * self.xpt[self.ix(k, j)];
            }
            for i in 1..=j {
                ih += 1;
                self.hq[ih] =
                    self.hq[ih] + self.w[self.rw(i)] * self.xopt[j] + self.w[rj] * self.xopt[i];
            }
        }

        for j in 1..=n {
            self.xbase[j] += self.xopt[j];
            self.sl[j] -= self.xopt[j];
            self.su[j] -= self.xopt[j];
            self.xopt[j] = 0.0;
            let (p1, p2) = (self.pa(1, j), self.pa(2, j));
            self.w[p1] = self.delta.min(self.su[j]);
            self.w[p2] = (-self.delta).max(self.sl[j]);
            if self.w[p1] + self.w[p2] < 0.0 {
                self.w.swap(p1, p2);
            }
            if self.w[p2].abs() < 0.5 * self.w[p1].abs() {
                self.w[p2] = 0.5 * self.w[p1];
            }
            for i in 1..=ndim {
                let b = self.ib(i, j);
                self.bmat[b] = 0.0;
            }
        }
        self.fbase = self.fval[self.kopt];

        let p1 = self.pid(1);
        self.w[p1] = sfrac;
        for j in 1..=n {
            let jp = j + 1;
            let jpn = jp + n;
            let pj = self.pid(jp);
            self.w[pj] = j as f64 + sfrac;
            let a1 = self.w[self.pa(1, j)];
            let a2 = self.w[self.pa(2, j)];
            if jpn <= npt {
                let pj = self.pid(jpn);
                self.w[pj] = j as f64 / np as f64 + sfrac;
                let temp = 1.0 / (a1 - a2);
                let (bjp, bjpn, b1) = (self.ib(jp, j), self.ib(jpn, j), self.ib(1, j));
                self.bmat[bjp] = -temp + 1.0 / a1;
                self.bmat[bjpn] = temp + 1.0 / a2;
                self.bmat[b1] = -self.bmat[bjp] - self.bmat[bjpn];
                let (z1, zjp, zjpn) = (self.iz(1, j), self.iz(jp, j), self.iz(jpn, j));
                self.zmat[z1] = 2.0f64.sqrt() / (a1 * a2).abs();
                self.zmat[zjp] = self.zmat[z1] * a2 * temp;
                self.zmat[zjpn] = -self.zmat[z1] * a1 * temp;
            } else {
                let (b1, bjp, bjn) = (self.ib(1, j), self.ib(jp, j), self.ib(j + npt, j));
                self.bmat[b1] = -1.0 / a1;
                self.bmat[bjp] = 1.0 / a1;
                self.bmat[bjn] = -0.5 * (a1 * a1);
            }
        }

        if npt >= n + np {
            for k in (2 * np)..=npt {
                let iw = (((k - np) as f64 - 0.5) / n as f64) as usize;
                let ip = k - np - iw * n;
                let mut iq = ip + iw;
                if iq > n {
                    iq -= n;
                }
                let pk = self.pid(k);
                self.w[pk] = ip as f64 + iq as f64 / np as f64 + sfrac;
                let temp = 1.0 / (self.w[self.pa(1, ip)] * self.w[self.pa(1, iq)]);
                let idx = [
                    self.iz(1, k - np),
                    self.iz(ip + 1, k - np),
                    self.iz(iq + 1, k - np),
                    self.iz(k, k - np),
                ];
                self.zmat[idx[0]] = temp;
                self.zmat[idx[1]] = -temp;
                self.zmat[idx[2]] = -temp;
                self.zmat[idx[3]] = temp;
            }
        }
        let mut nrem = npt;
        let mut kold = 1;
        let mut knew = self.kopt;
        let mut beta = 0.0;
        let mut denom = 0.0;

        // Labels 80 and 120, which need no values.
        'l80: loop {
            for j in 1..=n {
                let (a, b) = (self.ib(kold, j), self.ib(knew, j));
                self.bmat.swap(a, b);
            }
            for j in 1..=nptm {
                let (a, b) = (self.iz(kold, j), self.iz(knew, j));
                self.zmat.swap(a, b);
            }
            let (pk, pn) = (self.pid(kold), self.pid(knew));
            self.w[pk] = self.w[pn];
            self.w[pn] = 0.0;
            let rn = self.rw(ndim + knew);
            self.w[rn] = 0.0;
            nrem -= 1;
            if knew != self.kopt {
                self.vlag.swap(kold, knew);
                let rbase = self.rw(1);
                update(
                    n,
                    npt,
                    &mut self.bmat,
                    &mut self.zmat,
                    ndim,
                    &mut self.vlag[1..],
                    beta,
                    denom,
                    knew,
                    &mut self.w[rbase..],
                );
                if nrem == 0 {
                    return Label::L190Post;
                }
                for k in 1..=npt {
                    let r = self.rw(ndim + k);
                    self.w[r] = self.w[r].abs();
                }
            }

            'l120: loop {
                let mut dsqmin = 0.0;
                for k in 1..=npt {
                    let v = self.w[self.rw(ndim + k)];
                    if v > 0.0 && (dsqmin == 0.0 || v < dsqmin) {
                        knew = k;
                        dsqmin = v;
                    }
                }
                if dsqmin == 0.0 {
                    break 'l80;
                }

                for j in 1..=n {
                    let r = self.rw(npt + j);
                    self.w[r] = self.xpt[self.ix(knew, j)];
                }
                for k in 1..=npt {
                    let mut sum = 0.0;
                    if k == self.kopt {
                    } else if self.w[self.pid(k)] == 0.0 {
                        for j in 1..=n {
                            sum += self.w[self.rw(npt + j)] * self.xpt[self.ix(k, j)];
                        }
                    } else {
                        let (ip, iq) = self.pid_parts(k);
                        if ip > 0 {
                            sum = self.w[self.rw(npt + ip)] * self.w[self.pa(1, ip)];
                        }
                        if iq > 0 {
                            let iw = if ip == 0 { 2 } else { 1 };
                            sum += self.w[self.rw(npt + iq)] * self.w[self.pa(iw, iq)];
                        }
                    }
                    let r = self.rw(k);
                    self.w[r] = 0.5 * sum * sum;
                }

                for k in 1..=npt {
                    let mut sum = 0.0;
                    for j in 1..=n {
                        sum += self.bmat[self.ib(k, j)] * self.w[self.rw(npt + j)];
                    }
                    self.vlag[k] = sum;
                }
                beta = 0.0;
                for j in 1..=nptm {
                    let mut sum = 0.0;
                    for k in 1..=npt {
                        sum += self.zmat[self.iz(k, j)] * self.w[self.rw(k)];
                    }
                    beta -= sum * sum;
                    for k in 1..=npt {
                        self.vlag[k] += sum * self.zmat[self.iz(k, j)];
                    }
                }
                let mut bsum = 0.0;
                let mut distsq = 0.0;
                for j in 1..=n {
                    let mut sum = 0.0;
                    for k in 1..=npt {
                        sum += self.bmat[self.ib(k, j)] * self.w[self.rw(k)];
                    }
                    let jp = j + npt;
                    bsum += sum * self.w[self.rw(jp)];
                    for ip in (npt + 1)..=ndim {
                        sum += self.bmat[self.ib(ip, j)] * self.w[self.rw(ip)];
                    }
                    bsum += sum * self.w[self.rw(jp)];
                    self.vlag[jp] = sum;
                    let v = self.xpt[self.ix(knew, j)];
                    distsq += v * v;
                }
                beta = 0.5 * distsq * distsq + beta - bsum;
                self.vlag[self.kopt] += 1.0;

                denom = 0.0;
                let mut vlmxsq = 0.0f64;
                for k in 1..=npt {
                    if self.w[self.pid(k)] != 0.0 {
                        let mut hdiag = 0.0;
                        for j in 1..=nptm {
                            let z = self.zmat[self.iz(k, j)];
                            hdiag += z * z;
                        }
                        let den = beta * hdiag + self.vlag[k] * self.vlag[k];
                        if den > denom {
                            kold = k;
                            denom = den;
                        }
                    }
                    vlmxsq = vlmxsq.max(self.vlag[k] * self.vlag[k]);
                }
                if denom <= vlmxsq * 0.01 {
                    let r = self.rw(ndim + knew);
                    self.w[r] = -self.w[r] - winc;
                    continue 'l120;
                }
                continue 'l80;
            }
        }

        self.kpt = 0;
        Label::RescueNext
    }

    /// `rescue`, label 260: the next provisional point to evaluate.
    fn rescue_next(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        loop {
            self.kpt += 1;
            if self.kpt > npt {
                return Label::L190Post;
            }
            let kpt = self.kpt;
            if self.w[self.pid(kpt)] == 0.0 {
                continue;
            }
            if self.stop_evals() {
                self.rc = BobyqaStatus::MaxevalReached;
                return Label::L190Post;
            }

            let mut ih = 0;
            for j in 1..=n {
                let rj = self.rw(j);
                let x = self.ix(kpt, j);
                self.w[rj] = self.xpt[x];
                self.xpt[x] = 0.0;
                let temp = self.pq[kpt] * self.w[rj];
                for i in 1..=j {
                    ih += 1;
                    self.hq[ih] += temp * self.w[self.rw(i)];
                }
            }
            self.pq[kpt] = 0.0;
            let (ip, iq) = self.pid_parts(kpt);
            self.rip = ip;
            self.riq = iq;
            if ip > 0 {
                self.xp = self.w[self.pa(1, ip)];
                let x = self.ix(kpt, ip);
                self.xpt[x] = self.xp;
            }
            if iq > 0 {
                self.xq = self.w[self.pa(1, iq)];
                if ip == 0 {
                    self.xq = self.w[self.pa(2, iq)];
                }
                let x = self.ix(kpt, iq);
                self.xpt[x] = self.xq;
            }

            let mut vquad = self.fbase;
            let mut ihp = 0;
            if ip > 0 {
                ihp = (ip + ip * ip) / 2;
                vquad += self.xp * (self.gopt[ip] + 0.5 * self.xp * self.hq[ihp]);
            }
            if iq > 0 {
                let ihq = (iq + iq * iq) / 2;
                vquad += self.xq * (self.gopt[iq] + 0.5 * self.xq * self.hq[ihq]);
                if ip > 0 {
                    let iw = ihp.max(ihq) - ip.abs_diff(iq);
                    vquad += self.xp * self.xq * self.hq[iw];
                }
            }
            for k in 1..=npt {
                let mut temp = 0.0;
                if ip > 0 {
                    temp += self.xp * self.xpt[self.ix(k, ip)];
                }
                if iq > 0 {
                    temp += self.xq * self.xpt[self.ix(k, iq)];
                }
                vquad += 0.5 * self.pq[k] * temp * temp;
            }
            self.rvquad = vquad;

            for i in 1..=n {
                let p = self.xpt[self.ix(kpt, i)];
                self.x[i] = self.xl[i].max(self.xbase[i] + p).min(self.xu[i]);
                if p == self.sl[i] {
                    self.x[i] = self.xl[i];
                }
                if p == self.su[i] {
                    self.x[i] = self.xu[i];
                }
            }
            self.request(Label::RescuePost);
            return Label::Wait;
        }
    }

    /// `rescue` after the value of point `kpt`: the model update.
    fn rescue_post(&mut self) -> Label {
        let n = self.n;
        let npt = self.npt;
        let nptm = npt - n - 1;
        let kpt = self.kpt;
        let f = self.f;
        self.fval[kpt] = f;
        if f < self.fval[self.kopt] {
            self.kopt = kpt;
        }
        if self.stop_evals() {
            self.rc = BobyqaStatus::MaxevalReached;
            return Label::L190Post;
        }

        let diff = f - self.rvquad;
        for i in 1..=n {
            self.gopt[i] += diff * self.bmat[self.ib(kpt, i)];
        }
        for k in 1..=npt {
            let mut sum = 0.0;
            for j in 1..=nptm {
                sum += self.zmat[self.iz(k, j)] * self.zmat[self.iz(kpt, j)];
            }
            let temp = diff * sum;
            if self.w[self.pid(k)] == 0.0 {
                self.pq[k] += temp;
            } else {
                let (ip, iq) = self.pid_parts(k);
                let ihq = (iq * iq + iq) / 2;
                if ip == 0 {
                    let a = self.w[self.pa(2, iq)];
                    self.hq[ihq] += temp * (a * a);
                } else {
                    let ihp = (ip * ip + ip) / 2;
                    let a = self.w[self.pa(1, ip)];
                    self.hq[ihp] += temp * (a * a);
                    if iq > 0 {
                        let b = self.w[self.pa(1, iq)];
                        self.hq[ihq] += temp * (b * b);
                        let iw = ihp.max(ihq) - iq.abs_diff(ip);
                        self.hq[iw] += temp * a * b;
                    }
                }
            }
        }
        let p = self.pid(kpt);
        self.w[p] = 0.0;
        Label::RescueNext
    }
}

/////////////////
// Subroutines //
/////////////////

/// Powell's `TRSBOX`: an approximate minimiser of the quadratic model in the
/// intersection of the trust region and the box.
///
/// Arguments are zero-based slices of the one-based arrays; `xpt` is the
/// column-major `npt * n` matrix.
///
/// ### Params
///
/// * `n`, `npt` - Dimensions
/// * `xpt`, `xopt`, `gopt`, `hq`, `pq`, `sl`, `su` - The model and box
/// * `delta` - Trust-region radius
/// * `xnew`, `d`, `gnew`, `xbdi`, `s`, `hs`, `hred` - Outputs and work space
/// * `dsq` - Set to the squared step length
/// * `crvmin` - Set to the least curvature seen
fn trsbox(
    n: usize,
    npt: usize,
    xpt: &[f64],
    xopt: &[f64],
    gopt: &[f64],
    hq: &[f64],
    pq: &[f64],
    sl: &[f64],
    su: &[f64],
    delta: f64,
    xnew: &mut [f64],
    d: &mut [f64],
    gnew: &mut [f64],
    xbdi: &mut [f64],
    s: &mut [f64],
    hs: &mut [f64],
    hred: &mut [f64],
    dsq: &mut f64,
    crvmin: &mut f64,
) {
    #[derive(Clone, Copy)]
    enum At {
        L20,
        L30,
        L50,
        L90,
        L100,
        L120,
        L150,
        L190,
        L210,
    }
    let ix = |k: usize, j: usize| (k - 1) + (j - 1) * npt;

    let mut iterc: i64 = 0;
    let mut nact: usize = 0;
    for i in 1..=n {
        xbdi[i - 1] = 0.0;
        if xopt[i - 1] <= sl[i - 1] {
            if gopt[i - 1] >= 0.0 {
                xbdi[i - 1] = -1.0;
            }
        } else if xopt[i - 1] >= su[i - 1] && gopt[i - 1] <= 0.0 {
            xbdi[i - 1] = 1.0;
        }
        if xbdi[i - 1] != 0.0 {
            nact += 1;
        }
        d[i - 1] = 0.0;
        gnew[i - 1] = gopt[i - 1];
    }
    let mut delsq = delta * delta;
    let mut qred = 0.0;
    *crvmin = -1.0;

    let mut beta = 0.0;
    let mut stepsq = 0.0;
    let mut gredsq = 0.0;
    let mut itermax: i64 = 0;
    let mut ggsav = 0.0;
    let mut iact: usize = 0;
    let mut dredsq = 0.0;
    let mut dredg = 0.0;
    let mut sredg = 0.0;
    let mut itcsav: i64 = 0;
    let mut angbd = 0.0;
    let mut xsav = 0.0;
    let mut shs;
    let mut sdec;

    let mut at = At::L20;
    loop {
        match at {
            At::L20 => {
                beta = 0.0;
                at = At::L30;
            }
            At::L30 => {
                stepsq = 0.0;
                for i in 1..=n {
                    if xbdi[i - 1] != 0.0 {
                        s[i - 1] = 0.0;
                    } else if beta == 0.0 {
                        s[i - 1] = -gnew[i - 1];
                    } else {
                        s[i - 1] = beta * s[i - 1] - gnew[i - 1];
                    }
                    stepsq += s[i - 1] * s[i - 1];
                }
                if stepsq == 0.0 {
                    at = At::L190;
                    continue;
                }
                if beta == 0.0 {
                    gredsq = stepsq;
                    itermax = iterc + n as i64 - nact as i64;
                }
                if gredsq * delsq <= qred * 1e-4 * qred {
                    at = At::L190;
                    continue;
                }
                at = At::L210;
            }
            At::L50 => {
                let mut resid = delsq;
                let mut ds = 0.0;
                shs = 0.0;
                for i in 1..=n {
                    if xbdi[i - 1] == 0.0 {
                        resid -= d[i - 1] * d[i - 1];
                        ds += s[i - 1] * d[i - 1];
                        shs += s[i - 1] * hs[i - 1];
                    }
                }
                if resid <= 0.0 {
                    at = At::L90;
                    continue;
                }
                let temp = (stepsq * resid + ds * ds).sqrt();
                let blen = if ds < 0.0 {
                    (temp - ds) / stepsq
                } else {
                    resid / (temp + ds)
                };
                let mut stplen = blen;
                if shs > 0.0 {
                    stplen = blen.min(gredsq / shs);
                }

                iact = 0;
                for i in 1..=n {
                    if s[i - 1] != 0.0 {
                        let xsum = xopt[i - 1] + d[i - 1];
                        let temp = if s[i - 1] > 0.0 {
                            (su[i - 1] - xsum) / s[i - 1]
                        } else {
                            (sl[i - 1] - xsum) / s[i - 1]
                        };
                        if temp < stplen {
                            stplen = temp;
                            iact = i;
                        }
                    }
                }

                sdec = 0.0;
                if stplen > 0.0 {
                    iterc += 1;
                    let temp = shs / stepsq;
                    if iact == 0 && temp > 0.0 {
                        *crvmin = crvmin.min(temp);
                        if *crvmin == -1.0 {
                            *crvmin = temp;
                        }
                    }
                    ggsav = gredsq;
                    gredsq = 0.0;
                    for i in 1..=n {
                        gnew[i - 1] += stplen * hs[i - 1];
                        if xbdi[i - 1] == 0.0 {
                            gredsq += gnew[i - 1] * gnew[i - 1];
                        }
                        d[i - 1] += stplen * s[i - 1];
                    }
                    sdec = (stplen * (ggsav - 0.5 * stplen * shs)).max(0.0);
                    qred += sdec;
                }

                if iact > 0 {
                    nact += 1;
                    xbdi[iact - 1] = 1.0;
                    if s[iact - 1] < 0.0 {
                        xbdi[iact - 1] = -1.0;
                    }
                    delsq -= d[iact - 1] * d[iact - 1];
                    if delsq <= 0.0 {
                        at = At::L90;
                        continue;
                    }
                    at = At::L20;
                    continue;
                }

                if stplen < blen {
                    if iterc == itermax {
                        at = At::L190;
                        continue;
                    }
                    if sdec <= qred * 0.01 {
                        at = At::L190;
                        continue;
                    }
                    beta = gredsq / ggsav;
                    at = At::L30;
                    continue;
                }
                at = At::L90;
            }
            At::L90 => {
                *crvmin = 0.0;
                at = At::L100;
            }
            At::L100 => {
                if nact + 1 >= n {
                    at = At::L190;
                    continue;
                }
                dredsq = 0.0;
                dredg = 0.0;
                gredsq = 0.0;
                for i in 1..=n {
                    if xbdi[i - 1] == 0.0 {
                        dredsq += d[i - 1] * d[i - 1];
                        dredg += d[i - 1] * gnew[i - 1];
                        gredsq += gnew[i - 1] * gnew[i - 1];
                        s[i - 1] = d[i - 1];
                    } else {
                        s[i - 1] = 0.0;
                    }
                }
                itcsav = iterc;
                at = At::L210;
            }
            At::L120 => {
                iterc += 1;
                let temp = gredsq * dredsq - dredg * dredg;
                if temp <= qred * 1e-4 * qred {
                    at = At::L190;
                    continue;
                }
                let temp = temp.sqrt();
                for i in 1..=n {
                    if xbdi[i - 1] == 0.0 {
                        s[i - 1] = (dredg * d[i - 1] - dredsq * gnew[i - 1]) / temp;
                    } else {
                        s[i - 1] = 0.0;
                    }
                }
                sredg = -temp;

                angbd = 1.0;
                iact = 0;
                let mut back_to_100 = false;
                for i in 1..=n {
                    if xbdi[i - 1] == 0.0 {
                        let tempa = xopt[i - 1] + d[i - 1] - sl[i - 1];
                        let tempb = su[i - 1] - xopt[i - 1] - d[i - 1];
                        if tempa <= 0.0 {
                            nact += 1;
                            xbdi[i - 1] = -1.0;
                            back_to_100 = true;
                            break;
                        } else if tempb <= 0.0 {
                            nact += 1;
                            xbdi[i - 1] = 1.0;
                            back_to_100 = true;
                            break;
                        }
                        let ssq = d[i - 1] * d[i - 1] + s[i - 1] * s[i - 1];
                        let a = xopt[i - 1] - sl[i - 1];
                        let temp = ssq - a * a;
                        if temp > 0.0 {
                            let temp = temp.sqrt() - s[i - 1];
                            if angbd * temp > tempa {
                                angbd = tempa / temp;
                                iact = i;
                                xsav = -1.0;
                            }
                        }
                        let b = su[i - 1] - xopt[i - 1];
                        let temp = ssq - b * b;
                        if temp > 0.0 {
                            let temp = temp.sqrt() + s[i - 1];
                            if angbd * temp > tempb {
                                angbd = tempb / temp;
                                iact = i;
                                xsav = 1.0;
                            }
                        }
                    }
                }
                if back_to_100 {
                    at = At::L100;
                    continue;
                }
                at = At::L210;
            }
            At::L150 => {
                shs = 0.0;
                let mut dhs = 0.0;
                let mut dhd = 0.0;
                for i in 1..=n {
                    if xbdi[i - 1] == 0.0 {
                        shs += s[i - 1] * hs[i - 1];
                        dhs += d[i - 1] * hs[i - 1];
                        dhd += d[i - 1] * hred[i - 1];
                    }
                }

                let mut redmax = 0.0;
                let mut isav: i64 = 0;
                let mut redsav = 0.0;
                let mut rdprev = 0.0;
                let mut rdnext = 0.0;
                let iu = (angbd * 17.0 + 3.1) as i64;
                for i in 1..=iu {
                    let angt = angbd * i as f64 / iu as f64;
                    let sth = (angt + angt) / (1.0 + angt * angt);
                    let temp = shs + angt * (angt * dhd - dhs - dhs);
                    let rednew = sth * (angt * dredg - sredg - 0.5 * sth * temp);
                    if rednew > redmax {
                        redmax = rednew;
                        isav = i;
                        rdprev = redsav;
                    } else if i == isav + 1 {
                        rdnext = rednew;
                    }
                    redsav = rednew;
                }

                if isav == 0 {
                    at = At::L190;
                    continue;
                }
                let mut angt = 0.0;
                if isav < iu {
                    let temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
                    angt = angbd * (isav as f64 + 0.5 * temp) / iu as f64;
                }
                let cth = (1.0 - angt * angt) / (1.0 + angt * angt);
                let sth = (angt + angt) / (1.0 + angt * angt);
                let temp = shs + angt * (angt * dhd - dhs - dhs);
                sdec = sth * (angt * dredg - sredg - 0.5 * sth * temp);
                if sdec <= 0.0 {
                    at = At::L190;
                    continue;
                }

                dredg = 0.0;
                gredsq = 0.0;
                for i in 1..=n {
                    gnew[i - 1] = gnew[i - 1] + (cth - 1.0) * hred[i - 1] + sth * hs[i - 1];
                    if xbdi[i - 1] == 0.0 {
                        d[i - 1] = cth * d[i - 1] + sth * s[i - 1];
                        dredg += d[i - 1] * gnew[i - 1];
                        gredsq += gnew[i - 1] * gnew[i - 1];
                    }
                    hred[i - 1] = cth * hred[i - 1] + sth * hs[i - 1];
                }
                qred += sdec;
                if iact > 0 && isav == iu {
                    nact += 1;
                    xbdi[iact - 1] = xsav;
                    at = At::L100;
                    continue;
                }

                if sdec > qred * 0.01 {
                    at = At::L120;
                    continue;
                }
                at = At::L190;
            }
            At::L190 => {
                *dsq = 0.0;
                for i in 1..=n {
                    xnew[i - 1] = (xopt[i - 1] + d[i - 1]).min(su[i - 1]).max(sl[i - 1]);
                    if xbdi[i - 1] == -1.0 {
                        xnew[i - 1] = sl[i - 1];
                    }
                    if xbdi[i - 1] == 1.0 {
                        xnew[i - 1] = su[i - 1];
                    }
                    d[i - 1] = xnew[i - 1] - xopt[i - 1];
                    *dsq += d[i - 1] * d[i - 1];
                }
                return;
            }
            At::L210 => {
                let mut ih = 0;
                for j in 1..=n {
                    hs[j - 1] = 0.0;
                    for i in 1..=j {
                        ih += 1;
                        if i < j {
                            hs[j - 1] += hq[ih - 1] * s[i - 1];
                        }
                        hs[i - 1] += hq[ih - 1] * s[j - 1];
                    }
                }
                for k in 1..=npt {
                    if pq[k - 1] != 0.0 {
                        let mut temp = 0.0;
                        for j in 1..=n {
                            temp += xpt[ix(k, j)] * s[j - 1];
                        }
                        temp *= pq[k - 1];
                        for i in 1..=n {
                            hs[i - 1] += temp * xpt[ix(k, i)];
                        }
                    }
                }
                if *crvmin != 0.0 {
                    at = At::L50;
                    continue;
                }
                if iterc > itcsav {
                    at = At::L150;
                    continue;
                }
                hred[..n].copy_from_slice(&hs[..n]);
                at = At::L120;
            }
        }
    }
}

/// Powell's `ALTMOV`: two alternative new positions for interpolation point
/// `knew`, along a line through `xopt` and by a constrained Cauchy step.
///
/// ### Params
///
/// * `n`, `npt`, `ndim` - Dimensions
/// * `xpt`, `xopt`, `bmat`, `zmat`, `sl`, `su` - The model and box
/// * `kopt`, `knew` - Best point and the point to move, one-based
/// * `adelt` - Bound on the step
/// * `xnew`, `xalt` - The two candidates
/// * `alpha` - Set to `H(knew, knew)`
/// * `cauchy` - Set to the squared Lagrange value at `xalt`
/// * `glag`, `hcol`, `w` - Work space of `n`, `npt` and `2n`
fn altmov(
    n: usize,
    npt: usize,
    xpt: &[f64],
    xopt: &[f64],
    bmat: &[f64],
    zmat: &[f64],
    ndim: usize,
    sl: &[f64],
    su: &[f64],
    kopt: usize,
    knew: usize,
    adelt: f64,
    xnew: &mut [f64],
    xalt: &mut [f64],
    alpha: &mut f64,
    cauchy: &mut f64,
    glag: &mut [f64],
    hcol: &mut [f64],
    w: &mut [f64],
) {
    let ix = |k: usize, j: usize| (k - 1) + (j - 1) * npt;
    let ib = |k: usize, j: usize| (k - 1) + (j - 1) * ndim;
    let iz = |k: usize, j: usize| (k - 1) + (j - 1) * npt;
    let const_ = 1.0 + 2.0f64.sqrt();

    for k in 1..=npt {
        hcol[k - 1] = 0.0;
    }
    for j in 1..=(npt - n - 1) {
        let temp = zmat[iz(knew, j)];
        for k in 1..=npt {
            hcol[k - 1] += temp * zmat[iz(k, j)];
        }
    }
    *alpha = hcol[knew - 1];
    let ha = 0.5 * *alpha;

    for i in 1..=n {
        glag[i - 1] = bmat[ib(knew, i)];
    }
    for k in 1..=npt {
        let mut temp = 0.0;
        for j in 1..=n {
            temp += xpt[ix(k, j)] * xopt[j - 1];
        }
        temp *= hcol[k - 1];
        for i in 1..=n {
            glag[i - 1] += temp * xpt[ix(k, i)];
        }
    }

    let mut presav = 0.0;
    let mut ksav = 0;
    let mut stpsav = 0.0;
    let mut ibdsav: i64 = 0;
    for k in 1..=npt {
        if k == kopt {
            continue;
        }
        let mut dderiv = 0.0;
        let mut distsq = 0.0;
        for i in 1..=n {
            let temp = xpt[ix(k, i)] - xopt[i - 1];
            dderiv += glag[i - 1] * temp;
            distsq += temp * temp;
        }
        let mut subd = adelt / distsq.sqrt();
        let mut slbd = -subd;
        let mut ilbd: i64 = 0;
        let mut iubd: i64 = 0;
        let sumin = 1.0f64.min(subd);

        for i in 1..=n {
            let temp = xpt[ix(k, i)] - xopt[i - 1];
            if temp > 0.0 {
                if slbd * temp < sl[i - 1] - xopt[i - 1] {
                    slbd = (sl[i - 1] - xopt[i - 1]) / temp;
                    ilbd = -(i as i64);
                }
                if subd * temp > su[i - 1] - xopt[i - 1] {
                    subd = sumin.max((su[i - 1] - xopt[i - 1]) / temp);
                    iubd = i as i64;
                }
            } else if temp < 0.0 {
                if slbd * temp > su[i - 1] - xopt[i - 1] {
                    slbd = (su[i - 1] - xopt[i - 1]) / temp;
                    ilbd = i as i64;
                }
                if subd * temp < sl[i - 1] - xopt[i - 1] {
                    subd = sumin.max((sl[i - 1] - xopt[i - 1]) / temp);
                    iubd = -(i as i64);
                }
            }
        }

        let mut step;
        let mut vlag;
        let mut isbd;
        if k == knew {
            let diff = dderiv - 1.0;
            step = slbd;
            vlag = slbd * (dderiv - slbd * diff);
            isbd = ilbd;
            let temp = subd * (dderiv - subd * diff);
            if temp.abs() > vlag.abs() {
                step = subd;
                vlag = temp;
                isbd = iubd;
            }
            let tempd = 0.5 * dderiv;
            let tempa = tempd - diff * slbd;
            let tempb = tempd - diff * subd;
            if tempa * tempb < 0.0 {
                let temp = tempd * tempd / diff;
                if temp.abs() > vlag.abs() {
                    step = tempd / diff;
                    vlag = temp;
                    isbd = 0;
                }
            }
        } else {
            step = slbd;
            vlag = slbd * (1.0 - slbd);
            isbd = ilbd;
            let temp = subd * (1.0 - subd);
            if temp.abs() > vlag.abs() {
                step = subd;
                vlag = temp;
                isbd = iubd;
            }
            if subd > 0.5 && vlag.abs() < 0.25 {
                step = 0.5;
                vlag = 0.25;
                isbd = 0;
            }
            vlag *= dderiv;
        }

        let temp = step * (1.0 - step) * distsq;
        let predsq = vlag * vlag * (vlag * vlag + ha * temp * temp);
        if predsq > presav {
            presav = predsq;
            ksav = k;
            stpsav = step;
            ibdsav = isbd;
        }
    }

    // The C leaves `ksav` at zero when no line predicts a positive
    // denominator and then reads outside `xpt`; `stpsav` is zero then, so any
    // row gives `xnew = xopt`.
    let ksav = ksav.max(1);
    for i in 1..=n {
        let temp = xopt[i - 1] + stpsav * (xpt[ix(ksav, i)] - xopt[i - 1]);
        xnew[i - 1] = sl[i - 1].max(su[i - 1].min(temp));
    }
    if ibdsav < 0 {
        let i = (-ibdsav) as usize;
        xnew[i - 1] = sl[i - 1];
    }
    if ibdsav > 0 {
        let i = ibdsav as usize;
        xnew[i - 1] = su[i - 1];
    }

    let bigstp = adelt + adelt;
    let mut iflag = 0;
    let mut csave = 0.0;
    loop {
        // Label 100.
        let mut wfixsq = 0.0;
        let mut ggfree = 0.0;
        for i in 1..=n {
            w[i - 1] = 0.0;
            let tempa = (xopt[i - 1] - sl[i - 1]).min(glag[i - 1]);
            let tempb = (xopt[i - 1] - su[i - 1]).max(glag[i - 1]);
            if tempa > 0.0 || tempb < 0.0 {
                w[i - 1] = bigstp;
                ggfree += glag[i - 1] * glag[i - 1];
            }
        }
        if ggfree == 0.0 {
            *cauchy = 0.0;
            return;
        }

        // Label 120.
        let mut step = 0.0;
        loop {
            let temp = adelt * adelt - wfixsq;
            if temp > 0.0 {
                let wsqsav = wfixsq;
                step = (temp / ggfree).sqrt();
                ggfree = 0.0;
                for i in 1..=n {
                    if w[i - 1] == bigstp {
                        let temp = xopt[i - 1] - step * glag[i - 1];
                        if temp <= sl[i - 1] {
                            w[i - 1] = sl[i - 1] - xopt[i - 1];
                            wfixsq += w[i - 1] * w[i - 1];
                        } else if temp >= su[i - 1] {
                            w[i - 1] = su[i - 1] - xopt[i - 1];
                            wfixsq += w[i - 1] * w[i - 1];
                        } else {
                            ggfree += glag[i - 1] * glag[i - 1];
                        }
                    }
                }
                if wfixsq > wsqsav && ggfree > 0.0 {
                    continue;
                }
            }
            break;
        }

        let mut gw = 0.0;
        for i in 1..=n {
            if w[i - 1] == bigstp {
                w[i - 1] = -step * glag[i - 1];
                xalt[i - 1] = sl[i - 1].max(su[i - 1].min(xopt[i - 1] + w[i - 1]));
            } else if w[i - 1] == 0.0 {
                xalt[i - 1] = xopt[i - 1];
            } else if glag[i - 1] > 0.0 {
                xalt[i - 1] = sl[i - 1];
            } else {
                xalt[i - 1] = su[i - 1];
            }
            gw += glag[i - 1] * w[i - 1];
        }

        let mut curv = 0.0;
        for k in 1..=npt {
            let mut temp = 0.0;
            for j in 1..=n {
                temp += xpt[ix(k, j)] * w[j - 1];
            }
            curv += hcol[k - 1] * temp * temp;
        }
        if iflag == 1 {
            curv = -curv;
        }
        if curv > -gw && curv < -const_ * gw {
            let scale = -gw / curv;
            for i in 1..=n {
                let temp = xopt[i - 1] + scale * w[i - 1];
                xalt[i - 1] = sl[i - 1].max(su[i - 1].min(temp));
            }
            let a = 0.5 * gw * scale;
            *cauchy = a * a;
        } else {
            let a = gw + 0.5 * curv;
            *cauchy = a * a;
        }

        if iflag == 0 {
            for i in 1..=n {
                glag[i - 1] = -glag[i - 1];
                w[n + i - 1] = xalt[i - 1];
            }
            csave = *cauchy;
            iflag = 1;
            continue;
        }
        if csave > *cauchy {
            for i in 1..=n {
                xalt[i - 1] = w[n + i - 1];
            }
            *cauchy = csave;
        }
        return;
    }
}

/// Powell's `UPDATE`: revises `bmat` and `zmat` for moving point `knew`.
///
/// ### Params
///
/// * `n`, `npt`, `ndim` - Dimensions
/// * `bmat`, `zmat` - Column-major matrices, updated
/// * `vlag` - Zero-based, length `ndim`; its `knew`-th entry is decremented
/// * `beta`, `denom` - Updating parameters
/// * `knew` - The point being moved, one-based
/// * `w` - Work space, zero-based, length at least `ndim`
fn update(
    n: usize,
    npt: usize,
    bmat: &mut [f64],
    zmat: &mut [f64],
    ndim: usize,
    vlag: &mut [f64],
    beta: f64,
    denom: f64,
    knew: usize,
    w: &mut [f64],
) {
    let ib = |k: usize, j: usize| (k - 1) + (j - 1) * ndim;
    let iz = |k: usize, j: usize| (k - 1) + (j - 1) * npt;
    let nptm = npt - n - 1;

    let mut ztest = 0.0f64;
    for k in 1..=npt {
        for j in 1..=nptm {
            ztest = ztest.max(zmat[iz(k, j)].abs());
        }
    }
    ztest *= 1e-20;

    for j in 2..=nptm {
        if zmat[iz(knew, j)].abs() > ztest {
            let a = zmat[iz(knew, 1)];
            let b = zmat[iz(knew, j)];
            let temp = (a * a + b * b).sqrt();
            let tempa = a / temp;
            let tempb = b / temp;
            for i in 1..=npt {
                let temp = tempa * zmat[iz(i, 1)] + tempb * zmat[iz(i, j)];
                zmat[iz(i, j)] = tempa * zmat[iz(i, j)] - tempb * zmat[iz(i, 1)];
                zmat[iz(i, 1)] = temp;
            }
        }
        zmat[iz(knew, j)] = 0.0;
    }

    for i in 1..=npt {
        w[i - 1] = zmat[iz(knew, 1)] * zmat[iz(i, 1)];
    }
    let alpha = w[knew - 1];
    let tau = vlag[knew - 1];
    vlag[knew - 1] -= 1.0;

    let temp = denom.sqrt();
    let tempb = zmat[iz(knew, 1)] / temp;
    let tempa = tau / temp;
    for i in 1..=npt {
        zmat[iz(i, 1)] = tempa * zmat[iz(i, 1)] - tempb * vlag[i - 1];
    }

    for j in 1..=n {
        let jp = npt + j;
        w[jp - 1] = bmat[ib(knew, j)];
        let tempa = (alpha * vlag[jp - 1] - tau * w[jp - 1]) / denom;
        let tempb = (-beta * w[jp - 1] - tau * vlag[jp - 1]) / denom;
        for i in 1..=jp {
            bmat[ib(i, j)] = bmat[ib(i, j)] + tempa * vlag[i - 1] + tempb * w[i - 1];
            if i > npt {
                bmat[ib(jp, i - npt)] = bmat[ib(i, j)];
            }
        }
    }
}

/// [`BobyqaStepper`] driven by a closure.
///
/// ### Params
///
/// * `f` - Objective
/// * `x0` - Starting point, inside the box
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
/// * `params` - Stopping knobs, or nloptr's defaults
///
/// ### Returns
///
/// The result, or the error of [`BobyqaStepper::new`].
pub fn bobyqa<F>(
    mut f: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    params: Option<BobyqaParams>,
) -> Result<BobyqaResult, EdgeErrors>
where
    F: FnMut(&[f64]) -> f64,
{
    let mut stepper = BobyqaStepper::new(x0, lower, upper, params)?;
    while let Some(x) = stepper.ask() {
        let value = f(x);
        stepper.tell(value);
    }
    Ok(stepper.result())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// One of the bounded problems `tests/r/generate_fixtures.R` traces through
    /// `nloptr::bobyqa`.
    struct Case {
        /// Fixture suffix.
        tag: &'static str,
        /// Objective, written as R evaluates it.
        f: fn(&[f64]) -> f64,
        /// Start.
        x0: &'static [f64],
        /// Lower bounds.
        lo: &'static [f64],
        /// Upper bounds.
        hi: &'static [f64],
        /// nloptr's exit status.
        status: BobyqaStatus,
    }

    fn rosen(x: &[f64]) -> f64 {
        100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
    }

    fn corner(x: &[f64]) -> f64 {
        (x[0] - 3.0).powi(2) + 10.0 * (x[1] + 1.0).powi(2) + x[0] * x[1]
    }

    fn ripple(x: &[f64]) -> f64 {
        let u = 10000.0 * x[0] + 30000.0 * x[1];
        let t = (u - 2.0 * ((u + 1.0) / 2.0).floor()).abs();
        (x[0] - 0.25).powi(2) + 2.0 * (x[1] + 0.125).powi(2) + 0.5 * x[0] * x[1] + 0.5 * (t * t)
    }

    fn valley(x: &[f64]) -> f64 {
        (x[0] + x[1] - 1.0).powi(2)
    }

    fn chain3(x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2)
            + 100.0 * (x[1] - x[0] * x[0]).powi(2)
            + (x[2] - x[1]).powi(2)
            + 0.5 * x[2].powi(2).powi(2)
    }

    const CASES: [Case; 6] = [
        Case {
            tag: "rosen",
            f: rosen,
            x0: &[-1.2, 1.0],
            lo: &[-2.0, -1.0],
            hi: &[2.0, 3.0],
            status: BobyqaStatus::XtolReached,
        },
        Case {
            tag: "rosen_edge",
            f: rosen,
            x0: &[0.4001, 0.2],
            lo: &[0.4, 0.2],
            hi: &[2.0, 2.0],
            status: BobyqaStatus::XtolReached,
        },
        Case {
            tag: "corner",
            f: corner,
            x0: &[1.0, 1.0],
            lo: &[0.0, 0.0],
            hi: &[2.0, 2.0],
            status: BobyqaStatus::XtolReached,
        },
        Case {
            tag: "chain3",
            f: chain3,
            x0: &[0.5, 2.0, -1.0],
            lo: &[-3.0, -3.0, -3.0],
            hi: &[3.0, 3.0, 3.0],
            status: BobyqaStatus::XtolReached,
        },
        // The ripple degrades the interpolation set until `rescue` runs.
        Case {
            tag: "rescue",
            f: ripple,
            x0: &[-1.0, 0.25],
            lo: &[-2.0, -2.0],
            hi: &[2.0, 2.0],
            status: BobyqaStatus::XtolReached,
        },
        Case {
            tag: "roundoff",
            f: valley,
            x0: &[0.0, 0.0],
            lo: &[-2.0, -2.0],
            hi: &[2.0, 2.0],
            status: BobyqaStatus::RoundoffLimited,
        },
    ];

    /// Reads one trace: a row per evaluation, the point then the value.
    fn trace(tag: &str) -> Vec<Vec<f64>> {
        let path: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "tests",
            "data",
            "e2e",
            &format!("bobyqa_{tag}.csv"),
        ]
        .iter()
        .collect();
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
            .lines()
            .map(|l| {
                l.split(',')
                    .map(|v| v.trim().parse().expect("number"))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_bobyqa_retraces_nloptr_point_for_point() {
        for case in &CASES {
            let want = trace(case.tag);
            let mut got: Vec<Vec<f64>> = Vec::new();
            let res = bobyqa(
                |x| {
                    let v = (case.f)(x);
                    got.push(x.iter().copied().chain([v]).collect());
                    v
                },
                case.x0,
                case.lo,
                case.hi,
                None,
            )
            .expect("valid box");
            assert_eq!(got.len(), want.len(), "{}: evaluation count", case.tag);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{}: evaluation {i}", case.tag);
            }
            assert_eq!(res.evaluations, want.len());
            assert_eq!(res.status, case.status, "{}: status", case.tag);
        }
    }

    #[test]
    fn test_bobyqa_lands_on_an_active_bound() {
        let res = bobyqa(corner, &[1.0, 1.0], &[0.0, 0.0], &[2.0, 2.0], None).expect("valid box");
        assert_eq!(res.x, vec![2.0, 0.0]);
        assert_eq!(res.f, 11.0);
    }
}
