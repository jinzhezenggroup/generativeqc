# Decision: bound DF omissions after the metric and select work before storage

Status: implemented
Date: 2026-10-25

## Problem

The 96-atom pure-Coulomb endpoints repeated hundreds of millions of radial
source/target products per Fock build. The smaller r2SCAN basis could fit raw
three-center values, but retaining that tensor was slower than contracting
current Hermite densities. Lowering the public memory cap selected the faster
route experimentally, yet also rejected r2SCAN-3c when its complete plan no
longer fit. A work policy must use the actual user budget.

## Decision

For automatic restricted pure-J CUDA preparation with a positive screening
threshold and the existing optional Hermite consumer enabled, prefer streaming
when the packed raw tensor is at least 4 GiB and four times the Hermite reserve
is smaller than the tensor. The streamed planner uses the complete original
budget and still charges panels, metric, SCF and response reservations. Explicit
storage choices retain their meaning. Optional-owner or metric-rank refusal
retains the generic streamed provider; it may be slower but remains complete.

The optional full-rank Hermite value owner splits an absolute final-J omission
allowance, min(1e-15, screening_tolerance * 1e-3), equally between charge and
projection. Current source norms are recomputed for every density or auxiliary
potential, including Graph replay. Primitive Coulomb norms come from the generic
compiler algebra extended through combined angular order six. Norms use the
actual stored transform coefficients and outward positive arithmetic.

For charge, let B bound an AO-pair norm, C an auxiliary AO norm, Q the stored
metric eigenvectors, and lambda the smallest stored eigenvalue. The conservative
amplification bound is B * naux * C**2 * ||Q||_F**2 / lambda. Maximum shell
primitive multiplicities turn component maxima into contracted bounds. Divide
half the J allowance by this amplification, the total pair-primitive count,
and an additional factor four. Every removed current source is bounded by this
per-source limit, so omissions cannot silently accumulate with system size.
Invalid or nonfinite metric factors disable charge omission. The factor four
is slack, not a proof of unrelated FP64 evaluation rounding.

Projection uses the outward sum of all current auxiliary Hermite source norms
and each target component norm, dividing its half allowance by the total target
count. NaN/Inf norms retain work. Derivatives never use either primal omission:
response refreshes the exact-nonzero source list and traverses its full domain.
No force is inferred from an omitted primal contribution.

## Work, ownership and fallbacks

Norms are fields of the existing primitive inventory. Temporary scalars borrow
potentials whose previous contents are dead, on the same owning stream. The
capacity model includes the enlarged inventory. Optional allocations fail
transactionally; no CPU oracle enters production.

With GENERATIVEQC_DF_TRACE, non-captured calls count executed charge and
projection radial products separately from submitted unscreened upper bounds.
The diagnostic uses one dead pair-potential scalar, synchronizes before its
stack download expires, and does not add a persistent telemetry arena. It is
excluded from clean endpoint timings and skipped during Graph capture.

## Evidence and rejected alternatives

Independent norm/derivative algebra covers mixed f/f Hermites and extreme
exponents. Raw libcint/NumPy product and metric tests cover Cartesian/spherical
bases, signed and changed inputs, diffuse tails, huge density scales, nonfinite
fail-open behavior, Graph replay, exact-capacity refusal, derivative finite
differences and rank-deficient fallback. The 96/120-atom original, replay,
displaced and fresh-displaced complete endpoints pass independent GPU4PySCF
energy/force gates. Source-matched receipts and reproduction commands belong in
benchmarks/results/r2scan-endpoints-20261025/.

An initial fixed primitive threshold was rejected: it did not bound aggregate
omission or metric amplification. Artificially reducing the memory cap was
also rejected because it can turn a supported complete endpoint into an error.
The retained policy changes work selection without claiming that bounded
storage alone bounds recomputation.

Revisit the crossover and metric bound with complete cold/warm/moved evidence
on additional shapes and hardware; isolated radial-kernel speed is insufficient.
