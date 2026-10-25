# Proposal: bound current-density DF source work before radial contraction

Status: proposed
Date: 2026-10-25

## Remaining cost

After exact-zero source compaction and larger bounded force tiles, the remaining
large cold costs are the streamed r2SCAN-3c primitive domain and PBE96 resident
three-center preparation. Exact-zero compaction alone cannot improve resident
PBE, and increasing force capacity cannot remove SCF source work. On the
2026-10-25 research baseline, r2SCAN96 traversed 431952 orbital primitive
products against 2208 auxiliary primitives per dense charge/projection pass.
Earlier bound probes materially reduced this domain; they were experiments,
not an accepted default or a complete error certificate.

## Proposed numerical owner

Use the existing Hermite representation and generated algebra. For one
unnormalized Hermite Gaussian of exponent p and multi-index t, its Coulomb
self norm squared is

    sqrt(2) * pi**(5/2) * p**(|t|-5/2)
        * product((2*t_i-1)!!) / (2*|t|+1).

The existing independent libcint experiment covered 588 Cartesian/Hermite
comparisons through degree six and exponents 1e-4 through 1e4, with maximum
relative discrepancy 1.8757e-13. It has not yet qualified a compiled GPU bound.
The old helper's factor-two slack is not an interval-arithmetic proof.

A stronger next implementation should use upward positive sums/products and
square roots, a documented upper constant, and a downward denominator. An
unrepresentable or nonfinite norm must retain the source. Test the compiled
helper against the independent oracle, including cancellation, tiny/large
exponents, signed contractions and underflow/overflow boundaries.

Screen the *current precontracted Hermite density*, not a frozen bare AO-pair
coefficient alone. If h_t is the actual FP64 precontracted density, the positive
sum of abs(h_t) times the Hermite norm bounds its source norm. This responds to
arbitrary signed/large input densities and retains NaNs. A bare static cutoff
can hide a tiny coefficient multiplied by a huge input or hide NaN * zero.
The existing exact-zero compactor already refreshes the source list on every
call and captured replay; it is the appropriate execution/lifetime owner.

For charge, combine that density-weighted source norm with the auxiliary
component norm, sum all discarded contributions, and include auxiliary
contractions and the public AO convention. For projection, combine the current
auxiliary-potential source norm with each orbital target norm. Treat any scalar
error budgets as complete sums, not merely per-primitive cutoffs.

The enclosing full-rank metric owner must translate a charge-error budget into
a final-J budget. Metric conditioning cannot be ignored: delta J includes
A M^-1 delta rho as well as the projection error. Retained eigenvalues and
physical Coulomb norms can provide a conservative amplification bound. Keep
rank-truncated/unsupported owners on their current path; never fabricate
spectrum/rank information or silently change the existing rank threshold.

Force response should initially keep the complete nonzero source domain and
all raised orbital projections. A value bound does not justify derivative
screening. Qualify full cold/warm energy and analytic forces, original and moved
geometries, SCF work, graph refresh, exact byte admission and failure cleanup.
An accuracy fallback must be selected before execution; it cannot hide a
provider failure or consume uncharged storage.

## Residency and follow-up

Once screened streamed MD is independently qualified, compare it with resident
raw A for PBE96 using the original public DF memory allowance. Reducing the
allowance to 4 GiB was a useful ablation; it is not a production scheduling
policy. Choose a complete endpoint plan using measured source work and expected
reuse, preserving resident, rank and capacity fallbacks.

Do not reopen HF screening or replace the DF eigensolver before these larger
costs are addressed. The earlier inverse-root ablation saves about 0.4 seconds
for r2SCAN96 but needs an honest lazy-materialization/consumer contract; leaving
a poisoned or fake root is not an implementation.

## Retained experiments

The research checkout is `/home/jzzeng/codes/qc-gpu4-parity-research-20261025`;
ignored receipts are in `.artifacts/parity-research/`, mirrored under
`n1:/data/jzzeng/gpu4-parity-research-20261025/`. Relevant sources include
`md-bound-probe.patch`, `md-zero-probe.patch`, `hermite-norm-libcint.json`,
`md-memory-trials/`, and `md-moved-qualification-rerun/`. The complete MD libraries
have matching stationary AOT manifests; earlier identity-mismatch failures stay
retained and must not be mistaken for completed endpoint populations.
