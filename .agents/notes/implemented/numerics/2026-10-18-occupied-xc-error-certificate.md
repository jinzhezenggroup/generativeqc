# Decision: certify occupied-XC cancellation against the stored density

Status: implemented, opt-in; renewed numerical gates pass, default performance rejected
Date: 2026-10-18

## Why the exponent repair was insufficient

This supersedes the default/admission decision in the
[exponent-edge note](2026-10-17-occupied-xc-representability.md), not its historical
measurements. An ordinary orthonormal frame with occupied column `(0.6,0.8)`
and AO jets proportional to `(4e6,-3e6)` passes coefficient exponent checks and
has finite, zero projected jets. Its rounded stored-D consumer nevertheless
returns rho approximately 0.0027939677 and tau approximately 0.0363215804.
Neither a coefficient cutoff nor a nonfinite-panel test certifies cancellation.
These are generated-domain counterexamples, not evidence of a physical molecular
SCF failure. Agreement on molecular endpoints cannot exclude this admitted class.

## Ownership and scope

The existing native owner authenticates a coefficient frame that generated the
accepted density through the strict-FP64 structured Gram producer. Its D/C
pointers, spin/shape/occupation metadata, generation and stream lease must match.
An arbitrary caller-created D/C pair is not a density-factor proof. This rule
does not extend admission to held/mixed/nonlocal/response/chunk densities.

The compiler owns the certificate and both feature routes. RKS has occupation
weight `w=2`, UKS `w=1`; unequal and empty spins use the same bound with the
maximum occupied rank as an upper bound. Mapped AO spans restrict both D and C
to the same rows. Potential assembly and exact Direct J/K remain unchanged.

## Error certificate

Let `n` be the active AO span, `r` the maximum occupied rank, and `a_j` the
four AO jets. For each occupied orbital and jet compute

```text
p[j,o] = RN dot(C[:,o], a[j,:])
s[j,o] >= sum_mu abs(C[mu,o] * a[j,mu])
T[j] >= sum_o s[j,o]^2
```

Every product and addition in `s` and `T` rounds upward, including warp-tree
reductions. These are interval upper bounds, not RN magnitude estimates.
The RKS occupation factor is not put into the projection and is applied once
to the candidate feature and its error bound.

For an admitted point, every consumed nonzero C is in `2^-256..2^256`, AO
values are finite with magnitude at most `2^256`, and each `s` is either exact
zero or in `2^-256..2^256`. A positive product below the subnormal range rounds
up to a positive `s`, not a false exact zero. Other points use stored D.
Finite projected panels alone are never sufficient.

With `u=2^-53`, use upward-rounded

```text
K = 8*(n+r) + 128
alpha = K*u < 1/8
gamma = alpha / (1-alpha)
U = K^2 * 2^-810
M_rho = T[0]
M_grad[k] = 2*sqrt_up(T[0])*sqrt_up(T[k+1])
M_tau = (T[1]+T[2]+T[3])/2
B[term] = 2*gamma*w*M[term] + U
```

Counts are checked within exact integer-to-FP64 conversion range before this
formula. Subtraction in the denominator rounds downward. The radicand roots
are taken separately before multiplication, avoiding an unnecessary overflow
of `T[0]*T[k]`. Each subsequent positive operation rounds upward.

### Rounding envelope

The retained stored Gram differs from the exact `w*C*C.T` by the usual
absolute-product forward-error bound. The conservative K covers the producer,
stored half-before-add symmetrization, increasing-AO density dots (including
tile padding), feature bilinears and scalar/warp reductions. It also covers
the projected dots followed by the same feature bilinears and reduction.
For either route, its normal-rounding contribution is at most
`gamma*w*M`. Adding both routes gives the `2*gamma` term. Cauchy-Schwarz bounds
the cross-jet absolute products by the square-root expression above; rho and
tau use nonnegative squared norms. No gradient is certified from rho alone.

For the operation-count margin, a scalar Gram path uses at most `2*r`
roundings; matrix-AO and final AO/feature sums contribute at most `4*n`.
Half-before-add symmetrization, the three-axis kinetic sum, occupation scaling
and feature bilinears add fewer than 32. A tiled projection instead uses at
most `2*(n+15)` roundings including zero padding, followed by at most `2*r`
for an orbital reduction and fewer than 32 for feature arithmetic. Warp
reductions replace a scalar reduction by at most five additional levels.
Thus both routes fit `8*(n+r)+128`, including the minimum-span cases.
Combining normal relative-error factors uses the standard product-to-gamma
inequality with the sum of path lengths, not a separate relative-error
assumption on a cancellation-prone dot or gradient.

The coefficient range prevents underflow or overflow of individual Gram
products. Their dyadic products have no bit below `2^-616`, so the stored Gram
and its half factors are either zero or normal. Remaining gradual-underflow
errors arise in AO projection, matrix-AO products and feature arithmetic.
The admitted jet/dot magnitude envelope bounds propagation by `2^256`, with
at most quadratic K path multiplicity. `U = 256*2^-1074*2^256*K^2` encloses
these contributions with a conservative operation-count margin, including
occupation and gradient factors. It is not a numerical acceptance tolerance.

The same absolute-product bounds keep every admitted matrix/projection dot
and feature partial sum finite. In particular, the feature absolute sum is
bounded by `w*sum_o s[j,o]*s[k,o]`, not an unrelated product of the largest
AO and largest density entry. An outside-envelope input cannot suppress an
original matrix overflow: it is refused and evaluated by the original D route.
All these arguments use strict FP64/gradual underflow, not FP32/fast math.

### Pointwise admission budget

All five candidate features must individually satisfy

```text
B[term] <= 2^-45 * max(1, abs(candidate[term]))
```

This is 128 machine epsilons on a unit-or-relative atomic-unit scale, stricter
than the retained feature-oracle test envelope. It is a new internal admission
criterion, not a relaxed SCF/XC/energy/force/endpoint tolerance. It deliberately
permits a cancelled feature only when its absolute discrepancy is certified
small. A small rho does not authorize an unbounded gradient or tau.
No molecular, functional, AO-count or device-model switch chooses the budget.

## Storage and fallback

Projection and absolute-dot panels occupy the existing dead matrix-product
scratch. They require `2*r <= active`; otherwise the prepared/batch executor
retains its original matrix kernel. This is a capacity inequality, not a
molecule-size selector. No new allocation, full-grid buffer, scientific launch,
atomic reduction, host synchronization or CTA barrier is introduced.
The unoccupied/FP32 specializations compile without certificate arithmetic.

The existing feature kernel reduces all five candidates and four norm bounds,
then warp-votes refusal. Refused points recompute products from the still-live
original stored D and AO jets through the shared symmetric-density graph and
feature policy. The fallback uses scalar accumulation below 32 active AOs and
warp accumulation otherwise, matching the canonical consumer. It checks each
matrix product for finiteness, preserves half-before-add and does not apply
RKS occupation weight again. No compact panel is overwritten with a different
AO/rank stride. A missing certificate fails closed; if the original matrix
source is also unavailable, an error is published rather than an uncertified
factor result.

## Qualification and rejected shortcuts

The selector is returned to explicit `GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` opt-in.
Unset, empty and other settings select matrix evaluation. Default promotion
requires renewed independent science/resource/sanitizer and full E+F evidence;
the former frozen image is not relabelled with this implementation's identity.

Independent tests cover all five feature gates, finite cancellation, scalar
and warp dispatch, mapped/partial tiles, RKS/UKS/empty spins, representability
edges, missing certificates and genuine matrix errors. The host directed-round
counterparts deliberately widen an RN result and are conservative enclosures,
not bit-exact CUDA emulation. Real CUDA compares against the retained emitted
matrix consumer as well as independent stored-D scalar/array oracles.
An additional 800-digit Decimal oracle checks each route and their difference
against the derived bound for RKS/UKS with coefficient exponents from -250
through +250 and reciprocal AO scaling. This is independent test arithmetic,
not a production precision or benchmark-condition change.

The repaired frozen image has native identity
`ef2d34fcc1d1e87faedbd040c2de6f5af338d4815650b00fd65f94051574bd33`.
Finite Slurm job 10120 passed all 104 emitted-CUDA cases; memcheck, synccheck
and racecheck each passed their 84-case cohorts with zero errors/hazards.
Job 10133 passed all 16 native-owner/compact cases normally and under all
three sanitizers, also with zero errors/hazards. Directed FP64 arithmetic
and `--fmad=false` remain in force. These checks do not by themselves qualify
complete-endpoint performance or authorize default promotion.

Offline resource inspection retains zero declared LOCAL bytes. Standalone
projection registers increase 34 to 38; standalone feature registers increase
64 to 72. The occupied compact product increases 42 to 46 registers and
5376 to 9728 shared bytes, with a 56-byte stack. The unoccupied matrix and
inactive-FP32 product/feature resource envelopes remain unchanged. This is
bounded resource growth, not a claim that the extra certificate is free.

Rejected: magnitude-only patches, checking rho but not gradients/tau, using
rounded C*C.T as the oracle in place of stored D, rewriting packed projection
scratch with AO-stride products, loosening numerical gates, or changing tile/page
scheduling to conceal extra certificate/fallback arithmetic. Endpoint evidence
must count or explicitly mark unavailable that additional work, not infer a
win from projection admission or kernel launch counts alone.

## Renewal outcome

All 20 native populations (240 E+F calls) completed within their original
1800-second limits. Every full force array passes the independent reference;
maximum energy/force errors are `2.547e-11` Eh / `5.735e-11` Eh/Bohr.
The same-image primary 96-atom control/candidate cold times are
242.955053/279.241177 s, with matched 24/24 iteration/Fock counts; moved is
143.150417/162.011812 s with matched 13/13 counts. Warm/moved-warm medians
are essentially flat. The 48-atom cold/moved regressions are also about 16%.
The reverse 96-atom control changes convergence (candidate cold 25 versus
baseline 24, baseline moved 14 versus candidate 13); its elapsed time is
retained without iteration normalization or a pure-kernel interpretation.

The counted cold profile has the same 24-versus-25 convergence distinction.
Raw XC device sums increase 95.051359 to 135.075474 s. Candidate occupied
feature kernels consume 25.541778 s standalone plus 18.377863 s compact;
these contain certificate and original-D refusal work, whose per-point
refusal count was not instrumented. They must not be described as projection
work savings. Order-6/7/8 force work counters remain identical; warm device
times are approximately 2.611/0.491/0.454 s in both arms.

Therefore the mathematical repair is retained for explicit opt-in use, not
promoted by default. The next smallest performance step is to avoid paying
projection/certificate work followed by expensive per-point original-D
recomputation, using a proved bounded early refusal and the existing tiled
matrix consumer. It needs a separate generated implementation, independent
review and complete-endpoint requalification, not another page-width trial.
The [retained packet](../../../../benchmarks/results/occupied-xc-certificate-20261018/README.md)
contains lossless full-force reconstruction, all convergence counts and
mutually exclusive wall ledgers separately from kernel device sums.

The packet was already published and verified byte-for-byte at commit
`6c35f9ade60500f883d4ecf519b2ad9c554d6c0a`. A late three-way results-tree
audit exposed a 173,670-byte overflow against the newly advanced master even
though this branch individually passed. Retain that exact existing Git object
with a checksum-pinned, offline-only recovery reader rather than copying the
188,680 bytes into the current tree, raising the cap, deleting unrelated
evidence or creating an unauthorized Release/external backup. All 256 force
arrays still replay from the original bytes; storage recovery changes no
scientific source, measured endpoint or default decision.
