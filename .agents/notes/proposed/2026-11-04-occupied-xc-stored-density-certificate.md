# Proposal: current-source occupied-XC stored-density certificate

Status: proposed; feature/device qualification does not close #2751
Date: 2026-11-04

## Superseding conservative restriction

The [exact-identity decision](../implemented/numerics/2026-11-04-occupied-xc-exact-identity.md)
supersedes the candidate's numerical admission and unchanged-default proposal.
The roundoff rule below remains a prefilter, not an XC sensitivity contract:
explicit reuse now requires exact original-D feature identity, and automatic
KS requests are disabled pending a cheaper qualified contract. Earlier results
below retain their historical source scope; they are not new endpoint evidence.

## Problem and provenance

The existing owner witness authenticates D/C pointers, generation, strict-FP64
arithmetic, spin/occupations and a same-stream scratch lease. It does not prove
that rounded stored D and a reassociated C contraction have the same features.
The source-bound real-device counterexample retained in PR #2969 has finite,
orthonormal coefficients near 1e-200 and finite AO values near 1e200: stored
D00=0, dense rho=0, factor rho=2, accepted witness, no device error. Ordinary
molecular agreement cannot exclude that finite-input failure.

This candidate reuses the unmerged experiment at
`46e5f88d39645c0166264957548d1ec3b41d63c3`, not its historical acceptance. It
is adapted to parent `b735793be54caed18fb2e2cacfa9235ecb935f7c`: split compiler
templates, explicit FP64 FMA, current batch-map ownership and current resource
dispatch. In particular the fallback and independent dense GPU probe now use
the actual FMA-based tiled consumer, not the older separately rounded product.

## Candidate certificate

For active AO span n, maximum occupied rank r, and each consumed AO jet a_j,
the existing projection also computes

```text
p[j,o] = rounded dot(C[:,o], a[j,:])
s[j,o] >= sum_mu abs(C[mu,o] * a[j,mu])
T[j] >= sum_o s[j,o]^2
```

All positive products, sums and warp reductions in s and T round upward.
Nonzero coefficients must be in [2^-256, 2^256]; AO magnitudes are at most
2^256; nonzero s values must also be in [2^-256, 2^256]. An unsupported value,
NaN bound, overflow, missing certificate or insufficient scratch refuses the
candidate. These exponents define the proof domain, not empirical molecular
coefficient thresholds; they are insufficient without the absolute-dot test.

With u=2^-53, upward rounding and exact, range-checked integer conversion:

```text
K = 8*(n+r) + 128
alpha = K*u < 1/8
gamma = alpha / (1-alpha)          # denominator rounded downward
U = K^2 * 2^-810
w = 2 for RKS, 1 for UKS
M_rho = T[0]
M_grad[k] = 2*sqrt_up(T[0])*sqrt_up(T[k+1])
M_tau = (T[1]+T[2]+T[3])/2
B[term] = 2*gamma*w*M[term] + U
```

The stored Gram, symmetric half-before-add matrix contraction and feature
bilinears contribute one forward-error envelope; projected dots and their
feature bilinears contribute another. Their difference is bounded by the sum,
not a relative-error assumption on a cancelled feature. Cauchy-Schwarz bounds
cross-jet products. Taking square roots separately avoids overflowing T0*Tk.
K conservatively covers Gram formation, AO traversal including tile padding,
feature arithmetic, occupation scaling and scalar/warp sums. Explicit FMA
uses no more rounding operations than the earlier separate multiply/add path.

The coefficient range makes individual Gram products normal. Binary64 input
significands have no product bit below 2^-616 here, so exact Gram cancellation
cannot create a nonzero subnormal Gram or its half. Remaining gradual-underflow
errors arise in projection/matrix dots and feature arithmetic. Each can be
amplified by at most 2^256 within the admitted jet/dot envelope. The K-squared
path bound with the conservative 2^-810 margin dominates their accumulated
absolute contribution. Outside this proof domain the original D is consumed.

Every candidate feature must satisfy the historical internal admission rule
`B <= 2^-45 * max(1, abs(candidate))`. No scientific endpoint tolerance is
relaxed. **This feature rule is not itself a proved XC E/F sensitivity bound**:
the applicable point-program/domain and complete consumer acceptance remain
mandatory before treating the candidate as a completed #2751 repair.

## Scratch, fallback and ownership

Projection and absolute bounds require `2*r <= active`; both fit existing dead
matrix-product scratch. Other tiles retain the dense schedule. Refused points
recompute the original live D in registers without replacing compact scratch
by an incompatible AO stride. Mapped batches carry the same AO map through
projection and fallback; this required updating the current feature launcher.
The fallback retains half-before-add D, explicit ordered FMA and the original
small-span scalar/large-span warp feature schedule. It does not weight stored
RKS D twice. A missing original D does not authorize an uncertified result.

The ordinary matrix/FP32 specializations do not execute certificate arithmetic.
No extra host copy, synchronization, scientific kernel launch or global
allocation is introduced. Additional arithmetic, registers, warp votes and
possible original-D recomputation are real costs, not free reuse. The existing
automatic selector is not changed by this draft; merge/default acceptance must
include the actual extra work and its complete endpoint cost.

## Validation and outstanding gates

Independent tests exercise five separate feature gates, finite cancellation,
ordinary admitted data, RKS/UKS/empty spins, 800-digit Decimal forward-error
checks, original-matrix failures, mapped scratch offsets and compact batch
specializations. The new exact orthonormal underflow probe compares all five
features with the actual retained dense tiled consumer, including partial and
mapped inputs. Host directed-round helpers deliberately widen RN results;
they are conservative enclosures, not exact CUDA emulation.

The [source-bound capsule](../../../benchmarks/results/occupied-xc-current-certificate-20261104/README.md)
retains 156 passing host checks, 108 passing device checks normally and under
each of three zero-error sanitizer cohorts, plus the repaired exact finite
probe (dense rho=occupied rho=0, ordinary control rho=0.125). These are new
candidate observations, not old experiment receipts. Native full-plan ownership,
point E/Vxc/force, molecular/default-domain E+F, resource/codegen envelopes and
complete admission-inclusive cold/warm endpoint cost are separate remaining
gates. Do not close #2751, auto-merge this draft or relabel the old 256 force
arrays as current-source evidence merely because the feature/device tests pass.

## Rejected alternatives

Reject coefficient-only thresholds, finite-panel-only admission, a rho-only
certificate, unrounded C*C.T as the fallback oracle, double occupation scaling,
relaxed scientific gates and full dense reevaluation disguised as a free fast
path. Disabling the default alone would not repair the explicit factor route.
