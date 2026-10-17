# Decision: preserve stored-density semantics at occupied-XC exponent edges

Status: implemented; composed default has separate endpoint qualification
Date: 2026-10-17

## Problem

PR #2577 projects AO jets through an accepted occupied coefficient frame before
the existing compiler-owned rho/gradient/tau bilinears. Mathematical equality
with a density matrix does not remove the FP64 storage boundary: a coefficient
can survive projection through a large AO derivative after its squared density
element has underflowed to zero. An independent compiled reproduction with
coefficient `1e-308` and AO derivative `5e307` gives stored density zero and
matrix tau zero, but the original occupied feature body returns tau
`0.24999999999999994` without an error. This is not a valid default merely
because normal molecular endpoint tests pass.

## Decision

The generated projection marks nonzero coefficients outside `2^-256..2^256`
with a NaN work-panel sentinel. This is a conservative IEEE exponent admission,
not an AO/molecule/functional/GPU threshold or a numerical tolerance. Admitted
coefficient products remain normal and bounded: for INT_MAX-sized validated
views, squared-frame occupation sums cannot approach FP64 overflow. Nonfinite
projection panels also decline, including otherwise-safe factors whose AO
projection overflows.

Do not publish a numerical error solely for the temporary sentinel. The existing
feature kernel votes across the point's warp and, on refusal, evaluates that
point from the still-live original density and AO jets. It uses the existing
`symmetric_density_element` graph and `add_features` bilinears, without a second
scientific formula. Matrix density already includes occupations, so its fallback
must not receive the occupied RKS factor of two again. Invalid results from the
original matrix computation still reach the existing finite-error flag.

Fallback contractions stay in thread-local registers. Writing dense matrix work
back over compact occupied panels would race other points because their strides
differ. No allocation, extra launch, host synchronization or CTA barrier is
added; the additional normal-route synchronization is one warp vote per point.
The work remains bounded by the original active AO tile. Keep the original
mapped AO span, unequal/empty spin behavior, partial tiles and original potential
construction. The native owner still authenticates density/frame pointer,
generation, shape, occupations, precision and scratch lifetime before lending
the lease.

The candidate changes unset/empty selection to occupied-frame preference; `0`
and other nonempty values except `1` retain matrix preference. This default is
not admitted by this numerical note: it requires separate source-matched full
E+F performance qualification on the current composed Direct force stack.

The composed primary same-image 96-atom populations give cold
248.357339→227.781627 seconds with unchanged 24/24 iteration/Fock work, and
moved 146.753327→135.397015 seconds with unchanged 13/13 work. The reverse-order
repeat gives cold 257.012644→227.259084 seconds but 25/25→24/24 work; do not
attribute its whole gain to the contraction. Reverse moved is
146.107713→135.191427 seconds, both 13/13. Warm medians change
30.885827→30.592061 seconds in the primary and 29.975432→30.003422 seconds
in reverse. This is a cold/moved optimization, not a reproducible warm win or
achievement of the 10% warm target. Ordinary retained warm seeds do not acquire
an accepted-proposal frame merely because the preference is enabled.

Retain the small original-library control: three-atom cold increases
1.649248→1.753281 seconds (+6.308%, about 0.104 seconds), predominantly in
preparation; its same-image default/rollback difference is +0.454%. This cost
must not be hidden by comparing only two selectors in the larger binary or by
adding a molecule/AO-count exception. Original-library controls for the main
sizes, full independent endpoint receipts, diagnostic clocks and a fresh
same-card peer remain separate admission evidence; the historical #2577 packet
does not qualify this composition.

## Independent evidence

- 1099 focused host/compiler/owner tests pass, including default/rollback,
  stale/aliased frame handling, original FP32 scaled contractions, feature-export
  precision phases and publication lifetimes. Test cohorts overlap.
- Slurm 9986, assigned RTX 5090, runs the actual generated projection and feature
  kernels: 40 independent matrix comparisons pass, covering normal, completely
  underflowed, subnormal-density and large-but-finite factors; RKS/UKS, unequal
  and empty spins, maps and partial tiles. Memcheck/synccheck/racecheck each run
  the 20 underflow/subnormal cases with zero errors; racecheck reports zero
  hazards/warnings. GPU device and compiler-cache receipts are retained.
- Preserve negative trials: the first controller failed before any GPU test
  because the remote output parent was absent. Slurm 9981/9983 initially gave
  36/40 passes: four UKS subnormal comparisons used a different reference
  storage/symmetrization contract. The corrected independent oracle consumes
  the actual stored density, symmetrizing as `0.5*D + 0.5*D.T`, like the retained
  matrix consumer. Halving each subnormal before addition is not equivalent to
  halving their sum afterwards. No tolerance was relaxed and all failed logs
  are retained, rather than counting these attempts as passes.

The immutable candidate native identity is
`1d930f399b86d6aa4b2594584471284f262c8bb7436e88e69b2d113aa41c6dd6`.
It composes the actual PR #2577 head `b1851abc4` with PR #2596 delivery
`a0ca32ea7`, preserving intervening FP32 scale/admission and feature-export
changes. Baseline and subsequent complete-endpoint evidence remain distinct
from #2577's historical, uncomposed measurements.

## Native resource and safety qualification

The actual composed native library, not only the independent generated probe,
passes eight owner/fallback/scientific cases in finite Slurm job 9996. Memcheck,
synccheck and racecheck each pass the same eight cases with zero errors;
racecheck reports zero hazards and warnings. The immutable 1889 native-input
hashes are checked again after test, documentation and evidence-only changes.

Offline `cuobjdump` on the exact frozen original and candidate libraries gives:

| Consumer | Original registers/stack/shared | Candidate registers/stack/shared |
| --- | --- | --- |
| Compact matrix FP64 product | 34 / 0 / 5376 | 36 / 0 / 5376 |
| Compact matrix FP32 product | 48 / 0 / 3200 | 48 / 0 / 3200 |
| Compact matrix feature | 64 / 40 / 0 | 64 / 40 / 0 |
| Compact occupied FP64 product | absent | 42 / 56 / 5376 |
| Compact occupied feature | absent | 64 / 40 / 0 |
| Standalone occupied FP64 product | absent | 34 / 56 / 5376 |
| Standalone occupied feature | absent | 64 / 40 / 0 |

Stack and shared values are bytes; all listed entries report `LOCAL:0`. These
are static compiled resource facts, not hardware occupancy or traffic. Zero
LOCAL does not erase stack storage or imply zero local-memory cost. The
inactive FP32 product retains its original resource envelope. The admitted
occupied path still requires the existing batch/standalone resource evidence;
missing evidence fails closed.

The broad host evidence and default cohorts overlap: 1099 host/compiler/owner
cases, a separate 185-test default/resource/reproducer cohort, 325
documentation/structure/inventory cases (six subtests), and 37 retained-evidence
tests pass. They are not added together as a unique-test count. Compiler
structure checks report 567 modules and zero dependency errors. A broad local
production-codegen run retains 46 passes and one independent fresh-CMake OpenSSL
discovery failure; that environment prerequisite is not reported as passing.

The merged storage-only #2593 patch is replayed as a prerequisite because the
older composed branch otherwise exceeds the unchanged 64-MiB evidence budget
by 7721 bytes. It preserves the historical records and their scientific tests;
no archive, force coverage or acceptance gate is removed. Test-only #2596
repairs are also retained. Neither changes any frozen native production input.

## Revisit when

Changing the density storage, coefficient/occupation representation, matrix
symmetrization, AO mapping, work layout or precision requires renewed independent
edge-case and full-consumer qualification. Do not remove the matrix fallback
based only on typical coefficient magnitudes or ordinary endpoint agreement.

## References

- PR #2577: accepted occupied-frame infrastructure and original opt-in evidence.
- PR #2596: independently qualified default-on low-order Direct force stack.
- Ignored local evidence: `.artifacts/occupied-xc-default-composed-20261017/`.
- `tests/python/test_xc_occupied_features.py` and
  `tests/python/test_xc_occupied_features_cuda.py`: executable independent oracles.
