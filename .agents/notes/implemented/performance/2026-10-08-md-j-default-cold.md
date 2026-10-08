# Decision: default eligible FP64 Coulomb values to bounded MD-J

Status: implemented
Date: 2026-10-08

## Problem

The requested target is converged 96-atom / 768-AO PBE0 cold execution against
master with equal actual grid counts. Earlier independent fused MD J/K displaced
optimized normal K/XC consumers and was slower. A faster Coulomb kernel or
initialization stage alone cannot establish an endpoint advantage.

## Decision

Promote the native density-contracted MD-J route without changing normal K,
XC, SCF or finalization ownership. Strict-FP64 exact zero-order J admits s/p/d
bases with at least eight public AOs, within a charged optional 128-MiB cap.
Retain normal J for unsupported shapes, insufficient capacity and optional
allocation failure. Mixed J, fixed-mask response and derivative requests remain
on normal dispatch. Derivative-capable owners can prepare MD metadata but only
their value requests use it.

No enable environment variable is required. Creation-time
`GENERATIVEQC_DISABLE_MD_J=1` is the paired diagnostic opt-out, not a production
prerequisite. Optional MD storage is admitted only after the retained normal
owners are charged. Preparation and replay introduce no CPU integral oracle,
CPU linear algebra, AO four-index cache or production PySCF dependency.

Hermite setup consumes the existing normal `HermiteCoefficients`/`fill_hermite`
recurrence. Fixed bra/ket angular kernels reuse the charged primitive-order
buffer for GPU-compacted uniformly eligible primitive segments after geometry
transforms are complete. Partial AO screening remains a bounded, source-ordered
residual. The shared residual scatter helper can express J/K, but the production
MD caller requests J only; it never replaces normal K.

The SCF dependency ledger registers the borrowed MD launch ABI, numerical index
leaf and device consumers separately. Host providers may borrow the ABI, not
the device recurrence; device consumers cannot acquire host plans or drivers.
CUDA ownership counts the new scientific code explicitly rather than hiding it
as runtime code or claiming a compiler retirement.

## Error-budget invariant

For shell pair `p`, let `B[p]` be its maximum public-AO Schwarz bound and `L[p]`
an upper bound on its absolute density sum. Component count times maximum
absolute density bounds that sum; off-diagonal pairs include both input
orientations, and UKS bounds absolute alpha plus absolute beta without relying
on cancellation. A pair `q` contributes at most `B[p] * B[q] * L[q]` to any
output component of `p`.

Divide `min(screening_tolerance, 1e-12)` by the complete shell-pair census, not
active pairs, primitives or components. Uniform and residual quartet domains
are disjoint, so an output gets at most one allowance for each input pair.
Discard an unordered residual task only when both output orientations satisfy
their allowance. Round products upward and allowances downward. Refresh density
envelopes before both consumers; K never reads them. Screening zero keeps all
nonzero-density unscreened contributions. This bounds the additional J error;
it does not certify force/response endpoints or change their screening policy.

## Rejected alternatives

- Old fused MD J/K displaced the optimized normal pipeline and failed the cold
  comparison. Default promotion is J-only, not that earlier independent path.
- One bra row per CTA reduced candidate counts but starved high-angular classes.
- Flattened residual prefixes reduced probes from 209,520,685 to 19,830,192 per
  Fock, yet binary-search indexing raised the residual phase from roughly
  0.679 to 1.139 seconds. Retain dense unordered angular pages with GPU culling,
  bounded by 4096 workers and 128 candidates per page. Fewer probes alone are
  not a performance acceptance result.

## Qualification

The prototype on master `f316421f5c2cae0cbe736a81cb10c8649ead1564` achieved
75.562068 versus 64.218554 seconds median complete cold, ratio 1.176639.
This default-promotion branch rebases on
`ed8d21684910f65b656f5bf42c8ef0248aa7d40f` and reruns the same-library cohort
with the opt-out variable absent for every MD sample. The retained compact
receipt under `benchmarks/results/md-j-default-cold/` is authoritative for the
promoted source, not the older prototype number.

The promoted-source cohort (Slurm 6548) passes all gates: normal/default medians
74.273034/63.069519 seconds, ratio 1.177638, 15.084229% complete-time reduction.
The six times are 73.958711, 62.942511, 63.100250, 74.428092, 74.273034 and
63.069519 seconds in execution order. Normal uses 17 Focks and default MD uses
16; preparation medians are 3.328249/3.442795 seconds. Maximum independent
energy error is 6.730261e-11 hartree and maximum physical residual 1.148153e-11.
Each default sample has 16 MD calls, 32 density-bound refreshes and 4,718,592
grid-point visits. The residual candidate upper bound is 209,520,685 per Fock;
it is not a count of accepted integrals. Binary SHA-256 is
`119fc921535b5e6d05d11c3f69616e047f348ef14a5b27d312d5e880637011e8`.
The same versioned binary passes all 14 independent raw-J/K/fallback cases,
plus the spherical UKS partial-screening case under both memcheck and racecheck
with zero errors/hazards (Slurm 6549) and all nine direct CUDA response gates
(Slurm 6550). CPU cold/structure/ownership gates have 190 passes. The receipt
keeps exported input, oracle and complete samples, with only a documented
trailing-LF normalization of input/oracle bytes.

Every endpoint uses fresh processes/calculators/owners, no CUDA priming or
supplied density, spherical def2-SVP, 96 atoms / 768 AOs, actual native grid
24x8x16 / 294,912 points, FP64, screening `1e-12`, energy tolerance `1e-11`
and density tolerance `1e-9`. Timing includes construction, preparation, full
energy-only SCF and owner teardown; imports and process shutdown are excluded.
Normal and MD retain master's automatic initial-density policy. Acceptance
requires at least three alternating pairs, unique processes, the same binary
and Slurm device assignment, energy error at most `3e-9` hartree and physical
residual at most `1e-9`, plus a complete median normal/default ratio above one.

Native MD calls must equal complete Fock counts; normal calls must be zero.
Grid visits and density-envelope refresh counts are reported separately from
geometry-probe upper bounds. Different converged Fock counts are retained,
not disguised as a same-iteration kernel speedup. Independent public-AO raw-J
tests cover Cartesian/spherical, RKS/UKS, nonsymmetric densities, zero/tight/
partial screening, replay, default admission, normal-K retention and exact-budget
fallback. Force/response performance is not part of this energy-only promotion.

## Revisit when

Renew independent gates for any envelope, precision, screening, derivative or
response change. Revisit residual indexing only with complete paired endpoints.
Retire the native contraction only after a generated consumer passes the same
numerical/resource/endpoint gates without changing normal K ownership.

## References

- [Current Fock source behavior](../../../../docs/developer/fock_build.md)
- [Cold driver](../../../../benchmarks/md_j_normal_cold.py)
- [Raw-J gates](../../../../tests/python/test_md_j_normal_cuda.py)
- [Retained cold evidence](../../../../benchmarks/results/md-j-default-cold/README.md)


## Resource-admission correction

The initial promoted source `9c911d28004eb0baae07ef63d78ee9c8ffcda63f` added the
128-MiB optional cap to the public mandatory shape estimate. The public planner
has one incumbent candidate, so this made previously feasible normal-J budgets
infeasible before runtime could fall back. A one-water def2-SVP host probe
reproduced a 128-MiB increase per owner, including a three-owner 384-MiB increase.

Public ledger-bound owners now retain normal J. Their inventory and identity do
not depend on the diagnostic opt-out, and unused live bytes cannot be spent on MD
at the expense of later owners, force retention or rebuilds. Private unbudgeted
KS preparation explicitly requests the original optional cap through an internal
C++ reservation parameter; removing the cap everywhere would silently shrink
that provider's budget and risk losing the qualified default. Standalone Fock
owners still use their explicit budget. No public C ABI changes.

The MD allocation group now uses the incumbent rollback helper. It fences before
freeing, checks releases, clears allocation last-error state and propagates
unrelated CUDA/numerical errors. Host fault injection exercises the actual MD
preparation block at every allocation, with ledger binding/unbinding and opt-out
changes; public shape/planner tests retain multiowner and force reservations over
rebuilds. Original GPU receipt hashes remain pinned to the measured source.
Broader ledger admission requires a real optional candidate with simultaneous
capacity reservations, not a larger mandatory estimate or live-free-byte test.
