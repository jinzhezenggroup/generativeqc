# Decision: Retain full moving-grid partials once per right direction

Status: implemented
Date: 2026-10-16

## Problem

The native fixed-grid induced Fock and moving Gaussian/Becke geometry sources
do not themselves supply the frozen nuclear grid Fock or stationary mixed
energy. Differentiating only density-dependent coefficients omits AO assembly,
measure and pair-coordinate motion. Repeating VV10 for every left coordinate
would be memory-bounded but add an avoidable atom-count factor to pair work.

## Decision

Extend `CudaWb97mvGridResponse` with an opt-in nuclear arena, not a second
functional implementation. Build-time scalar Graph and TensorIR AD reuse the
incumbent feature primal, its density VJP, mixed feature map, Cartesian XC
coefficient policy and complete VV10 pair energy. The shared compiler owns
the beta normalization and affine schedule work-count emission.

The field ABI is `[rho,gx,gy,gz,tau]` for each physical spin, followed by weight
and three coordinates. The reference holds all field partials; a right action
retains their complete JVP and copies AO/D/weight/point directions. Its moving
potential map differentiates both AO assembly and full feature cotangents.
Left actions have zero left/mixed density and contract
`dot(left_fields,right_partials) + dot(mixed_fields,base_partials)`.
Their mixed feature AD still includes the right density response. Mixed point
motion vanishes because owner-atom coordinates are affine in nuclear motion.

Actual point motion is gathered by the geometry CUDA kernel and lent only after
a successful action. No CPU reconstruction of point motion or recovery of
atomic measures from partitioned weights enters production. Geometry motion
borrows expire on every prepare/mixed attempt. The nuclear right cache expires
on every prepare/apply/right attempt, not merely successful replacements.

## Invariants

- Fixed-grid consumers retain their default storage and pair-work contract.
- All numerical runtime work is CUDA; Python only emits AOT source at build time.
- Screening is fixed from reference total rho on both VV10 legs. Pair energy
  has the ordered-pair half; first-leg partials do not, since the symmetric
  second leg supplies the other half. Weight and position partials lift once,
  whereas total-density feature partials lift to both physical UKS spins.
- Preserve the incumbent floors: functional-Hessian/energy-density seeds use
  floored directions; Cartesian gradient-coefficient pullbacks retain raw
  gradients. The physical tau feature retains its one-half.
- Signed/indefinite D motion is supported without PSD factorization.
- Left contractions never traverse pairs. Explicit workspace, pair-work and
  affine-work bounds are separate; affine counts include materialized AD primal
  dependencies, not just output size. They are term visits, not measured FLOPs.
- Retained sources, caller arena and outputs are disjoint. Device, stream,
  capture, finite-value, symmetry and live-owner gates precede publication.
  A CUDA failure during final publication invalidates that output.

## Rejected alternatives

Do not hand-differentiate XC/VV10 again in C++, call Python at runtime, replace
analytic response with finite differences, or cache borrowed caller directions
across subsequent geometry actions. Do not project signed directions through
Libxc's optional FHC work extension: it changes the declared derivative domain
at rank-one densities. Independent signed molecular tests instead customize
separate pinned Libxc handles, as in the fixed-grid qualification. Full-rank
interior tests keep the unmodified upstream oracle.

## Evidence

`tests/python/test_wb97mv_nuclear_grid.py` includes independent bilinear feature
and potential contractions, pinned Libxc plus separately evaluated real-space
VV10 energy/potential differences, native-vs-AD lowering, zero/frozen/signed
directions, empty pair screening, tile variants and caller-buffer overwrite
tests. The native probe checks alias, finite, stream, capture, cache invalidation
and late-revocation publication. Geometry tests check actual owner-gathered
point motion and its failure lifecycle.

Pre-final Slurm node2/main/PRO6000 job 3113, assigned visibility 0, passes 67
synthetic/compiler/native and geometry cases. Job 3116 on the same allocation
type passes all eight actual native RKS/UKS Direct/DF geometry and moving-grid
cases. Complete qualification and sanitizer evidence is retained under ignored
`.artifacts/wb97mv-hessian-aot-20261015/nuclear-*`; these are component tests,
not complete Hessian endpoint timings.

## Consequences

The opt-in arena retains full field partials and right directions, trading
linear storage for one VV10 traversal per right direction rather than per
left/right coordinate pair. Existing dense affine schedules remain explicitly
work-bounded rather than advertised as optimized molecular Hessian performance.
Complete integral shell coverage, nuclear H/S/J/K, raw DF metric/factorization
and ECP response, stationary orchestration and production CLI migration remain
unqualified. CLI Hessian/HVP still use the Python bridge. This change neither
registers a native molecular Hessian capability nor supports a GPU4PySCF speed
claim.

## Revisit when

The complete stationary native endpoint consumes these partials, or a separately
qualified streamed/fused feature schedule reduces the admitted affine work.
