# Decision: tile stationary cotangent programs under the existing element cap

Status: implemented
Date: 2026-11-04

## Problem

The spherical NiH4(2-) B3LYP probe on 08738a0b8 converged shared response in
17 iterations / 34 actions with true residual 8.110805132056192e-12. After an
explicit independent CPU prewarm, its one-electron second-integral phase
completed in 1338.877 s. The next Coulomb first-integral phase failed with
`stationary-HVP integral block exceeds element budget`: one Cartesian ffff
quartet has 10000 ordered components. Preparing density pullbacks and caching
programs by term count did not make that full symbolic block admissible.

## Decision

`StationaryHVPPlan` owns a lazy disjoint ordered-term schedule and uses one
capacity calculation for both its schedule and block admission. The original
65536-element default, spin-resolved accounting and source algebra remain
unchanged. The runtime gathers indices and density feeds only for a slice,
executes the ordinary compiler-generated fixed or response weight program, and
scatters into its ephemeral shell-local output. All-AO one-electron/Pulay pair
weights use the same schedule. Public spherical density pullbacks, arbitrary
signed values, same-spin exchange and both response product terms are retained.
Only mathematical programs are retained by tile extent, never shell tensors.

## Evidence and limits

The failed warm probe is retained in
`/tmp/gqc-2843-2849/ni-hvp-bounded-warm.log` (1502.59 s total). A preceding
1800 s cold/invalidation run timed out in the one-electron phase. The explicit
CPU prewarm completed 74 missing artifacts out of a 218-artifact inventory in
358.40 s with three workers, unchanged O3/identities and finite compiler limits;
it is separate qualification work, not hidden endpoint work. The first raw
matrix probe was intentionally interrupted before any column completed once
the inevitable weight-block failure was identified. No complete Ni HVP or raw
Hessian pass follows from these receipts.

Executable tests retain independent signed density-product cotangents and JVPs
for restricted/unrestricted full-range and range-separated methods, including
large pair arrays and 10000-component ffff shells. Compiler schedule tests check
disjoint coverage, the original admission boundary and eager invalid-limit
errors. The full raw-matrix diagnostic also uses actual common public-method
eligibility rather than advertising low-level global-hybrid work as public.

The weight/plan suite passes 72 cases in 44.66 s, and compiled native/input,
orbit, orbital-gap and response guards pass 98 cases in 324.70 s (28 CUDA cases
intentionally unexecuted without Slurm). The compiler structure check covers
586 modules with zero dependency errors; all applicable staged hooks pass.
An initial molecular replay passes 51 cases but fails one with a missing
temporary `compiler.log`: the local warm-cache fixture pointed inside an old
pytest retention directory while another pytest session collected that
directory. This is a qualification-harness lifetime failure, not a numerical
pass or an integral-cap failure. Remaining owned cache bytes were moved outside
pytest retention without overriding source/artifact identity; the failed log
and subsequent rerun are retained separately. No production fallback was added.
The rerun outside pytest retention passes all 52 complete global-hybrid and
WB97M-V direct/full-rank-DF RKS/UKS molecular cases in 781.71 s (eight JUnit
warnings). This is a cache-enabled replay with missing artifacts rebuilt, not
a wholly warm-cache timing or an installed-public Ni qualification.

## Subsequent complete low-level evidence

On eb051d0d1 the complete spherical Ni HVP passes all three independent
moving-grid gradient-difference steps with normalized maximum errors
2.6253e-8, 7.2917e-9 and 1.0926e-8, below the frozen 1e-5 Eh/Bohr^2 gate.
All eight stationary sources occur exactly once, with one response solve,
17 iterations / 34 actions / true residual 8.1108e-12 and no displaced production
SCF. Assembly takes 1875.44 s; the full 1895.66 s probe and 1021276 KiB peak
include independent reference setup and are not isolated native performance.
This is explicit 5760-point-grid, source-checkout, low-level qualification,
not default-grid or installed/public admission. Receipts are
`/tmp/gqc-2843-2849/ni-hvp-weight-tiles-acceptance.json` and the associated log.

The raw native replay passes its first column against all three independent
matrices (errors 2.1983e-6, 5.5001e-7 and 1.9838e-7), then is intentionally
interrupted at 2522.25 s to avoid repeating diagnosed metadata work. The partial
column and logs are retained with the eb051d0d1 suffix; they do not qualify a
full raw matrix. The follow-up immutable graph metadata rationale is recorded
in `2026-11-04-immutable-tensor-graph-metadata.md`.

## Rejected alternatives

Raising the global element cap would hide a consumer scheduling error and weaken
other bounded programs. Hand-writing density products in the runtime would
duplicate MethodIR/TensorIR coefficients and response algebra. Neither is needed.

## Consequences

The retained full output is still subject to its existing source/shell budget;
this change bounds symbolic programs and feed gathering, not aggregate endpoint
work. In particular it does not resolve expensive scalar graph reconstruction,
grant default-grid admission, or qualify a complete installed/public Ni endpoint.
