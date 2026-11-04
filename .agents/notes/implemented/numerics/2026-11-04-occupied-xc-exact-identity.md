# Decision: require exact original-D features before occupied XC reuse

Status: implemented and scoped native endpoints qualified in the repair branch;
CI/review and merged delivery pending
Date: 2026-11-04

## Problem

The [absolute-dot proposal](../../proposed/2026-11-04-occupied-xc-stored-density-certificate.md)
repairs the finite underflow counterexample but admits bounded nonzero feature
differences. Its `2^-45 * max(1, abs(feature))` rule is not a proved error budget
for any applicable XC energy/potential/force consumer. A point program can
amplify small errors or take different branches. Neither molecular agreement
nor a passing forward-error bound discharges that missing contract.

## Decision

Keep the outward-rounded rule only as a prefilter. Every explicit occupied
feature invocation independently evaluates the original live stored D using
half-before-add symmetrization, ordered FP64 FMA and the retained built-in
scalar/warp reduction. Preserve that vector in registers, not in compact
scratch shared with other point owners. Select the candidate only when all
five finite values are exactly equal, including signed zero; otherwise publish
the entire original-D vector. Missing original inputs fail closed, and genuine
original-D nonfinite arithmetic remains an error even if the candidate is finite.

The published point-program input is therefore identical to the original-D
built-in consumer. Deterministic downstream point algebra sees no perturbation:
no estimated Lipschitz constant, coefficient cutoff, density threshold or
functional-specific sensitivity assumption is needed. This does not promise
bitwise agreement between different optional matrix providers, grids or SCF
schedules, nor does it repair independent errors in the original consumer.

Exact verification repeats dense work, in addition to projection and bounds.
Stop requesting that experimental route by default in KS. Only the explicit
value `GENERATIVEQC_CUDA_KS_OCCUPIED_XC=1` requests it; ordinary default, `0` and
invalid controls retain the unchanged matrix schedule. The explicit route is
correctness-restricted, not newly promoted or claimed faster.

## Invariants and cost

- Every published feature is original-D identical; passing a nonzero error
  budget alone is never authorization to reuse a candidate.
- RKS weighting is applied once, UKS/empty spins retain their scientific domain,
  and compact AO maps/offsets and generation/lease checks remain authoritative.
- Both panels fit dead scratch only when `2 * rank <= active`. No host oracle,
  global allocation, synchronization or new kernel launch is introduced.
- The exact reference adds real matrix arithmetic and register pressure. Any
  later performance claim must include it and any refusal/fallback work.

## Validation

The portable suite adds an admitted, one-ULP stored-D discrepancy whose original
rho lies on the opposite side of a branch boundary, and strengthens ordinary
small-feature comparison to bitwise equality with an independent FP64/FMA
oracle. Device tests require bitwise equality against the independently launched
retained dense CUDA consumer, including finite cancellation/underflow and
nonfinite original arithmetic. Additional cases form D with the actual emitted
GPU SCF Gram kernel from orthonormal frames and exercise RKS, UKS, empty spins,
mapped/full AO domains and both sides of scalar/warp dispatch.

Actual run identities/results belong in the source-bound benchmark capsule,
not inferred from test collection or the historical candidate. Full native
owner/batch integration, independent molecular/default-grid E+F, whole-runtime
resources and complete admission-inclusive endpoint costs are still required
before merge/issue closure; this note alone does not assert those gates passed.
The source-matched follow-up below records the subsequently executed gates.

The [new restriction capsule](../../../../benchmarks/results/occupied-xc-exact-identity-20261104/README.md)
records 162 focused host cases, 168 additional consumer cases and Slurm job
15811: 138 real-device cases normally and under each of three zero-error
sanitizer cohorts, including actual GPU Gram and compact batch kernels. The
two new executable bitwise regressions fail against the preceding template.

The follow-up at `def31af3f6c9b8476e8ff9d892f7f74589d168c7` builds and executes
the actual native library after merging the required native SCF-default runtime.
Slurm 15825 passes four full native lifetime/default/explicit E+F tests and 160
alternating complete endpoints; Slurm 15827 passes the same four tests under
zero-error memcheck. Independent default-grid PySCF energy/force vectors pass
the unchanged 2e-9 Eh / 1e-7 Eh/Bohr gates. The capsule retains actual-library
changed/fallback launch resources and all timed vectors/work counts. It does
not infer whole-force-region profitability, achieved occupancy or allocator peak.

The explicit verified route costs about 22% more for RKS cold/moved and 51–53%
more for UKS cold/moved on this three-atom fixture; warm UKS costs about 16%
more. These are complete, admission-inclusive endpoint comparisons, not kernel
timings or universal performance bounds. They independently reinforce the
decision to retain the matrix default. A future agent should not restore the
unverified historical default merely because exact verification is slower.

## Rejected alternatives and revisit conditions

Disabling only the default leaves explicit reuse unprotected. Accepting only
small feature errors without a consumer contract leaves scientific boundaries
unproved. Retaining unconditional default exact verification would add needless
work to production. Removing original-D/cancellation checks or relabelling old
force arrays as new results is not acceptable.

Revisit the exact reference cost and default only with a proved point/domain
E/Vxc/force contract, independent boundary regressions, complete native resource
and endpoint evidence, and an unchanged bounded original-D fallback. The earlier
roundoff proposal remains useful input to that work, not its acceptance.

Refs #2751 and PR #2971. No claim about the cause of #2915 is made.
