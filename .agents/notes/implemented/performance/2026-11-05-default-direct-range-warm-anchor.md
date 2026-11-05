# Decision: enable qualified dense-Direct warm-anchor deferral by default

Status: implemented; promoted-image A100/RTX 5090 endpoint qualification complete
Date: 2026-11-05

Supersedes the default-selection policy in
[the original opt-in trial](2026-11-05-direct-range-warm-anchor.md).

## Problem

The matched A100 and RTX 5090 trial removes one unused incumbent Fock from
water96 full-Direct WB97M-V without transferring it to the first warm endpoint.
Cold times improve 9.222% and 7.349%; cold-plus-three-warm times improve 6.152%
and 5.700%. All warm observations remain one iteration/one Fock and pass the
predeclared 1.05 timing guard. Independent energy/force gates pass. Requiring
users to discover an environment switch leaves this qualified benefit unused.
The user explicitly requests default selection after reviewing this evidence.

## Decision

Unset `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY` and explicit `1` both admit dense
full-Direct seeds under the existing guards. Explicit `0` restores the eager
path; other explicit values also conservatively retain eager preparation.
Fitted defaults and partial-fitting exclusions do not change.

This is resource- and owner-based selection, not A100 model-name dispatch.
Require strict FP64 RKS, device-fused XC, an immutable audited replayable range
source, compatible policy/lifetime, and sufficient actual following-force
headroom. Source loss reconstructs the saved density's matching incumbent energy
as counted preparation; force eviction preserves the current stationary token.
No CPU/reference work enters production. Do not change the Hamiltonian,
convergence criteria, force tile, or Direct-versus-fitting contract.

## Evidence and limits

The original [compact publication](../../../../benchmarks/results/direct-range-warm-anchor-20261105/README.md)
retains the matched explicit-on/eager trial and independent reference. It is one
instrumented population per policy/device, not a statistical zero-regression
claim. RTX 5090 warm ratios are 1.0022–1.0084; the A100 native cold endpoint still
trails GPU4PySCF. Do not reinterpret those timings as measurements of the new
default binary.

Qualification of the promoted image must authenticate the changed source and
loaded ELF, exercise unset/explicit-on/explicit-off small-system controls on both
actual scheduled devices, and run one unset-default water96 cold/three-warm
lifecycle per device. Reuse the authenticated independent large reference from
the trial, with provenance stated explicitly. Do not repeat the successful
matched populations or relax numerical/work gates.

### Promoted-image qualification

The new default is qualified separately at production commit
`00ec18f032a1574ed495d0cdc9615657ec204687`, native source identity
`c69a04e5c7bc79220e4e52516c79ab86d76ed42b42b446de657e66a8264f7f32`, and
core ELF SHA256
`3cf435f6fb2ae36cc411d27a44a88498544ec896bc6892ecf793e9a898c04dcb`.
An exclusive incremental build actually recompiles `cuda_ks.cpp` and build
identity, relinks the core, and refreshes companion manifests. The verified
private sccache 0.16.0 launcher records two misses/two compilations/no failures;
existing SM80/SM120 Release CUDA objects remain unchanged. Do not describe this
as an identity-only relink or claim cache hits.

One unset-default water96 cold/three-warm E+F lifecycle per device completes:

| Device/allocation | Cold (s) | Warm 1 (s) | Warm 2 (s) | Warm 3 (s) |
| --- | ---: | ---: | ---: | ---: |
| A100, LSF 225544 | 118.503707 | 21.692416 | 21.598975 | 21.600666 |
| RTX 5090, Slurm 15959 | 203.074699 | 19.287336 | 19.304196 | 19.325214 |

These are new-default observations, not another paired eager/on comparison.
Both cold calls use 18 iterations/Focks, 18 J and 36 K applications; every warm
call uses one iteration/Fock. All scopes in the eight execution traces close,
without incumbent rebasing or deferred preparation Focks. Relative to earlier eager samples,
A100 warm ratios are 0.998309/1.000232/1.001367; RTX 5090 ratios are
1.011313/1.011175/1.010297. They pass the existing 1.05 per-call timing guard;
these single populations do not establish statistical zero regression.

On each device, two Direct capacity/lifetime tests cover frozen/updating warm
seeds with eager/on/unset controls and an independent same-grid GPU4PySCF
oracle. Two additional public default/strict/moved-geometry E+F controls pass.
Host qualification passes 106 checks; six GPU cases skip outside allocation.
The unchanged large independent reference from LSF 225523 is explicitly reused,
not recomputed: all 16 new native/reference observable pairs pass. Worst errors
are 7.1481e-12 Ha/atom, 3.5522e-7 Ha/Bohr force max and 1.5571e-7 force RMS;
strict paired parity against the retained optimized image also passes.

LSF 225544 is terminal DONE on `gpu_chem/gpu22`, eight CPU slots/one exclusive
GPU, finite 15 minutes. Slurm 15959 runs on `main/node1`, eight CPUs/one 5090,
finite 12 minutes, with driver and scheduler client returning zero. Assigned
visibility is preserved. Slurm accounting storage is disabled; do not invent
a terminal `sacct` record. The
[separate compact publication and CPU reader](../../../../benchmarks/results/direct-range-default-20261105/README.md)
retain all observations, trace summaries, identities and receipt hashes.

## Revisit when

A qualified owner transfers work into a later endpoint, violates independent
scientific gates, or shows a reproducible endpoint regression. Retain explicit
disable and bounded eager fallback rather than broadening unsupported owners.

## References

- Engineering memory: issue #2976; implementation PR #3006.
- The previous fitted default and headroom rationale:
  [guarded fitted anchor](2026-11-01-default-range-warm-anchor.md).
