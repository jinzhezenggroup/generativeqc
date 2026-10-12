# Decision: opt-in single-pass RHF shell nuclear cross derivative

Status: implemented, default promotion pending real-device qualification
Date: 2026-10-12

The initial opt-in policy is superseded by
[the retained-shell default decision](2026-10-12-rhf-shell-bilinear-default.md).
The historical canonical fallback and pending-validation descriptions below
record the original implementation, not the current default policy.

## Problem

RHF frame response evaluates `P:G'(D)` by symmetric polarization of the
quadratic energy derivative: `[E2'(D+P)-E2'(D-P)]/2`. This is two traversals of
the same unscreened derivative source. The earlier canonical bilinear consumer
already reduced this to one traversal, but lost shell primitive/component reuse;
fewer traversals alone did not qualify a faster endpoint.

## Decision

The existing `bilinear_derivative` opt-in now tries the retained shell consumer
before the bounded canonical consumer. Differentiate the compiler-owned
quadratic density products rather than introducing another HF weight formula:

```text
q(A,B) = c A B / 2
directional_q(D,P) = c (D_left P_right + P_left D_right) / 2
```

The existing Coulomb/exchange coefficients, unique eightfold ERI orbit,
normalization and force scatter remain authoritative. The canonical bilinear
helper also delegates to this same product-rule implementation. Scaling precedes
multiplication, preserving the original quadratic arithmetic order and disabled
source guards.

Project `D` and `P` through the owner's two already-admitted density slots. A
Cartesian seed pointer lives only in a copied `DeviceBatch` launch view, not the
retained geometry/topology owner. Every low-order, higher-order, generic,
pair-materialized and cooperative shell coefficient consumer forwards it.
Shell classification, primitive caches and generated derivative recurrences
are unchanged. Combined output needs only one atom-gradient channel.

Only one restricted unscreened item is admitted. The density-bound preparation
can use the two-slot projection ABI, but its spin-density bounds must not screen
signed cross weights. Screened, batched, unrestricted, separate-source and
missing-seed requests are rejected before projection/submission. Both resident
public input matrices are device-checked and finite-audited before evaluation;
the complete output is finite-checked before publication.

Missing shell storage returns `NOT_IMPLEMENTED` and selects the existing bounded
canonical consumer. Other failures propagate without a numerical or submission
retry. The two-pass default and three-pass validation identity remain available.
The new consumer remains opt-in. Scoped GPU measurements are recorded below;
they do not establish a general default-promotion envelope.

## Invariants

- Both operands belong to the same fixed unscreened source and AO frame.
- Preserve FP64 weights/accumulation, distinct canonical permutations, Cartesian
  projection, normalization and translation reconstruction.
- Preserve stream-owned temporary lifetimes and failure-atomic publication.
- Do not mutate the retained source identity or allocate another density owner.
- Report shell versus generic passes honestly; the canonical quartet/jet census
  is not a measured ledger for a shell launch.

## Evidence and acceptance gates

Host probes compare all canonical repeated-index orbits against independent
dense signed/nonsymmetric contractions, compare scaled cross weights against
polarization, and retain overflow/disabled-source regressions. An extreme-scale
test checks a finite cross derivative when `D+P` and `D-P` would round identically.
Production host-wrapper fault injection checks the same shell route, angular
route, borrowed topology, one output channel, owner recovery, unchanged retained
batch and pending-download lifetime. Unsupported cross contracts must submit no
work and publish no output.

The real-device probe forces shell and canonical APIs separately, so fallback
cannot conceal a broken shell implementation. Its independent libcint and
finite-difference tests cover signed symmetric operands, Cartesian/spherical
bases, through-f sources and two/four nonplanar centers. These tests are Slurm
gated; they are not replaced by host coefficient tests.
Their derivative acceptance is `atol=2e-11, rtol=2e-10`, with translational
sum `atol=2e-12`. The existing complete RHF independent force gate remains
`atol=2e-8, rtol=2e-8`, the Z residual ceiling is `1e-10`, and the complete
frame stationarity ceiling is `1e-8`.

The final selected host/compiler/reference regression suite reports **667
passed, 69 skipped**. The skip set includes the Slurm-gated real-device cases;
it is not numerical qualification of CUDA forces. The compiler ownership audit
reports 521 modules and zero dependency errors; Ruff lint/format and changed-line
clang-format checks pass. The extended native response probe compiles through
the verified ccache launcher.
The focused CMake build of `direct_jk.cpp`, `direct_coulomb.cpp` and
`rhf_frame_response.cu` succeeds. The complete `generativeqc` CUDA target does
**not** finish within the explicit 20-minute limit: the large
`direct_bounded_fallback.cu` compilation is still pending when the build is
interrupted. This is not a successful complete native build or link. Retain the
build logs and before/after ccache statistics; complete that target before any
real-device run.

Raw local validation belongs in `.artifacts/rhf-shell-bilinear-20261012/`.
The initial inventory checked only the local Slurm controller, which exposes
drained maintenance node `node3`. That was not sufficient to infer fleet GPU
availability. Remote controller checks on 2026-10-12 identify usable allowed
nodes, including idle `node4` with an RTX 5090. Real-device qualification and
paired endpoints use that node's own `srun`, preserving assigned visibility;
maintenance node `node3` remains excluded.

The follow-up benchmark freezes the master observed at its start,
`531364e133ae145097b402d66fec5036ed051365`, plus the focused candidate patch.
A fresh source archive avoids changing the user's dirty branch or mixing old
build outputs into the measured library. Source reconstruction, verified cache
launcher receipts, independent reference inputs and Slurm reproduction scripts
are retained under `.artifacts/rhf-shell-bilinear-20261012/remote/`.

Default promotion requires independent complete analytic forces, Z residuals,
translation/numerical-repeatability gates and randomized repeated endpoint
comparisons of one/two/three passes on the same qualified source revision and
allowed GPU. Include preparation/projection, semantic source visits, derivative
evaluations and all allocations/transfers, not only kernel time. Recheck static
register/local-memory resources and existing quadratic force performance because
the shared coefficient gains an optional runtime seed branch.

## Real-device follow-up: frozen master on node4

The fresh frozen-master candidate builds and links successfully with verified
ccache 4.5.1 and CUDA 12.9.1, targeting sm_120. Its measured library SHA256 is
`b667312cbf533644505f71e932de788e220334c58f660a2b8f0dbba15aae169e`.
This library is not the stale output from the initial interrupted build.
Slurm job 775 runs all **96** RHF response/derivative CUDA tests successfully,
including independent libcint signed-density jets, finite differences,
Cartesian/spherical through-f cases and complete RHF/Pulay/response checks.
There are no skipped cases in that device suite.

Slurm job 776 runs two separate fresh-process ABBA cohorts, eight complete
native DF-CCSD(T) energy-plus-force endpoints in total. Both cohorts use the
same library, ethane230 input, strict FP64, automatic primal Q32, Lambda Q32,
DIIS history 8, cadence 30 and unchanged independent acceptance gates. The
assigned physical RTX 5090 is
`GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`, driver 580.178.04. The controller
reports `COMPLETED`, exit `0:0`, with runtime 33:02 inside the finite 75-minute
allocation. Neither visibility nor GPU clocks are overridden.

| Isolated ablation | Control median s | Candidate median s | Time reduction |
| --- | ---: | ---: | ---: |
| Two-pass to shell single-pass, complete native E+F | 250.485922 | 227.308465 | 9.253% |
| Two-pass to shell single-pass, complete process wall | 250.725 | 227.550 | 9.243% |
| Two-electron orbital nuclear-derivative phase | 46.161287 | 22.998957 | 50.177% |
| Lambda immutable frontier off/on, complete native E+F | 259.792111 | 250.477915 | 3.585% |
| Lambda immutable frontier off/on, complete process wall | 260.035 | 250.720 | 3.582% |
| Lambda immutable frontier off/on, Lambda phase | 88.415161 | 79.064964 | 10.575% |

The derivative cohort retains core reuse in both arms. The frontier cohort
retains two derivative passes in both arms. These percentages must not be
added or presented as a directly measured combined speedup. The single-pass
samples are 227.384955 and 227.231976 seconds; its controls are 250.382180 and
250.589664 seconds. Frontier-on samples are 250.430281 and 250.525549 seconds;
its controls are 259.797941 and 259.786282 seconds.

Measured shell derivative passes decrease 2 to 1, with zero generic passes.
Quartet/primitive jet counts remain unavailable for shell launches and must
not be inferred from the canonical census. Every endpoint retains 19 CCSD
iterations/evaluations, 21 Lambda iterations, 22 solver operator actions,
13 Z operator actions and 416 reported Lambda auxiliary batches. Core reuse
changes Lambda GEMMs from 15,033 to 14,307, with one preparation and 23 reuse
actions; it does not reduce the true solver action count. Its retained storage
is 396,517,056 bytes. One-second device sampling observes a maximum of 26,500
MiB with retention versus 26,120 MiB without; these are sampled physical peaks,
not allocator high-water measurements. Single-pass does not lower that observed
complete-endpoint peak.

All eight endpoint gate receipts pass. Energy differs from the retained
independent PySCF energy by at most `2.615e-12` Ha. Full forces differ from
the retained independently qualified native force by at most `8.164e-10`
Ha/bohr; the retained independent finite-difference audit covers two nonzero
components at two steps, not all large-system components, with maximum error
`3.094e-8`. Across all new runs, energies are identical, maximum componentwise
force span is `1.601e-12`, and maximum translation sum is `2.080e-13`.
Maximum Lambda and Z residuals are `6.117e-13` and `1.358e-13`; maximum
stationarity is `3.976e-13`. No discarded/retried endpoint is accepted as a
matched work receipt.

Raw source reconstruction, build/cache receipts, binary identities, allocation,
GPU samples, all eight JSON endpoints, numerical gate receipts and machine-readable
summary are retained in `.artifacts/rhf-shell-bilinear-20261012/remote/`.
The source snapshot is `/data/jzzeng/qc-rhf-shell-bilinear-master-20261012-531364e1/source`
on the build host and node4. The original dirty checkout and Lambda-history
prototype are preserved.

This is one ABBA cohort per optimization, two samples per arm, one molecule and
one physical GPU. The control is the opt-out schedule in the same patched
library, not a separately rebuilt untouched-master binary. It isolates pass
selection but does not quantify seed-branch overhead versus untouched master.
Broader workloads/hardware, unchanged quadratic performance, sanitizer and
static register/local-memory evidence remain separate default-promotion gates.
Q-consumer fusion and f-jet tiling were not implemented or measured here.

## Lambda-history follow-up

Preserve the existing uncommitted CPU experiment and proposed note without
changing native production policy. Rerunning its default H2/H2O supplied-MO
endpoints reproduces shared physical Lambda action counts `3/5/5` and
`29/29/29` for diagonal/Jacobian-fit/inverse-fit preconditioners. These results
do not support an unconditional native default change or predict the large DF
adjoint's speedup. Native history ownership, budgeted retention, metric-aware
transpose, rank admission and bounded fallback remain separate promotion gates.
The shifted/damped water rerun (`level_shift=0.4`, `damping=0.15`) reproduces
`35/33/35` actions and `17/16/17` iterations. Across these nine supplied-MO
reference endpoints, independent Lambda residuals remain below `8e-12` and
weighted multipliers differ from the diagonal solution by less than `3e-12`.
These are action/numerical checks, not repeated native GPU endpoint timings.
See `../../proposed/2026-10-12-cc-lambda-history-preconditioning.md`.

## Rejected alternatives

- Promote canonical bilinear solely because its traversal count is smaller:
  it discards precisely the shell reuse the change is intended to retain.
- Handwrite an RHF cross formula in each native kernel: permutation and
  normalization changes would gain multiple independent scientific owners.
- Construct quadratic `D+P` and `D-P` weights inside one consumer: it introduces
  avoidable cancellation and does not express the shared product-rule contract.
- Enable the new default before device/endpoint evidence: host algebra and a
  successful build cannot qualify GPU forces or a performance claim.
