# Decision: optional angular-bucketed canonical public-AO CUDA J/K

Status: implemented; opt-in, not the production default
Date: 2026-10-01

This historical opt-in stage is superseded by
[automatic screened through-f dispatch](2026-10-01-default-screened-through-f.md).
Its retained measurements and binary identities are not measurements of the
superseding default.

## Problem

PR #1637 admits f-shell geometry composition, but its generated shell J/K
owner still requires complete SPD coverage. One f shell sends the whole
molecule's resident value contractions to ordered public-AO loops. Full J,
full K and the long-range correction then repeat expensive ERI work. Reducing
scratch alone does not fix this source-work amplification.

## Decision

`GENERATIVEQC_CUDA_CANONICAL_JK=1` prepares unordered public-AO pairs, bucketed
by total shell angular momentum. Each pair-of-pairs bucket has a fixed
recurrence order. A symmetry-unique full ERI supplies both raw J and raw K;
SR/LR exchange uses the same inventory with the original radial operator.
The existing scientific compiler owns contracted ERIs and exact permutation
scatter, including nonsymmetric density orientation and restricted/unrestricted
spin. No second recurrence, spherical transform or contraction algebra is added.

The benchmark runner enables this candidate explicitly and records the setting.
The general provider default remains unchanged until larger complete endpoints
and candidate UKS force qualification are available. Set the flag to `0` to
retain the old schedule. An insufficient optional budget or mixed-J consumer
also retains the existing bounded generic path. Allocation, driver and numerical
failures remain failures, not successful fallback results.

Optional storage consists of two int32 entries per unordered AO pair and six
batch spatial-matrix equivalents of FP64 scratch, all charged before allocation.
No molecular four-index tensor is retained. Prepared state is geometry-bound
and explicitly rebuilt on geometry changes. Derivative owners and their shape,
host/device, precision and final-state-export limits are unchanged.

## Work accounting

For N AOs, P=N(N+1)/2 pairs partition exactly P(P+1)/2 candidates per radial
pass. Full J and K share that pass. These are pre-screen candidates, not a claim
that every candidate evaluates an ERI. The optional
`GENERATIVEQC_CUDA_CANONICAL_JK_CENSUS=1` counts candidates actually visited and
radial ERIs actually evaluated on the provider stream, resetting each enqueue.
It is disabled for timed endpoints. The dedicated `--canonical-work-only`
native fixture checks both counts at 58 and 116 AOs with zero screening;
its single-primitive basis is a scheduling diagnostic, not OMol25 timing.

## Evidence

- Experimental worktree Slurm 11943 passes the independent CPU full/SR/LR
  value matrix gates through f, with both AO representations and spins,
  nonsymmetric densities, two geometries, output masks, invalid-input recovery
  and minimum-budget fallback. Both named def2-TZVP and full local def2-TZVPD
  RKS complete energy/force and displaced-energy gates pass. That job's OH UKS
  reference did not converge; the overall job did not pass, and this is not
  candidate UKS force qualification.
- Slurm 11950 completes all twelve schema-v3 endpoints for the full 58-AO
  def2-TZVPD water case against the already-qualified independent matched-domain
  reference. Original/moved cold and five frozen-density warm calls all pass
  1e-8 Eh / 1e-7 Eh/Bohr gates. Maximum errors are 7.92e-12 Eh and
  5.75e-11 Eh/Bohr. Both clocks include host force return.
- On that PR-head preview, the original warm median changes from 46.2553 s
  to 14.3832 s; cold from 879.220 s to 64.289 s; geometry refresh from
  400.485 s to 45.682 s. Five original and moved samples are retained, not
  selected or divided by iteration counts. GPU4PySCF's original warm median
  is 16.4248 s. This is one complete small-molecule point, not a scaling or
  universal parity claim. Preview library SHA256:
  `8d38dbe2a85f38a730fc4d468cbb6440307ae4e812a78248980daf8d1d27db6c`.
- The worktree subsequently incorporates master
  `8154ab3df56a10900dd267039b857021a1054721` while retaining #1637. The preview
  timing is not relabeled as a measurement of that later build. New source,
  binary and scheduler identities must accompany later measurements.

The integrated master build additionally passes Slurm 11955's independent
value gates, now checking candidate/evaluation counts for every output mask
and radial operator. The explicit 58/116-AO work fixture visits and evaluates
1,464,616 / 23,028,291 ERIs for each full-JK and LR-K pass, with 340,046 /
1,353,364 bytes of explicit device storage. These are actual instrumented
value-source counts, not full-endpoint times. The current library hash is
`096147bb0e1d95dd90005c899dc163fc0fc9789e97b419ad3830edaf93d4c5ab`.
The focused host suite passes 110 tests with 16 opt-in skips. See retained
`schedule-qualification.json` for exact source/patch/log hashes and scope.

An initial census fixture incorrectly inherited HF's derivative-order default
while preparing a value-only plan. The provider correctly rejected it; the
fixture now requests order zero explicitly. Failed jobs 11951/11952 are not
reported as passing jobs; 11951's separate value-matrix gate did pass.

## Rejected alternatives

Do not widen the generated SPD class mask without exact consumers, drop the
diffuse/f basis, borrow reference densities, relax gates, remove derivative
resource caps, or treat a timeout as warm latency. Do not promote the candidate
globally just because one small endpoint is faster.

## Consequences and revisit conditions

Canonical scatter uses FP64 atomics, so reduction order can change without
changing the mathematical contraction. Retain independent tolerance gates and
finite-value checks. Public-AO expansion and high-angular derivatives still
dominate some endpoints: the preview's last force assembly attributes about
9 seconds to native integral derivatives, without a kernel-level time split.
Investigate those sources next; do not move oracle derivative work into setup.

Promote an automatic policy only after larger complete cold/warm/moved endpoints,
UKS forces, screening/resource-bound behavior and complete endpoint speedups
qualify. Existing generic and generated-shell paths remain explicit fallbacks.
