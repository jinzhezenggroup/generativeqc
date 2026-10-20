# Decision: accelerate HF density admission and retain independent audit seeds

Status: implemented
Date: 2026-10-20

## Problem

The 96-atom def2-SVP direct-HF README diagnostic spent 176.997 seconds restoring
and validating a suspended Fast candidate density, then 43.579 seconds on a
cold independent precision audit. The complete warm endpoint took 226.93 seconds.
HF fleet restoration supplied no eigen callback to the common physical validator,
which selected the legacy maximum-pivot host solver. KS already supplies a CUDA
callback. Precision fleets also discarded every auditor's converged density.

## Decision

CUDA HF admission uses a temporary ordinary-stream eigensolver, with the shared
small-native/provider dispatch, explicit matrix storage, bounded queried provider
workspace, checked returned frames, and exception-safe draining. It needs no Fock
source and cannot overwrite a resident seed. All supplied states are validated
before the fleet's existing transactional commit. CPU admission is unchanged.
The common validator retains its finite/symmetry/charge/occupation gates and its
legacy fallback for accepted nearly asymmetric proposals; inputs are not repaired.
Python checkpoint and precision restoration bind the prepared device ledger.

A precision fleet may retain one independent auditor density per item, only for
the same geometry and model/solver signature. Every audit still executes SCF and
all requested observables, and every comparison/error gate still runs. Tighter
fallback verification remains independently cold. Auditor seeds never enter the
candidate history, including when audited observables replace a rejected result.
Freeze/clear/close apply to both histories; failed audits invalidate their history.

The existing seed byte cap bounds the combined staged candidate/auditor arrays.
Candidate staging has priority. Optional audit history is dropped on insufficient
space; an audit still runs cold. Host reservations include detached histories and
validation scratch while only one candidate/auditor device owner is live.

## Rejected alternatives

Skipping physical density validation, disabling unsupported-domain audits, copying
the auditor density into the candidate, and caching complete audit observables
would weaken or change the measured contract. None is used. Retaining two device
fleets would violate the sequential residency contract; only host seeds survive.

## Evidence

Finite Slurm allocations on n1, RTX 5090, CUDA 12.9.1, sccache 0.16.0. The same
frozen README driver, inputs and default controls measured cold plus three warm
repeats on master eac2de2b plus the auxiliary-g and warm patches. The profiled core
SHA-256 was `f2bf0f0bd00bdc069fed476ddc5c013424dd2b50dae7aebefdfc04f6fe893c00`;
subsequent trace/accounting edits have a separate final build identity.

- Candidate restore: approximately 1.2 seconds, versus 176.997 seconds.
- Audit: 8.72, 10.51, 8.74 seconds, versus 43.579 seconds. Audit iterations fell
  from 21 to 2–3, and Fock builds from 22 to 3–4. The cold audit still takes 21
  iterations/22 builds; this change only avoids repeating its initialization.
- Complete warm endpoints: 16.254, 18.029, 16.235 seconds; median 16.254 seconds.
- Every call retains two complete solves. The returned energy differs from the
  pre-fix observation by at most 4.55e-12 Eh; maximum force difference is below
  5e-13 Eh/Bohr. These are diagnostic comparisons, not new README rankings.
- 166 precision-policy/ownership/work-validity host tests passed.
- Four real CUDA RHF/UHF physical-admission cases passed, spanning small-native
  and provider dimensions. An independent PySCF overlap constructs valid and
  negative-occupation ensembles; rejected imports preserve all prior states.
  The near-asymmetric legacy acceptance is also exercised.
- A real tetramer audit replay passes independent PySCF energy/force gates and
  verifies frozen, separate candidate/auditor histories over three endpoints.
- All 58 checkpoint tests pass on CUDA, including fresh-process RHF/UHF direct/DF
  references, corrupt density rejection, atomic restore, limits and lifecycle.
  Checkpoint subprocess tests run in a separate process from the audit suite.

Full profiles, source/build/cache receipts and test logs remain under ignored
`.artifacts/` and `/data/jzzeng/df-g-auxiliary-20261020/receipts` on n1. The original
pre-fix diagnostic is retained in the README campaign's diagnosis directory.

## Revisit when

A qualified Fast provider covers large direct HF, or a native owner can safely
suspend GPU residency while retaining its bounded solver state. Either may remove
additional full-solve/setup work; this patch makes no universal Fast speed claim.
