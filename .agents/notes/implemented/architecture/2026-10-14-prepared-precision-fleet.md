# Decision: native prepared precision fleets with candidate-owned seeds

Status: implemented
Date: 2026-10-14

## Problem

The application-budget facade originally exposed only scalar endpoints. Feeding
README cold/frozen/moved benchmarks through repeated scalar calls would remove
native plan/seed reuse, change the work and unfairly label a different protocol
as Fast. Conversely, retaining a device candidate while building each FP64
auditor would silently exceed a single-owner resource allowance.

## Decision

Share the existing precision-policy controller between scalar and real native
fleet execution. The prepared wrapper does not implement chemistry or repeat
singlepoints. Its initial candidate supports the facade's full observable set;
energy-only AUTO admission does not promote an unavailable force consumer.
Qualified mixed and untargeted FP64 calls retain the native fleet unchanged.

Only when an item requires an independent observable audit, detach the fleet's
own warm proposals to bounded host storage, close the candidate device owner,
then prepare independent cold one-item FP64 auditors sequentially. Resume the
candidate lazily on the next execute. An auditor never seeds a candidate, even
when verified reference observables replace an over-budget candidate. Frozen
post-cold/post-move proposals retain their source geometry and Python provenance.
Seed metadata may legitimately be absent after a Python force failure while the
native SCF still owns a valid proposal; do not invent checkpoint provenance or
drop that proposal merely because it lacks a successful Python force record.

The historical HF-named warm ABI supports KS through `BatchWarmStarts`. Native
restore checks density/occupations against source overlap before committing
states; target execution always reruns SCF. There is no PySCF production oracle.

The old getter's size query can materialize a resident density. Therefore its
memory gate must precede even that query: compute a conservative all-slot bound
from prepared shell representation and spin, enforce `maximum_seed_bytes`, and
reserve staging/validation scratch above the current host resource plan. Actual
descriptor sizes must fit that bound. Audit resource budgets reserve retained
host seed storage, including when only an explicit `ResourcePlan` was supplied.
Disabled warm starts require no optional seed ABI. Missing transport, insufficient
headroom and restore failures are explicit failures, not silent loss of history.

## Timing and failure invariants

- Snapshot coordinate updates once, retaining native malformed-input semantics.
- Failed candidates skip policy audits. Failed audits publish no usable energy
  or forces, and cannot leak a successful candidate's gradient evidence.
- Process successful peers before raising in strict mode. Do not roll back their
  candidate-owned warm updates.
- Candidate timing belongs to the fleet. Schema-2 item reports use `None` for
  shared candidate/endpoint seconds, never equal division or scalar replay times.
- Batch reports retain actual candidate precision/Fock work, complete elapsed
  time, resume/staging cost and all audit attempts, including setup failures.
  First-audit duration includes suspension, so breakdowns are not additive.
- Unknown native work counts stay unknown. In particular, ordinary HF fleets
  may lack the KS per-item Fock total; iteration counts are not a replacement.
- The README CLI's ordinary FP64 baseline remains `Calculator`. Fast records
  use a distinct schema and explicit budget gates, retaining the unchanged FP64
  parity gate separately. Existing FP64 reducers reject the Fast schema.

## Rejected alternatives

Repeated scalar calls would invalidate the README warm protocol. Auditing once
and extrapolating to moved geometries has no supported error bound. Importing an
auditor's density would couple the engines' warm history and bias later work.
Keeping all auditors resident would multiply capacity; dropping seeds on every
audit would hide a cold restart behind a nominal warm call. Native handles must
not be fabricated or queried after suspension merely to fill diagnostic fields.

## Evidence

- Host tests exercise the real controller with fake native owner/seed ABI ports:
  shared timings, qualified work, optional ABI/memory failures, geometry snapshots,
  rejected candidates, cold independent audits, frozen proposal restore and close.
- Slurm 8376: all 32 prepared/regression tests passed on the assigned RTX 5090.
  This includes independent cold, two frozen replays, moved and two moved-frozen
  E/F calls for direct PBE/PBE0/B3LYP/r2SCAN/M06-2X, their public DF variants,
  RHF and canonical r2SCAN-3c at the existing small qualified domain. Separate
  tests exercise real KS seed export/restore after an energy-only audit and
  tight-budget per-item audits with a malformed neighbor.
- The first run, 8375, is retained: its HF test incorrectly required a non-null
  KS-specific work count, and its 3c oracle dependency was absent from PYTHONPATH.
  The corrected test preserves unknown work and the independent numerical gates;
  the existing isolated dftd4 oracle installation is now included.
- Initial fleet qualification did not rebuild native code. Its library SHA256 is
  `5b7fc0ef88c97ee56246c91c55428b3cb8df972ca5d68fe576a1ce5792bd9599`.
  Source hashes, scripts, logs and XML are in ignored
  `.artifacts/readme-budget-fast-20261013/` and the same-named n1 run directory.

The actual runner subsequently exposed excess mixed warm SCF work. The native
correction and its separate source-matched remeasurement are recorded in
[the resident warm-epoch decision](../performance/2026-10-14-fast-resident-warm-epoch.md).

This adds a truthful measurement/API path, not all-FP32 arithmetic, all README
domain qualification or a universal acceleration claim. WB97M-V, HF DF,
corrected/RSH DF and broader method/size qualification remain separate work.

## Final source-matched regression

Slurm 8382 passed all 75 prepared/application/public-DF/native-batch checks on
the rebuilt library, then 14 selected ordinary AUTO regressions. Separately,
Slurm 8384 passed **all 24** ordinary AUTO tests in a fresh process; the selected
8382 run is not substituted for that full receipt. Final host qualification
passed 263 tests with 11 explicit skips. Matching library SHA256 is
`f993f2ba55c378b4c8102a682226ad8ed7193903d1dc674632445b3fec886bf6`.

Retain the earlier 8379 combined-run failure: after the larger prepared tests,
three PySCF J-only oracle calls failed inside `temporary_env(vhfopt=None)`.
The reference sets `direct_scf=False`; PySCF selects incore versus out-of-core
from `_is_mem_enough`, and its out-of-core J-only path attempted to change a
prescreener on `None`. The same three tests passed on the unchanged old binary
in a fresh 8382 process, and the complete fresh current-binary run passed 8384.
This is evidence of a memory-dependent reference path, not proof of a native
memory leak or numerical regression. No production math, oracle tolerances or
unrelated legacy fixtures were changed to suppress it.

The source-matched dirty patch, new files and complete overlay are retained as
`prepared-8380-{working.patch,new-files.tar.gz,overlay.tar.gz}` in the ignored
local/remote artifact directories. `final-prepared-8380.so` preserves the
matching native library on n1. The losing 8378 binary and Python snapshot remain
separate; do not restore the older public-DF checkpoint over this prepared work.
