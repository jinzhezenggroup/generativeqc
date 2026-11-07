# Decision: share GPU4PySCF-aligned energy defaults across public SCF callers

Status: implemented
Date: 2026-11-05

## Problem

This supersedes the default-selection whitelist in
[Standard cold stopping](2026-10-22-standard-cold-scf-convergence.md), while
retaining [native policy ownership](../architecture/2026-11-04-required-native-scf-defaults.md).

The public Calculator selected energy/density tolerances `1e-9/1e-5` only for
CUDA RHF and a model/grid whitelist. Other backends, spins and grids selected
`1e-10/1e-8`; raw native SCF, FockPlan and CLI defaults also differed. Thus a
backend or model change silently changed the requested iteration controls.

## Decision

`src/scf/default_controls.hpp` owns the shared public values: energy `1e-9`,
density `1e-5`, screening `1e-12`. The native metadata policy is version 2;
Python rejects older receipts rather than silently retaining the split policy.
Model/grid metadata validation, the C ABI layout, explicit controls and actual
method/provider admission are preserved. Valid HF/KS metadata always selects the
same defaults; an absent/default identity grants no scientific execution capability.
GFN2-xTB SCC has no GPU4PySCF counterpart and preserves its `1e-10/1e-8` controls.

The CPU UHF handoff additionally checks the current physical spin commutator
against the existing strict final-state ceiling. Finalization validates the
converged determinant before adopting a fresh physical-Fock projector; frame
and density generations distinguish those two witnesses correctly.
The new water-cation oracle exposed premature handoff with the looser density
default: bounded canonical correction exhausted its 32 corrections, while the
explicit strict SCF request succeeded. Continue ordinary DIIS within the original
SCF iteration budget instead of expanding final correction, silently retrying
with a different profile, or publishing a failed determinant. An AO commutator
ceiling alone did not fix this case while finalization still pre-projected D:
that additional projection amplified a slow/unstable spin fixed-point mode.
Retaining D for the unchanged canonical/fixed-point proof fixes the actual
handoff rather than simply requesting progressively tighter scalar tolerances.
Prechecking the full scaled DIIS eigen-frame residual was also unnecessary once
the premature projection was removed, so its duplicate dense products were not
retained. The handoff guard adds no new Fock build or eigensolve. Energy-only UHF
uses the same physical handoff condition as force/reference consumers.

GPU4PySCF inherits PySCF's `conv_tol=1e-9` and `conv_tol_grad=None`. Its ordinary
SCF loop resolves the latter to `sqrt(conv_tol)` and tests the physical MO-gradient
norm in addition to the energy change. Its density change is diagnostic. Native
density `1e-5` is therefore an additional condition, **not** that gradient norm
and not a claim of identical stopping trajectories. Retain native physical
commutator, canonicality, trace, idempotency, finite-value and lifetime gates.
Neither new defaults nor the metadata receipt expand their qualification.

Post-HF reference clamps, CC/lambda/response residuals, CBS component budgets and
Hessian-specific stricter defaults remain scientific endpoint contracts. In
particular, GPU4PySCF/PySCF CC amplitude-update norms cannot be substituted for
physical CC residuals by merely copying their numerical tolerance.

## Rejected alternatives

- Copying `sqrt(1e-9)` into density tolerance: these are different quantities.
- Letting uniform defaults authorize looser physical checks on UHF, ECP, custom
  models or unsupported grids: selection of a default grants no owner capability.
- Relaxing post-HF canonical reference gates: those protect a different consumer.
- Changing screening, max-cycle budgets, grids or DIIS to chase iteration parity:
  those are independent numerical/resource policies, not this default migration.

## Evidence

Executable native tests compare public metadata and actual ScfOptions defaults;
Python tests cover backend, spin, model/grid changes, overrides and receipt/cache
failure handling. CLI and FockPlan callers use the same native policy. Independent
PySCF endpoint tests cover restricted/unrestricted HF and PBE, cold/warm calls,
displaced geometries, energies and analytic forces. CUDA cases require a finite
Slurm allocation and explicit enabling; host-only metadata is not GPU evidence.

Local validation used a Release CPU build with sccache 0.16.0 after transferring
only this change onto upstream master at `4478c16c1`. The focused Python suite
passed 375 tests with 8 optional cases skipped; all 9 selected native
SCF/common-options/UHF/final-state/response tests passed. Four CPU oracle cases check energies
within `1e-7` Eh and every force component within `2e-6` Eh/Bohr, including actual
warm-start use and displaced geometries. Existing UHF torque and semantic-work
reconciliation gates remain intact; a valid retained determinant is allowed to
need zero corrections. Analytic spin-state tests still force the correction
path explicitly rather than requiring every physical fixture to take it.
Raw logs, JUnit output, build commands and before/after compiler-cache statistics
are retained locally under `.artifacts/gpu4pyscf-defaults/`.

No real-GPU validation was performed: the visible Slurm GPU node was drained.
CUDA oracle cases remain explicitly Slurm-gated and are not replaced by host
metadata tests or an unscheduled GPU run. This is not GPU endpoint qualification.

Upstream sources inspected on 2026-11-05:

- GPU4PySCF `gpu4pyscf/scf/hf.py`: SCF attributes and `_kernel` stopping policy.
- PySCF 2.14.0 `pyscf/scf/hf.py`: inherited SCF defaults.

## Revisit when

A separately validated common physical MO-gradient implementation is available
for every admitted native owner, or a new explicitly named precision profile
changes the shared public convergence contract.
