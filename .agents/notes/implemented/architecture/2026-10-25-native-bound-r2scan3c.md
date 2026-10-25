# Decision: compile the defining r²SCAN-3c basis and correction owner

Status: implemented
Date: 2026-10-25

## Problem

The complete named KS owner initially omitted r²SCAN-3c because its scientific
identity includes an orbital basis and gCP as well as the electronic/D4 graph.
Publishing only the electronic program under that identity would silently change
the model. Requiring Python to build and combine the remaining pieces also left
the installed CLI unable to execute a common composite method.

This extends [complete named KS ownership](2026-10-25-complete-named-ks-owners.md)
without changing its explicit compatibility or backend boundaries.

## Decision

The build-time named catalog compiles the existing audited H–Ar def2-mTZVPP
snapshot into a method-owned native element template. Its generator validates
the existing immutable basis binding. The ordinary bundled basis pack/checksum
contract is unchanged. `generativeqc_system_create_named_ks_v1` uses the shared
native template expander and normalizer; it needs atoms, charge and multiplicity,
not runtime Python or a basis file.

Both named single and batch preparation compare the actual normalized basis
against that template at the same geometry. Matching a name/checksum is
insufficient. Cartesian/ECP/changed primitive inputs fail before execution.
Equivalent explicit canonical data remain accepted. All inputs are copied before
preparation returns.

The native KS owner selects the r²SCAN-3c EEQ profile, runs D4 and gCP only after
SCF convergence, and publishes the combined correction together with its two
component receipts. Failure revokes the transaction's receipts. Energy-only gCP
omits gradient storage and derivative arithmetic. A conservative per-system
reserve covers atomic numbers and two gradient outputs before D4 receives its
remaining byte budget; its diagnostic remains the D4 component's inventory.

Python's exact CPU route presents native results without energy arithmetic.
Offline basis metadata stays native-independent. Native selection refusals do
not retry a Python correction owner. Custom graphs, older libraries, global
ResourcePlan and unqualified electronic force consumers retain the existing
explicit compatibility path.

## Qualification boundaries and rejected alternatives

The new complete named r²SCAN-3c endpoint admits CPU energies. The current gCP
provider is CPU-only; promoting the same named endpoint on CUDA would introduce
an implicit host numerical component. Existing Python CUDA composition explicitly
reports `cuda+cpu` and remains available. A future native CUDA/mixed contract
must expose and qualify that ownership instead of silently changing the backend.

CPU native r²SCAN electronic forces are not qualified by adding a correction.
Python's established DF force route therefore retains its compatibility owner;
the CLI must refuse force/optimization requests for this context. No local force
allowlist is widened. Other named D3/D4 methods retain their already-qualified
native electronic force domains.

Extending the ordinary basis catalog was rejected because its checksum describes
a different source pack and is part of Python/native compatibility. A second
normalizer or a Python-only identity check was also rejected: independent C/C++
callers need the same admission boundary.

## Evidence

- `test_native_r2scan3c.py`: compiled H–Ar shell/column/exponent/coefficient parity
  against the pinned input; native construction/copy lifetime and altered-basis
  refusal; no external Python correction owner on direct/DF energy execution;
  budget refusal, replay failure isolation and revoked receipt recovery.
- Independent PySCF/Libxc + dftd4 4.3 + audited independent gCP checks: H2 RKS and
  H2+ UKS, canonical spherical basis, unpruned 8 × 4 × 8 product grid, 2e-8
  Hartree total gate. These establish implementation parity, not grid convergence.
- `test_cli_dft_df.py`: the same independent energies without Python in PATH,
  implicit/explicit canonical basis, identity metadata, wrong basis and
  Cartesian/force/budget refusal.
- `test_gcp_r2scan3c.cpp`: energy-only equals analytic energy on MB16-43/06,
  independent upstream energy gate 5e-8 Hartree and two-step gradient FD gate 2e-8.
- Existing r²SCAN-3c CPU composition, DF force and corrected method gates remain.
  The existing CUDA H2 canonical energy/force composition gate also passed on
  n1 through a separate finite Slurm allocation; it retains its explicit
  CUDA D4 plus CPU gCP compatibility contract.

## Revisit when

A native r²SCAN electronic force owner or explicitly identified CUDA gCP/mixed
owner is qualified, or another basis-bound composite requires a generated list
of defining templates. Preserve exact basis admission and same-transaction
component publication when extending this program.

## Subsequent force and CUDA extension

[Complete native meta-GGA forces](2026-10-25-native-meta-forces.md) supersedes
this note's CPU-energy-only and CUDA gCP exclusions. It retains the defining
basis and transaction contract while qualifying native analytic forces and a
device gCP provider in explicit bounded contexts.
