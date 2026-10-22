# Decision: bind independent CBS energy oracles to frozen full-DF inputs

Status: implemented
Date: 2026-10-23

## Problem

The homepage C1 and C3–C7 CBS points previously compared Fast to the same
project's Standard policy. That comparison is not an independent oracle. The
existing native receipts predate current master and must retain their source
identity rather than inherit subsequent Lambda or SCF changes.

## Decision

Use a standalone PySCF-only energy driver with independently reconverged RI-JK
RHF, DF-MP2 and DZ DF-CCSD(T), sharing the exact JKFIT auxiliary per basis. Match
the original all-electron cc-pVDZ/cc-pVTZ model and original QM7 geometries before
comparing each total-energy constituent and the composite to Standard and Fast.
The energy gate stays 5e-8 Eh for both policies.

The driver reads geometry and basis identities but no native energies or seeds.
Raw convergence logs, runtime library hashes, individual errors and failed
attempts accompany the results. Replay reconstructs the historical native source
and uses committed LF bytes; Windows text transformations are not measurement
identities. A source patch preserves the exact standalone oracle snapshot.

## Rejected alternatives and invariants

- Standard is a control, never an independent reference.
- Correlation-only exact-HF/RI auxiliaries are a different model and cannot
  substitute for the full RI-JK/JKFIT reference and response contract.
- A handwritten alkane geometry cannot replace the pinned dataset indices.
- Compile, allocation, process completion and CI success do not establish an
  energy gate. Missing references remain INCOMPLETE.
- Reference wall times are separate from native performance comparisons.
- Energy qualification cannot close the remaining directional-force/translation
  coverage or certify a newly installed master build.

## Evidence and consequences

See `benchmarks/results/cbs-independent-energy-20261023/` and its replay driver,
publication manifest and per-molecule status records. The first C1 diagnostic
failure is retained; its corrected attempt uses a fresh independent MINAO guess.
The original native campaign is immutable. No production code, public tests,
shared benchmark index or homepage series is changed.

Refs #2679. Revisit the numerical qualification after any change to the exact
scientific model or when independently qualifying a new native source snapshot.
