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

The completed six-molecule energy cohort passes all 72 constituent/composite
comparisons across Standard and Fast at the unchanged 5e-8 Eh gate. The largest
absolute error is 2.37639596889494e-9 Eh. The accepted energy publication's replay
command uses `--require-pass`; missing or failing molecules cannot satisfy that
command. This numerical acceptance applies to the frozen native records, not a
new master build, force qualification, CI or review completion.

A post-measurement wrapper fix refuses pre-existing per-molecule artifacts under
the lease before opening stdout or outcome redirects. The Python receipt guard
alone was too late to preserve companion files on retry. The original measured
wrapper remains compressed and is the version reconstructed by the source patch;
the revised wrapper is reproduction tooling, not a retroactive measurement-source
identity. Bounded fixture checks retain both the original failure and fixed
byte-preservation results without rerunning scientific calculations.

Refs #2679. Revisit the numerical qualification after any change to the exact
scientific model or when independently qualifying a new native source snapshot.

## Lossless storage packaging

The completed flat publication exceeded the aggregate evidence budget. Its 29
Git-original files (211606 bytes) now have explicit mode, size, SHA-256 and storage
bindings in `canonical.json`; all their bytes, including the original publication,
sources, failed attempt and retry fixtures, are recoverable from files in the
same delivered version. The old PR commit is informational rather than required
for recovery. No historical evidence owned by another task or budget policy is
changed, and no Release, remote backup or unmerged-history pointer is used.

Use a real JSON/XZ record container, not a tar archive disguised under a different
suffix. The outer publication inventories the actual stored files and uses the
existing compressed-evidence reader. Its safe entry validates the storage and
canonical inventories before writing only regular files into a new directory;
it rejects traversal, collisions, unsupported modes, stale bytes and overwrites.
The explicit strict replay separately validates the restored original publication
and executes its unchanged replay against the original native source history.
The shared validator does not itself interpret the file container; the entry's
canonical and numerical checks are additional required reproduction steps.

New storage negative tests and exact deterministic repacking were executed
locally. The unchanged runner/checker bytes retain their real CPU-image 1/9
negative and 9/9 corrected execution records; no fresh POSIX or cloud execution is
claimed for this packaging step. No mathematical formula, scientific receipt,
gate, native timing or current-master device qualification changes.
