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

## Local lossless packaging experiment

The completed flat publication initially exceeded the aggregate evidence budget.
Local commit `9204370a8f7fc87e527c8ca22fa40963479245d5` tested a lossless JSON/XZ
layout with explicit mode, size, SHA-256 and storage bindings for all 29 original
files (211606 bytes). Every original byte, including the publication, sources,
failed attempt and retry fixtures, was recoverable from that candidate. The
candidate required no old-PR recovery pointer, external archive, policy change
or modification of another task's historical evidence.

The candidate used a real JSON/XZ record container, not a tar archive disguised
under a different suffix. Its outer publication inventoried actual stored files
using the existing compressed-evidence reader. Its safe entry validated the storage and
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

The final 153942-byte local candidate passed the actual integrated retention
gate at upstream `839a260002504954f7c0457b67683b53eadab084`. Subsequent upstream
growth caused the normal integration at `bfc18215ccea3ae536c26ee933689c470981b79d`
to fail again: 67134496 > 67108864 bytes. Both outcomes remain recorded; no
packing result is described as a permanent capacity reservation or an
irreducible size bound. Earlier public-head CI run `39837866197` failed its one
retention test at 67320282 > 67108864; that failure is not reclassified as PASS.

## Forward restoration after delivered headroom

On 2026-10-24, upstream `be28a867c143d27bf39138ec189b0697ccfb2364` made sufficient
headroom available under the unchanged 64 MiB policy. A subsequent forward
commit restores all 29 public `7c55c844d` file blobs and modes exactly. Directly
reviewable original files avoid maintaining the additional unpack entry when it
is no longer needed. The local packing commits, including `9204370a8` and
`bfc18215c`, remain in branch ancestry; no reset, rebase or history rewrite is
used. The original measured wrapper, post-measurement retry fix, failed attempt,
publication and all numerical inputs/results are unchanged. CI and exact-head
review must still qualify the new public head before protected merge.
