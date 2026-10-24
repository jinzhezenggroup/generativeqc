# Decision: qualify frozen C4/C7 forces with fresh displaced PySCF energies

Status: implemented (frozen-native C4/C7 two-step numerical qualification PASS)
Date: 2026-10-24

## Problem and decision

The delivered six-molecule independent ENERGY qualification does not establish
the C4/C7 analytic forces. Extend the original frozen full-DF protocol using the
seed-709 normalized direction at both 1e-4 and 3e-5 Bohr, retaining each of the
eight displaced geometries and independently reconverging both bases from fresh
MINAO. Reuse the committed central ENERGY evidence without recomputing it.

The new oracle is PySCF-only. It does not import native code or accept native
energies, densities, orbitals or amplitudes as initialization. Shared JKFIT for
RI-JK and correlation, all-electron spherical cc-pVDZ/cc-pVTZ and all CBS weights
remain the original protocol. Force is minus the energy derivative; constituent
and composite directional errors must each meet 5e-6 Eh/Bohr at both steps for
Standard and Fast. Translation remains 1e-6 Eh/Bohr. The central energy gate is
unchanged at 5e-8 Eh. There is no native displaced-energy qualification here.

## Evidence boundaries and retained alternatives

The new replay reconstructs and verifies the historical source rather than
attaching current master identity to old native results. Original vectors stay
in the immutable native family. Neither a directional check nor translation
proves full-vector accuracy, new-source device acceptance or true CBS accuracy.
Reference runtimes are provenance, never native speedup denominators.

Each task has a finite timeout, an exclusive lease and a fresh output directory
claimed before stdout redirection. No automatic retry overwrites failures or
checkpoints. A timeout without a completed receipt is FAIL; absent references
remain INCOMPLETE. Raw text is preserved without newline or signed-zero
normalization in JSON/XZ records with size, SHA-256 and observed original modes.
Unknown POSIX modes remain UNAVAILABLE. Host negative tests exercise sign,
nonfinite data, altered/missing geometry, timeout retention, raw restoration,
corruption, path containment, collisions and no-clobber behavior.

The initial host replay verifies native translation and reports both molecules
INCOMPLETE because no displaced reconvergence has run. Tooling/host checks do
not imply a scientific PASS or PR delivery. The coordinating Chat owns the
cross-slice #2679 audit; this family uses Refs and keeps the parent OPEN.

## Revisit when

The scientific protocol changes, full-vector qualification is requested, a real
native force defect appears, or a new native source/device campaign is authorized.

Refs #2679; `benchmarks/results/cbs-independent-forces-20261024/`.

## Resource and source receipt

After the ENERGY owner's release, the coordinating Chat handed off only the
early CPU object to this slice. Fresh platform and runtime guards matched its
creation time 1792689025000, cpu-nat-576, container --0020ef93afaa-, HIGH/6 and
4 CPU/16 GiB cgroup. The visible 128-CPU cpuset is not the allocation. The new
root and environment are independent of the immutable ENERGY root. The opaque
Notebook ID and an unrelated global OFD lock's owner remain UNAVAILABLE; neither
is fabricated, cleared or treated as a resource reservation.

Nine remote runner and two outer-controller diagnostic checks passed without
scientific calls. The measured source was then frozen at
`6478a8fd82a5fa4605e29e1f9f5cf88485fa4ed2` and the suite launched at 06:22:55 CST
on 2026-10-24. Each of eight serial calls has a 21600-second cap plus 30-second
grace; the outer controller also has a finite 180000-second cap. Four numerical
threads and an 8000 MB PySCF bound stay fixed during execution. Process completion
still requires all raw/protocol/convergence/algebra checks and both force steps.

Related task records share one logical JSON/XZ collection to compress repeated
basis and log context without dropping, normalizing or splitting original raw
bytes. Restoration validates the full collection before creating its output
directory. Publication and scientific replay remain separate checks; an
inconclusive publication must fail strict scientific verification.

A reader-side negative test exposed a task-directory traversal before
publication: membership in an input-provided task set alone accepted
`../escape`. The RED test failed with `ValueError not raised`. The fix validates
the fixed C4/C7, h0/h1, minus/plus namespace independently before creating any
restore directory. Three restoration tests then passed. This changes only the
offline reader; measured oracle/runner/input/environment bytes stay frozen.
An actual four-task C4 roundtrip retained all 20 original raw files byte for
byte and replayed C4 PASS while correctly refusing overall PASS with C7 missing.

The reader also rejects a native receipt whose stored HF/correlation/delta
force terms contradict the weighted constituent forces. A corrupted HF-term
fixture first failed to raise; after adding executable algebra checks, all six
replay tests passed. The independent directional gates remain the original five
constituents plus composite, at both steps; no gate or oracle input changes.

## Completed scientific cohort

All eight displaced reconvergences and both controllers completed with exit 0.
Strict replay passes both steps for both policies and all constituents/composite:
C4 maximum directional error is 2.3297564630958556e-8 and C7 maximum is
1.4372056532721555e-8 Eh/Bohr, below the unchanged 5e-6 gate. Maximum translation
residual is 1.2624909798142347e-12 Eh/Bohr, below 1e-6. The central 5e-8 Eh ENERGY
proof remains the separate delivered publication, reused without recomputation.

The final postflight observed unchanged measured source, dataset and numerical
libraries, no remaining scientific worker, and an available own suite lease;
the unrelated global OFD lock remains UNAVAILABLE. Its 13 raw records and the
initial 114 records remain byte-bound. The scientific collection restores all
40 original receipt/log/outcome files with modes and raw identities intact.
Missing terminal source/environment proof prevents strict PASS even when the
directional numbers pass. Source/library drift, missing tasks, active workers,
mode errors and controller failure have executable negative coverage.

Budget audits pin a commit and results-tree before summing Git-LF blobs. An
early observation queried the shared moving `origin/master` but attached an old
334e label; the original observation and correction remain in local audit
receipts. Fixed objects confirm 334e results=66970250 bytes and e85e=66957527
bytes. No historical data, old baseline value or scientific input was rewritten.
Final delivery still requires its actual integrated tree budget, exact-head CI
and review, and protected merge; numerical PASS alone is not that delivery.
