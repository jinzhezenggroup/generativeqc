# Decision: qualify frozen C4/C7 forces with fresh displaced PySCF energies

Status: implemented (qualification tooling; scientific results INCOMPLETE)
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
