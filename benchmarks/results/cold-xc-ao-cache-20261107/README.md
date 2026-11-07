# Rejected cold XC AO-geometry cache

This experiment does **not** change production defaults. It tests the complete
96-atom water WB97M-V/full-Direct/Standard cold energy+analytic-force+host-export
endpoint and three warm repeats seeded only by that owner's frozen cold density.

| Device / actual job | Rollback cold / s | Candidate cold / s | Rollback warm median / s | Candidate warm median / s |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF227134, exit0 | 82.695270 | 82.042750 | 13.816298 | 13.686628 |
| RTX5090 / Slurm16644, exit1 | 168.310181 | **incomplete: OOM** | 12.755009 | **not reached** |

A100's one cold observation saves0.652520s (-0.79%) while retaining
6,989,834,240 bytes of AO/jet panels. The tighter paired energy differences are
zero and maximum force differences are1.01e-12 Ha/Bohr. All complete exports
pass the retained independent static E/F gates. The 5090 independent native
E/V/features/nonlocal-pullback/retirement probe and memcheck pass, but do not
qualify its failed complete endpoint. No failed-call duration is ranked.

Both clean arms keep the effective LR/composite source enabled, strict FP64,
the same grids, screening, convergence and analytic-force response. The sole
rollback is `GENERATIVEQC_CUDA_XC_AO_CACHE=0`; the candidate genuinely leaves
it unset. Cold retains16 Focks (11 provisional+5 target), four SCF VV10 sweeps,
16 J/16 K actions and1,873,221,716 resident source outputs. Each warm repeat
retains one J/K/Fock and zero new source construction or SCF pair sweeps.

## Provenance and limits

The GPU-executed candidate is source `5efc4cf4e871609dc6f6819ea45232a0c2bfe8b672e457a950d8b6a3ae57e2df`,
ELF `5a9331731c148f589819f7dfd87ba396d9e333ff3bba7d7e0b8f85c442bb2793`,
over PR3033 head9ec79b779, which already contains current master22b33b081.
Separate intrusive Slurm16645 reproduces OOM in the force lifetime with
6,280,069,120 AO bytes retained. Diagnostic scopes are nested, not additive
endpoint costs, and are never substituted for the clean timing observations.

The unconditional handoff-retirement alternative is a **different compile-only**
source `de9848f4287633822eced0125b8a3368bf1c08295308a3f739d9a7dfd81ee063`,
ELF `d0255e88db9eefc4592a750b96b5ca877d4ceb6b8f15a6441a3c7f762e3b0391`.
Its921-case host population passes, but its waiting GPU drivers were stopped
before submission. It has no GPU performance or numerical qualification.
The initial missing-import host collection and build-transport setup failures
are not counted as passes. Both final cached builds succeed and restore the
borrowed original runtime; compiler-cache statistics are retained. Slurm
accounting is disabled: runner exit receipts are not invented sacct records.
The owned finite A100 transfer is stopped, and credentials are not retained.

Reject the candidate rather than ship an opt-in or unqualified default. This is
a cost/benefit/resource decision, not a claim that all sub-percent gains are
worthless. There is no fresh gpu4pyscf latency ranking and no statistical claim
from a single cold pair. The next meaningful VV10 effort needs actual work
reduction beyond the already retained symmetry/reciprocal optimizations.

## Replay

Run `python benchmarks/results/cold-xc-ao-cache-20261107/replay.py`.
`evidence.json.xz` retains complete exported observables, immutable source
new-file overlays and lossless patches against the recorded base, build/ELF/
scheduler receipts, independent probe logs, complete component traces and selected
cache/force progress events. Full verbose progress remains in ignored artifacts
with its SHA256 retained here; it is not duplicated into Git. `summary.json` is derived. The reader pins
scientific inputs/settings to the existing independent reference and incumbent,
checks actual work and refuses promotion or relabelling the incomplete5090 call
or compile-only repair. Its corruption gates remain active under Python `-O`.
The retained driver blocks production oracle imports. Both archived patches
were separately reconstructed against the recorded base and reproduce their
exact native source identities; the one-off receipts are retained in the archive.

See the [rejection rationale](../../../.agents/notes/rejected/2026-11-07-cold-xc-geometry-cache.md)
and tracking issue #2976. No production implementation or selector is retained.
