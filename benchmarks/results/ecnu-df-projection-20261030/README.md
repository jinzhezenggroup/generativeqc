# Native DF projection and nonlocal force follow-up

These are **portable-profile default-grid endpoints**, not a universal speed or
accuracy claim. Three native devices each completed all 64 README combinations:
eight methods, Direct/DF, and 3/12/48/96 atoms, cold plus three same-owner warm
calls from each engine's frozen cold density. Math libraries use one thread.
GPU4PySCF 1.9.0 references use the same A100 and another RTX 5090 node; no PRO
6000 reference was measured. Scheduler receipts retain genuine LSF/Slurm IDs.

Run `python benchmarks/results/ecnu-df-projection-20261030/report.py` for the
96-atom medians and failed gates. `observations.json.xz` retains every endpoint
time, iteration/Fock count, grid/nonlocal work, candidate energy, paired maximum
errors, per-repeat independent oracle errors, provenance and trace counters.
Baseline receipt-set hashes bind the complete case-to-receipt map; candidate
and reference records also retain individual receipt hashes. This compact
summary does not reconstruct omitted force vectors. Exact raw receipts and the
producer script remain in ignored `.artifacts/ecnu-stationary-parity-20261030/`.
The independent oracle file/hash and explicit v2 environment audit are embedded.

All devices pass 58/64 common independent gates. HF DF fails the dense-direct
energy gate at all sizes (GPU4PySCF DF has the same fitting error); M06-2X at
three atoms fails the default-grid RMS force gate in both modes. No threshold
was relaxed. Native WB97M-V DF uses fitted full-range plus exact long-range
exchange, while GPU4PySCF fits both.

Old Direct HF 48/96 baselines omit force contributions and are invalid speed
baselines. The corrected candidate restores that work. Other warm changes
exceeding 5% were absent on PRO 6000/5090; the A100 PBE DF three-atom cohort
contains a 47-to-70 ms increase, with samples decaying from 82 to 64 ms,
requiring a paired repeat before a no-regression claim.

See [the DF note](../../../.agents/notes/implemented/performance/2026-10-30-final-df-projection-and-pressure-scratch.md)
and [the map note](../../../.agents/notes/implemented/performance/2026-10-30-nonlocal-stationary-ao-maps.md).
Auto-profile and portable generated-force experiments are separate. Embedded
native source identities bind measured images; the Git parent alone does not
describe their uncommitted source deltas.
