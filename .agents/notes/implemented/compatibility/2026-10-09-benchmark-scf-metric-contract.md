# Decision: explicit benchmark SCF metric and endpoint contracts

Status: implemented
Date: 2026-10-09

## Problem

The PBE0 iteration investigation exposed a mislabeled GPU4PySCF density norm,
incomparable stopping criteria and incomplete first-cycle/cold-seed diagnostics.
Its rationale and causal DIIS evidence remain in
[the separate investigation](../../proposed/2026-10-09-pbe0-gpu4pyscf-diis-tail.md).
The user chose not to report upstream and requested corrections to our tooling.

## Decision

`compare_gpu4pyscf_batch` emits schema v3. Its convergence rows carry
`residual_schema_version=2`: GPU4PySCF `density_frobenius` preserves the
reported `norm_ddm`, while `density_rms` divides by the square root of the
total density-array entry count, including spin blocks. Unknown dimensions
produce null RMS. Shape inspection introduces no device read, matrix product,
or additional reduction. Nonfinite optional residuals are unavailable, not
invalid JSON numbers. First-cycle energy change uses `last_hf_e` when present;
explicit `de` retains priority. Cold default guesses are not called warm seeds.

Policy metadata explicitly states that native density/physical-residual gates
and the stock global orbital-gradient gate are different. Shared iteration
branches do not establish equal work: GPU4PySCF has an initial potential
evaluation before its reported cycles. Native physical-residual RMS is retained
when available, never substituted with the orbital gradient. No tolerance is
relaxed, and stock reference DIIS is unchanged.

An explicit full-Fock request clears both `dm_last` and `vhf_last`, rather
than assuming `direct_scf=False` is sufficient on every backend. Reconfiguration
does not stack wrappers and can restore the original reference method. The
explicit request is recorded separately from density fitting's own switch.
Both cold and warm reference E+F timers now include downloading host forces;
JSON/list assembly stays outside timing, like the native caller's serialization.
SCF tolerances must be positive and finite before GPU imports or preparation.

## Compatibility and rejected alternatives

Summary selection accepts both outer schema v2 and v3. The README reducer
preserves new stopping-policy metadata and residual version markers. Archived
v2 values are neither rewritten nor silently interpreted as normalized RMS.

We reject loosening the reference gradient gate to manufacture iteration
parity, changing upstream DIIS silently, adding expensive per-cycle physical
audits to clean endpoint timers, and treating matching cycle counts as an
equal-work speedup. Callback residuals are labeled as callback observations,
not as an independent final-state audit.

## Evidence

CPU unit coverage includes restricted/spin-block RMS normalization without
device reads, missing dimensions, nonfinite diagnostics/tolerances, first-cycle
energy differences, cold seed labels, explicit full-Fock input suppression and
restoration, v2/v3 summary compatibility, and host-force transfer inside timing.
The CPU tests use the existing frozen `build/readme-cpu/libvibeqc.so`.

Finite Slurm job 2745 on n2/node2 (`main`, `gpu:pro6000:1`, 15 minutes)
validates the changed tracker/payload/configuration helpers using installed
GPU4PySCF and an independent CPU PySCF PBE0/def2-SVP oracle on the same explicit
48-by-16-by-32 moving grid. Both stock and explicit full-Fock energy/force
endpoints pass `1e-8 Eh`/`1e-7 Eh/Bohr`; computed RMS also matches a direct
array oracle. The helper source snapshot and checksum are retained, not assumed
from a future mutable working tree. The subsequent timer correction has a
separate deterministic transfer-latency regression test.

Raw validation/experimental evidence is retained under
`.cache/benchmark-scf-fix-20261009/` locally and
`/data/jzzeng/benchmark-scf-fix-20261009/runs/2745` on n2. Native optimization
experiments there use the independently identified supported CUDA source
`4385f72751b829883407c01106186917c344317b`, not this older dirty worktree.

## Consequences and revisit conditions

New timings include the actual reference public-output transfer boundary and
can differ from archived timings. Cross-version raw numbers require that
distinction. Optional normalization fallback and older artifacts remain
explicit. A scientifically matched stopping/work protocol or transparently
labeled alternative-DIIS experiment requires its own qualification; neither
is implied by this corrective change.
