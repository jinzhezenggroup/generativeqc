# Decision: reserve the composite owner's requested phased Becke storage

Status: implemented; 3/12/48-atom endpoint qualified, 96-atom endpoint unresolved
Date: 2026-10-14

The completed 96-atom frozen-VV10 trial (`003e5c7b0`, Slurm 8298) reports a
70.001-second first warm endpoint. Its force endpoint consumes 44.959 seconds,
including 34.076 seconds in the geometry drain and 10.606 seconds in native
integral derivatives. Both semilocal and nonlocal geometry owners report zero
phased storage and zero phased batches, despite the existing ordinary source
policy requesting phased Becke at 48 or more atoms.

The composite dry planner forwarded only the experimental primitive selection.
It consequently returned an ordinary source arena. Each runtime source received
that exact arena size as its budget, so its later phased request had no space
and fell back. Spare capacity in the complete composite budget could not help.

Forward the same resolved ordinary phased selection to the dry planner and both
runtime owners. Charge both optional phase panels inside the existing complete
device bound, keeping the source planner's no-room fallback and point lanes.
Resolve the primitive selection once too, and include both selections in the
prepared identity. No new device equation, grid, precision, convergence rule,
primitive default or user budget is introduced.

Host resource checks cover both spin layouts, exact double-owner accounting and
bounded fallback. Independent complete molecular energy/forces and source-bound
48/96 endpoint populations must qualify the actual selected route before
performance claims or issue closure. The original unphased measurements remain
retained; a requested policy alone is never evidence of executed phased work.

## Qualification and fixture limits

Assembled source `948fb10586f1ac73ad872a80f9de63a99169d276` (library SHA256
`4a5ccdabd0a3c67eeab53281a52ae4d0c556e364f9e421d0d29b1f2f16fb5e39`) passes all
36 default 3/12/48-atom observations. It also includes the separately proposed
exact SCF/force reuse changes; these timings do not represent this isolated PR
on its own. No initializer or device-final-validation experiment was enabled.
The original 1800-second point limit, 100-iteration bound, FP64 grid and
1e-8 Eh / 1e-7 Eh/Bohr independent gates were retained.

At 48 atoms, cold/warm/moved/moved-warm are
575.813462 / 15.547547 / 405.619519 / 15.579747 seconds (Slurm 8313).
The preceding SCF-reuse source `3b15b73d7` gives
579.670301 / 18.435429 / 409.448652 / 18.401237 seconds. Each new owner reserves
41,706,304 phased bytes and visits all 1,179,648 grid points in 1,152 actual
phase batches per endpoint. The earlier owner reports zero bytes/batches.
Maximum 48-atom errors are 4.3201e-12 Eh / 6.5203e-11 Eh/Bohr.

The small regression uses 24 separated H2 fragments, the unchanged automatic
48-atom threshold, original and moved geometries, repeated forces, and two
reconverged energy-difference steps. This tests an independent derivative,
while the original-grid water-48 population supplies absolute external E/F
acceptance. It passes on Slurm 8324; memcheck (8325) and synccheck (8326) each
pass with zero errors. Their receipts are retained with the benchmark publication.

Do not reuse coarse sto-3g water-48 as an absolute-reference acceptance fixture
without resolving its existing discrepancy: GridSpec(8, 6, 12) differs from
full-Fock GPU4PySCF by 9.88294e-7 Eh on both the preceding unphased source and
this source. The 48-H cluster also has absolute differences of 1.35320e-8 Eh
on GridSpec(8, 6, 12) and 1.01675e-8 Eh on GridSpec(12, 4, 8); neither is an
accepted external energy comparison. Logs and fixture definitions remain in
qualification receipts. These failures were not relabeled as setup problems
or addressed by relaxing the gate. The derivative test is a different,
explicitly scoped observable. The 96-atom timing issue #2426 remains open.
