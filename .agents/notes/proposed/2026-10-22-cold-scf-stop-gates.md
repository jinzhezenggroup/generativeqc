# Diagnosis: cold SCF counts and the actual stopping norm

Status: proposed
Date: 2026-10-22

## Finding

Standard's additional cold iterations are real for ordinary HF/DFT. A strict
physical stationarity gate, in addition to the public energy/density gates,
often controls the tail. Iteration counts alone do not explain the endpoint
gap: native per-iteration work and setup/force costs remain substantial.

This diagnosis changes no solver policy. A temporary native journal records
scalars that `CudaKsState::finish()` has already copied to the host, with no
extra device work. A GPU4PySCF callback observes its existing stopping scalars
without changing the default initial guess, convergence, screening or grids.
The traces are diagnostic populations, not replacement README timings.

## Fresh endpoint evidence

Seven native Standard DFT populations and four default GPU4PySCF populations
run on n1/node1 through finite Slurm allocations. Each includes a complete
cold construction-to-force-export call and three warm calls from its own
frozen cold density. All 44 energy/force calls pass the existing independent
gates: energy 1e-5 Eh/atom, maximum force 1e-4 Eh/Bohr, RMS force 3e-5
Eh/Bohr, with the original quarter-gate reference checks. XC/VV10 defaults
and complete moving-grid force response remain unchanged.

The following are 96-atom Direct Standard measurements. The first E/D pass
means the first complete target iterate satisfying energy and density-change
conditions; it does **not** certify final physical/canonical acceptance.
Cold and warm durations below are from these same instrumented v1 receipts;
warm is the median of the three frozen-density calls.
Tail time sums disjoint completed iteration/finish host intervals after that
pass. It is observed work, not a promised removable cost.

| Method | Total iterations | First E/D pass | Later iterations | Later seconds | Cold seconds | Warm median seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PBE | 15 | 11 | 4 | 3.540 | 27.684 | 6.867 |
| PBE0 | 15 | 11 | 4 | 7.214 | 41.007 | 8.332 |
| B3LYP | 16 | 11 | 5 | 9.264 | 44.193 | 8.784 |
| r2SCAN | 14 | 11 | 3 | 3.242 | 29.270 | 7.398 |
| r2SCAN-3c | 16 | 12 | 4 | 9.471 | 67.915 | 19.088 |
| M06-2X | 16 | 11 | 5 | 9.989 | 46.590 | 8.949 |

GPU4PySCF repeats HF/PBE/PBE0/r2SCAN-3c in 10/11/9/11 cycles. Its default
energy tolerance is 1e-9 and its orbital-gradient norm tolerance resolves
to sqrt(1e-9), about 3.1623e-5. Density change is logged but is not a separate
stock stopping condition. The earlier qualified native HF baseline takes
14 iterations; the rejected common-DIIS-scale experiment did not change it.

## The hidden distinction between residual norms

Native Standard requests energy 1e-10 and density-change tolerance 1e-8.
Ordinary CUDA KS additionally requires both the maximum spin RMS residual
and the maximum absolute AO commutator entry below
`min(1e-9, density_tolerance)`. HF's corresponding physical maximum gate is
`min(1e-8, density_tolerance)`. Neither is GPU4PySCF's occupied/virtual
orbital-gradient norm, so matching their numerical threshold values is not
a meaningful equivalence.

`ks_diagnostic.history.physical_residual_max` contains the **maximum over
spin RMS residuals**, not the AO-element maximum used by the second KS gate.
For PBE0 at iteration 14 the reported spin RMS maximum is 2.8611e-11, but
the actual AO maximum is 1.1441e-9; the latter correctly prevents stopping.
Iteration 15 reduces it to 1.0254e-10. At the first E/D pass, iteration 11,
the AO maximum is still 7.9600e-8. PBE's iteration-11 AO maximum is
4.3000e-7, versus the logged spin RMS maximum 6.7412e-9.

Do not infer an unnecessary extra solve from the misleadingly terse history
field name. Future diagnostics should expose the norm explicitly. Earlier
same-binary public-tolerance scans did not remove these internal gates:
PBE remained at 15 iterations even at requested E/D = 1e-7/1e-6; HF and
PBE0 likewise retained 14/15 in their scans.

## Counts that require separate interpretation

The fresh native omegaB97M-V endpoint has 28 iterations, but 23 are
provisional semilocal/frozen-VV10 iterations and only five evaluate the full
target. Its nonlocal ledger records four VV10 pair sweeps. Comparing 28
directly with ten GPU4PySCF cycles treats unlike work as equivalent. The
native endpoint is 315.188 seconds in this instrumented population; it does
not show an ordinary E/D-passing residual tail like the six methods above.

The r2SCAN-3c initial-guess capacity repair in #2668 reduced Standard/Fast
Direct counts from 22/14 to 16/10. The corrected lifetime estimate admits
Huckel under the unchanged cap. This is distinct from changing convergence;
all seven fresh native cold cases use their admitted Huckel seed with no
discarded target iterations.

Fast Direct PBE/PBE0/B3LYP already use 9/9/9 iterations in the retained
qualified runs, versus GPU4PySCF's 11/9/10. Their remaining cold losses
therefore require per-step/setup/force work improvements too. DF Fast can
explicitly fall back to Standard and must be described by its executed route.
Large DF raw-source regeneration remains expensive even when counts fall.

Density-change convergence is already enabled. The disabled incremental
J/K option uses delta D to reduce operator work; it is not the density
stopping check. Earlier incremental ablations did not establish a universal
win, and can add full-density confirmation work or hurt warm endpoints.

## Next decision and rejected shortcut

Investigate final-state-aware stopping/closure and J/K cost separately.
Retain current physical/canonical acceptance until any revised accuracy
contract has independent qualification. A previous diagnostic changing the
ordinary KS residual ceiling from 1e-9 to 1e-8 failed canonical-density
export for PBE and r2SCAN: errors 1.15914e-8 and 1.02152e-8 exceeded the
unchanged 1e-8 gate. A prospective earlier stop must check the true F[D]
frame and continue safely when that check fails; simply replacing the
constant would reintroduce those failures. See the Huckel lifetime note for
the retained failed receipts.

## Provenance

Base `4906be1ad1755577c9a96adf5674938b633e9486` is master
`976f756b391bc65edb79f76df1a8c3fe807518a7` plus the #2668 Huckel repair.
The diagnostic-only patch SHA-256 is
`360ade1118e5fd942a0d872a0e7b4dd5de60842e2487b2b000dd9e22913c2c9e`;
core SHA-256 is
`d4734f3a1c9799495246c61f8ec3b58d22b593ce6048640c9e84f6ba8755e763`.
Both mutable build and immutable runtime verify all 12,146 source inputs.
The build uses verified sccache 0.16.0 and retains before/after statistics.
Stationary AOT libraries and manifests are built together from the same
source; older unrelated manifests must not be copied into a new runtime.

Ignored `.artifacts/cold-gates-v1/` in the diagnosis worktree and the matching
`n1:/data/jzzeng/scf-performance-20261021/cold-gates-v1/` retain the patch,
source manifest, build evidence, launchers, all endpoint receipts, raw
stopping/host/device journals and independent qualification. `summarize.py`
recomputes the table, checks complete zero-drop CUDA traces, and preserves
source/receipt identities. The frozen runtime is the adjacent
`runtime-cold-gates-v1/source/` directory. No default promotion or complete
README leadership claim follows from this diagnosis.
