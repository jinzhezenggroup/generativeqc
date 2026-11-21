# Decision: Retain an AO-jet prefix before replaying the grid tail

Status: implemented
Date: 2026-11-21

## Problem

The 48 AO/226 auxiliary PBE0 UKS water dimer cation passes both fitted HVP
branches, but the default 128 MiB grid arena could not retain the complete
order-three AO jets and mixed outputs together. The former all-or-nothing
fallback replayed all 13,824 points for each mixed action. With 116 bounded
AO primitive/polynomial incidences per point, the 342 mixed actions of the
dense matrix charge 10,968,514,560 visits, above `2^33`. Slurm `1786`
rejected the request before column work. Increasing the default budget or
admitting the redundant replay would not address the cliff.

## Decision

Reserve mandatory outputs first and one replay tile if needed. Use the
remaining spatial capacity for an immutable point prefix evaluated once;
the unchanged generated evaluator replays the nonresident suffix. The
point ordering and mixed-output layout do not change, and all FP64 and
finite-publication gates remain. `spatial_jets()` still lends only a
complete resident panel. Minimum capacity still permits pure replay when
the per-request work budget allows it. Plan the full replay count before
submitting the matrix; report the responsible grid when admission fails.

Do not spend a shared independent-VV10 geometry budget on the first
owner's optional prefix: retain the existing full-resident or tile-only
choice for that owner, then admit the second owner's work with the actual
remainder. No work or memory cap increases.

## Evidence

Node5 RTX 5090 geometry replays passed 37 tests (four other probe families
were not enabled) on the final build. The fixture compares full residence,
partial residence and pure tiled execution elementwise and against
independent Gaussian and partition derivatives.

The final shared-budget build also passed Slurm `2218` on node5: the
dual-grid native HVP at the exact combined moving-geometry minimum agrees
with the resident result, and a one-byte smaller budget leaves the output
unchanged. Five public CLI/Python cases passed for separate VV10 meshes,
tight grid selection (default level 3 and explicit level 0), and r2SCAN-rVV10
RKS/UKS. The staged CLI, library and native probe passed SHA-256 verification;
see `build/hessian-gpu-sm120/r2scan-orbital-f/partial-ao-shared-vv10-n5.log`.

Slurm `1787` qualified all 18 columns of the spectral UKS dimer matrix using
an explicit 192 MiB grid allowance and the default 64 MiB fitted allowance.
Elapsed time was 905.39 s with 630 induced Fock actions, 18 DF value passes,
18 DF contraction passes and 324 mixed grid contractions. Maximum asymmetry
and translational defect were `3.88e-10` and `3.34e-10 Ha/Bohr²`, without
post-hoc symmetrization. The two projected errors against the independent
energy oracle were `3.10e-8` and `2.20e-8 Ha/Bohr²`. No host numerical fallback,
intermediate download or displaced SCF occurred in the native endpoint.

At the default 128 MiB grid limit, the single-grid plan retains 7,970 of
13,824 points and replays 5,854 per action. The 342-action request thus
bounds replay at 4,644,797,760 AO primitive/polynomial visits, below `2^33`.
Slurm `1788` completed all 18 columns in 907.71 s, with the same endpoint
work counts as `1787`. Raw asymmetry was `3.88e-10`, the translational
defect was `3.34e-10`, and the two projection errors were `3.10e-8` and
`2.20e-8 Ha/Bohr²`. The same device-residency and publication gates passed.
This run used the single-grid implementation before the final independent
VV10 budget guard; the guarded build was exercised separately by `2218`.
See
`build/hessian-gpu-sm120/r2scan-orbital-f/pbe0-uks-dimer-spectral-full-matrix-default-n4.log`
and `build/hessian-gpu-sm120/r2scan-orbital-f/pbe0-uks-dimer-spectral-full-matrix-resident-n4.log`.

On the final shared-budget build, Slurm `1789` ran the full-rank variant
(all 226 metric roots retained) at the same default grid and fitted limits.
All 18 columns completed in 907.43 s, with 630 induced Fock actions,
18 DF value and contraction passes each, and 324 mixed grid contractions.
Raw asymmetry and translation defects were `1.74e-10` and `2.17e-10`,
and the two independent projected errors were `4.36e-8` and `2.69e-8 Ha/Bohr²`.
The device-residency and publication gates passed. See
`build/hessian-gpu-sm120/r2scan-orbital-f/partial-ao-shared-fullrank-matrix-n4.log`.

On the final build, Slurm `2219` passed 25 public CLI/Python tests: M06-2X
and MN15 Hessians and HVPs over all selected s/d/f orbital, RKS/UKS and
direct/fitted modes (including auxiliary g in the polarized-basis fitted
cases), plus direct r2SCAN with f orbitals. This confirms frontend replay
for the fixtures, not an independent oracle at every angular combination.
See `build/hessian-gpu-sm120/r2scan-orbital-f/partial-ao-shared-angular-cli-n5.log`.

Together `2218`, `2219`, `2220`, `2221`, and `2222` passed all 128 cases from
`test_wb97mv_hessian_cli_cuda.py` on the final library. The first broad run
passed 92 of its 98 selected cases; three passed after correcting the
test deployment (`sccache` on `PATH`, Python dependencies and fixtures).
The default-grid r2SCAN-3c test passed after replacing its obsolete zero
residency assertion with a partial-preparation and bounded-replay gate,
without changing its raw matrix/HVP checks. The two auxiliary-g ECP cases
were initially skipped without the native probe, then passed with that probe
and their independent reconverged-energy checks enabled. These are SM120
results; they do not qualify other CUDA architectures or the global resource
planner. See `partial-ao-shared-{remaining-cli,cli-retry,grid-and-fg-ecp}-n5.log`
in the same log directory.
