# Evidence: WB97M-V bridge baseline versus GPU4PySCF

Status: implemented (benchmark driver and negative baseline; not a speed claim)
Date: 2026-10-15

## Scope and provenance

The user requested an endpoint comparison, then clarified that #2527 must have
a Python-free runtime. This evidence measures the earlier explicit Python/CuPy
CLI bridge, **not** the still-incomplete native runtime or AOT leaf package.
The source commit recorded for staged Python is
`8126eaa93a42fa6c3a7da987f942ee1ee2e63d23`; staged native CLI/DSO artifacts were
built from numerical candidate `80dfd1bf08f4051c94f883c53896b294d55b3ff1`.
Do not attribute these timings to the later PR head.

Compact raw H2 timing/matrix samples are retained in
`benchmarks/results/wb97mv-hessian-bridge-h2-20261015.json`.
Native CLI SHA256 was
`6cd50d355dbb9278711985fbf3be4b1ea4228e1a808bb17c8dea4227f1ea9af6`;
DSO SHA256 and the measured comparator SHA256 are in the receipt. The exact
measured comparator snapshot is retained locally under the ignored
`.artifacts/wb97mv-hessian-gpu4pyscf-20261015/measured-comparator.py`.
The committed driver additionally fixes owned-worker cleanup/timeout slack,
labels accelerator metadata as collected before timing and rejects nonfinite or
broadcast-mismatched matrices before comparison. These do not change the
measured mathematical work or the retained valid matrices/acceptance result.
The compact receipts retain hashes of their full original JSON, which remains
in the same ignored local artifact directory. Repeated invariant native work
counts are stored once; all six H2 timing/matrix samples remain with exact
matrix-table deduplication (no rounding or symmetry repair). The water
receipt retains its failure and reference-only timing, not a fictitious native
matrix. This keeps the new evidence within the aggregate retention budget
without deleting unrelated historical data or publishing an archive.

## Endpoint and acceptance

Slurm node1 job 8729, RTX 5090, GPU4PySCF CUDA12x 1.8.1, PySCF 2.14.0 and
CuPy 13.6.0. Both engines use spherical STO-3G, direct J/K (not differing default
DF-RSH models), original unpruned three-iteration Becke quadrature 8x4x8 per
atom and VV10 rho threshold 1e-8. The grid is intentionally coarse; there is no
grid-convergence claim. Both include moving-grid, VV10 and orbital response.

Each measured sample includes a fresh process, fresh SCF, complete host Hessian
and JSON. Persistent compilation caches are reused. Sample zero is first
observed use, **not** empty-cache cold. Two subsequent fresh-process samples
per engine alternate order. All repeat pairs must satisfy max absolute Hessian
error <=3e-6 Eh/Bohr^2 before a speed ratio is accepted.

For H2 (two AOs, 512 logical points, raw 6x6 Hessian), artifact-primed endpoint
medians are 20.2581391893 s for GenerativeQC and 4.5815144610 s for GPU4PySCF.
Observed timing is about 4.42 times longer for this bridge, but the accuracy
gate **fails**: max matrix difference is 9.3267070329e-5 Eh/Bohr^2. The qualified
ratio remains null. This is not an equal-accuracy performance comparison.
Native work includes six columns, no nonzero CPKS iterations/actions for this
minimal H2 fixture, 43,308 primitive/component evaluations, 1,440 scalar launches,
841 tensor programs and 3,686,400 VV10 pairs. Do not extrapolate its lack of
orbital response work to larger molecules or UKS.

## Independent mismatch diagnosis

Outside timing, displaced reconverged **forces** were checked at +/-1e-3 and
+/-3e-4 Bohr for atom-0 z. At 8x4x8, the native analytic Hzz is
0.24699973004378606; GPU4PySCF analytic Hzz is 0.2470929971141147. At the smaller
step, force-difference Hzz values are 0.24699983814435725 (native forces) and
0.24699983630962116 (GPU4PySCF forces). Energies/forces agree across engines at
the sampled geometries. Thus the native AD column is supported by both force
oracles, while this GPU4PySCF analytic entry is inconsistent with its own forces
for these exact settings. This does not establish a universal reference bug or
validate every matrix entry.

Repeating the diagnostic at 12x6x12 gives GPU4PySCF analytic Hzz
0.2904924706853845 versus force differences 0.29057426910043355 (native) and
0.29057426655820784 (GPU4PySCF) at 3e-4 Bohr. This is still a diagnostic, not a
grid-convergence or accuracy-equivalence claim. Do not relax the common gate to
publish a speed ratio. Force differences are oracles only, never production AD.

## Water timeout and scheduling cleanup

Water samples during jobs 8729 and 8741 are discarded: cancelling the first
bridge job left its separately grouped Python CUDA worker alive, which could
overlap a subsequently allocated device. Owned CLI/worker identities were
verified through their Slurm environment before termination; all identified
orphans were verified gone. Never override CUDA_VISIBLE_DEVICES to work around
this scheduler/process-group boundary.

The corrected driver kills owned separately grouped direct workers before its
supervisor, handles SIGTERM and allows 30 s beyond the native worker deadline
for cleanup. A CPU regression exercises the separate-group timeout case.
Uncontaminated water job **8747** runs the reference first (8.9681338631 s,
first observed process) but the bridge fails its 300 s worker deadline with no
Hessian publication. Its receipt is
`benchmarks/results/wb97mv-hessian-bridge-water-timeout-20261015.json`.
There is no completed native water latency, warm median or qualified ratio.

## Consequence

Keep the negative baseline. Replace Python orchestration with the complete native
compiler-AD consumer, qualify scientific equivalence independently and then
remeasure the actual required endpoint. Native kernel/AOT compilation success
or leaf microbenchmarks cannot substitute for that endpoint evidence.
