# WB97M-V preliminary-density lifecycle control

This frozen RTX 5090 experiment composes resident AO/indexed forces with #1807's
GPU checkpoint admission. At 24 atoms, adding a private GPU LDA preliminary
density reduces complete cold from **242.278002 to 192.881119 s** (20.3885%),
including the entire 12.213284-s source lifecycle. The strict target takes
18 / 13 iterations. Paired GPU4PySCF cold is 118.652617 / 117.618048 s, so cold
is still slower. Native warm medians are 26.063857 / 26.088192 s; reference
warm medians are 27.672955 / 27.719961 s. No warm gain from the source is claimed.

These are single ordered no-source/LDA cold samples in n1 Slurm 5716 with eight
CPUs and 16 GiB allocated host memory. Reverse-order repetition and larger
combined controls remain necessary before a general policy recommendation.
The public CUDA preliminary-source capacity/lifetime API is unqualified. This
record changes neither defaults nor the README's separately frozen large-system
curve. Earlier #1807 import percentages and other GPUs' results are not added.

Complete cold means synchronized preparation plus the first complete SCF energy
and analytic-force execution, following the existing comparator. It is not a
fresh operating-system/process or empty disk-cache startup measurement. Both
semilocal and VV10 reference grids use the same full unpruned 48×16×32 atomic
rule. Target settings are spherical def2-SVP, FP64, DIIS 8 and energy/density/
screening tolerances 1e-11 / 1e-9 / 1e-12. Each reference XC component is on GPU.

The source uses GPU LDA, the same basis/geometry, a 16×8×16 rule and 1e-6 / 1e-4
energy/density controls. At 24 atoms its synchronized phases are: Calculator
construction 0.002748 s, owner preparation 0.114243 s, solve 12.033391 s,
export 0.000640 s, import 0.055422 s, owner destruction 0.006478 s and remaining
bookkeeping 0.000363 s. These non-overlapping phases sum to 12.213284 s and are
included in native preparation. The target's first E/F call falls from
241.421601 to 179.812750 s; it is not an isolated SCF-only measurement.

The source takes 18 iterations and 18 actual XC submissions. Its Fock-build
count is unavailable and stays null. Per source XC submission the dense
point-AO-square domain is 1,811,939,328; this is not a FLOP count. The exported
density and coordinates occupy 294,912 and 576 bytes. No unmeasured peak memory
or primitive/quartet counts are inferred from those values.

The 3-atom pilot also passes: cold 10.432424 / 8.742931 s, target iterations
15 / 12 and LDA lifecycle 0.665268 s. Every report retains cold, priming and
three warm pairs; all 20 pairs pass 1e-8 Eh / 1e-7 Eh/Bohr gates. Maximum
energy/force errors are 2.956e-12 Eh / 1.177e-9 Eh/Bohr. The target uses the
imported seed without fallback; all priming/warm replays take one iteration.

## Provenance and independent verification

`manifest.json` binds exact stored and decoded report bytes, native library,
source identity and frozen source commit c472684c9a6c680e58a40fb276ff061702b50ace.
That source composes master fc7d5e2d8 and #1807. Integrating master 0fb5fdeea's
already present #1756 component preserves all 1,356 production inputs and the
expanded native derivative tests. It introduces no extra measured speedup.

`support.json.xz` retains the original measurement scripts as text, their hashes,
build command, Slurm/device/source receipts and source-bound qualification.
Slurm 5713 passes three native programs, four admission cases and seven complete
E/F/state cases in each sparse/zero-cache mode. Each mode has 66 successful
native calls and 467 XC submissions. Failed deployment 5712, which referenced
an unavailable compiler path, is also retained. Scripts are data here; the
offline verifier never executes the archived reproduction scripts.

From a checkout with NumPy, verify all compressed hashes, qualification,
source/library identity, source phase sums, actual seed use, reference backend,
AO work, complete-cold totals and every E/F pair without a GPU:

```bash
python -O benchmarks/results/wb97mv-cold-admission-20261004/verify.py 3
python -O benchmarks/results/wb97mv-cold-admission-20261004/verify.py 24
```

For a new measurement, reconstruct the frozen Git source and build with the
retained ccache configuration. Extract `support.json.xz`'s `runtime_scripts`
into a new run directory and adjust its environment paths to that directory.
Rebind script hashes after path changes; preserve source/library checks and the
numerical qualification gates. Submit the retained runner through finite Slurm
`srun --partition=main --gres=gpu:5090:1` with eight CPUs and the same allocation
and scientific settings. Preserve the allocated device visibility and write new
reports separately. Retained records and the verifier are read-only.
