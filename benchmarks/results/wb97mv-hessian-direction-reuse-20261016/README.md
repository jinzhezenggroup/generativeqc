# Native WB97M-V Hessian direction/symmetry reuse

Measured candidate: `5aa703721fdce6a220838676af615d574f0fed05` (clean compiled inputs).
Native baseline: `f37f96a1c3fee0da04ea0cc2c490f04140d65ae6` (clean merged master).
Both source-matched builds include the same concurrent master changes: this A/B
isolates the optimization rather than comparing against the older PR branch.
Slurm 3265 reserves node2 exclusively and exposes one RTX PRO 6000 Blackwell
to the benchmark step. CUDA 12.9/sm120, strict FP64, canonical WB97M-V, spherical
STO-3G, direct RKS, self-consistent VV10 and full moving-grid/partition/CPKS
response are common. Primary/NLC/CPHF use the same 8/4/8 quadrature policy.
This is a **small-system implementation-latency diagnostic**, not a chemically
grid-converged or large-molecule scaling benchmark.

## Complete endpoints

Each sample starts a new process and fresh SCF and ends after the complete host
Hessian and JSON serialization, including teardown. The native child cannot
discover Python or a compiler; PTX JIT is disabled. All engines have the same
600-second deadline. Retain first-observed samples separately; medians use all
five later artifact-primed fresh processes per engine/case in rotated,
interleaved order. Caches are not cleared; there is no empty-cache cold claim.

| Case | Old native (s) | Optimized native (s) | GPU4PySCF (s) | New/old time | New/GPU4 time |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2, 2 AOs | 1.768047589 | 1.667805509 | 5.423468762 | 0.943303517 | 0.307516385 |
| Water, 7 AOs | 15.490395554 | 8.679433998 | 10.932610709 | 0.560310676 | 0.793903142 |

Water uses **44.0% less time than baseline native** (1.785x speedup) and **20.6% less
than GPU4PySCF** (1.260x). H2 uses 5.67% less than baseline native and is 3.252x faster
than GPU4PySCF. These are bounded measured ratios, not confidence intervals or
statistically qualified broad default promotions. All 36 endpoints complete.
No sample is dropped for timing or accuracy.

Every old/new raw matrix pair passes `3e-12 Eh/Bohr²`; maximum errors are
`5.551115123e-17` for H2 and `6.883382753e-15` for water. Every native/reference
pair passes the unchanged `3e-6` gate; optimized/reference maxima are
`1.834633334e-9` and `3.603278126e-8`. Matrices are not symmetrized after assembly.
GPU4PySCF 1.8.1/PySCF 2.14.0/CuPy 13.6.0 are unmodified; the common reference
adapter retains `cphf_grids = grids` and the established response tolerance.
Reference response-action counts are unavailable and remain null, not inferred.

## Executed work and resource boundary

The source retains bounded last-direction primitive jets and shares ERI eightfold
shell symmetry. Both Cartesian component strides and gradient/HVP center slots
are permuted. D/dD are always reprojected; identical direction values, not pointer
identity, control cross-action reuse. Public reductions preserve incidence order
with barriers and no floating-point atomics. The cap is 64 MiB; zero or insufficient
arena/aggregate capacity retains the original ordered streaming source. Mandatory
DF/ECP storage is reserved first.

| Case | Old primitive components | New primitive components | Source builds / hits | Resident jet bytes |
| --- | ---: | ---: | ---: | ---: |
| H2 | 31,104 | 5,832 | 6 / 6 | 2,816 |
| Water | 7,001,316 | 793,152 | 9 / 9 | 217,856 |

Water executes 88.67% fewer primitive components. Its nine Hessian columns,
20 native SCF iterations, 108 induced-Fock actions, 39,690 one-electron components,
81 grid mixed contractions and all VV10 work are unchanged. The stationary
owner's accounted storage changes from 4,771,912 to 5,049,928 bytes; this scope
is not total process/device memory including SCF/context/library allocations.
The response is still scalar-column, not a new multi-RHS solver.

Historical pre-master instrumented full-water profiles (Slurm 3244 / 3262,
source17 / source616, with exact revisions and binaries in the evidence) reduce launches
from 434,900 to 196,813 and total kernel duration from 13.94498 to 7.18305 s.
Factorized integral launches change from 46,512 / 7.23659 s to 4,644 / 0.752914 s.
Grid AD pairs remain 35,316 launches and about 2.14 s, exposing the next bottleneck.
These profiles explain work removal; they are not relabeled as final-source
executions and do not replace the clean wall timings.
CUDA API durations overlap/backpressure and must not be added to kernel time or
treated as a pure memory-bandwidth cost.

## Independent qualification

The same source-matched binaries pass Slurm 3266 on an RTX PRO 6000 Blackwell:
50 independent nuclear-J/K component, four-distinct-center permutation,
residency/streaming/budget-fallback and lifecycle tests; four stationary
resident/streamed full-Hessian comparisons; four real public C++ CLI/API tests;
and three additional independent molecular HVP tests. The entire 35-case
stationary suite is not claimed as newly rerun. Coverage includes RKS/UKS, Direct/full-rank
DF, ECP, Cartesian/spherical, orbital f and auxiliary g. Four stationary cases
compare complete resident/streaming Hessians at `3e-12` absolute/relative tolerance.

Every final benchmark matrix is freshly gated against the retained independent,
reconverged values-only Libcint/Libxc/direct-space-VV10 molecular-energy stencils
from Slurm 3240 at source17 (the prior matched-CPHF publication). Those independent
reference energies are unchanged and are not claimed as rerun on source5aa. All
36 H2 and 81 water entries pass the unchanged `.004/.002` Richardson gate:
`atol=8e-7`, `rtol=1e-7`. Both stencil matrices and every extrapolated entry are
retained, as are molecular HVP energy stencils. Finite differences belong only
to the independent oracle, never the production derivative path.

Slurm 3267 additionally passes memcheck, initcheck, racecheck and synccheck on
the native J/K probe with zero errors, warnings or race hazards. Python/compiler
discovery is disabled for that numerical child. Correctness, sanitizer and
exclusive timing allocations are separately identified. Earlier unchanged AOT leaf qualification remains in the prior default-fg
publication, not relabeled as newly rerun here.

`samples.json.xz` retains complete timings, raw matrices, commands, actual native
work, environment and binary/source identities. `qualification.json.xz` retains
recomputable numerical errors, retained independent energy matrices, fresh HVPs, diagnostic
kernel tables and test/sanitizer/build receipts. The publication manifest
authenticates every member. The measured comparison driver is losslessly stored
as `compare-native-ab.py.xz`, preserving its original SHA-256 and bytes rather
than reformatting a historical execution artifact. The accompanying host evidence tests independently
recompute matrices, all medians and both ratios without requiring favorable timing.

## Reproduce

Keep this bundle outside the source checkouts; prepare clean checkouts of the
two measured revisions and set `BASELINE_CHECKOUT`, `CANDIDATE_CHECKOUT`,
`REFERENCE_PYTHON`, CUDA and reference shared-library search paths. On the node2
Slurm submission host, run `bash /path/to/reproduce.sh`. It verifies the compiler
cache, builds both full default AOT inventories and uses an exclusive allocation
with one Slurm-visible benchmark GPU. Adapt node and GRES together elsewhere;
never override device visibility. Build-time Python is allowed; native runtime
Python/JIT is not. The standalone script is syntax checked, not independently
replayed. Independent qualification commands and prerequisites are in the
publication manifest. `qualify.sh` retains the exact selected tests;
`qualification-env.sh` retains the actual deployment environment. For that
qualification, stage the built executables/library, default inventory and identity
header beside the source tests as in the recorded deployment, then call the
environment wrapper with that source root and `bash /path/to/qualify.sh` inside
a finite `srun`. Adapt paths, node and GRES together elsewhere. No release assets
or expiring external archives are needed.

For the sanitizer replay, restore `qualification.sanitizers.input_base64` from
the evidence record to `sanitizer-fixture.bin` (verify `input_sha256`), then run
`sanitize.sh` through the same environment wrapper inside a finite `srun`.
