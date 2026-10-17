# WB97M-V: shorten the frozen-VV10 inner solve, not remove VV10

Follow-up: the conservative policy is promoted by
[`wb97mv-vv10-default-20261018`](../wb97mv-vv10-default-20261018/README.md).
This directory remains the original experiment-only evidence; its source,
selector, default-change flags and timings are historical, not measurements
of the production-default binary.

Frozen master: `e4aff7810b0462204831eb5168f19b6e217ef520`, fetched on
2026-10-17; device measurements run on 2026-10-18. Master advanced after the
earlier `3ad2eb791` campaign with a Direct-force implementation change. Use
the new same-revision control, not the earlier 1199-second observation, for
endpoint comparisons.

The fixture is the README water-96 proxy: spherical **def2-SVP / 768 AOs,
Direct RKS, FP64**, grid `48 x 16 x 32` (2,359,296 points), unchanged energy,
density and screening tolerances `1e-12 / 1e-10 / 1e-12`. This is not DF,
def2-TZVPD, an actual OMol25 sample, or a different functional without VV10.

## Policies

The master inner tolerance, using the latest full-target AO maximum residual
`r` and `g = min(1e-9, density_tolerance)`, is:

```text
clip(1e-3 * r, min(1e-12, 0.1 * g), 1e-7)
```

The experiment changes only `src/dft/application_precision.hpp` in an isolated
source archive. `experiment.patch` adds an experiment-only selector:

| Policy | Residual factor | Lower bound | Upper bound |
| --- | ---: | ---: | ---: |
| `baseline` | `1e-3` | `min(1e-12, 0.1*g)` | `1e-7` |
| `factor100` | `1e-2` | unchanged | `1e-5` |
| `factor10` | `1e-1` | unchanged | `1e-5` |
| `factor100-floor` | `1e-2` | `0.1*g` | `1e-5` |
| `factor10-floor` | `1e-1` | `0.1*g` | `1e-5` |

These are proposal/refresh thresholds, **not publication tolerances or
precision changes**. The nine-step initial semilocal solve is unchanged.
Every Fock uses FP64; the full-target energy/density/residual gates, DIIS
history eight, maximum 100 iterations, finite inner limits and ordinary
full-target fallback tail are unchanged. Application-budget behavior is not
modified. This selector does not exist in unpatched master, and no production
source or default in the working checkout is changed.

## Water-96 results

| Policy | Provisional SCF Focks | Full-target SCF Focks | Total iterations / endpoint Focks | SCF VV10 pair sweeps | Complete cold E+F (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Master control | 31 | 5 | 36 / 36 | 4 | 1165.549 |
| Moderate `factor100` | 23 | 5 | 28 / 28 | 4 | 995.149 |
| Looser `factor10` | 20 | 6 | 26 / 26 | 5 | 1023.311 |

The moderate policy changes the provisional-stage lengths from `9+8+7+7`
to `9+4+5+5`: it saves eight frozen-potential steps, leaving the initial
semilocal solve and all five full-target checks intact. Provisional work
falls by 25.8%, total SCF iterations by 22.2%, and the observed complete
endpoint by 14.6%. One exact component-cache hit leaves four pair sweeps
for five full-target SCF Focks; force derivative work is separate and
included in endpoint timing.

Against the same-revision control, the moderate policy's energy, maximum
force and maximum density-element differences are `9.09e-13 Eh`,
`7.80e-12 Eh/Bohr` and `8.50e-12`, respectively. Both pass the unchanged
physical convergence gates. The retained first control endpoint takes
1165.484 s and also passes the paired gates; it is not subtracted from any
timing or used as a hidden warm density.

The looser policy reaches `9+4+2+3+2 = 20` provisional steps, a 35.5%
reduction from master, and 26 total iterations. It needs one additional
full-target check/pair sweep. Its complete endpoint is 12.2% below the control
but 28.161 s slower than the moderate policy in this observation: fewer inner
iterations alone are not the optimization objective. Its paired energy,
maximum force and maximum density-element differences are `4.55e-13 Eh`,
`2.87e-11 Eh/Bohr` and `6.80e-11`; the original physical gates also pass.

Prefer the moderate policy for further qualification of this fixture; retain
the looser result as the iteration/per-expensive-operator tradeoff. No default
is changed. Successful screen and endpoint allocations are Slurm jobs `1120`
and `1121`; the complete endpoint-suite `srun` wait returns zero. This is not
a claim based on historical `sacct` accounting.

## Small-system qualification

All five water-12 E+F policies pass the retained independent GPU4PySCF reference
with `1e-8 Eh / 1e-7 Eh/Bohr` energy/maximum-force gates. Reference data is read
only after native execution and never initializes a target. The moderate
policies reduce provisional Focks from 31 to 25 without increasing the four
SCF VV10 pair sweeps; the looser policies use 22 provisional Focks and five
pair sweeps. Both floor variants have the same iteration/work counts as their
unchanged-floor counterpart at water-12, so only the latter advance to 96.

These sequential screens contain first-use compiler/JIT cache effects.
Their recorded elapsed times are retained, **not treated as a speedup
qualification or used to select on timing alone**. Water-12 reports one extra
post-SCF endpoint Fock; do not equate endpoint Focks with SCF history length.

## Protocol and limits

- Native libraries are built on `n1`, Release, official `sm_120`, fast-compile
  and portable profiles off, using actual verified `sccache 0.16.0` launcher
  commands. Before/after cache statistics are retained; no cache is cleared.
- GPU execution uses node `n4`'s RTX 5090, CUDA 12.9, eight CPU threads, 32 GiB
  host allocation and a common 2 GiB LR-resident ceiling. Every real-device
  call runs through finite `srun`; Slurm device visibility is never replaced.
- Both libraries require glibc 2.39 from their build host. Execution on n4 uses
  a fingerprinted private matching loader/libc/C++ runtime for Python only;
  compiler subprocesses retain the host runtime. No system libraries change.
- Each endpoint is a fresh process, Calculator, owner and native cold guess,
  with no imported or warm density/orbitals, retries or target fallback. The
  preliminary native atomic Focks are counted separately from target work.
- A complete first water-96 endpoint is retained as a shape/cache preflight.
  The comparison endpoints then use fresh cold owners and the existing
  compiler/JIT cache. No priming cost is subtracted from any recorded call.
- Complete E+F timing includes preparation, SCF, forces and synchronized host
  publication; it excludes imports, CUDA-context setup, Calculator construction,
  optional density export, JSON serialization and owner destruction.
- There is no independent water-96 oracle. Paired native energy/force/density
  checks do not establish absolute 96-atom accuracy or the ground state.
  Measurements are single cold observations, not medians or a statistical
  performance qualification, and do not qualify arbitrary molecular systems.

## Retained artifacts and reproduction

`evidence.json` binds the compact results to the raw bytes and acceptance
gates. Full source archives, exact consumer scripts, histories, density/force
arrays, source/library authentication, independent-reference binding,
compiler commands/cache statistics and Slurm receipts remain locally under:

```text
.artifacts/wb97mv-inner-20261017-mastere4aff7810/
```

Matching execution artifacts and libraries remain on `n4` at:

```text
/data/jzzeng/wb97mv-inner-20261017-mastere4aff7810/
```

Build receipts and the compiled source/build also remain at that absolute
path on `n1`. Its canceled queued job and the initial n4 runtime-load failures
produce no scientific endpoints and are not pooled with successful results.

On n4, replay one moderate endpoint with a fresh receipt directory:

```bash
srun --partition=main --nodelist=node4 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=02:15:00 \
  bash -c 'root=/data/jzzeng/wb97mv-inner-20261017-mastere4aff7810; \
    bash "$root/run-policy.sh" factor100 96 "$root/recheck-factor100-job$SLURM_JOB_ID"'
```

Use `baseline` or `factor10` for the other policies and a distinct receipt
prefix. The retained `analyze.py` checks the original campaign's physical,
independent small-system and paired full-system gates without starting a GPU
job. Do not combine additional populations into a single-sample timing claim.
