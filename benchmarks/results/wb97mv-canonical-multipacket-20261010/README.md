# WB97M-V bounded canonical multi-packet qualification

Measured base: fetched master `7f342546d887796e6a92a005ba033029ed73ce2a`,
2026-10-10. This extends the merged
[automatic order-five consumer](../wb97mv-canonical-default-20261010/README.md)
to orders six and seven, rather than replacing the mathematical owner.

## Default behavior and protocol

Orders five/six/seven own one/two/three register packets per lane: their complete
s/p/d/f component domains have at most 162/324/648 elements. Admission follows
existing immutable primitive-cache capability and remaining provider budget,
not an environment opt-in, method, basis name or atom-count whitelist. The
original component-level AO Schwarz comparison, full/SR/LR source identity,
positive independent J/K scatter, and Cartesian/spherical projection remain
unchanged. Four bounded row-prefix planes include equal-bucket sorted triangles.
Missing cache, insufficient budget and allocation denial retain the incumbent
bounded consumer. MD-J keeps priority; combined full-J/range-K, fixed screening,
compensation, resident values, other orders and forces keep their existing owners.

The fixture is the README **water proxy**, not an OMol25-distribution sample.
It uses the complete spherical def2-TZVPD basis (including diffuse/f shells),
strict FP64 exact Direct RKS WB97M-V, grid 48 x 16 x 32 per atom, three Becke
iterations, E/density/screen thresholds 1e-12/1e-10/1e-12, 100 maximum cycles,
VV10 density threshold 1e-8 and the existing 4 GiB incremental force budget.
Independent acceptance is 1e-8 Eh / 1e-7 Eh/Bohr.

Both measured native arms and the matched 12-atom reference use one finite
Slurm allocation on node1/RTX 5090, preserving assigned device visibility.
Compilation uses verified ccache 4.5.1 with shared artifact/compiler caches
preserved. Complete cold timing includes prepare, SCF, physical forces,
synchronization and host publication. Imports, context and Calculator
construction precede the clock; teardown/serialization follow it. Allocation
first-use samples are separate from three alternating fresh-owner 3-atom
repeats. This is not an empty-cache compilation benchmark.

## Complete endpoints

| Metric | Master (order five automatic) | Orders five through seven automatic |
| --- | ---: | ---: |
| 3-atom complete E+F median, seconds (three repeats) | 12.742789 | 11.956485 |
| 3-atom iterations / Focks, each repeat | 15 / 15 | 15 / 15 |
| 12-atom complete E+F, seconds (one matched pair) | 532.670114 | 347.357041 |
| 12-atom prepare, seconds | 0.984814 | 0.941837 |
| 12-atom SCF/publication, seconds | 492.808174 | 307.730481 |
| 12-atom physical forces, seconds | 38.877125 | 38.684722 |
| 12-atom iterations / Focks | 21 / 21 | 21 / 21 |

The 3-atom median reduction is **6.17%**; the 12-atom matched reduction is
**34.79%**, not a repeated median. All independent native E/F gates pass.
The 12-atom candidate errors are 5.287e-12 Eh / 5.271e-11 Eh/Bohr. The saving
is in SCF, not fewer iterations, a reduced force domain, or a cheaper force cache.
Runtime work/traffic records retain their scopes; unavailable post-screen
integral quartet counts are not inferred from capacities.

### GPU4PySCF comparison

Installed GPU4PySCF 1.8.1 / PySCF 2.14.0 / CuPy 13.6.0 supplies an independent
exact Direct reference, not native initialization, production integrals or
forces. The stock incremental 12-atom solver did not converge in 100 cycles.
Disabling incremental Fock updates converges at the **unchanged** grid,
thresholds and cycle limit, with an independent engine-local density seed:
45 iterations / 46 Focks, complete E+F **181.018748 seconds** (prepare 0.755571,
SCF/publication 169.173544, forces 11.089633).

The candidate is still **1.92x slower** than this explicitly **full-Fock**
GPU4PySCF reference. This is not a stock-default GPU4PySCF endpoint ratio.
The diagnostic retains its full-Fock override separately from the original
protocol object's incremental-policy field; the actual solver policy, not
that inherited field, governs this comparison. No numerical gate is relaxed,
and no density seed is borrowed between engines.

## Scope and next work

A 96-atom source capture selects first-density actions 9/13/37/41, the
order-six/seven full and LR blocks. Other source classes are deliberately
omitted and execution stops before publishing E/F. Such a capture is **not**
a complete 96-atom SCF, force endpoint or independent physical gate. The
baseline outer domain counts AO quartets; the candidate counts shell quartets.
Those units are not interchangeable or a same-unit work-reduction ratio.

| Selected 96-atom action | Master GPU seconds | Candidate GPU seconds |
| --- | ---: | ---: |
| Order six, full J/K | 128.526039 | 55.596551 |
| Order seven, full J/K | 195.898844 | 69.696141 |
| Order six, LR K | 124.488688 | 56.311395 |
| Order seven, LR K | 189.604234 | 69.756438 |
| Sum of these four actions only | 638.517805 | 251.360523 |

These selected actions improve **2.54x**. All six requested J/K matrices are
finite and agree within 3.320e-14 maximum absolute difference, below the
explicit 1e-8 native/native consistency gate. That is not an independent
96-atom E/F gate. Extra baseline J-buffer dumps for LR K-only actions are
unrequested ABI buffers, not additional J results; only requested channels
are compared. Resource attributes for candidate orders six/seven are
174/198 registers, 7,984/8,944 shared bytes and 512 local bytes, versus 182
registers and 6,200/7,160 local bytes in the corresponding scalar kernels.
Static local-memory attributes are not measured spill traffic.

All 56 first-density 12-atom source actions total 20.838681/12.021160 seconds.
Orders six/seven fall from 5.189928/5.201477 to 0.917266/0.657018 seconds,
while retained orders three/four remain essentially unchanged. Remaining
orders eight/nine consume 5.113911 seconds, about 42.5% of candidate source
time. This decomposition is separate from complete endpoint timings.

A rejected order-three/four prototype regresses their first-density source
times from 0.972357/1.796859 to 2.220674/1.921670 seconds. It is not promoted.
The retained route requires enough component reuse to amortize shared
preparation and the 256-lane schedule. Next examine the remaining higher
angular orders and then the integral force endpoint; do not promote larger
packets without complete ownership, register/local-memory, numerical and
complete-endpoint qualification.

## Acceptance and source identity

Four native CUDA suites pass independent full/SR/LR matrices, RHF/UHF masks,
Cartesian/spherical projection, signed/diffuse components, displaced geometry,
equal-bucket triangles, packet tails, and canonical work checks. They include
an s/f-only case with order six but no order-five seed, incumbent MD-J budget
priority, and real-ledger allocation denial. The allocation fixture now retains
completed derivative metadata if a later optional-index allocation fails.
Four independent PySCF/Libcint runtime cases pass. The pair qualifier checks
production two/three-slot ownership, whole-shell/tail output and exact primitive
recurrence work. All four memcheck/initcheck/synccheck/racecheck tools execute
orders six and seven under one-CTA persistent shared-storage reuse and report
zero errors/hazards.

Integration is additionally built/tested on refreshed master
`b9f344b1d84a419c59fb2833fcaac76919311a65`: four native suites, four Libcint
cases and an independent complete 3-atom E/F gate pass (15 iterations/Focks;
E/F errors 8.356e-12 Eh / 1.837e-11 Eh/Bohr). Its allocation-first-use endpoint
is 27.990426 seconds, **not** a new matched performance arm. The six compiled
patch/test files match the PR tree byte-for-byte. Reported paired timings stay
attached to the frozen measured base, not relabelled as measurements of the
newer binary. Focused source-executing host/source tests report 1,163 passed;
compiler structure checks 511 modules with zero dependency errors. Pre-commit
checks pass.

The canonical/generated-J census fixtures isolate their intended owner with
the existing diagnostic `GENERATIVEQC_DISABLE_MD_J=1`; materialized budget
fixtures restore/test MD-J priority. Complete production endpoints set neither
that diagnostic nor the legacy materialization selector. Sanitizer tracing is
separate from clean timings.

## Retention and reproduction

The [compact machine-readable receipt](evidence.json.gz) records source/library
identities, exact endpoint times, scoped work/traffic, selected-source resource
attributes, matrix hashes, qualification hashes and unavailable 96-atom physical
gates as null. Large raw matrices and traces are not expanded into the Git diff.
The receipt locates the node1 `raw-evidence.tar.gz`, its SHA-256/size and the
complete per-member SHA-256 inventory; every archived regular file is streamed
and verified against that inventory. No originals, caches or release assets
are deleted/published. Measured sources are recoverable from the immutable Git
base plus the qualified patch; the refresh patch separately identifies the
integration build.

Frozen build/environment, endpoint, profiler, reducer and qualification recipes
are retained under
`/data/jzzeng/wb97m-canonical-orders-20261010-7f342546d/input/` on node1, and
locally in ignored `.artifacts/canonical-orders-20261010/`. Reproduce into a
fresh checkout/build and fresh output root, never write new experiments into
the historical archive. Keep scheduler visibility, finite allocation limits,
unchanged science/acceptance settings and verified compiler launchers. For a
configured fresh CUDA build, the focused diagnostic gates run as:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env GENERATIVEQC_DISABLE_MD_J=1 \
  ctest --test-dir build/cuda-release-sm120 --output-on-failure \
  -R 'generativeqc_cuda_(fock_(canonical|materialized)|generated_j_budget|lr_domain)_tests'
```

See the [multi-packet decision](../../../.agents/notes/implemented/performance/2026-10-10-bounded-canonical-multi-packet-reuse.md)
and [current recurrence contract](../../../docs/developer/direct_pair_recurrence.md).
