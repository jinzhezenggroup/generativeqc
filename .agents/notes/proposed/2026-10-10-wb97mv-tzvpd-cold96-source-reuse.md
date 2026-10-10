# Decision: prioritize bounded canonical component reuse for cold WB97M-V/TZVPD

Status: proposed; no production optimization or new default qualified
Date: 2026-10-10

Follow-up: [indexed canonical qualification](../implemented/performance/2026-10-10-indexed-canonical-pair-materialization.md)
and [automatic admission](../implemented/performance/2026-10-10-automatic-canonical-pair-reuse.md)
implement this route. This proposal preserves the original investigation and
its incomplete 96-atom endpoint evidence.

## Problem and revision boundary

The request is to profile the latest master at 96 atoms with the OMol25
WB97M-V basis, select the next optimization route, and repair a performance
bug if one is demonstrated. The local checkout contains unrelated user work
and is not the benchmark source.

The first frozen master is `15bc697009d191a88120607e0e50a15561d35d61`.
A later fetch finds `cd0eb059fd1d0ba205c7db160e50c1145cb8488e`, adding
PR #2190's optional ordered incremental DIIS Gram reduction. The canonical
direct J/K source, source-contraction compiler, and public force path are
unchanged between these revisions. DIIS options are unset in these runs;
the added reducer is not on the first-Fock path. Do not relabel the earlier
binary's measurements as measurements of the later binary.

At the closing fetch, master is `82c166caca181fc0df51b1ff9378b972e3b1eba8`.
Since the measured `cd0eb059f` revision it adds allocation-receipt tooling
(#2187) and opt-in GFN2 work receipts (#2180). Direct J/K kernels, their
scientific compiler, KS solver and force equations are unchanged. Generic
allocation-ledger bookkeeping does change; no complete binary timing or
numerical qualification of `82c166cac` is claimed. The measured source-action
diagnosis remains applicable to its unchanged canonical integral source.

## Scientific and timing contract

- The fixture is the existing README **water-96 proxy**, not an actual sample
  of the OMol25 molecular distribution or the ORCA grid.
- RKS WB97M-V, complete spherical def2-TZVPD, including diffuse and f functions;
  the offline H/O basis is
  `benchmarks/results/omol25-wb97mv-20261001/def2-tzvpd-ho.json`.
- 1,856 public spherical AOs, 2,048 Cartesian source AOs, 768 shells and
  1,184 packed primitives. Shell counts for s/p/d/f are 384/256/96/32;
  primitive counts are 704/352/96/32.
- The unpruned moving grid has 48 radial, 16 polar and 32 azimuth points per
  atom: 2,359,296 total points, with three Becke iterations.
- Exact direct FP64, not DF/COSX; native E/density/screening tolerances are
  1e-12/1e-10/1e-12, maximum 100 iterations, VV10 density threshold 1e-8.
  The reference retains GPU4PySCF incremental Fock and its stock convergence
  policy, with E/gradient tolerances 1e-12/1e-10 and direct screening 1e-14.
- A fresh owner and target density define cold here. Import, CUDA context and
  Calculator construction precede timing, as in the README comparator;
  preparation, SCF, physical forces and synchronized host publication are
  included. Serialization and owner destruction are excluded. Existing
  compiler/JIT caches are reused, not cleared: this is not a forced cold-cache
  installation benchmark.
- The private force admission wrapper supplies an explicit 4 GiB incremental
  host/device budget. It does not change the functional, screening, grid or
  force equations, and it is not a promoted resource-policy change.

All GPU work uses finite Slurm `srun` jobs on node1's RTX 5090s, preserving
the allocated device visibility. The long native and reference arms use
different devices (visible 0 and 2); the isolated split/shared arms use
devices 3 and 4. Other scheduled work on the host is not canceled.

## Evidence and its limits

The compact evidence receipt is
`benchmarks/results/wb97mv-cold96-master-20261010/evidence.json`.
The retained ignored directories contain the raw journals, launch logs,
Nsight reports/SQLite exports, binary checksums, compiler commands and
before/after compiler-cache statistics. Native compilation uses a verified
ccache 4.5.1, Release/sm_120, without clearing the shared cache.

### Clean complete endpoints: right-censored, not finished timings

The first frozen binary's native job 6998 and independent reference job 6997
both reach the 45-minute Slurm limit without publishing a complete 96-atom
energy/force result. Slurm reports TIMEOUT, not convergence or success.
Native preparation is 15.0492 s; the retained stage is `cold.execute`.
Reference preparation is 0.7769 s; the retained stage is `cold.scf`.

Thus complete endpoint times, 96-atom numerical errors, SCF iteration/Fock
counts and the native/reference endpoint ratio remain **null**. The absence
of reference cycle journal entries is not evidence of zero cycles: the first
diagnostic callback was replaced by the work tracker. Later smoke runs
compose the callbacks correctly. No timing or method conclusion depends on
the missing cycle journal.

### Valid partial CUDA captures

A diagnostic CUDA launch interposer records the actual canonical arguments,
fences completed selected launches, and stops profiling only after completion.
The scientific library is unchanged. Nsight Systems 2026.4.1 captures valid
activities; the old unsupported profiler and forcibly terminated in-flight
captures do not. Zero retained CUDA activities in those aborted reports
must not be interpreted as GPU idle time or a CPU scientific fallback.

The first three sequential canonical buckets retain indexed prefix tables:

| Total order / bucket | Indexed row entries | Dense bucket capacity | GPU seconds | Registers/thread |
| --- | ---: | ---: | ---: | ---: |
| 0 | 191,170,858 | 2,732,120,160 | 0.52955 | 192 |
| 1 | 1,823,825,648 | 21,799,895,040 | 5.82157 | 194 |
| 2, sp/sp | 4,351,701,942 | 43,486,691,328 | 10.38862 | 254 |

Their summed kernel duration is 16.73974 s. The count is the measured
geometry-screened row-prefix domain, **not** a primitive-product, radial
evaluation, density-screened-work or FLOP count. This subset is not an angular
class distribution of the complete endpoint. Block size is 64, grid size 4096.

An isolated total-order-5 bucket (pair orders 3 and 2) admits 26,062,583,408
row entries from a dense capacity of 291,939,287,040. Other canonical source
classes are explicitly omitted in this diagnostic, so it does not produce
physical energies or forces. At the same first target density:

- Separate full/LR actions take 553.67302 s combined CUDA duration in two
  launches; maximum single-launch duration is 276.86550 s, 174 registers/thread.
- Existing opt-in shared full/LR takes 429.98020 s in one launch,
  180 registers/thread: approximately 1.288x source-action speedup, or 22.34%
  lower duration. This is a single exploratory comparison on two RTX 5090s,
  **not** a repeated same-device or complete-endpoint qualification.

The full 2,048-Cartesian-AO unique quartet capacity is 2,201,172,312,576.
Do not claim this dense capacity is executed work. Indexed filtering works;
the admitted residual source domain itself is still very large.

A separate all-bucket census reads all 56 prepared row prefixes, then omits
every canonical ERI action and exits before publishing a method result.
The 28 full-range buckets total **163,496,862,218** geometry-admitted row
entries; the 28 LR buckets have identical domains. This is approximately
7.43% of the dense full-range capacity. It is a real prefix-domain census,
but **the 56 canonical J/K value actions perform zero radial/primitive
evaluations**. Preparation still computes bounds and one-electron quantities;
those are not omitted or included in this source-action work count.
Aggregate row entries by total order are:

| Order | Row entries | Order | Row entries |
| --- | ---: | --- | ---: |
| 0 | 191,170,858 | 7 | 21,313,765,872 |
| 1 | 1,823,825,648 | 8 | 10,773,380,288 |
| 2 | 7,590,701,642 | 9 | 4,001,237,056 |
| 3 | 18,533,805,416 | 10 | 1,030,584,184 |
| 4 | 30,332,682,498 | 11 | 170,583,808 |
| 5 | 35,906,046,880 | 12 | 11,125,000 |
| 6 | 31,817,953,068 | | |

The large order-3/4 domains also deserve profiling; candidate-domain fractions
are not time fractions. The three-launch table above covers only the sp/sp
order-2 bucket, not all order-2 work.

Nsight Compute hardware counters are unavailable (`ERR_NVGPUCTRPERM`). Static
register counts are known, but achieved occupancy, local-memory spills and
hardware memory traffic are unmeasured. Sampled host stacks in module/function
lookup followed by the canonical launch and point GPU-utilization observations
do not identify a CPU scientific fallback or prove a specific occupancy cause.

### Independent small numerical smoke

A fresh 3-atom complete E+F endpoint passes the unchanged 1e-8 Eh / 1e-7
Eh/Bohr independent gates: E error 8.3560e-12 Eh and maximum force error
1.8177e-11 Eh/Bohr. Native is 46.8825 s (15 iterations/15 Focks), reference
45.5427 s (35 iterations/36 Focks). Native force preparation alone is
36.1910 s of its 37.4625 s public force time; it must not be attributed to
steady-state derivative kernels. The smoke qualifies the small workflow,
not 96-atom accuracy, a changed default, or matched stopping rules.

### Later-master recheck

The separately built `cd0eb059f` binary uses ccache in all 480 C++/CUDA compile
commands and has SHA256
`2e346139439b2ecc2f41b6f710b9d3332ad784c318b1c11d14245b3d734eaaff`.
All 56 row-prefix buckets exactly match the initial revision's census.
Its three completed canonical CUDA durations are 0.53230/5.80847/10.38480 s,
summing to **16.72557 s**, with the same static register counts.

Latest-binary 3-atom default and existing shared-RSH routes both pass the
independent E/F gates, with 15 iterations and 15 Focks each. Default E/F errors
are 8.3560e-12 Eh / 1.7535e-11 Eh/Bohr; shared errors are 8.3702e-12 Eh /
1.7944e-11 Eh/Bohr. Their complete times are 28.7947/12.3618 s, but force
setup/cache states differ: **do not interpret their ratio as a shared-RSH or
revision speedup**. These are small accuracy smokes, not default promotion
or 96-atom acceptance.

The first later-master capture's profiler-controlled shutdown returns 143
after all three fenced kernels finish and the report is exported. It is a
valid completed-activity capture, not a successful solver run. The reproduction
recipe now stops capture without requesting target shutdown; a separate finite
Slurm job successfully completes both numerical smokes. This is diagnostic
process control, not a production performance fix.

## Decision and next route

No accidental unindexed fallback, duplicated full Coulomb J, forbidden CPU
oracle work, or independently demonstrated production performance bug has
been established. Do not change precision, screening or default routing merely
to manufacture a patch. The retained canonical f-capable value route is
intentional; the current MD-J admission excludes f shells.

Prioritize a **compiler-owned, bounded shell/primitive component-reuse**
experiment within the canonical exact value path, beginning with the measured
order-5 class and then independently profiling orders 6--8. The current scalar
AO-quartet evaluator repeats primitive loops and geometry/recurrence setup for
different Cartesian components of the same shell quartet. This is a source-level
reuse opportunity, not a measured operation-count speedup.

1. Instrument actual canonical candidates, radial evaluations, shell-quartet
   visits, primitive products, shared setup counts and both range-specific
   evaluations before changing their schedule. Preserve each consumer's masks.
2. Reuse common primitive geometry and appropriately bounded Hermite/radial
   intermediates across admitted Cartesian components. Keep FP64 normalization,
   permutation scatter and the existing compiler-owned mathematics. Charge
   classification, buffers, atomics, preparation and resource occupancy. Do not
   materialize the entire admitted AO-quartet domain in resident storage.
3. Treat existing shared full/LR as a separately measurable axis, not a new
   invention. Compare split versus shared on the same device and include
   register/resource effects; do not enable it globally from this one class.
4. Retain the current indexed canonical path when bounded reuse is inadmissible
   or loses. Do not resurrect the through-f bounded default rejected on measured
   complete endpoints, or introduce repeated angular scans without accounting.
5. Gate promotion with independent E/F, complete cold/warm/moved endpoints,
   actual work counters, larger sizes, batching and constrained-memory cases.
   The current 96-atom timeout is not a passed numerical or speedup gate.

## Rejected shortcuts and revisit conditions

Further VV10-only tuning cannot resolve a first-value action of this size;
however this capture does not establish the eventual force/VV10 fraction.
Blind register caps, interpreting an aborted empty profile as a CPU bottleneck,
switching to DF/COSX, thinning the basis/grid, loosening screening, and enabling
an already-rejected bounded route do not address the measured contract.

Revisit the priority if a completed current-source profile reveals a different
dominant path, counter access explains the resource cliff, or a qualified exact
component-reuse implementation reduces complete endpoint work and time.

## References

- [Earlier TZVPD integral profile](2026-10-04-tzvpd-warm-integral-diagnosis.md)
- [Full/LR reuse work contract](2026-10-04-tzvpd-shared-full-lr-work.md)
- [96-atom capacity is not work](2026-10-04-tzvpd-96-ao-capacity.md)
- [Rejected bounded value default](../implemented/performance/2026-10-02-through-f-value-policy.md)
- `src/scf/cuda/direct_jk_kernels.cu::canonical_jk_kernel`
- `python/generativeqc_compiler/integral/direct_source_contraction_cuda.py`
- `docs/maintainer/performance_engineering.md`
