# Current-source exact Direct force recurrence/contraction reuse

Clean scientific exports: master `5348b9e66` after #2531/#2532/#2540 and
incumbent `4bdffe3ce` (#2541: preweighted order-six Boys roots and existing
order-seven Hermite-axis convolution). Later test/docs commits preserve native
inputs. These are the campaign-cutoff revisions, not later master changes.
1753-input native inventories match loaded images; hashes and environments
are retained. Both arithmetic improvements are automatic in their admitted
generated consumers on the branch, not an opt-in. No PR is merged here.

This supersedes the frozen-composition evidence at this working-tree path.
The original full bundle, including its independent arrays, rejected DAG patch
and detailed historical source-algebra counters, remains at immutable commit
`6369de44fe1f30759b3d07c8f267f4a02d0d70dd`, same path. No historical result is
relabeled as current. Only this PR's own publication is replaced; no unrelated
evidence, storage policy, Release, tag or external archive is changed.

## Complete E+F

RTX5090, CUDA12.9.1, official Release/sm_120, strict FP64, portable=0,
fast_compile=0. Canonical def2-mTZVPP,544/1088 spherical AOs, unchanged moving
48×16×32 Becke grid/atom, original screening1e-12/SCF convergence. HF/PBE0:
def2-SVP. Finite Slurm allocations preserve device visibility. Primary96 uses
the same n4 GPU; primary48 and separate diagnostic profiles each use a matched
n1 allocation. Diagnostic stationary caches are isolated from clean endpoints.
Every population retains cold + five warm + moved + five moved-warm calls and
the unchanged1800-second process timeout. Times are seconds, warm medians:

| Case | Phase | Master s | Incumbent s |
| --- | --- | ---: | ---: |
| r2scan-3c-48 | cold | 75.817234 | 69.934962 |
| r2scan-3c-48 | warm | 21.438233 | 15.226590 |
| r2scan-3c-48 | moved | 56.877101 | 50.849639 |
| r2scan-3c-48 | moved-warm | 21.439160 | 15.253462 |
| r2scan-3c-96 | cold | 411.282020 | 386.116851 |
| r2scan-3c-96 | warm | 90.798641 | 65.477918 |
| r2scan-3c-96 | moved | 260.105874 | 234.961981 |
| r2scan-3c-96 | moved-warm | 90.800574 | 65.707564 |
| pbe0-48 | cold | 23.735617 | 23.372648 |
| pbe0-48 | warm | 4.559350 | 4.258826 |
| pbe0-48 | moved | 22.465948 | 22.088725 |
| pbe0-48 | moved-warm | 4.575266 | 4.257219 |
| hf-48 | cold | 11.001145 | 10.937647 |
| hf-48 | warm | 0.742660 | 0.741202 |
| hf-48 | moved | 7.206905 | 7.204627 |
| hf-48 | moved-warm | 0.745415 | 0.742078 |

incumbent/master96 warm/moved-warm reductions: 27.89%/27.64%.
Incumbent remains 2.46× historical GPU4PySCF26.59s; no fresh competitor measurement.

Actual cold/moved iteration/build counts (every warm call1/1):

| Case | Master | Incumbent |
| --- | --- | --- |
| r2scan-3c-48 | 19/19; 13/13 | 19/19; 13/13 |
| r2scan-3c-96 | 24/24; 13/13 | 24/24; 13/13 |
| pbe0-48 | 17/18; 12/12 | 17/18; 12/12 |
| hf-48 | 21/22; 16/17 | 21/22; 16/17 |

All 96 primary calls complete and pass independent gates1e-8Eh/1e-7Eh/Bohr. Maximum errors: 3.8653525e-11Eh and 6.0142558e-11Eh/Bohr. Every trajectory/count remains explicit, including any moved convergence difference.

## Diagnostic profile

Separate intrusive warm captures, not substituted for clean endpoint times:

| Device group | Master s | Incumbent s |
| --- | ---: | ---: |
| direct_j | 8.402367 | 8.353921 |
| cuda_ks_xc_shared | 4.538393 | 4.550966 |
| direct_force_generated_low | 4.678396 | 4.678237 |
| direct_force_order6_spatial | 21.968210 | 17.549077 |
| direct_force_order7_cooperative | 28.218229 | 7.375439 |
| direct_force_order8_materialized | 5.809917 | 5.824592 |

`profile-summary.json.xz` preserves mutually exclusive host owners and separate
device windows for SCF J/K, XC/AO/grid, order6/7/8 J′, other force/Becke,
snapshot/decode/final validation, preparation and synchronization. Device times
are not added to host waits. Resource rows retain actual register, launch and
shared-memory bounds, not achieved occupancy or full process peak memory.

master entered order6/7/8 primitive products: 2,400,504,912/1,250,544/95,976.
incumbent entered order6/7/8 primitive products: 2,400,504,912/1,250,544/95,976.

All other census counters are retained, including order-seven bra/ket/Coulomb
preparations and contractions. Unexposed counters remain null. No machine
instruction/FLOP reduction is fabricated: prior Nsight Compute reports
ERR_NVGPUCTRPERM. The immutable frozen evidence retains entered source-algebra
counts and resource/sanitizer qualification for these unchanged generated
arithmetic components; current-source native profiles and independent complete
endpoint gates are retained here.

## Validation and replay

588 focused host checks pass after master synchronization, including independent
70-digit Wick sums and correlated-Gaussian integral/derivative tests. Six actual-
linked independent Libcint GPU cases pass both spins, Cartesian/spherical and
separate/combined J′/K′. Compiler structure541 modules/0 errors. Overlapping
suites are not summed. Existing bounded fallback, exact Direct semantics,
FP64, screening, symmetry/signs and force accumulation remain unchanged.

Two follow-up primitive-reuse probes are rejected: warp-local Boys/prefactor
sharing cuts actual radial preparations32-fold but slightly regresses both
partitions; subgroup Wick-term sharing gives no consistent useful gain. Both
have exact zero derivative differences for full/partial/singleton masks;
neither is enabled. Different AO layouts are not compared across trials.
The accurate but slower scalar DAG and earlier cooperative-six/page-width
experiments remain rejected. A separate response-factoring candidate is tracked
in #2549; no result for that additional code is included in this two-arm table.

Every primary E/F array and both independent geometry oracles are retained
losslessly in standard `numeric-arrays.npy.xz`; `numeric-index.json.xz` keeps
protocols, shapes, offsets, per-call timings/counts/gates and reference metadata.
`python verify-arrays.py` recomputes all gates; `--restore-references` reconstructs
oracle inputs. Other records use ordinary JSON/XZ. Detailed native histories,
inventories, build/cache/probe/Nsight receipts remain hash-bound under ignored
`.artifacts/direct-reuse-next/` and remote
`/data/jzzeng/gqc-direct-next-20261016` on n1/n4. This is scoped numerical and
matched endpoint acceptance, not formal global interleaved performance promotion.

Build exact exports with `SOURCE_ROOT=... bash build.sh`, verified ccache4.5.1,
explicit CXX/CUDA launchers and checkout-root CCACHE_BASEDIR; shared cache is
retained. Place source under `ARM/source`, libraries/manifests under
`ARM/runtime`; restore references and decompress `def2-svp.json.gz`. Adapt the
cluster Python wrappers retained in the immutable frozen bundle. HF's wrapper
retains already-returned valid native physical-build counters after timing.

```bash
GQC_DIRECT_REUSE_ROOT=/path/to/campaign srun --partition=main \
  --gres=gpu:5090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=28G \
  --time=00:35:00 bash run-gpu.sh incumbent r2scan-3c 96 endpoint
```

Use `profile` for diagnostics (remove local `defer-profiles` scheduling marker),
`tests` for Libcint checks and `counted` for HF. No independent oracle runs in
production. Next: generated contraction/preparation arithmetic reuse, not
CTA/page tuning. Historical GPU4PySCF26.59s is a target, not a fresh measurement.
