# Guarded restricted Direct-force orbit contraction and tuning coverage

Status: implemented coefficient fast path; endpoint performance qualification incomplete
Date: 2026-11-02

## Scope and source

Investigate fully Direct energy plus analytic forces independently of DF and
DF J/K with exact long-range exchange. The baseline is master `cd2b4514b`,
scientific identity
`4b8ab33e04c4347589fdaa232c6eb3a3e8a079c5385d827f0a39dea564a01f51`.
It includes #2902 and excludes the pending #2900 and #2932 changes.
The earlier fitted-LR panel/Rys work is a separate follow-up PR, #2933.

## Actual autotune coverage

`generativeqc-tune` accepts RHF/UHF workloads. Its worker requests
`shell_class_profiling`; native KS batches explicitly reject these HF-specific
profiling flags. Extending the CLI method choices alone would not make Direct
DFT tuning work.

More importantly, `launch_bounded_shell_energy_derivative` passes an empty
generated-force mask into the bounded scaled derivative consumer. The existing
HF AOT force launch ABI has no general J/K coefficients or separate-source
publication contract. Enabling its mask without adding the correct consumer
would change the scientific calculation. A production HF schedule winner is
therefore not evidence that this DFT force path was tuned.

The default KS SCF/force tile policy currently admits fitted RKS only. Ordinary
Direct RKS retains 256 points even on the same device and grid. Explicit public
tiles can investigate that independent scheduling dimension without changing
the method, precision, grid or screening.

## First A100 endpoint and diagnostic population

LSF 224875, A100-PCIE-40GB, README water-96, default grids and method-specific
bases, strict FP64. Each clean arm has one constructor/prepare/SCF/force/export
cold call and three same-owner replays of its own frozen cold density. Native
oracle imports are prohibited. The production library has all native stationary
bindings; a build with only the WB97M-V binding is not interchangeable for these
comparisons. All source/library/helper hashes and scheduler visibility are kept
in `.artifacts/direct-autotune-20261102/`.

| Fully Direct method | Default cold / warm (s) | Force tile 1024 cold / warm (s) | GPU4PySCF Direct cold / warm (s) |
| --- | ---: | ---: | ---: |
| r2SCAN-3c | 22.1039 / 12.4055 | 20.9681 / 11.2709 | 15.0120 / 5.6743 |
| PBE0 | 16.7494 / 6.4927 | 15.8342 / 5.4806 | 11.3097 / 3.2626 |

Native cold/warm iteration counts are 10/1 for r2SCAN-3c and 9/1 for PBE0;
GPU4PySCF reports 11/1 and 9/1. The engines' stopping rules differ. All sample
pairs pass README observable gates. Maximum reference force discrepancies are
1.35e-6 and 4.55e-7 Eh/Bohr, respectively. The force-tile change in r2SCAN-3c
agrees with the default to 1.06e-11 Eh/Bohr and 9.48e-15 Eh/atom. This initial
single-pair sample is not default-promotion or grid-convergence evidence.

Separate instrumented owners show r2SCAN-3c Direct J/K force time of 7.94 s:
low orders through five take 4.98 s, order six 2.21 s, seven 0.393 s and eight
0.344 s. PBE0 Direct force time is 2.55 s, dominated by low orders at 2.22 s.
Increasing the grid tile leaves these derivative times unchanged. Inclusive
stationary-force and Direct-force scopes must not be added together.

Two bootstrap attempts failed before endpoint execution: a transfer filename
containing digits violated the task server's filename allowlist; a subsequent
script exported the private newer-glibc search path to system utilities. The
successful runner scopes that library path to Python only. These failures are
retained and are not timing observations.

## Next acceptance boundary

Search SCF and force tiles independently on complete endpoints, then address
the measured scaled J/K force consumer with independent integral-derivative
oracles and source-specific work counts. Preserve arbitrary exchange fractions,
pure J, separate channel exports, spin contracts, bounded scratch and runtime
omega behavior. Do not substitute HF kernels, relaxed precision, or DF for
fully Direct. New defaults need paired cold/warm and changed-geometry evidence,
larger-size/resource controls and device-specific acceptance; an A100 starting
schedule is not an all-GPU optimum.


## Implemented coefficient contraction

The generic scaled Direct-force consumer previously enumerated the eight ERI
permutations even for restricted symmetric densities, although exact-class HF
workers already use the closed orbit identity. The compiler emitter now tries
that identity for restricted callers without a directional seed: one Coulomb
product and two exchange products, multiplied by the canonical orbit stabilizer.
It checks every used transpose pair for exact equality; mathematical symmetry
alone is insufficient because Cartesian projection can introduce roundoff
asymmetry. Canonical pair order and bounded finite factors are also required.

Nonzero factors must have magnitude in `[2^-128, 2^128]`. This conservative
window keeps the reordered products away from overflow and underflow. Zero
factors are admitted, and disabled sources do not read unused density entries.
Noncanonical, asymmetric, seeded, unrestricted, extreme and subnormal inputs
retain the original permutation consumer. This changes neither integral
formulas, screening, precision, source publication, scratch capacity nor device
selection. It does not enable HF force masks for DFT or change batch defaults.

The independent host test deduplicates eight explicit index tuples with a set
and accumulates in long double. It covers canonical coincidences, signed J/K
fractions, offsets, inactive sources, unrestricted and directional calls, plus
bitwise preservation of the general asymmetric/extreme fallback. The relevant
host suite passed 32 tests; the final offset test also passed separately.
Compiler structure checking passed for 583 modules.

Candidate scientific identity:
`13f12658f4d7821c246cf95121b554e28127edc74297a3a72834932ce0bb328f`.
The source archive is retained locally with SHA256
`b3c5591915dc96dd09a9df82df041ba296e8c13f915108f055259d1f5ac3ec3c`.
At handoff this candidate has only the limited 5090 endpoint observation below;
it has not qualified all methods, changed geometries, larger sizes or other GPUs.

## Completed default-grid batch investigation

These are fully Direct README water-96 endpoints on baseline `cd2b4514b`, not
#2900, #2932, #2933 or the coefficient candidate. Cold includes construction,
preparation, SCF, analytic forces and host export; warm is the median of three
same-owner calls using that owner's frozen cold density. Strict FP64 and stock
method grids/bases are retained. Different engines keep their stopping rules.
Every recorded sample passed the existing default-grid observable gates. These
static-geometry observations do not establish dense-grid accuracy qualification
or a general production ranking.

| Method | A100 master cold / warm (s) | A100 GPU4PySCF Direct (s) | 5090 master cold / warm (s) | 5090 GPU4PySCF Direct (s) |
| --- | ---: | ---: | ---: | ---: |
| PBE | 10.1211 / 4.3252 | 8.9891 / 2.6573 | 13.1845 / 5.2756 | 16.0630 / 5.5674 |
| PBE0 | 16.7494 / 6.4927 | 11.3097 / 3.2626 | 18.8355 / 7.2485 | 20.3292 / 6.6962 |
| B3LYP | 16.9169 / 6.5209 | 12.1884 / 3.3994 | 19.3076 / 7.4999 | 21.4385 / 6.6943 |
| r2SCAN | 11.2930 / 4.7785 | 11.1463 / 3.4854 | 15.5928 / 6.3599 | 18.9907 / 7.5244 |
| r2SCAN-3c | 22.1039 / 12.4055 | 15.0120 / 5.6743 | 31.2006 / 14.3237 | 32.7794 / 14.8220 |
| M06-2X | 19.5664 / 6.7311 | 14.3792 / 4.1105 | 23.1089 / 7.9150 | 25.0608 / 8.6568 |
| WB97M-V | 163.9202 / 21.7467 | 88.5699 / 28.5905 | 247.9636 / 18.8878 | 371.2877 / 119.5902 |
| HF | 17.7083 / 2.4141 | 5.9720 / 1.5054 | 18.0503 / 2.9262 | 11.8432 / 3.3923 |

A100 WB97M-V uses 28 cold iterations versus GPU4PySCF's 10; both warm calls
use one. This is a solver-work difference, not evidence of a hardware limit.
A100 retains portable schedules; the SM120 image includes measured HF profiles.
Neither fact means the generic DFT force consumer is covered by HF autotuning.

Explicit force tile 1024 improves all seven measured DFT cases on both devices.
On 5090 the seven cold/warm populations beat GPU4PySCF in this sweep, including
PBE0 at 17.9667/6.4001 s and B3LYP at 18.1593/6.4501 s. A100 still has large
remaining gaps: PBE0 is 15.8342/5.4806 s and r2SCAN-3c is 20.9681/11.2709 s.
Joint SCF/force 1024 gives 14.9764/5.4194 s and 19.8473/11.2206 s respectively;
force 2048 is worse. These public-option sweeps are not an integrated DFT CLI
autotuner. No tile default is changed by this patch.

Evidence remains in `.artifacts/direct-autotune-20261102/`: A100 jobs 224875,
224885, 224903 and 224910; Slurm jobs 15695, 15696 and 15699; all-sample
observable summaries, iteration lists, original JSON/NPZ populations and
separate event/work traces. Concurrent 5090 jobs used distinct physical GPUs.

## Rejected angular partitioning

A separate experiment split low orders 0-3, 4 and 5 into separate kernels.
Independent Libcint checks passed on both GPUs. A100 warm r2SCAN-3c improved
12.4081 to 11.8916 s and PBE0 6.4942 to 6.3332 s; 5090 changes were negligible.
However, candidate quartet scans increased 2.5-2.8 times without changing admitted
or primitive work. Registers remained at 255 despite the lower stack size.
That source change was removed; it is not part of this implementation.
`low-partitions.patch`, Slurm 15697 and A100 224912 preserve the rejected
experiment. Native event tracing remains usable; Nsight hardware counters were
unavailable (`ERR_NVGPUCTRPERM`), so no counter-based conclusion is made.

## Handoff validation limits

A100 build 224915 failed before producing a candidate image: CMake selected the
system Python 3.10 and MP2 generation imported `typing.Self` through the runtime,
which requires a newer interpreter. This is a build-environment failure, not a
numerical result. The dependent GPU job 224918 was cancelled. The failure archive
is retained as `build-224915.tar.gz`; no A100 candidate speedup is claimed.

The user requested an immediate PR and stop, so broad additional tuning,
all-method qualification, default promotion and the build-environment follow-up
are deferred. Resume only with a new task. Any further performance claim needs
paired source-bound endpoint results and unchanged semantic work counts.


## Final bounded 5090 validation

The full normal SM120 Release build completed successfully. Slurm job 15700
passed six independent Libcint derivative checks (PBE RKS, PBE0 RKS/UKS, each
Cartesian and spherical). To finish the requested handoff, the endpoint matrix
was reduced before execution to PBE0 with unchanged default batches, master
versus candidate, one cold plus five warm calls and separate diagnostic owners.

| PBE0 fully Direct, water-96 | Cold (s) | Warm median (s) | Warm min-max (s) |
| --- | ---: | ---: | ---: |
| Frozen master `cd2b4514b` | 18.768262 | 7.223582 | 7.220597-7.247139 |
| Guarded orbit candidate | 18.639202 | 7.094968 | 7.082448-7.104988 |

The sampled reduction is 0.69% cold and 1.78% warm. Both owners take nine cold
SCF iterations/Fock builds and one per warm call. Both separate work-census
receipts are exactly equal, including traversal and all available per-class
admission/execution/primitive counters; unreported counters remain null.
Candidate versus master maximum paired force difference is 3.01e-13 Eh/Bohr.
Every candidate sample passes the default-grid gates against the retained
GPU4PySCF PBE0 population (same scientific inputs): maximum energy error
2.80e-12 Eh/atom, force maximum 4.55e-7 and force RMS 1.99e-7 Eh/Bohr.
These are default-grid comparisons, not independent dense-grid qualification.

This small result does not close the GPU4PySCF gap, and no A100 or all-device
performance claim follows. The full result archive is retained locally as
`.artifacts/direct-autotune-20261102/5090-orbit-15700.tar.gz`, SHA256
`db9c976d9e262f1f51bbad0bed3e63eaeb60bbe3867c72cd8f5da1f0f34bd469`.
All pre-commit checks passed. The local GPU job completed successfully, the
A100 dependent job was cancelled, and the task-owned transfer server stopped.

## Follow-up on 2026-11-03

A newly authorized task resumes the deferred A100 build/qualification and
automatic scheduling work. The [guarded A100 automatic-tiling decision](2026-11-03-a100-direct-automatic-tiles.md)
records its separate source identities, controls, retained fallbacks and new
evidence. This follow-up does not reattribute the historical measurements here
to a newer source or turn the earlier handoff into an A100 performance result.
