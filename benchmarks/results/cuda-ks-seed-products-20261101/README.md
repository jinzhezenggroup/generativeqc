# CUDA KS seed-validation products

This campaign measures borrowing the existing FP64 GPU matrix provider for
large CUDA KS seed validation. The seed, scientific equations, quadratures,
stopping criteria and final-state checks are unchanged. Final production source
identity: `b6b63f9063949ed70191c54b4d0f0b7efb788957274eb680c98f4242bbcf74f3`.
Every receipt retains the actual library hash; the runner's Git revision alone
does not identify then-uncommitted changes.

Replay from the repository root with its Python/reference dependencies:

```bash
PYTHONPATH=python:. python benchmarks/results/cuda-ks-seed-products-20261101/report.py --verify
```

The report authenticates the retained original JSON, original parent population
bundle, independent oracles and audited reference-environment reuse contracts.
Parent JSON is read from that hash-bound existing archive without duplication.
It recomputes independent complete energy/full-force gates, matching scientific
tasks, medians and paired native output differences. Native comparisons also
require unchanged target/preliminary work, quadrature and nonlocal work counts.
Positive percentages mean the candidate is slower. Incomplete populations and
instrumented traces are not used in the final timing comparisons.

## Measurement contract

The final cohort contains all seven README functionals at water96, spherical
def2-SVP (mTZVPP for r2SCAN-3c), DF, Standard and stock quadratures. The force tile
is explicitly 1024 points; the SCF tile remains 256. Both engines use one math
thread. Each population includes a complete cold constructor/preparation,
energy/full-force endpoint and three warm endpoints from its own frozen cold
density. GPU4PySCF keeps its stock convergence and initial guess, includes grid
response, and includes D4/gCP for r2SCAN-3c. No automatic tile policy is promoted.

Device work uses finite Slurm allocations on n4/n2, or finite ECNU LSF exclusive
GPU allocations on gpu22. Comparisons are cross-allocation observations, with
the same node, PCI address and UUID where available. A100 UUID is unavailable;
the report verifies node, PCI address and assigned visibility and retains the
distinct LSF IDs. These are not interleaved controls or statistical guarantees.
The final endpoint jobs are A100 224116, RTX 5090 1392 and PRO6000 3821.
The separate RTX 5090 component trace is job 1390; A100's trace is a separate
instrumented population within job 224116.
The final Blackwell allocations explicitly request eight CPUs, matching their
retained parent/GPU4PySCF controls, while keeping all math libraries at one
thread. Jobs 1389/3819 and their pilot parents have an unset CPU-count field;
their native-only pairs are retained separately. The replay check refused to
equate those records with the explicitly eight-CPU reference, prompting the
matched final runs. Scheduler accounting is disabled and expired jobs cannot
be reconstructed through `scontrol`; the original fields are not rewritten.

The formatting-only pilot has source identity
`fdf1573b4b55c9301279c61550c6a940c4da0710967d805319f8484829b88602`.
It is retained separately and supplies no final candidate timing. Parent
controls use source identity
`1714f26e6f6bd4ecac0867f29744ebbf57ae717d3e7ceeda533a7a57365bca8b`.

## A100 final results

Seconds measure complete cold setup/energy/full-force endpoints. Warm ratios
use the median of all three frozen repeats. Positive overhead is slower.

| Method | Native cold (s) | GPU4PySCF cold (s) | Cold overhead | Warm overhead |
| --- | ---: | ---: | ---: | ---: |
| pbe | 8.495481 | 7.950443 | +6.86% | -3.71% |
| pbe0 | 12.773257 | 11.400412 | +12.04% | +1.75% |
| b3lyp | 12.862240 | 9.010638 | +42.75% | +3.18% |
| r2scan | 10.660616 | 10.186962 | +4.65% | -15.77% |
| r2scan-3c | 18.530423 | 12.012433 | +54.26% | +4.53% |
| m06-2x | 16.241185 | 11.547772 | +40.64% | -8.98% |
| wb97m-v | 146.256421 | 93.354615 | +56.67% | -18.53% |

Against the preceding native source, A100 cold improves by 0.93–15.47%.
Warm differences range from -0.10% to +0.66% with identical semantic work.
These sampled differences are not a statistical nonregression proof.

## Blackwell final results

Positive overhead is slower than the original GPU4PySCF population, with
matching eight-CPU reservations and single-threaded math libraries.

| Method | RTX 5090 cold | RTX 5090 warm | PRO6000 cold | PRO6000 warm |
| --- | ---: | ---: | ---: | ---: |
| pbe | -4.77% | -12.85% | -17.35% | -8.86% |
| pbe0 | -6.65% | -43.09% | -12.87% | -41.70% |
| b3lyp | -10.28% | -42.48% | -15.79% | -40.90% |
| r2scan | -10.68% | -31.44% | -17.87% | -28.69% |
| r2scan-3c | +4.63% | -31.31% | -0.83% | -29.17% |
| m06-2x | -9.69% | -48.62% | -14.21% | -47.35% |
| wb97m-v | -38.76% | -54.00% | -36.21% | -63.13% |

RTX 5090 retains only the r2SCAN-3c cold gap (4.63%). PRO6000 is faster in
all sampled cold/warm endpoints, but its r2SCAN-3c cold margin is only 0.83%.

## State, numerical and resource checks

- The native 512-AO sparse oracle checks all transpose combinations, dimension
  refusal and refusal to borrow an active owner's storage. Small MINAO verifies
  the host-product fallback. Host failure injection checks product parity and
  failure propagation; 238 Python initial-guess tests and 70 catalog tests pass.
- The public 768/960-AO tests use an independent PySCF overlap and analytic
  ensemble spectra. A rejected batch import preserves the previously published
  final-state token and both warm snapshots. Frozen replay and changed geometry
  agree with independently prepared native owners at unchanged 1e-8 energy and
  1e-7 maximum-force gates.
- These state tests use an explicit small diagnostic grid, energy/density
  tolerances of 1e-12/1e-10 and DF budgets of 512/1024 MiB respectively. They
  test ownership, not default-grid accuracy. Complete benchmark populations
  retain stock grids and ordinary convergence controls.

Both large cases pass on A100 (224116), RTX 5090 (768 AO in 1388, 960 AO in
1389) and PRO6000 (768 AO in 3818, 960 AO in 3819). The four existing small
public RKS/UKS cases pass on all three devices. Original native probe and test
logs retain each source and allocation binding.

## Rejected MINAO experiment

A separate native-only MINAO diagnostic keeps its own source identity
`79d2d71b16c817744dcc85d5f68b9cc7a69a10a62b8f86ff5330812e0f57f523`.
The public Python constructor still refuses this combination. The retained
adapter installs the diagnostic policy before native planning/preparation;
native admission, bounded storage, strict validation and fallback still apply.
This is not a supported public configuration or a promoted default.

Same-device n5 RTX 5090 jobs 1937/1938 give these complete cold observations:

| Method | MINAO (s) | Default Huckel (s) | Target iterations, MINAO/default |
| --- | ---: | ---: | ---: |
| PBE0 | 29.350035 | 22.341775 | 11/9 |
| r2SCAN-3c | 34.301417 | 26.814275 | 12/10 |
| WB97M-V | 240.274770 | 234.852630 | 28/28 |

All six populations pass the independent endpoint gates. MINAO does not reduce
WB97M-V work and increases PBE0/r2SCAN-3c work. Its one-warm-repeat controls are
retained separately from the main three-repeat production cohort. Earlier
experiments rejected by the Python guard remain failed admission records.

## Remaining WB97M-V work

The separate A100 trace has 23 provisional Focks, five full-target Focks and
four actual VV10 pair sweeps. Its cold KS entry takes 21.925 seconds including
optional exact-range source preparation. Inclusive SCF XC takes 56.051 seconds,
of which 22.620 seconds is nonlocal pair work. These nested scopes must not be
added together or substituted for uninstrumented endpoint timings.

An additional incumbent exact-LR action takes 12.151 seconds after convergence
to prepare the fallback warm-energy baseline. It executes 1,873,221,716 radial
recurrences; the subsequent A100 warm action instead reuses the cached values
and takes 0.714 seconds. This identifies a candidate for deferred fallback
preparation, not permission to remove the energy-baseline contract. Cache
eviction, policy changes, frozen warm densities, checkpoint export and failed
solves must all retain a valid baseline for the selected provider. This PR does
not change that behavior.

## Retention and limits

`receipts.json.xz` contains original populations, settings, full force vectors,
work counters, source/binary/scheduler identities and qualification results.
`provenance.json.xz` contains launch scripts, source patches, cache/build/test
logs and component traces. Large originals remain in ignored
`.artifacts/cuda-ks-seed-products-20261101/` and
`.artifacts/df-minao-cold-start-20261101/`.

The retained failures include budget refusal before seed admission (64 MiB for
96 atoms and 512 MiB for 120 atoms), and changed-versus-fresh force differences
up to 4.794e-7 with ordinary convergence controls. Tightening only the diagnostic
SCF controls retains the force gate. Slurm cancellation of pilot 1384 left its
owned child running; it was explicitly terminated after transient OOM failures
in jobs 1385/1386. Its incomplete WB97M-V candidate is never ranked. Successful
reruns do not overwrite these failed records.

These observations qualify the sampled static default-grid endpoints, not every
geometry/size, Direct/Fast workload or unmeasured GPU. The previous HF/DF fitting
and small M06-2X numerical gaps remain outside this seven-functional water96
cohort. WB97M-V retains the disclosed model distinction: native full-range DF
with exact LR exchange versus GPU4PySCF fitted LR exchange. Independent gates
do not erase that distinction. The overall goal of outperforming GPU4PySCF for
every README functional on every target GPU remains open.

See the [decision note](../../../.agents/notes/implemented/performance/2026-11-01-cuda-ks-seed-products.md)
for scratch lifetime, bounded fallback and discarded alternatives.

## Source integration boundary

After rebasing onto #2908 (`04d316ce0`), the complete source identity was
`8b75da56f4efef0a2b8b2dde33541ac1e0151df5ac5aeb95830820c29cba02d6`.
At that point, native sources, CMake inputs and DFT compiler sources were unchanged
from the measured implementation; incoming edits concerned Libxc provenance/cache
identity and consumer validation.

The subsequent checkpoint-routing fix (`385f90ac8`) changes native sources and
the complete source identity to
`9e99ec9b6f144963f3d87faa02d5c7e553f03a0a21d29818c14700c9e73c32e7`.
It forwards optional products through checkpoint validation for both spin
conventions. The original GPU receipts do not qualify this later routing change.
All receipts retain their original measured identity; these tables are not fresh
measurements of the current checkout. The independent evidence reader is replayed
without changing recorded inputs or gates.
