# Unified default-grid README comparison (2026-10-21)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions and differences](conditions.md)

All eight HF/DFT methods were remeasured in Direct and DF at 3/12/48/96 atoms: **192 populations and 752 complete energy-plus-force calls**. Every timed population uses native source `0fe06f31c6c96182973405b58f5047f3f177fff4` and the same measurement driver. There is one GPU4PySCF curve, native Standard/Fast, and no moved-geometry or CBS rerun.

## Verified observations

Direct ωB97M-V on n1 RTX 5090; seconds for complete E+F, with setup included in cold. Warm is the median of three calls, each from the engine’s frozen cold density. All samples below pass the common independent energy/force gates.

| Atoms | Engine | Cold (s) | Warm median [min–max] (s) | Warm reference/native |
| ---: | --- | ---: | ---: | ---: |
| 48 | GenerativeQC Standard | 88.014 | 7.036 [7.027–7.052] | 4.617× |
| 48 | GenerativeQC Fast | 72.227 | 6.879 [6.815–6.912] | 4.722× |
| 48 | GPU4PySCF | 96.758 | 32.482 [32.473–32.510] | 1.000× |
| 96 | GenerativeQC Standard | 638.889 | 45.180 [45.161–45.203] | 2.609× |
| 96 | GenerativeQC Fast | 306.896 | 40.186 [40.167–40.217] | 2.934× |
| 96 | GPU4PySCF | 365.004 | 117.886 [117.857–117.897] | 1.000× |

These are scoped observations, not all-method superiority or a statistically interleaved cross-device study. For example, Standard M06-2X DF96 warm is 56.974 s versus 15.152 s for the reference. All methods and sizes remain in the figures or the explicit failure inventory; no slower valid point is removed.

## Complete accounting

| Profile | Attempted | Complete | Accuracy qualified | Accuracy failed | Incomplete |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 64 | 62 | 56 | 6 | 2 |
| fast | 64 | 62 | 56 | 6 | 2 |
| gpu4pyscf | 64 | 64 | 58 | 6 | 0 |

**170 populations qualify, 18 fail accuracy, and four remain incomplete.** HF DF fails the common direct-target energy allowance at all four sizes for all three profiles; M06-2X at three atoms fails grid accuracy in both Direct and DF for all three profiles. The default grids, auxiliaries and gates remain unchanged. Raw values and every repeat are retained. A complete cold phase and complete warm population are qualified independently; unqualified phases are gaps in the plots.

| Incomplete populations | Outcome |
| --- | --- |
| Standard/Fast ωB97M-V DF96 | Native prepared-force resource limit; no complete endpoint |
| Standard/Fast r²SCAN-3c DF96 | 25-minute Slurm timeout; no complete endpoint |

All allocations have ended. The two externally terminated producer JSON files retain their original `status: running`; their scheduler exit 143 and time-limit messages are retained in `jobs.json.xz`. They are not still running and are not plotted. Native auxiliary **g** (angular momentum l=4) is now admitted, but generic g response and large DF value/storage limits still affect performance. This is distinct from a grid level or a GPU restriction.

## Fast limitations isolated by actual work

Supported Direct KS warm calls now reuse a strict validated state. The ωB97M-V/96 Fast warm calls above execute one strict Fock and one force evaluation, without an implicit independent endpoint audit. The Fast range-cache repair is included in the measured source.

DF Fast still executes a Standard-threshold FP64 candidate plus a tighter independent audit, with their forces and owner reconstruction charged inside the endpoint. For M06-2X DF96, the first warm call has two candidate and four audit Focks and two force evaluations; Standard uses one Fock and one force evaluation. The full solve census, rather than only primary SCF iterations, is in [work.md](work.md). This fallback is not a speed improvement.

A separate same-build, same-card diagnostic (Slurm 11629) reproduces the r²SCAN-3c Direct96 cold slowdown: Standard 82.284 s and Fast 274.864 s, despite 22 versus 15 Focks. Native preparation is 7.349/7.372 s. Traced Coulomb J totals are **16.801/224.291 s**; XC totals are 27.317/11.741 s. The mixed Coulomb route, rather than preparation or a larger iteration count, dominates the regression. Warm strict J is 0.759 s in both profiles. Both diagnostic endpoints pass the same independent gates; their instrumented times never replace the official samples. `diagnostics.json.xz` retains exact traces, public results, launcher and checks. The remaining mixed-J dispatch and DF audit costs require further production optimization.

## Conditions and provenance

See the [complete condition table](conditions.md), including native stopping residuals, XC/VV10 grids and response, VV10 signed-weight screening, default auxiliary tables, DF metric treatment and the ωB97M-V fitted/exact long-range distinction. Auditing all 752 completed samples confirms the declared default controls and grid point counts. Six active processes were additionally checked for inherited native runtime overrides; only the explicit library path was set.

The frozen build combines master `28d4bf6980092cd34fd7c4036672f4cbb2e8f6ef` with the qualified g-response, VV10 density-domain and Fast range-cache fixes (#2651, #2652, #2653). All 12,047 archived source entries match the executed tree. Core SHA-256: `f89e0b13b8c5e028c33fe087501f65aeb919e5789841599d5feab59840d6dff0`. Build controls: GCC 13.3, CUDA 12.9.1, Release/sm120, FastCompile OFF, verified sccache 0.16.0. Package versions: PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 13.6.0 and NumPy 2.4.6. Per-job GPU visibility, native images and environment records are retained.

Forty original independent references are reused through `reference-reuse.json`, binding their full receipt hashes, original driver hashes and reviewed scientific inputs. Original reference revisions are preserved. Driver input functions, basis tables/loaders and correction owners were checked against the frozen producer. Every sample error, consecutive denser-grid stability check and matching CPU/GPU spot is recomputed. Reuse never changes the energy 1e-5 Eh/atom, maximum force 1e-4 Eh/Bohr, RMS force 3e-5 Eh/Bohr, or quarter-gate reference stability requirements.

The provenance bundle retains source/binary/cache records, executed launchers, configuration checks and reference-input review. The complete compiler-command listing remains at `n1:/data/jzzeng/readme-unified-20261021/build-receipts/compiler-commands.txt`, with its hash and representative cache-wrapped commands retained. Warm variability describes three calls in one process; cold has one observation. Persistent compilation caches are retained. No method-wide Fast or global performance promotion is inferred.

## Offline replay

From the repository root (NumPy and Matplotlib required; no GPU or SCF solve):

```bash
PYTHONPATH=python:. python benchmarks/results/readme-unified-20261021/reproduce.py \
  --output .artifacts/readme-unified-replay
```

This recomputes every acceptance gate and regenerates both 2×4 SVG/PNG figures, the full timing table and the executed-work table. Lossless raw measurements, references, auxiliary inputs, job failures and provenance remain in the adjacent `.json.xz` files.
