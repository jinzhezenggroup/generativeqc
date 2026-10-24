# Small direct RHF cold source reuse, 2026-10-23

Three alternating fresh-process pairs on n1 / RTX 5090, CUDA 12.9.1,
Release sm120, FP64 RHF/def2-SVP spherical, default Hückel seed. Compare the same
binary with `GENERATIVEQC_RHF_RESIDENT_VALUES=0` versus genuinely unset selection.
Cold includes constructor, preparation, SCF, analytic forces and host export.
Each process then makes three frozen-density warm calls. No tracing in this
population; warm medians pool all nine calls per arm.

| System | Public / Cartesian AOs | Disabled cold (s) | Default cold (s) | Reduction | Disabled / default warm (s) |
|---|---:|---:|---:|---:|---:|
| 12-atom water cluster | 96 / 100 | 1.75543 | 1.55980 | 11.1% | 0.11679 / 0.11712 |
| 24-atom water cluster, early fallback | 192 / 200 | 2.84449 | 2.81269 | no source admitted | 0.25361 / 0.25440 |
| Benzene | 114 / 120 | 2.41090 | 2.11356 | 12.3% | 0.21755 / 0.21731 |

These are limited static-geometry observations, not universal speedups or a
statistical confidence claim. Both water cases execute 10 SCF iterations / 12
physical Focks; benzene executes 9 / 11. Every warm call executes one Fock.
The optimization removes repeated integral production, not convergence work.

Separate traced default diagnostics count 8,266,864 / 25,024,440 resident FP64
values for water12 / benzene, 66,134,912 / 200,195,520 value bytes, and
82,355,764 / 222,845,154 total additional numeric capacity bytes. They feed
12 / 11 Fock actions with zero remaining streamed source values per action.
Derivative integrals still use the existing force path. Traced 3/48/96-atom
fallback receipts contain no resident-preparation scope; their disabled/default
cold times are 1.49083/1.52504, 7.00165/7.08227 and 22.55133/22.54920 seconds.
Instrumented observations are excluded from the clean timing population.

## Numerical and execution qualification

All 104 final endpoints (72 clean, 24 fallback, four diagnostic, four memcheck)
pass independent gates: energy 1e-5 Eh/atom, maximum force error 1e-4 Eh/Bohr,
force RMS 3e-5 Eh/Bohr. Across the final clean population, observed maxima are
1.81e-13 Eh/atom, 6.88e-8 Eh/Bohr and 2.52e-8 Eh/Bohr, respectively.
Water12, water24 and benzene use fresh CPU PySCF 2.14.0 full-density RHF oracles,
energy tolerance 1e-11, orbital-gradient tolerance 1e-8, one math thread;
achieved gradient norms are below 3e-9. Larger fallback records use the retained
[tight CPU/GPU reference campaign](../readme-final-20261022/README.md) with its
explicit reference-reuse contract. Exact geometries and all final force arrays
are retained. Oracles are outside production and timing.

Real-device regression: spherical/cartesian cold, frozen warm, moved and moved
replay, one-byte optional-capacity refusal, and existing UHF forces: **3 passed**.
These use stricter 1e-8 Eh / 1e-7 Eh/Bohr integration gates. Physical-reference
resident exports: **3 passed**. Compute Sanitizer memcheck: **zero errors** on
the complete default water12 cold plus three warm endpoints. Host policy,
lifetime and structure checks: **175 passed**.

The first forced prototype had correct energies but incorrect cold forces
(roughly 15 Eh/Bohr error): resident Fock bypassed Cartesian derivative metadata.
It is rejected, retained separately, and fixed by producing metadata from the
selected final density. Correct forced broad-domain trials still regressed:
48 atoms 7.00814 to 8.48815 s; 96 atoms 22.49069 to 27.58572 s. Repeated water24
calibration also regressed. These populations motivated the small-domain guard;
they are never pooled with final default timing. Early CPU-oracle convergence
failures and a reference-probe CUDA include-path failure remain in archived
logs; subsequent passing runs used corrected harness controls/paths.

## Provenance and replay

[evidence.json.xz](evidence.json.xz) contains 40 complete receipts, independent
CPU arrays, compact summaries of 17 other exploratory receipts, selected trace
events, complete scope-name inventories, raw file hashes, original scripts,
test logs and source patches. All final receipts retain full numerical arrays,
settings, work, device and environment metadata. Large raw artifacts remain in
ignored `.artifacts/` and `/data/jzzeng/direct-cold-resident-20261023` on n1.

The measured source is **base `4a783a27e104e9392a7227b7efca79abcfba60ae` plus
archived `default.patch`**, SHA256
`ad2a9d9b2d9a279c540e18e798a10de6a4ba1acef914dc59e46664971b6e68c9`.
The raw `source_revision` field denotes that base, not the modified build.
Final library SHA256:
`2383ceb2c19805382c25a7462bc08850e6a8680a6386d38f7e3efcec18736612`.
All 2,404 production inputs matched the frozen manifest; all 553 actual compiler
commands used verified sccache 0.16.0. Before/after shared-cache stats are retained
and do not imply exclusive build hit counts. Pilot patches and source mapping
are separate; both pilot library hashes were not independently retained.

```bash
PYTHONPATH=python:. python benchmarks/results/direct-cold-resident-20261023/replay.py \
  --output .artifacts/direct-cold-replay.json --extract .artifacts/direct-cold-scripts
```

Replay recomputes every retained full-array gate and the clean medians, including
the expected rejected v1 force result. It performs no GPU work and imposes no
permanent source-fingerprint CI test. To reproduce measurements, reconstruct the
archived base plus patch in an isolated checkout and adapt the extracted scripts'
local paths. `build.sh` selects/verifies sccache and records cache stats;
`measure-default.sh` preserves pair order and exact geometry via `calibrate.py`.
The direct arm uses built-in def2-SVP; its `--inputs` path is unused. Execute GPU
scripts only inside a finite allocation, preserving Slurm visibility, for example
on n1:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 bash measure-default.sh
```

No later-master performance qualification is implied. The accompanying
water24 historical JSON-to-gzip migration is byte-identical and frees evidence
space without changing measurements or increasing the repository budget.
