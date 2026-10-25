# Final HF/DFT README refresh (2026-10-25)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **128 populations / 512 complete energy-and-force calls** were measured
at frozen source `7a7320099d7eb03f022e6ee55e06ae3408bc091f`, following the
**#2786 default-only framework**. Eight methods, Direct/DF and 3/12/48/96 atoms
each have one fresh process per engine, with one cold and three frozen-density
warm calls. HF is included throughout. This refresh promotes no further
optimization. The original 2×4 format, colors and cold/warm encoding are
preserved; CCSD(T)/CBS is unchanged.

## Observations

| Mode | Phase | GenerativeQC wins / measured pairs |
| --- | --- | ---: |
| direct | cold | 29 / 32 |
| direct | warm | 32 / 32 |
| df | cold | 30 / 32 |
| df | warm | 32 / 32 |

GenerativeQC is faster in **64/64 measured warm cases** and **59/64 cold cases**.
Direct ωB97M-V96 warm is **4.19×** faster in this cohort.
The five observed cold gaps remain visible (seconds per complete E+F call):

| Case | GenerativeQC cold | GPU4PySCF cold |
| --- | ---: | ---: |
| direct HF96 | 18.015 | 11.828 |
| direct r²SCAN-3c96 | 33.432 | 31.773 |
| direct ωB97M-V48 | 104.877 | 96.647 |
| df HF96 | 21.286 | 20.564 |
| df ωB97M-V48 | 103.481 | 89.683 |

DF PBE0-96 is 26.895 s versus 27.300 s here. Its previous sample had the
opposite ordering; this small difference is not a demonstrated optimization
or a statistical significance claim. No repeated-process confidence interval
or universal superiority is claimed. All three warm samples and ranges remain
in the tables and full-precision receipts.

## Accuracy and measurement scope

Both engines have **58 qualified / 6 failed / 0 unverified** populations.
Every complete point is displayed, including all independent default-grid or
fitting failures. The gates remain energy ≤1e-5 Eh/atom, maximum force
≤1e-4 Eh/Bohr and RMS force ≤3e-5 Eh/Bohr, with quarter-gate reference-grid
stability and CPU/GPU checks. Original oracle producer identities are preserved;
exact geometry, orbital basis, representation, charge, spin and auxiliary
inputs are audited before reuse. This reuses numerical references, not old
timing samples. Qualification here covers sampled static geometries.

The first six scientific pairs and the remaining 58 pairs use the same frozen
source, runner and native library. All six initial clean pairs are retained
regardless of timing. Both engines run consecutively on the same allocated card
within each pair, with alternating first-arm order. Thirteen finite Slurm
allocations cover the cohort. Existing explicit spatial-order and FP64-validation
execution controls are disclosed in [conditions](conditions.md). The previously
qualified implementation and its separate displaced-geometry checks remain in
[the preceding campaign](../r2scan-endpoints-20261025/README.md).

Six separate intrusive profiles are retained in `diagnostics.json.xz`; none
supplies a figure timing. The unexplained DF ωB97M-V48 clean/profile cold-time
difference remains a diagnostic limitation. No further optimization was promoted.

## Reproduction and validation

```bash
PYTHONPATH=python:. python benchmarks/results/readme-refresh-20261025/reproduce.py \
  --output .artifacts/readme-refresh-replay --verify --check-readme
```

Replay recomputes every independent accuracy gate, timing statistic, work table
and win count, checks publication identities, and redraws both SVGs byte for
byte. `--check-readme` also verifies the current homepage images and unchanged
CBS checksum; omit it when replaying this historical cohort after a later
refresh. The previous campaign likewise verifies its own original figure hashes.

The source audit checks **12,975 files** in both the frozen measurement tree and
actual build inputs with no mismatch. `provenance.json.xz` retains source/core/AOT
identities, build configuration, verified sccache 0.16.0 and shared-cache
before/after statistics; raw compiler streams, traces and binaries remain in
ignored artifacts. `jobs.json.xz` retains all commands, allocations and successful
worker exits. All **122 offline runner, reducer, qualification and output-retention
tests** pass. This documentation refresh adds no new sanitizer or moved-geometry
claim; those implementation checks retain their original source identities.
