# Historical HF/DFT README cohort after the cold endpoint improvements (2026-10-26)

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **128 populations / 512 complete energy-and-force calls** were newly
measured with the **#2786 default-grid framework** and one frozen implementation.
Eight methods, Direct/DF and 3/12/48/96 atoms each have one fresh process per
engine, with one cold and three frozen-density warm calls. HF is included.
The original 2×4 format, colors and cold/warm encoding are preserved;
**all completed accuracy-failing points remain visible**. CCSD(T)/CBS is unchanged.

## Integration scope

This is a retained measurement of the frozen `r2-default-v2` image described
below, not a benchmark of current master. Its then-default Direct HF route used
screened proposals; current source preserves that route only with the explicit
HF opt-in after the stronger force diagnostic failed. The original samples,
figures, source patch, gates and failed records are unchanged. In particular,
these historical HF timings do not establish the restored ordinary-HF default's
accuracy or speed. No integrated-image GPU rerun is claimed.

The homepage labels these exact historical figures in the surrounding heading,
alt text and caption. The SVG bytes and pinned-renderer replay remain unchanged;
new current-source measurements should be published as a separate cohort.

## Observations

| Mode | Phase | GenerativeQC wins / measured pairs |
| --- | --- | ---: |
| direct | cold | 32 / 32 |
| direct | warm | 32 / 32 |
| df | cold | 31 / 32 |
| df | warm | 32 / 32 |

This cohort is faster in **63/64 cold** and **64/64 warm** cases.
Direct ωB97M-V96 warm is **4.21×** faster. These are descriptive latency
observations from one process per engine/case, not a universal performance
claim or a repeated-process confidence interval.

Every slower observation remains in the figures (complete E+F seconds):

| Case | Phase | GenerativeQC | GPU4PySCF |
| --- | --- | ---: | ---: |
| df pbe0 96 | cold | 26.9349 | 26.2575 |

DF PBE0-96 is slower in this fresh single-process cohort, while its independent
three-round default campaign has a faster native cold median. Its native warm
calls also differ from the three-round campaign despite equal iteration counts.
The cause is not established. This complete population is retained, not
replaced by one of the faster qualification samples. The
[six-endpoint evidence](../cold-r2-defaults-20261026/README.md) retains all three
rounds and the distinction between default timing and scientific qualification.

## Accuracy and source identity

GenerativeQC has **58 qualified / 6 failed / 0 unverified** populations;
GPU4PySCF has **58 qualified / 6 failed / 0 unverified**.
The unchanged gates are energy ≤1e-5 Eh/atom, maximum force ≤1e-4 Eh/Bohr
and RMS force ≤3e-5 Eh/Bohr, with quarter-gate reference stability and CPU/GPU
checks. Default-grid and fitting failures are displayed rather than omitted.
References preserve their original producer identities and bytes; exact geometry,
basis/representation, state and auxiliary inputs are checked before reuse.
Numerical references are reused; every timing in these figures is new.

All populations use frozen `r2-default-v2`: base
`42b266917f6676ecf0559fb8965a4e19f60d0fe3` plus the exact `source.patch.xz`.
Its archive SHA-256 is
`e4af728d14fc6279faa146748b65fcc38d87b98dbe5e0ee0233d3df5658b6b71`.
This dirty source is not represented as the unmodified parent commit. The
13,178 archived files match the frozen tree, and all 1,858 compiler inputs
match the actual build tree. All 558 C++/CUDA compile commands invoke verified
sccache 0.16.0. Core/AOT identities and shared-cache statistics are retained.

Seven finite Slurm workers run preassigned scientific pairs consecutively on
the same allocated RTX 5090, alternating first-arm order. No research selector
or profiling control enters these figures. Existing spatial-order and validation
GEMM controls, resource limits and timer boundaries are disclosed in
[conditions](conditions.md). Intrusive work counts and separate 96/120-atom
moved-geometry tests belong to the implementation evidence, not figure timings.
The inherited stronger default-HF force diagnostic failure remains documented
in [its original evidence](../cold-hf-defaults-20261026/README.md).

## Reproduction

```bash
PYTHONPATH=python:. python benchmarks/results/readme-refresh-20261026/reproduce.py \
  --output .artifacts/readme-refresh-20261026-replay --verify --check-readme
```

Install `render-requirements.txt` in an isolated environment before byte replay;
it pins the original Matplotlib 3.11.2 renderer. Replay checks all 128 populations / 512 calls, unchanged scientific inputs,
same-job pairing, source/library identities, independent gates, work and timing
tables, and redraws both figures byte for byte. `--check-readme` additionally
checks the current homepage assets and unchanged CBS checksum; omit it for
historical replay after a later refresh. All **128 offline runner, reducer,
qualification and output-retention tests** pass. Raw compiler output and
binaries remain in ignored artifacts; no external release asset is required.

The unchanged repository storage budget is preserved by indexing the older
r²SCAN exploratory archive and pinning its existing Git recovery revision.
All accepted historical endpoints, reference inputs and ordinary figure replay
inputs remain in the checkout. The original archive restores byte for byte;
its historical figures also replay byte for byte with the original renderer.
