# Historical HF/DFT README cohort after the DF screening-headroom repair

[Direct figure](../../../assets/readme-direct-defaults.svg) · [DF figure](../../../assets/readme-df-defaults.svg) · [All timings](timings.md) · [Executed work](work.md) · [Conditions](conditions.md)

All **128 populations / 512 complete energy-and-force calls** were freshly
measured on 2026-10-26 with the **#2786 default-grid framework**, including the
DF screening-headroom repair. Eight methods, Direct/DF and 3/12/48/96 atoms
each have one fresh process per engine, one cold and three frozen-density warm
calls. Both axes are linear, with latency starting at zero. The 2×4 layout,
colors and cold/warm encoding are preserved.
**All completed accuracy-failing points remain visible.** CCSD(T)/CBS is unchanged.

## Integration scope

This is a retained measurement of the frozen `46b916ac` source and its qualified
incrementally linked library described below, not a benchmark of current
master. Its then-default Direct HF route used screened proposals; current
source preserves that route only with the explicit HF opt-in after the
stronger force diagnostic failed. The original samples, linear figures,
scientific gates and failed records are unchanged. These historical HF timings
do not establish the restored ordinary-HF default's accuracy or speed. No
integrated-image GPU rerun is claimed.

The homepage labels these exact historical figures in the surrounding heading,
alt text and caption. The linear SVG bytes and pinned-renderer replay remain
unchanged; new current-source measurements should be a separate cohort.

## Observations

| Mode | Phase | GenerativeQC wins / measured pairs |
| --- | --- | ---: |
| direct | cold | 32 / 32 |
| direct | warm | 32 / 32 |
| df | cold | 32 / 32 |
| df | warm | 32 / 32 |

This cohort is faster in **64/64 cold** and **64/64 warm** cases.
Direct ωB97M-V96 warm is **4.22×** faster.
These are descriptive observations from one process per engine/case, not
confidence intervals or a universal performance claim. All repeats enter the
reported warm medians and ranges; no timing comes from an earlier campaign.

All measured native medians are below their paired GPU4PySCF medians.

## Accuracy and source identity

GenerativeQC has **58 qualified / 6 failed / 0 unverified** populations;
GPU4PySCF has **58 qualified / 6 failed / 0 unverified**.
Unchanged gates are energy ≤1e-5 Eh/atom, maximum force ≤1e-4 Eh/Bohr and RMS
force ≤3e-5 Eh/Bohr, with quarter-gate reference stability and CPU/GPU checks.
Default-grid/fitting failures remain plotted. Original reference bytes and
producer identities are preserved; exact scientific inputs and all measured
samples are rechecked before reference reuse. Timing data are all new.

Source is the clean committed tree `46b916ac6df0082d56fbc48639bc1261aecc7d04`. Its archive
SHA-256 is `c5ae0e48c72409ef6bfd91c37cc77c3fb4c4c26f814428c0b4aa5c26ecf02b73`. All 2,770 relevant
source/header/compiler/runner files match the qualified candidate source.
The same core/AOT libraries are reused; full hashes and source parity are in
`provenance.json.xz`. The core's inherited embedded source identity is explicitly
distinguished from this commit. The
[repair evidence](../df-screening-headroom-20261026/README.md) retains the verified
incremental compilation/relink, independent references, three-round same-card
comparisons, moved geometry, work counts and constrained-memory fallbacks.

Seven finite Slurm workers pair both engines consecutively on each assigned
RTX 5090 and alternate first-arm order. CUDA PCI addresses and GPU UUIDs bind
each pair. Existing spatial-order and validation-GEMM controls, solver policies
and timer boundaries are disclosed in [conditions](conditions.md). Intrusive
diagnostics do not enter the figures. The inherited stronger default-HF force
diagnostic limitation remains in [its evidence](../cold-hf-defaults-20261026/README.md).

## Reproduction and previous cohort

```bash
PYTHONPATH=python:. python benchmarks/results/readme-refresh-20261026/reproduce.py \
  --output .artifacts/readme-screening-refresh-replay --verify --check-readme
```

Install `render-requirements.txt` for the original Matplotlib 3.11.2 renderer.
Replay checks all 128 populations / 512 calls, same-card pairing, complete
inputs, source/library identities, independent gates and work/timing tables,
then redraws both figures byte for byte. `--check-readme` also checks current
homepage images and the unchanged CBS checksum. The existing 52 offline runner,
qualification and publication tests pass.

The previous complete cohort and figures, including its **26.9349-second native
versus 26.2575-second GPU4PySCF DF PBE0-96 cold** observation, remain unchanged
in existing Git history. The repair addresses the norm-storage capacity cliff;
this refresh replaces every timing population together. `previous-cohort.json`
pins all original file hashes and the existing recovery commit:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/readme-refresh-20261026/previous-cohort.json \
  --output .artifacts/readme-before-screening-repair
```

Recovery was checked byte for byte, and the original figures still replay
byte for byte. A full clone restores offline; shallow clones may need the
recorded commit. This preserves the unchanged evidence-storage budget without
removing a slow observation from the historical record or publishing a release.
