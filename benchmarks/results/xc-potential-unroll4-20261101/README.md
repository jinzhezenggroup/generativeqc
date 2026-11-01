# Portable CUDA XC potential loop unrolling

This campaign measures bounded unrolling of the generated CUDA XC potential
consumer. The production matrix tile stays 16, with the same ordered reduction,
FP64 finite-cancellation fallback, quadrature, selected AO maps and SCF policy.
No GPU-name whitelist is introduced. The change also removes the compiled
register-resource launch failure of the existing explicit tile32 schedule;
that schedule is not promoted.

Replay from the repository root with the Python/reference dependencies:

```bash
PYTHONPATH=python:. python benchmarks/results/xc-potential-unroll4-20261101/report.py
PYTHONPATH=python:. python -O benchmarks/results/xc-potential-unroll4-20261101/report.py
```

The reader authenticates every original member/reference hash, replays the
independent energy/full-force gates, verifies paired semantic work and all
sample counts, and recomputes the tables. Integrity and acceptance checks remain
active with Python assertions disabled. Positive changes mean slower.

## Measurement contract

Frozen parent identity:
`b6b63f9063949ed70191c54b4d0f0b7efb788957274eb680c98f4242bbcf74f3`.
Frozen candidate identity:
`ce5f498a8d173d214bc46d686256caba37836541445f6a089601a550066d6ad0`.
Both use water96, DF Standard, spherical def2-SVP (mTZVPP for r2SCAN-3c),
stock quadratures, SCF tile256 and explicit force tile1024. Each population
includes setup and the first complete energy/analytic-force call, followed by
three calls with its frozen warm seed. Eight CPU slots are reserved and math
libraries use one thread. These are default-grid endpoint comparisons, not a
search for either engine's fastest accuracy-qualified configuration.

Every method/device has two source pairs in candidate,parent,parent,candidate
order, in one finite GPU allocation: A100 LSF 224136, RTX 5090 Slurm 1399 and
PRO6000 Slurm 3826. Raw per-sample variability remains in `analysis.json` and
the original receipts. The declared 1% median regression-review threshold is a
practical noise screen, not statistical equivalence or a universal guarantee.

GPU4PySCF values in the following tables reuse retained same-device populations
from earlier allocations. The reader checks node, PCI address and UUID where
available; A100 lacks a UUID and uses its retained node/PCI/assigned-visibility
locator. This limitation is unchanged. Separate fresh reference controls for
the narrow Blackwell r2SCAN-3c gap are described below.

The publication branch is based on #2910 head 2ce541b5. It contains subsequent
source-provenance/tuning-publication fixes absent from the frozen measurement.
`publication-production-boundary-summary.json` records byte-equal native source, public
headers, CMake, runtime Python, DFT compiler and generation tools. Those later
changes do not change the measured scientific implementation. Original whole
build identities and Git revisions are retained verbatim rather than relabelled
as the eventual publication commit.

## A100 final source comparison

| Method | Parent cold (s) | Candidate cold (s) | Cold change | Warm change | Cold vs retained GPU4 | Warm vs retained GPU4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pbe | 8.565316 | 8.233427 | -3.87% | -1.73% | +3.56% | -5.13% |
| pbe0 | 12.785386 | 12.408617 | -2.95% | -1.66% | +8.84% | +0.48% |
| b3lyp | 12.871708 | 12.511784 | -2.80% | -1.46% | +38.86% | +1.93% |
| r2scan | 10.665129 | 9.470407 | -11.20% | -5.12% | -7.03% | -20.02% |
| r2scan-3c | 18.509905 | 15.923120 | -13.98% | -5.65% | +32.56% | -1.27% |
| m06-2x | 16.245696 | 14.880940 | -8.40% | -3.88% | +28.86% | -12.65% |
| wb97m-v | 146.303877 | 140.223764 | -4.16% | -0.92% | +50.21% | -19.26% |

## RTX 5090 final source comparison

| Method | Parent cold (s) | Candidate cold (s) | Cold change | Warm change | Cold vs retained GPU4 | Warm vs retained GPU4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pbe | 11.301493 | 11.205523 | -0.85% | -0.29% | -5.71% | -13.08% |
| pbe0 | 21.388076 | 21.322268 | -0.31% | -0.12% | -7.07% | -43.17% |
| b3lyp | 21.545461 | 21.481531 | -0.30% | -0.16% | -10.57% | -42.57% |
| r2scan | 13.475060 | 13.071458 | -3.00% | -1.07% | -13.45% | -32.17% |
| r2scan-3c | 24.419874 | 23.515794 | -3.70% | -1.37% | +0.95% | -32.24% |
| m06-2x | 25.018921 | 24.558534 | -1.84% | -0.74% | -11.29% | -49.00% |
| wb97m-v | 234.501232 | 232.645430 | -0.79% | -0.15% | -39.25% | -54.06% |

## PRO 6000 final source comparison

| Method | Parent cold (s) | Candidate cold (s) | Cold change | Warm change | Cold vs retained GPU4 | Warm vs retained GPU4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pbe | 10.618125 | 10.534365 | -0.79% | -0.35% | -18.25% | -8.96% |
| pbe0 | 20.018085 | 19.925729 | -0.46% | -0.23% | -13.33% | -41.71% |
| b3lyp | 20.165718 | 20.087481 | -0.39% | -0.12% | -16.26% | -40.95% |
| r2scan | 12.615146 | 12.236230 | -3.00% | -1.07% | -20.67% | -29.42% |
| r2scan-3c | 22.997003 | 22.138203 | -3.73% | -1.30% | -4.45% | -29.89% |
| m06-2x | 23.363514 | 22.948099 | -1.78% | -0.71% | -15.96% | -47.70% |
| wb97m-v | 223.487028 | 221.870637 | -0.72% | -0.15% | -36.67% | -63.18% |

## Fresh-reference and correctness controls

| Device | Native cold (s) | GPU4 cold (s) | Cold overhead | Native warm (s) | GPU4 warm (s) | Warm overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5090 | 23.547403 | 23.215340 | +1.43% | 6.805996 | 10.032914 | -32.16% |
| pro6000 | 22.125869 | 22.811003 | -3.00% | 6.422191 | 9.159576 | -29.89% |

Both engines first execute an explicit complete cold/one-warm cache-conditioning
population. Those original samples are retained but not ranked. Two additional
fresh-owner cold/three-warm ABBA pairs then form each same-allocation comparison.
The independent n1 control is retained under `tile32-followup/`; its tile16
cold comparison is +0.14%, so the narrow 5090 gap is not a stable win.

The main cohort contains 128 independently qualified populations / 440
complete E/F calls, with 26 further independently qualified populations in the
separate tile/reference experiments. Paired native differences are at most
3.64e-12 Eh and 2.84e-12 Eh/Bohr.

All three devices pass 510 independent contraction cases across tiles 8/16/32
and both FMA compiler modes. The n4/n2 controls additionally pass the independent
RKS/UKS occupied-XC default/forced/off/invalid/moved CPU-oracle tests (512 total).
Native matrix/tail/spin/variational/capture and point/ragged suites pass;
Blackwell memcheck and synccheck report zero errors. Sanitizers and instrumented
traces are excluded from the endpoint rankings.

The 120-atom r2SCAN controls compare both sources at an explicit 8 GiB DF budget,
including cold, frozen warm, moved coordinates and frozen replay at the moved
coordinates. Each v2 sample records the actual coordinate array. Earlier v1
controls accidentally passed `None` on the last call, which returns to the
original geometry; their misleading `moved-warm` label is not corrected in place.
They remain valid move/return differential evidence, and all three devices have
separate corrected v2 populations. Numerical/work checks cover both cohorts.
This larger differential test supplements, rather than replaces, the independent
README 96-atom oracles and contraction-level numerical tests.

Original unsuccessful attempts remain visible: A100 pilot 224133 uses an
abbreviated Git revision and is rejected by the normal gate; matched 224134
omits the pinned Libxc tree and fails before a complete endpoint; checks 224135
mixes runtime and compiler loader paths; 1/4 GiB larger controls explicitly refuse
insufficient workspace. Checks 224137 passes 510 numerical cases and the native
matrix suite before the later 4 GiB request fails. Later jobs repair deployment
and test-driver issues without altering scientific gates or prior records.
The originally missing local native-library host-test failures and configured
successful rerun are also retained.

## Rejected scheduling experiments and limits

The original fully unrolled 32x32 schedule fails real launches on both A100 and
5090. The repaired unroll4 tile32 build passes independent numerical and native
launch tests, but its n1 pilot is 2.67–4.51% slower cold and 1.03–1.46% slower warm
for four functionals. It remains an explicit experiment. The archive preserves
its distinct CMake/generated/library hashes; source identity alone cannot
identify a matrix-tile build override. See the
[matrix-tile decision](../../../.agents/notes/rejected/2026-11-01-xc-potential-matrix-tiles.md).
The preceding [cooperative-map experiment](../xc-map-groups-rejected-20261101/README.md)
also remains rejected with all original measurements retained. The additional
8x8 n1 pilot 15034 passes all eight independent qualifications, but cold changes
are +0.16%/+2.25%/+3.05%/+1.19% for PBE/r2SCAN/r2SCAN-3c/M06-2X and warm
changes +0.07% to +1.99%. It also remains unpromoted. Exact configuration,
library, source and raw receipts are preserved under `tile8-followup/`.

A100 still trails the retained GPU4PySCF cold baseline in six functionals and
its PBE0/B3LYP warm medians remain behind. PRO6000 leads all sampled endpoints;
5090 retains the narrow 3c cold gap. This change does not establish the requested
all-functional win or absence of regressions on unmeasured GPU architectures.
The four separately qualified A100 traces identify remaining component costs;
their instrumented times are diagnostic and excluded from rankings.

## Retention

All original measurement populations, numerical outputs and failures remain in
this compact archive. The full publication patch (which duplicated the staged
source and another compressed evidence family) and the full per-file publication
inventory remain ignored local diagnostics with exact byte counts and hashes.
Their source changes and verified boundary summary are retained without copying
those redundant payloads into the archive.

To stay within the existing aggregate checkout budget, the superseded
[structural-XC qualification](../xc-structural-default-20261028/README.md#replay)
uses its already-merged Git snapshot for the unchanged 811,824-byte receipt bundle.
Its complete byte recovery and original numerical reader were verified before
retirement; summaries, negative results, reader and recovery manifest remain.
No storage limit, acceptance gate or external publication boundary changed.

The receipt container now separates exact FP64 words from JSON structure. The
shared decoder reconstructs all 603 original members and verifies their original
SHA-256 before running this report. No sample, failed attempt, timing or force
component is rounded or removed. The storage change reduces the 1,045,252-byte
container to 629,544 bytes in two XZ payloads; original producer identities and
scientific acceptance remain unchanged.
