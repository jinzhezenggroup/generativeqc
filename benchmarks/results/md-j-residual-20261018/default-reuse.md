# Angular-three MD-J default-path validation (2026-10-18)

The user authorized default-on promotion after the measured benefit in the
[original opt-in trials](contraction-trials.md). This new same-binary pair tests
explicit `GENERATIVEQC_MD_J_RECIPROCAL_HIGH=0` against an **unset** selector,
not explicit `1` relabeled as default. Parent reciprocal and residual-tile
selectors are also unset. The
[decision note](../../../../.agents/notes/implemented/performance/2026-10-18-md-j-high-reciprocal-default.md)
documents fallbacks and the scope of this promotion.

## Identity and protocol

- Parent source: `58759558834e446dad8c0c9bae2d92d76a9bad0c`; policy/header/test
  patch SHA-256:
  `0cf886fbd54c8a5db5e97dd8fae45fc05c674127a84084475000b4f5b9f24b44`.
- Native source identity:
  `8f77cd561ff6deb0a626b96d5b48aff5fd651db1177cf6830b8353014928e27b`.
- Native library SHA-256:
  `6c22be30124b914406268624ccd7b8899bebe582aa6dfa7b3e4877483bb610c1`.
- Companion force DSOs retain their original source-matched identities:
  RKS `06beac3af56c4a6a500951ccc1b12275cea2f536fdea678007ecc1e9051d90be`,
  SPD `9fc738752cb6e1270482bff33fefba29e36a810ed3c23db3b109f07203300e01`.
- Remote n1, RTX 5090, Slurm `10136`, `main`, one GPU, eight CPUs, 64 GiB,
  finite 40-minute allocation; scheduler-assigned visibility is preserved.
  Assigned GPU UUID: `GPU-d0b9e1b7-2083-6198-da50-c3e55693691c`.
- CUDA 12.9.1, GNU 13.3, sm_120, Release, verified ccache 4.5.1 reused.
- Same 96-atom / 1088 spherical-AO r2SCAN-3c case, no density fitting, same
  independent reference and basis as the original trial. Diagnostic counters
  and CUDA event fences are disabled in the complete endpoint pair. Each arm
  starts a fresh process on the same assigned GPU.

## Complete endpoint pair

All values are the driver's `complete_seconds`, including cold preparation.

| Phase | High off s | Unset default s | Fock builds off/default |
| --- | ---: | ---: | ---: |
| Cold | 270.286147 | 265.244123 | 24 / 24 |
| Warm | 57.696494 | 57.328674 | 1 / 1 |
| Changed geometry | 171.058952 | 168.529296 | 13 / 13 |
| Changed-geometry warm | 57.435933 | 57.329108 | 1 / 1 |
| Sum | 556.477525 | 548.431200 | 39 / 39 |

Complete cold improves 1.87%; the four-phase sum improves 1.45%. Both arms
converge and pass every independent energy/force gate. Maximum independent
energy/force errors are `2.319e-11` Eh / `7.665e-11` Eh/bohr. Maximum off/default
differences are `9.095e-13` Eh / `3.639e-11` Eh/bohr. Neither fewer SCF builds
nor relaxed scientific settings explains the improvement.

Together with Slurm 10122's original off/on pair, both measured cold and
four-phase comparisons improve. This is still a small, single-device sample,
not statistical or cross-device qualification. The separate draft PR's older
residual-tile endpoint limits are not resolved by this uniform-only promotion.

## Separate scientific and work gates

The unset-default scheduled suite passes **114 GPU tests**, including independent
screened raw-J oracles for cartesian/spherical, RKS/UKS, nonsymmetric density,
residual-tile selectors and explicit/unset high reuse. It also covers bounded
normal/K fallback and frozen replay policy, including parent reciprocal disable.

Separate instrumented 96-atom density runs compare high `0`, high `1` and unset.
Explicit `1` and default agree per angular class on radial-source counts. All
three preserve density consumers, Hermite summands, residual admitted tasks,
primitive products and the complete candidate domain:

| Work | High off | Explicit 1 / unset default |
| --- | ---: | ---: |
| Uniform radial evaluations | 2,225,962,468 | 2,038,308,216 |
| Uniform density directions | 4,013,012,720 | unchanged |
| Uniform Hermite summands | 95,323,936,432 | unchanged |
| Residual candidate domain | 526,842,030 | unchanged |
| Residual admitted tasks | 43,920,982 | unchanged |
| Residual primitive products | 242,399,046 | unchanged |

Maximum independent raw-J error is `2.445e-11`. Candidate-check versus
tile-screened partitions differ between fresh compacted owners; their sum and
admitted scientific work remain conserved. Instrumented runs are not timing
samples.

Focused host suites pass **400 tests**, with 114 real-GPU cases skipped locally
and run through Slurm instead. Formatting, default inventory (69 entries / 119
controls), compiler structure (565 modules), CUDA ownership (400 files), evidence
retention and warnings-as-errors documentation checks pass. The change adds no
scientific CUDA equations; combined PR handwritten scientific CUDA growth
remains +151 maintained lines versus its master base, runtime growth zero.

## Retention

Original opt-in receipts and archive remain immutable. New default-build source
patch, scripts, build/cache receipts, scheduler/device identities, raw endpoint
JSON/logs, per-class work and checked summary are retained separately under
ignored `.artifacts/mdj-remote/default-results/` and `default-receipts/`.
Remote binary snapshots are preserved at
`/data/jzzeng/gqc-mdj-static-20261018/trials/high-optin/` and `trials/high-default/`;
their checksums identify which measured selection policy each implements.
No Release, release asset or new external storage host is created.

The separate ignored archive
`.artifacts/mdj-remote/high-default-validation-20261018.tar.xz` retains 49
verified members, 120,356 bytes, SHA-256
`47989ce1f8396e7c79284635cbda76ca6e42b6c499a830006a77e5558c0322ea`.
Its sibling `high-default-validation-manifest.json` records every member's
size/checksum; all archive bytes are verified against the original receipts.
