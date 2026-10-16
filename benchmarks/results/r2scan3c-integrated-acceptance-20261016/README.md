# Integrated-head r²SCAN-3c acceptance

Date: 2026-10-16. This packet qualifies the actual integrated PR #2553 native
image. It does not relabel the [pre-integration repair campaign](../r2scan3c-stationary-confirmation-20261016/README.md)
or the [original failed cohort](../r2scan3c-final-validation-20261015/README.md).
The original `none/96/moved` +15.985% regression remains rejected.

## Scope and unchanged gates

The strict-FP64, all-electron, restricted exact-direct pure-J repair holds an
already stationary density for a fresh complete physical F[D] when only its
finite energy-change gate is missing. It neither replays energy nor accepts an
orbital proposal, increases iteration limits, publishes a provisional owner or
relaxes convergence. Both actual residual norms must already pass. Method,
precision, ECP and resource exclusions, the bounded resident fallback, lease and
accepted-generation guards, and explicit selector `0` remain unchanged. See the
[decision note](../../../.agents/notes/implemented/numerics/2026-10-16-stationary-pure-j-energy-confirmation.md).

Each comparison uses the same library and GPU for explicit
`GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION=0` versus a **genuinely unset**
selector. `GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS` is absent in both arms. All
complete cold/warm/moved/moved-warm endpoints must pass existing nonregression,
including any changed-work population; warm semantic SCF work must also match.
The unchanged 2%/MAD benefit label is descriptive, not an enablement threshold.
Positive Direct/48 results below that label are not grounds to disable it.

The scientific protocol and pinned independent GPU4PySCF references are
unchanged: spherical def2-mTZVPP, original grid/screening and D4/gCP, five warm
repeats per geometry, **1e-8 Eh** energy and **1e-7 Eh/Bohr** maximum-component
force-error gates. SCF energy tolerance remains **1e-12 Eh**, with both actual
physical residual gates unchanged. No reference-oracle work is introduced into
the production endpoint; the existing contracted D4/gCP terms remain timed.

## Complete endpoint results

All eight phase gates pass. Seconds include the complete synchronized
energy-and-force endpoint, including returning forces to the host. Warm rows
are medians of five calls; cold and moved rows each contain one complete call.
Benefit is `(control - default) / control`, never a per-iteration normalization.
Work is `SCF iterations / Fock builds`, equal in both arms in every row.

| Direct atoms | Phase | Selector 0 (s) | Unset default (s) | Benefit | Work, both arms |
|---:|---|---:|---:|---:|---:|
| 48 | cold | 76.850894 | 76.566748 | +0.370% | 19 / 19 |
| 48 | warm | 21.912034 | 21.446858 | +2.123% | 1 / 1, each of 5 |
| 48 | moved | 57.683561 | 57.138393 | +0.945% | 13 / 13 |
| 48 | moved-warm | 21.927157 | 21.456218 | +2.148% | 1 / 1, each of 5 |
| 96 | cold | 413.839568 | 409.679554 | +1.005% | 24 / 24 |
| 96 | warm | 91.372078 | 87.522406 | +4.213% | 1 / 1, each of 5 |
| 96 | moved | 261.452944 | 257.768177 | +1.409% | 13 / 13 |
| 96 | moved-warm | 91.367928 | 87.557654 | +4.170% | 1 / 1, each of 5 |

All **48 comparison calls**, plus **24 candidate-only 3/12-atom calls**, pass
the independent scientific gates. Across these 72 complete qualification calls,
maximum absolute energy error is **3.819878e-11 Eh** and maximum-component
force error is **6.043033e-11 Eh/Bohr**. Actual energies and complete force arrays
are retained for all 72 calls, including the small systems, so the verifier
recomputes every error rather than trusting stored pass/error flags.

Separate intrusive cold/first-warm diagnostics execute valid resident
final-state proofs at both 48 and 96 atoms without CUDA errors. Their calls are
excluded from the 48-call timing comparison and the 72-call qualification
population. Native final-state and KS-library lease fixtures also pass.
The retained lease fixture replaces only a removed transport getter with
`BatchKsTransportDiagnostics`; its numerical and lifetime assertions are
unchanged. This is not a claim that the stock fixture compiles unmodified.

These are same-image descriptive comparisons, not statistical significance or a
controlled old-image/new-image patch-only speedup. In particular, the older
repair image's 96-atom moved trajectory has 14 iterations/builds, whereas this
image has 13 in both arms. Neither archive nor its plots are regenerated.

## Immutable image and execution

- Frozen clean Git head: `65cb37d58bcd384f46bf94f9056eb24cd9d294de`, including
  `master` through `4a36f6f3ef5d072eabba0f0fe02a4521d76ecdd0`.
- Native source identity:
  `e84f242a2260dc9cf6e75782b265f32a9fa6b830404f6b8843156edeb93423f9`.
- Core library SHA256:
  `c521f58d53e99d2c164c399b39c1ee90e927a906c867579328acf4987ef79f14`.
- Complete native inventory: **1753 files**, reconstructed losslessly against
  the pinned historical inventory and checked against the committed Git head.
  The **610 upstream inputs** are bound to its Git tree and inventory digest.
- The actual force AOT loader verifies both current contracts and binary hashes
  before every campaign. `r2scan_rks` contract:
  `4e6fef19f184b58d335e2bb0442bee09732403e5e6ac02f68d3da1bc443cf17d`;
  binary `4f0db500ac20c768eac45ffd8972c399872157fe06df0eea883fab859f4044fc`.
  `r2scan_rks_spd` contract:
  `5e7acf0fe671d85605928750694de9333ee9ad70ac9e36c99cde5c6c9b8a3d45`;
  binary `ce4f01ed4ff9e9ebb714eebc26fa4fe37c24a542fecbef44a61db420d119ea81`.
- Slurm **1018 / node4** qualifies 3/12/48 atoms and the 48-atom resident route;
  **1639 / node5** qualifies 96 atoms and its resident route. Both use RTX 5090,
  partition `main`, `--gres=gpu:5090:1`, eight CPUs and finite `srun` limits
  (01:00:00 and 01:30:00), preserving assigned `CUDA_VISIBLE_DEVICES`.
- Slurm **1020 / node4** runs both targeted test modules with their separate
  opt-ins: **53 passed, no skips**, including **45 actual device cases** and
  eight host checks, under a 00:10:00 limit. Its completed scheduler receipt
  records exit `0:0`. The first job's **30 passed / 23 skipped** result is also
  retained; it did not qualify the separately opted-in compensated-XC module.
- Integrated host/policy/lifetime/evidence/README tests: **290 passed, 36 skipped**;
  device-only skips are not represented as host passes. CUDA 12.9.86, GCC 11.4,
  CMake 4.2.1 and verified **ccache 4.5.1** are recorded with actual launchers,
  build logs and before/after cache snapshots. Shared cumulative cache counters
  are not exact per-build hit attribution; no cache was cleared.

The first core build used the preceding integrated head. A new upstream solver
commit then required a shared-code conflict resolution, a nine-action core
rebuild and a new clean source freeze. No GPU timings qualify that preceding
image. Both old and final build logs are retained.

Setup negatives remain explicit: the initial missing rsync parent, and stale
force AOT manifests after building only library targets. The real loader rejects
both stale contracts. Building the normal `*_manifest` targets regenerates
metadata from the current generated sources/binaries; no identity field is
patched. Before/after loader receipts retain both the rejected and passing
states. These setup observations are not scientific qualification attempts.

## Retention and reproduction

`qualification.json.xz` retains observations and receipts: full phase samples,
protocols, work counts, source inventories, numerical data, separate resident
proofs, build/cache/test/job logs, AOT manifests, fixture sources and launch
scripts. Comparisons use lossless dictionary deltas against the unchanged
historical packet. Comparison forces use little-endian binary64 XOR deltas
against the previous same-geometry call, starting from its independent reference;
every bit is decoded before recomputing force errors. Small-system energies and
force arrays are retained directly. Nothing is rounded in the evidence data.

`manifest.json` authenticates the compressed artifacts. `source.patch.xz` has
zero decoded bytes because the qualified freeze is already a clean committed
head; later report/retention edits are not native compiler inputs.
`verify.py` is an **opt-in immutable-evidence check**, not a permanent required-CI
fingerprint test of production source spelling:

```bash
PYTHONPATH=.:python python \
  benchmarks/results/r2scan3c-integrated-acceptance-20261016/verify.py
```

It checks every committed native input, upstream inventory, protocol, selector
absence, image, complete-call count and phase gate, independently recomputes
all 72 energy/force errors, authenticates retained documents and successful
finite jobs, and requires both resident proofs. It also requires that the
original Direct/96 moved failure **still fails**. It works under `python -O`.

To reproduce new measurements, rebuild the pinned clean head and both force
libraries **and their `*_manifest` targets**, adapt retained deployment paths,
and run finite `srun` jobs with the same references/repeats/selectors. The
retained core/force hashes identify the observed images, not a promise that
another compiler/path produces a bit-identical binary. Preserve any new failed
attempts separately; do not overwrite or pool either historical cohort.

The unchanged 64 MiB aggregate evidence cap is respected by losslessly
compressing the archived #409 source patch: **246001 → 54212 bytes**. Its original
decoded SHA256, every source byte, native identities, scientific arrays and
incomplete/deferred statuses remain unchanged; both manifests and recovery
instructions are updated. The one-off byte-preservation check and result are
retained here. No external archive, Release, asset or tag is published.
