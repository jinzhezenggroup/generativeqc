# Rejected H100 orthonormal-DIIS cold screen, 2026-11-06

The restricted FP64 DIIS history-metric candidate is **not promotable** for the
H100 cold-endpoint goal. This one-round screen completed six fresh processes and
all 24 energy-plus-analytic-force outputs. Every output passes the original
independent accuracy gates and both cases pass every warm guard, but the
candidate saves no cold molecular iteration or Fock build and remains slower
than the paired reference. The five-case three-round stage and coherent 30-case stage
were not run. **The 30/30 goal is not achieved.**

| H100 Direct /96 atoms | Native selector 0 cold (s) | Native selector 1 cold (s) | GPU4PySCF cold (s) | Reference / selector 1 | Native iterations / Focks, off → on |
| --- | ---: | ---: | ---: | ---: | --- |
| WB97M-V | 68.91083470801823 | 68.74741227203049 | 46.2739117750898 | 0.673100415648897 | 18 / 18 → 18 / 18 |
| HF | 11.914148230105639 | 11.759328447980806 | 6.355696065118536 | 0.5404812097250224 | 11 / 11 → 11 / 11 |

These are individual cold calls, not medians of independent repetitions. The
selector 1/0 ratios are 0.9976284943190693 and 0.9870053839238275; their small
differences do not establish an effective solver improvement. Warm median
selector 1/0 ratios are 0.9901803257080073 and 1.0042353674721318. All six matched
warm-call ratios and both process warm-median ratios satisfy the <=1.05 guard.
The original win rule still requires three paired ratios, median >1.05 and each
>1; this screen supplies one pair per case and neither reference/candidate ratio
exceeds 1.

## Source, execution and scientific scope

- Measured candidate: `d785a10d1e3d43e7e8167c7efa5c78a3b6f3594c`, tree
  `c346c923a8a8f51fd5fed97ae154ae236a60ef18`, based on
  `32b4942de70ff4ac6ec4701063f4b537daf096a9`.
- Native source identity:
  `4b7a246cb162b8dfbd72a43a82a1f4788e85eea0f6eafb82dd85601a0c8d0a5f`.
- Executed core SHA256:
  `2610999fc1d6d714e58af533a546b5cd7534ad2523322dd172d00a9878796cbf`.
- One H100 80GB/SM90, MIG disabled, UUID
  `GPU-d36c3a44-fb07-cfa1-bf66-7872e07d3366`; exact platform, driver, package,
  resolved controls, native/AOT mappings and producer hashes remain in the capsule.
- Same-image `GENERATIVEQC_SCF_ORTHONORMAL_DIIS=0/1`, Standard FP64, spherical
  def2-SVP, neutral restricted water96, original default grids and controls.
  This is an explicit experimental selector comparison. The original reducer's
  `final_policy_kind=PUBLIC_DEFAULTS` field is retained as original metadata;
  it does not make selector 1 a public default or authorize promotion.
- Cold includes constructor, preparation, SCF, analytic forces and host export;
  process startup/imports and separately prepared compilation are excluded.
  Each process then performs three warm E/F calls from its own frozen cold density.
  No oracle density enters native production, and timing traces are disabled.

All 24 complete vectors are independently rechecked against the original dense
GPU references; applicable WB dense-grid stability and CPU/GPU small-system
spots are also rechecked. The original
oracle and audited environment-reuse bytes are read from the already-published
`h100-cold-force-fixed30-20261105` capsule; no reference is regenerated or copied
into a new timing population. Gates remain energy <=1e-5 Eh/atom, force max
<=1e-4 Eh/Bohr and force RMS <=3e-5 Eh/Bohr. Worst observed errors are
4.860357734060017e-7, 2.2241209941320506e-5 and 1.0528722960727949e-5, respectively.

The pre-data DIIS-only v2 amendment retains native/native thresholds 1e-9 Eh/atom,
1e-7 force max and 3e-8 force RMS as diagnostics. WB misses those diagnostic
thresholds in all four calls: force max is about 1.25932e-7 and RMS 6.97814e-8.
HF passes all four. Every original error and pass flag is retained; the misses
are not relabeled as passes. Physical/canonical convergence, settings identity,
independent algebra/Gram/scratch tests, independent molecular accuracy and timing
guards remain hard requirements. This amendment does not alter #3019's separate
original failures or transfer its tighter-control checks to H100.

## Retained admission failure and continuation

The first small admission attempt contains four passing HF Direct/DF off/on
owners, followed by a PBE0 selector 0 `FileNotFoundError` for
`upstream/libxc/7.0.0` before an E/F call. PBE0 selector 1 was not run. That attempt
remains **failed**, with its raw record, plan, stderr, source/image identities and
four HF traces preserved.

The source archive had omitted tracked upstream assets. A new readonly runtime
root added 610 exact d785 Git-tracked inputs (1,334,522 bytes, including 45 selected
LibXC7 and 562 fulltree inputs) to the unchanged 3384 frozen files. Native identity,
native/AOT binaries, build receipt and scientific tables were unchanged. The
runtime manifest SHA256 is
`f4dac491bd4d629882d9f92cfe7d74eb2a906a2bc1f0f31edf460c2cd12eb1ab`.
Actual imports resolve to that root; all pilot off/on processes use it.

A targeted continuation authenticated and reused the four original HF receipts
and ran only two fresh PBE0 owners. It passed admission with a distinct
continuation status. It is not one original six-success attempt, independent
accuracy qualification, a timing comparison or an all-30 cohort. Capture counters
describe captured-body admission; their values are not replay execution counts.

## Offline replay and source reconstruction

```bash
uv run --offline --no-env-file --no-project --python 3.12 python \
  benchmarks/results/h100-orthonormal-diis-rejection-20261106/replay.py
```

Python 3.10+ is required; a suitable existing interpreter may also run the script
directly. The standard-library reader verifies the compressed evidence and patch, restores
the exact original pilot JSON bytes, rechecks all 24 full E/F vectors against
the pinned existing references, recomputes matched diagnostic flags/timings/work
and verifies the failed/continued admission scope. It performs no native import,
GPU, compiler, solver, network or Git subprocess. `summary.json` is derived;
`cohort.json` binds the payload and source patch.

`candidate.patch.xz` losslessly expands to `candidate.patch`, which reconstructs
the exact unmerged candidate tree on the stated base, including its tests and trace-only follow-up. Offline temporary-index
verification reproduced tree `c346c923a8a8f51fd5fed97ae154ae236a60ef18` exactly.
The patch is historical replay input under this evidence directory; experimental
production code and defaults are not added to main. All pilot timing producers,
protocols, full vectors, negative observations and original reduction metadata
remain in `evidence.json.xz`. Original and continuation admission-driver sources
are represented by their original producer hashes. Redundant runtime tree/SDK censuses and unrelated
transient build streams are represented by their original bindings and scope.

This H100 screen complements the distinct A100/5090 rejection in
[#3019](https://github.com/jinzhezenggroup/generativeqc/pull/3019). The open cold-gap
tracker remains [#3004](https://github.com/jinzhezenggroup/generativeqc/issues/3004).
Decision and revisit conditions:
[rejected note](../../../.agents/notes/rejected/2026-11-06-h100-orthonormal-diis.md).
