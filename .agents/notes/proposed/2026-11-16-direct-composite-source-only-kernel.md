# Proposal: isolate geometry-only composite source preparation

Status: proposed; first isolated resource/runtime gate passes, final qualification pending
Date: 2026-11-16

## Observed remaining bottleneck

The complete clean sm120 native build at
`bd9d6cc30ec70c8fe05fbbc948ef6cd535a69247` spends 2,130.439 task-wall seconds
in `direct_jk_kernels.cu`, versus 1,514.488 in bounded fallback. These tasks
overlap other work; they are not isolated cold-wheel measurements. The retained
Direct-JK object has 565 device-function resource records. They include 156
`canonical_jk_kernel` specializations: 13 angular orders times 12 combinations
of spin, Cartesian layout, paired-range, mixed and composite-source modes.
There are also 59 materialized canonical kernels and other scientific owners.

The composite-source mode was added by `9d9083606` on November 7, 2026
(PR #3033). It contributes two spin variants across all 13 orders. Its runtime
benefit is qualified separately in
`../implemented/performance/2026-11-07-direct-composite-source-build.md`.
Neither its extra specializations nor their names establish removable science.

## Distinct hypothesis

For a nonzero full-range coefficient with a nonnull `source_values` destination,
the composite mode computes a geometry-only paired radial value, stores
`selected + coefficient * full`, then continues before density and Fock scatter.
The generic kernel nevertheless also compiles its ordinary J/K branch because
the source pointer is a runtime argument. This is a potential ownership boundary
for a dedicated source-only kernel, shared across spins, rather than another
noinline extraction of density preparation or per-order range dispatch.

Keep the existing ordinary, paired-Fock, mixed and materialized consumers. Do
not route zero-coefficient source preparation through a paired evaluation:
that would add scientific work. Do not remove the qualified composite-source
optimization or its bounded raw fallback merely to shrink the binary.

## Contracts to establish before implementation

- Audit every caller, especially raw/component fallbacks and source-window
  builders. Keep existing null-pointer behavior: a nonzero coefficient with no
  source destination currently selects ordinary Cartesian J/K work. A new
  geometry-only path must not accidentally discard that behavior.
- Preserve ordered AO rows, item/angular domains, screened pair enumeration,
  prefix clipping, bounded work windows, `work_begin`, output indexing and
  all existing finite/error handling. Never add density-based skipping to
  geometry-only source preparation.
- Preserve paired scientific evaluation and FP64 expression order, including
  runtime range/omega, zero and signed zero, finite signed coefficients,
  repeated-shell/component domains and admitted constant-AOT cases.
- Retain diagnostic semantics explicitly: the existing source store continues
  before ordinary evaluated-work accounting. Do not silently change work
  counts or mistake candidate-domain counts for executed recurrence work.
- Keep the nonzero-coefficient admission and null-pointer fallback explicit;
  do not depend on undocumented source-lease lifetime or a hidden host oracle.

## Required qualification and rejection conditions

Freeze the complete affected owner and verified compiler-cache provenance.
Compare full compiler-worker resources and all retained architectures; helper
text sizes and an empty-compile sample are insufficient. Keep independent
native source construction/replay, bounded-window, screening, failure-injection
and lifetime tests, then use scheduled real-GPU complete source and public
cold/warm endpoint evidence with semantic work counts. Pure forwarding mocks
cannot qualify scientific arithmetic or the resource/lifetime contract.

Reject if removing the generic branches does not materially reduce complete
owner cost, or if the new kernel harms complete source construction/replay or
method behavior. The rejected density-preparation and shared range-dispatch
experiments remain rejected; see
`../rejected/2026-11-16-direct-shared-density-preparation.md` and
`../rejected/2026-11-16-direct-range-order-sharing.md`.

The original resource dump, demangled census and build timing context are
retained in `/tmp/gqc-wheel-g-value-orbits/merged-direct-jk-specializations.json`,
`merged-direct-jk-resources.txt` and `merged-native-build-census.json`.

## First prototype checkpoint, not final acceptance

The isolated prototype replaces the two spin variants of composite-source
preparation with one geometry-only kernel per order. It retains the existing
paired scientific evaluator, FP64 expression order, single-range preparation
for zero coefficients and Cartesian Fock work for null destinations. No method,
angular order, architecture, screening, bounded fallback or numerical gate is
removed.

Both complete-owner requests use the existing Release sm120 CMake optimization,
RDC and `--disable-optimizer-constants` settings. Verified sccache 0.16.0 records
zero hits and four CUDA-stage misses for each; both owned foreground servers
exit normally. Ninja records 1,118 inputs; 93 repository inputs are additionally
frozen. Header contents stay unchanged. Initial preparation incorrectly expected
a Ninja-consumed depfile; `ninja -t deps` recovers the evidence without filling
the live baseline's additional empty include paths or restarting compilation.

| Complete sm120 owner | Baseline | Prototype |
| --- | ---: | ---: |
| Object bytes | 81,438,704 | 72,337,552 |
| Embedded PTX bytes, same extraction | 167,815,479 | 148,700,298 |
| Reaped compiler-worker maximum RSS, KiB | 5,723,164 | 5,144,964 |
| Device-function resource records | 565 | 552 |
| Ordinary canonical specializations | 130 | 130 |
| Composite source specializations | 26 | 13 |
| Single observed request seconds | 2,126.034 | 1,784.791 |

Object size falls 11.18%, embedded PTX 11.39%, and the single worker maximum
10.10%. Overlapping host observations are not controlled cold-wheel timings.
All 130 ordinary canonical and 409 other function identities/resource records
remain identical in sm120. Dedicated source order 10 uses 196 rather than 184
registers; other dedicated orders do not increase registers. All corresponding
source stacks remain zero. Do not claim universally lower registers or faster SCF.

The baseline is the previously frozen clean `f6159c523` library, SHA256
`b8d8ba567b0bb74d702beefea1872c425c76dcb09052311798b4dab0b7dde7d3`, source ID
`0c5fbc8027a7dde3eeb72c16d49d9d6a8f801be2baea3bbef42f7a8100764bec`.
The prototype links the actual completed Ninja object at **dirty** `f6159c523`,
source patch SHA256
`edb7965b94995e3d5af5867818ebb435ed60e33cfe01bffd57f28077faf0cf91`.
Its ELF SHA256 is
`77847cf8381ad8cbd0032a1ad6be7ca838018b984c46afd9b3073a9201a6d51d`, source ID
`91cbcd83c54efd18cdef40a4e9d252b6f8beb617c36b7bde23074bceb391c490`.
The 119-second incremental relink is not a cold native build.

### Executed scientific and endpoint checks

- Slurm 17532 qualifies complete composite construction/replay independently
  against PySCF Full/LR matrices for def2-SVP, four waters, displaced geometry
  and through-f def2-TZVP. Both coefficient signs, screening, immutable identity,
  admission/capacity refusal and usable raw fallback checks remain.
- Both artifacts pass 21,840 source-window cases and 26 null-destination Fock
  comparisons: all thirteen orders, both spin flags, dense/ordered/prefix layouts
  with empty rows, clipping, nonzero `work_begin`, local guards, screened untouched
  stores and initialized counters. Full/SR/LR, zero/signed-zero omega and
  coefficients, adjacent-to-0.3 omega and custom omega 4.7 are covered. Evaluated
  work intentionally stays unchanged. The ordinary Full/selected arithmetic
  comparison has maximum absolute error `1.78e-15`; old/new 11,808,000-byte
  traces differ by at most `1.07e-19` in FP64 payloads.
- Slurm 17533 runs ABBA complete source/replay and default public Direct
  WB97M-V cold E+F plus first-own-density warm E+F in fresh processes. Each
  through-f case retains 692,076 completed outputs, 1,384,152 **represented**
  radial terms and 6,074,473 retained bytes. This is not an executed primitive
  recurrence count. Independent matrix error is at most `1.60e-15`; warm replay
  does not reconstruct sources.
- Public endpoints retain 15 cold/one warm Fock build, precision operator counts,
  J/K applications, VV10 work and grid counts. Maximum paired energy difference
  is `5.69e-14` hartree and force difference `6.31e-14` hartree/bohr. Baseline
  cold samples are 4.741/4.651 seconds; prototype 4.633/4.803. Warm is 0.644/0.649
  versus 0.639/0.641. These small populations show no useful runtime ranking:
  claim compilation-resource reduction, not accelerated SCF. The public runner
  rejects production oracle imports; independent validation stays separate.

Initial public attempts lacked CuPy and compiler provenance assets. Retain them
as prerequisite failures, not scientific receipts. Successful observations use
the existing reference Python environment and a complete source-asset staging
root, not missing assets or a mock backend.

The maintained `composite-source-windows` pytest mode also passes against each
actual library (52.88/52.29 seconds), rather than only the scratch probe. Its
first pytest invocation skipped because the compiler-cache fixture was not
configured; that skip is not qualification. Rerunning with the verified
`CCACHE=/home/jzzeng/.local/bin/sccache` prerequisite executes the test. Seventeen
focused admission, auto-policy, incremental-source and resident-retry host gates
pass separately, as do source/test pre-commit checks. Host doubles establish
admission/lifetime behavior, not the independent device numerics above.

### Remaining gates and evidence

The full `80;90-real;120-real` complete-owner requests are still running, keeping
compute80 PTX. The maintained local patch adds formatting, rationale comments
and a permanent executable window test. Do not relabel the dirty prototype ELF
as a clean artifact of that final patch. Finish full-matrix resources, rebuild
and qualify actual final source, and retain final profile/ABI and host/format
checks before publication or changing this proposal's status. This checkpoint
does not complete manylinux build/link/repair/installed-wheel acceptance, the
separate Libxc lifecycle receipt or stale GitHub merge metadata recovery.

Ignored evidence is in `/tmp/gqc-wheel-direct-source-only/`:
`owner-comparison.json`, complete owners/cache/process-tree receipts,
`candidate-native/`, `qualification/paired-report.json`,
`qualification/window-comparison.json`, frozen dependencies and
`evidence-sha256.txt`. Full-matrix logs/objects are in `baseline-wheel/` and
`candidate-wheel/`; preserve those running requests on observation timeouts.

## Actual master audit correction

The PR REST payload still names base `010687258`, but a fresh SSH fetch of
`refs/heads/master` resolves to `718d5b32a` (November 16, 2026, 02:45:34 UTC).
At `d31343b55` the branch is two commits behind/twenty ahead. A read-only
merge against this real tip finds a genuine binary conflict in
`manifests/libxc/native-qualification-snapshot.json.xz`, not a CUDA source
conflict. Do not infer zero-behind or only an external metadata problem from
the cached PR base. Retain incoming #3101 CI ownership and #3102 scoped Libxc
dependency fixes; regenerate the snapshot with the merged offline validator
rather than choosing one binary side. Preserve historical receipts, stale
contexts and current resource/lifetime acceptance requirements.

The actual merge regenerates that snapshot with the incoming validator and
passes all 1,093 shard-ownership/dependency-qualification tests. All 28
revalidation contexts retain their products and non-input fields; only the
seven upstream-authorized post-HF-only generator inputs disappear. Historical
campaigns, measured identities, executions, resource acceptance and blockers
are unchanged. The snapshot remains stale with zero currently qualified
products; passing catalog integrity is not a new Libxc lifecycle receipt.
