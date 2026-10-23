# Native DF cold-factor endpoint qualification

## Integration status

This is a retained alternative implementation and qualification branch, **not a
merge-ready change against the latest upstream**. During qualification, PR #2692
merged algebraic native-cold density-factor recovery and other endpoint changes.
PR #2696 incorporated that work at
`9f008cff0631834d10c4520d33a9b86cfc162fa6`. Its existing default already avoids the
large-system dense first K. The timings below compare the earlier dense-first
baseline and do not establish an additional gain over that new default.

No duplicate production PR was opened. Reconsider the producer-coefficient
handoff only after measuring against the current algebraic recovery path and
reconciling its admission, rollback and noncanonical-factor lifetime contract.
The tested implementation and every historical population remain available on
`codex/fast-df-endpoint-perf`; do not apply it over the new policy unchanged.

The native Hückel factor replaces one dense first K with occupied K in eligible
restricted FP64 DF cold starts. Standard and Fast share the implementation.
This receipt qualifies default-grid solver behavior on the sampled workloads;
it does not establish quadrature convergence or a universal performance ranking.

## Complete endpoints

PBE0/def2-SVP spherical water clusters, identical DF auxiliary inputs, stock
initialization and native grids. Cold includes calculator construction, preparation
and the first returned energy/analytic-force call. Each fresh process also returns
two frozen-density warm E/F calls. `GENERATIVEQC_CUDA_KS_COLD_FACTOR=0` is the
same-binary baseline; the candidate leaves it unset. Tracing/auditing is disabled
for all clean timing populations. Values below are medians in seconds; negative
changes mean lower latency.

| Public owner | Atoms | Cold off | Cold default | Change | Warm off | Warm default | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 12 | 3.180 | 3.151 | -0.91% | 0.317 | 0.316 | -0.06% |
| Standard | 48 | 15.220 | 15.011 | -1.37% | 4.891 | 4.897 | +0.12% |
| Standard | 96 | 36.952 | 33.275 | -9.95% | 11.100 | 10.977 | -1.11% |
| Fast | 12 | 3.218 | 3.078 | -4.36% | 0.318 | 0.318 | -0.16% |
| Fast | 48 | 14.981 | 15.014 | +0.22% | 4.895 | 4.896 | +0.01% |
| Fast | 96 | 36.868 | 33.381 | -9.46% | 11.050 | 11.077 | +0.24% |

48/96 atoms have two alternating-order pairs per owner. At 12 atoms, the initial
Standard population was **5.18% slower** (off 3.027/3.067 s; default 3.338/3.072 s).
This triggered the declared regression investigation: four additional alternating
pairs for **both** owners, retaining all samples. The table uses all six pairs;
no outlier was removed. Small-size startup variability limits causal attribution
of small timing differences. Fast48's +0.22% is retained explicitly. All warm
median changes remain within 1.2%.

There are **56 retained native processes / 204 complete E/F endpoints**: 40 clean
processes (120 endpoints), 12 lifecycle processes (72 endpoints), and four separate
matrix/trace processes (12 endpoints). Every retained numerical/admission gate
passes. All 28 off/default pairs have identical phase and physical-Fock counts.
12/48/96 cold calls use 10/9/9 Focks; frozen warm calls use one. PBE0, B3LYP and
M06-2X each cover both owners, moved geometry, moved warm and cleared-state reset.

## Independent numerical and work evidence

Independent tight GPU4PySCF DF references use the same frozen orbital/auxiliary
inputs, full-density Focks, energy tolerance 1e-12, orbital-gradient tolerance 1e-9,
screening 1e-14 and grid-response forces. Original and displaced references are
separate from production; they never provide a native initial density. Existing
PBE0 references are reused with their original identities. B3LYP/M06-2X DF
references were generated separately in job 13524. The predeclared maximum gates
are 2.5e-6 Eh/atom, 2.5e-5 maximum force and 7.5e-6 RMS force in Eh/Bohr.

Across all retained endpoints, maximum errors are 3.074e-12
Eh/atom, 1.112e-06 maximum force and
4.96e-07 RMS force. Four separately instrumented 96-atom
runs show the first physical K changing from dense to occupied, with no other
change to the physical Fock count. Native D reconstruction differs by at most
9e-16. At the first cold K, dense/occupied K and full-Fock differences are at most
6.22e-14 and 7.8e-15; including warm audits, maxima are 2.772e-13 and 3.47e-14,
all below the unchanged 2e-11 matrix gate. Diagnostic first occupied K takes about
536 ms versus 3883–3897 ms for dense K; these nested trace intervals are **not**
added to endpoint timings. Audits execute/count extra dense and occupied K work
and are excluded from clean timing populations.

## Source, build and retention

- Base: `bfb1584b3e6aed39c8e7f6276f553d4f5ee7d8ec` (PR #2696).
- Core SHA256: `14726734e9063465028cb8edeb11de3cad93ee64f6a94b4e534ffdef324c59a2`.
- Native source identity: `49c96d71aab8908b30a315d6dbf7ec49c4847fd3ba7dd41476fca4f8122f2378`.
- 2,441 production/build/driver inputs matched between worktrees; aggregate SHA256:
  `c9c5092f98ae5ebe93cff62e703ef33f6a05e42f65cf007babeee917aec5b4f3`.
- GCC 13.3.0, CUDA 12.9.86, Release `120-real`, PCH disabled; verified sccache 0.16.0
  launchers. Matching SPD force AOT contracts were verified before reuse. Full
  PBE/PBE0 AOT for small-basis tests was built from current inputs.
- n1 RTX 5090 through finite Slurm allocations: job 13519 (30-minute limit,
  visibility 2), continuation/12-atom repeats in 13524 (15-minute limit, visibility
  0). Scheduler-assigned visibility was preserved. Other host jobs may coexist.
- 398 host tests, native cold/warm factor suites and 33 distinct public DF/seed
  tests passed. New native coverage is also registered as
  `generativeqc_ks_df_cold_factor_tests`. All repository pre-commit hooks passed.
- Handwritten scientific and runtime CUDA translation-unit ownership deltas are
  zero; the change reuses existing kernels/providers through C++ runtime plumbing.

[`qualification.json.gz`](qualification.json.gz) stores every retained endpoint's
timing, error gates, physical work, controls, raw checksum and binary/source
identity, plus trace counters, reference/input identities and the full protocols.
Raw arrays, logs, source-input manifest, cache statistics and scripts remain in
ignored local `.artifacts/` and `n1:/data/jzzeng/fast-df-perf-20261023/`.
The previous baseline build and PBE0 reference tree remain unchanged at
`n1:/data/jzzeng/pr2696-fast-df-20261023/`.

The initial lifecycle campaign stopped when it looked for a pre-existing B3LYP
**DF** oracle, while only its Direct oracle existed. That failed attempt's log is
retained; its endpoint arrays/timings were not published by the driver and are
not included in the 204 retained endpoints or timing table. New independent DF
oracles were generated before resuming only unfinished populations. Initial
small-basis GPU tests also exposed missing full AOT and a PySCF test-helper import
order dependency; both were corrected without relaxing numerical gates.

## Reproduction

With the verified build, force AOT and frozen inputs/reference files, run each
selection in a fresh process inside a finite allocation, for example:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:05:00 env GENERATIVEQC_CUDA_KS_COLD_FACTOR=0 \
  python -m benchmarks.ks_incremental_default --arm standard --mode df \
  --method pbe0 --atoms 96 --selector unset --inputs "$INPUTS" \
  --reference "$DF_REFERENCE" --source-revision "$SOURCE" \
  --output .artifacts/cold-factor/off.json
```

Unset only the cold-factor selector for the candidate. Repeat in alternating
order for both owners and sizes. Use `--lifecycle` for the 12-atom hybrid checks.
Generate any missing references with `--arm reference --mode df` before native
runs. For separate diagnostic processes, set `GENERATIVEQC_DF_TRACE`,
`GENERATIVEQC_DF_HOST_TRACE` and `GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT=1`.
The archived protocol records the original populations and the 12-atom amendment.

See the [current contract](../../../docs/developer/ks_df_warm_seed.md) and
[decision rationale](../../../.agents/notes/implemented/performance/2026-10-23-ks-df-native-cold-factor.md).
