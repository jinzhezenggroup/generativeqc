# Generated source census at `d154546d2`

This source-only audit for [#3171](https://github.com/jinzhezenggroup/generativeqc/issues/3171)
replayed 41 generator commands at
[`d154546d2d6629ab8c865ca08f680002d68b4152`](https://github.com/jinzhezenggroup/generativeqc/commit/d154546d2d6629ab8c865ca08f680002d68b4152).
The selected inventory contains 416 distinct C++/CUDA artifacts and 4,168,879
nonblank generated lines, including three checked-in headers verified by replay.
The census records source emission and historical qualification. It makes no new
endpoint timing, numerical accuracy, or code-maintenance claims.

| Selected method-family sources | Spec / IR | Hand-written runtime | Compiler support | Generated total | Derivative / response files | Mixed files |
|---|---:|---:|---:|---:|---:|---:|
| Direct HF energy and forces | 1,634 | 19,211 | 7,946 | 1,480,071 | 858,187 | 293,597 |
| Direct DFT energy and forces | 3,805 | 35,640 | 11,399 | 1,623,465 | 948,402 | 339,060 |
| DF HF energy and forces | 1,047 | 19,984 | 5,041 | 3,357,591 | 750,851 | 2,023,098 |
| DF DFT energy and forces | 3,218 | 36,413 | 8,494 | 3,500,985 | 841,066 | 2,068,561 |
| DF-MP2 energy and forces | 2,049 | 27,232 | 5,651 | 3,364,362 | 753,649 | 2,026,536 |
| DF-CCSD(T) energy and forces | 2,696 | 28,563 | 5,950 | 3,434,852 | 784,779 | 2,033,457 |
| CBS composite with full-DF components | 2,767 | 33,626 | 6,193 | 3,434,881 | 784,779 | 2,033,457 |

All table cells are nonblank **lines**. The last two columns count lines in
artifacts dedicated to derivatives/response or containing a mix of roles,
respectively. The generated total also includes value and support artifacts.
Rows reuse shared groups and cannot be added together. For example, the mean-field
group accounts for 889,467 generated lines and appears once in each row.

- DF-MP2 adds the 29-line CUDA RI pair energy header. The shared post-HF
  group includes the 2,847-line DF source bundle with emitted MP2 energy and
  response entries (`mp2_energy_cuda` and `mp2_response_cuda`), plus RHF
  frame response and DF factor/source code. The mixed bundle also contains
  CC source code. The restricted native RI and correlation-only routes are both
  represented. The six conventional MP2 tile/dispatch companions are inventoried
  in the raw output with the `excluded` role.
- DF-CCSD(T) adds 70,519 generated lines beyond the shared groups. The
  31,130-line DF Lambda bundle includes generated `run_transpose_cuda`,
  `run_independent_transpose_cuda`, staged transpose/adjoint actions, RHS
  components, declarations and workspace queries. The checked-in
  `src/cc/df_lambda_cuda.cu` owns dispatch and composition of those actions.
  The 31,130 measures the entire bundle, so it gives no isolated transpose
  line count. The occupied-triples bundle includes both value and response.
- The selected CBS wrapper and extrapolator add 958 hand-written runtime
  lines and zero CBS-specific generated C++/CUDA lines. The generated lines
  in that row come from reused HF, DF-MP2 and DF-CCSD(T) components.
  The row represents the restricted full-DF composite route.

## Counting rules and scope

[scope.json](scope.json) is the allowlist. It identifies the source regions,
native runtime files, generator CLI arguments, expected artifacts, and shared
groups. The audit counts representative owners and a fixed build-time artifact
selection. It omits the full transitive source closure, other architecture
images, the full named/libxc registry and stationary profile inventory,
unrestricted post-HF implementations, runtime JIT inventories, and unrelated
methods. Direct AOT uses the accepted shell manifest, one SM90 real image and
eight stable shards. The native stationary XC sample is RKS/UKS for PBE0,
r²SCAN and ωB97M-V. The DF shell invocation covers all 80 emitted orbital
s/p/d/f and auxiliary s/p/d/f/g combinations.

- **Spec / IR** counts the selected first-party definitions of mathematical
  inputs and graphs. A `path:symbol` includes the top-level Python
  definition and its decorators. A whole-module selection includes the
  surrounding helpers and imports. This is a source count, so a loop
  constructing many IR nodes is counted once and vendored Libxc inputs are
  outside the authored inventory. First-party adapters to those inputs are
  eligible.
- **Hand-written runtime** counts selected checked-in execution, provider,
  binding, and orchestration files, including the CBS Python API. The
  runtime files can include code shared with adjacent routes.
  **Compiler support** counts selected emitters, templates, schedules, and
  generation manifests. It does not cover the general compiler or every
  generator dependency. Generator entry scripts themselves are listed and
  hashed in the raw record.
- **Generated** counts C++/CUDA emitted by the listed commands, including
  template expansions, declarations, dispatch tables, comments and repeated
  specializations. Hand-written template source and emitted expansions
  describe different stages of the same build. Calls to libraries are counted
  where emitted. The library implementation and included headers are outside
  that artifact's line count. Large files reflect such factors as shell
  unrolling, derivative components, and multiple precisions. These counts
  cannot estimate code avoided, engineering effort, or GPU execution frequency.
- Physical and nonblank counts are both retained. The main tables omit blank
  and whitespace-only lines and include comments, preprocessor lines, and
  docstrings. `response` is an artifact-level label for derivative/response
  owners and includes their glue. `mixed` includes artifacts carrying
  multiple roles, so the response column is not the total amount of
  derivative code in the corpus.

## Historical r²SCAN extension

Five first-parent merge diffs capture the milestones from generated r²SCAN
point energy through public CUDA energy/force qualification. The reported
public CUDA test passed two RKS/UKS cases on RTX 5090 against matched-grid
PySCF 2.11.0/Libxc on the PR-head build. The report was posted 38 h 40 min
46 s after the first visible implementation commit's author timestamp. This
measures elapsed repository history. The commit may postdate exploratory
work, and the interval includes CI, waiting and review.
[First commit](https://github.com/jinzhezenggroup/generativeqc/pull/537/commits/6fb870c431c72659cb48b88f2b680fa23e490b67)
· [Public qualification receipt](https://github.com/jinzhezenggroup/generativeqc/pull/747#pullrequestreview-5261652043)

| PR | Milestone | Implementation additions / deletions | Validation additions / deletions |
|---|---|---:|---:|
| [#537](https://github.com/jinzhezenggroup/generativeqc/pull/537) | Generated meta-GGA point energy and feature derivatives | +530 / −75 | +424 / −28 |
| [#620](https://github.com/jinzhezenggroup/generativeqc/pull/620) | Public RKS/UKS SCF and CUDA point integration | +791 / −172 | +194 / −81 |
| [#646](https://github.com/jinzhezenggroup/generativeqc/pull/646) | CPU stationary τ geometry | +77 / −22 | +215 / −7 |
| [#650](https://github.com/jinzhezenggroup/generativeqc/pull/650) | Internal CUDA stationary τ geometry | +193 / −39 | +189 / −3 |
| [#747](https://github.com/jinzhezenggroup/generativeqc/pull/747) | Public matched-grid CUDA E+F gate | +0 / −0 | +14 / −2 |
| **Total** | | **+1,591 / −308** | **+1,036 / −121** |

Git numstat counts changed physical lines in these PR deltas, including
shared infrastructure. [census.json.gz](census.json.gz) retains the full parent and
merge SHAs, every per-path diff count, and separate generated, vendored
and other changes. Binary fixture changes have no text count. The internal
CUDA run reported three passing analytic/finite-difference cases and used
a different gate from the public endpoint. The public review reports an
earlier environment attempt without NVCC/PySCF with no numerical result.
[Internal qualification receipt](https://github.com/jinzhezenggroup/generativeqc/pull/650#pullrequestreview-5260455404)
· [Public qualification receipt](https://github.com/jinzhezenggroup/generativeqc/pull/747#pullrequestreview-5261652043)

## Reproduce

Run from the PR checkout with Git and CPython 3.12. Set distinct paths for
`PIN_CHECKOUT` and `CENSUS_VENV` outside that checkout.

```sh
git worktree add --detach "$PIN_CHECKOUT" d154546d2d6629ab8c865ca08f680002d68b4152
python3.12 -m venv "$CENSUS_VENV"
"$CENSUS_VENV/bin/python" -m pip install numpy==2.4.6 array-api-compat==1.15.0 \
  typing-extensions==4.15.0 clang-format==23.1.5
"$CENSUS_VENV/bin/python" benchmarks/results/codegen-census-20261125/count_codegen.py \
  --source "$PIN_CHECKOUT" --jobs 4 --output /tmp/gqc-codegen-census.json
# Compare the two source-only records on the recorded environment.
gzip -dc benchmarks/results/codegen-census-20261125/census.json.gz | cmp - /tmp/gqc-codegen-census.json
```

The driver runs generators in fresh per-command directories and checks their
artifact names against the allowlist. It records both raw and formatted
replay hashes for the checked-in MP2 headers, and uses the checked-in line
counts. An unsuccessful, timed-out, or mismatched in-scope generation stays
in the JSON and produces a nonzero exit status. [attempts.json](attempts.json)
retains the initial quadrature inventory omission and the pre-formatting
MP2 byte mismatches. The later run reconciled both. Two independent replays produced identical
JSON bytes. The committed gzip preserves that JSON without resampling. Its
uncompressed SHA-256 is
`08c17f00802ee3d74e19714d0a8010e86de7ac00c2c0a5a33525d79a0cb74dc8`.

Host provenance is Apple M5 Pro, Darwin arm64, macOS 26.6 (25G72), Git 2.56.0
and CPython 3.12.14. Exact package versions and the Git tree ID are in
[census.json.gz](census.json.gz). CUDA was a source-generation target. This audit used
no native builds, GPU execution, or remote allocations. qz projects and job names
are inapplicable. GPU card-hours and remote cost were both zero.
