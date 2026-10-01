# CPU ERI Coulomb-root reuse: focused incremental qualification

Date: 2026-10-01. The primary result is the **bytecode-normalized 960-endpoint
campaign**, comparing a fresh build of the exact #1667 merge `f678b01` with
compiler-owned Coulomb-root reuse on the same shared CPU host. This is the next
increment after shell-geometry reuse, not a repeat of the earlier interpreter
or geometry-reuse speedup claim.

**Formaldehyde/def2-SVP warm endpoints support a small improvement:** pooled
medians are **298.606 → 286.679 ms**, or **4.0% less elapsed time / 1.042×
speedup**. The separate median of paired-process candidate/baseline ratios is
**0.95849**, a 4.15% reduction, with percentile-bootstrap 95% interval
**[0.92444, 0.98672]**. Sixteen of twenty process pairs favor the candidate.
Cold and changed-geometry paired intervals also favor the candidate.

The water warm results are directional or inconclusive, and H2 does **not**
establish either a gain or guaranteed non-regression. No timing observation or
outlier was removed. These data support a narrow formaldehyde result on this
host, not broad CPU/PySCF parity, scaling, DFT/force speedups, or CUDA qualification.

## Complete endpoints and uncertainty

Milliseconds, pooled medians. Each of twenty independent rounds runs each of
four cases with baseline, candidate, and PySCF. Engine order reverses on alternate
rounds. Each process makes one cold, two fresh-calculator warm, and one
changed-geometry call: **240 processes, 960 endpoints**. Cold/changed each have
20 calls per engine/case; warm has 40 calls but only 20 independent process units.

The final column uses the **median paired-process ratio**, not the ratio of the
pooled medians. It resamples twenty process pairs, reducing the two correlated
warm calls to one median per process. NumPy default RNG seed 20261001, 20,000
resamples, percentile 95% intervals. These are pointwise intervals, without a
multiple-comparison correction; formaldehyde warm is the primary claim.

| Case | Phase | #1667 baseline | Coulomb reuse | PySCF in-core | Paired candidate/baseline [95% interval] |
|---|---|---:|---:|---:|---:|
| H2/STO-3G | cold | 23.595 | 23.820 | 4.501 | 0.9843 [0.8921, 1.1373] |
| H2/STO-3G | warm | 1.389 | 1.461 | 2.257 | 1.0295 [0.8939, 1.1215] |
| H2/STO-3G | changed | 1.348 | 1.346 | 1.916 | 1.0426 [0.9812, 1.1460] |
| Water/STO-3G | cold | 24.397 | 23.920 | 10.536 | 1.0060 [0.9033, 1.0944] |
| Water/STO-3G | warm | 3.042 | 2.967 | 8.134 | 0.9985 [0.8749, 1.1043] |
| Water/STO-3G | changed | 3.049 | 3.057 | 24.562 | 1.0256 [0.9333, 1.0721] |
| Water/def2-SVP | cold | 76.554 | 71.319 | 28.066 | 0.9270 [0.8770, 0.9689] |
| Water/def2-SVP | warm | 38.156 | 36.837 | 35.003 | 0.9584 [0.8724, 1.0233] |
| Water/def2-SVP | changed | 36.065 | 35.725 | 22.515 | 0.9923 [0.8912, 1.0588] |
| Formaldehyde/def2-SVP | cold | 339.988 | 326.863 | 72.891 | 0.9619 [0.9101, 0.9856] |
| Formaldehyde/def2-SVP | warm | 298.606 | 286.679 | 77.042 | 0.9585 [0.9244, 0.9867] |
| Formaldehyde/def2-SVP | changed | 291.365 | 288.216 | 66.648 | 0.9694 [0.9342, 0.9916] |

Water/def2-SVP warm is 38.156 → 36.837 ms, but its paired interval includes 1.
H2 warm is 1.389 → 1.461 ms, a 5.2% pooled-median slowdown; its wide paired interval
also includes 1. Neither overlapping ranges nor unchanged scientific work proves
non-regression. The smallest cases are particularly susceptible to host noise.
For example, formaldehyde warm ranges are 269.931–790.783 ms baseline and
254.131–641.978 ms candidate; all tails remain in the evidence. The candidate
formaldehyde pooled warm median is still **3.72× PySCF's** on this campaign.

`summary.json` is the readable reduction, including ranges, iteration/energy/work
inventories, and every process pair. `paired-process-confidence.json` retains all
ratios and intervals. The full original summaries and every process output/log
are retained in the gzip capsules; `confidence.py` recomputes the intervals from
the underlying per-process calls rather than trusting the summary.

## Scientific contract and bytecode control

- Single-thread CPU FP64, spherical AOs, neutral singlets, energy-only exact
  in-core four-center RHF; density fitting is off
- Identical frozen BSE exponent/coefficient decimals and coordinates in Bohr
- Fresh calculator and core-Hamiltonian guess on **every** call; no native or
  cross-engine converged-density reuse
- DIIS history 8, maximum 100 iterations, |delta E| < 1e-10 Ha and density RMS
  < 1e-8; frontend screening 1e-13, without claiming identical screening work
- Changed geometry moves the last atom's x coordinate by 0.02 Bohr
- Complete timing includes calculator setup, integral preparation, SCF, and
  result collection. Import and process wall times are separately retained.
  The `scf_wall_seconds` field includes the full singlepoint operation, not an
  isolated SCF or integral kernel. Process-cold does not mean disk-cache-cold
- PySCF is labeled `pyscf-auto` by the frozen runner, but actual `_eri` presence
  is asserted; this is an in-core comparison, not integral-direct scheduling
- OMP, OpenBLAS, MKL, and NumExpr are one thread; all loaded numerical pools and
  provider paths are checked. No timing processes overlap. The parent observed
  builds, tests, profiling, and auxiliary checks finish before this campaign;
  a process-namespace scan alone cannot prove exclusive access to this shared host

The first nine-round, **432-endpoint** campaign is retained in
`pilot-campaign.json.gz` under `matched-endpoints-attempt-01`. Its baseline source
snapshot had 57 Python bytecode files while the candidate had 289. The apparently
large cold improvement was confounded and is **not a causal cold-speedup result**.
No baseline scientific source or native library was changed to remedy this.

Before the fixed twenty-round campaign, all three source trees were compiled to
477 bytecode files each (`repo`: 174 → 477, detached baseline: 57 → 477, candidate:
289 → 477). The normalized campaign selected the actual baseline build's `repo`
Python/source tree. `bytecode-control.json`, full inventories, imported paths,
and binary hashes are retained in `supporting-evidence.json.gz` and the campaign.
Both campaigns use the same frozen endpoint inputs and native libraries. The
earlier pilot remains visible but is not pooled into the primary estimator.

## Numerical and regression qualification

All **960/960 normalized endpoints converge** and pass the 1e-10 Ha all-repeat
energy gate. Maximum candidate/baseline error is **8.526512829121202e-14 Ha**;
maximum candidate/PySCF error is **4.547473508864641e-13 Ha**. Numerical errors use
every cross-repeat comparison, without iteration or timing-based filtering.
Native iteration counts are identical in every phase: H2 2, water/STO-3G 8,
water/def2-SVP 13, formaldehyde 14. Native Fock counts are source-derived; PySCF
J/K calls and its extra post-SCF checks are recorded. Matching native counts does
not assert equal work with PySCF.

Qualification receipts, retained with the exact build/source identity:

- Baseline and candidate: **57/57 native tests** each
- Candidate: **39 compiler/codegen tests**, including 60,000 independent libcint
  values through each of two CPU paths, reused scratch at changed geometry,
  initialization/order bounds, permutations, and fallback coverage
- **186 compiler regressions passed, 44 skipped**; skips are not qualification
- **9 public CPU HF/PBE/PBE0 energy/gradient checks** passed
- Final pre-commit, compiler ownership/structure, complexity, CUDA ownership
  inventory, and default-promotion checks passed; this is no CUDA runtime claim
- Independent review: **313 exact component-local fallback body matches**,
  **6,334 valid shared-root reference sites**, all six axis-remapping rows, and
  **26 exact value/first-derivative shell-DAG matches**
- Generated artifacts: 107 byte-identical files; the expected ERI header and
  build identity differ. Eight Libxc translation units differ only in
  source-bound `ExpressionIdentity` strings; scientific source matches after
  masking only those identity fields. The audit record is retained

Separate **untimed** water/STO-3G and water/def2-SVP HF/PBE/PBE0 checks pass all
six cases in `hf-pbe-pbe0-attempt-03`. Maximum native pair error is 8.53e-14 Ha,
maximum independent-oracle error 2.42e-13 Ha, with identical native iteration
counts. DFT uses explicit 24 × 8 × 16 unpruned version-1 equal-radius Becke grid
points/weights; PySCF independently evaluates integrals, XC, J/K, and SCF on that
quadrature. This does not independently test grid construction or convergence.

Both failed supplementary pilots are retained, rather than relabeled as passes:
attempt 01's extra-tight 1e-12/1e-10 PySCF checks failed its post-SCF density gate;
attempt 02 at 1e-10/1e-8 failed the STO-3G PBE post-SCF density check. Attempt 03
retains failed oracle attempts and allows at most two self-refinement solves at
unchanged criteria; STO-3G PBE needed one, starting from **its own oracle density**.
Every native solve still starts from a fresh core guess. No density crosses
engines, and no refinement affects a timed benchmark. This supplementary oracle
must not be described as core-guess-only. Exact scripts and all checks are kept.

## Work, ownership, and costs

The compiler's shared `_coulomb_derivative` mathematical owner supplies
roots to two generated CPU schedules. The new schedule prepares a graded prefix
of Cartesian Coulomb derivative roots once per primitive geometry and reuses it
across shell components. The component-local schedule remains exact. Mixed f+ systems still call
`primitive()` one component at a time for their s/p/d quartets. Eagerly preparing
the full graded root prefix inside that compatibility wrapper would repeat the
table for each component. Retaining the byte-identical local factored DAG avoids
that work amplification and is why this change accepts two-schedule code growth.
No new
recurrence, native HF/DFT scientific fork, cross-geometry cache, or molecule-wide
root cache is introduced. f+ and derivative fallback behavior remains unchanged.

| Case | Primitive quartets, unchanged | Components, unchanged | Component-local root demands | Shared prepared root values |
|---|---:|---:|---:|---:|
| H2/STO-3G | 486 | 486 | 486 | 486 |
| Water/STO-3G | 9,720 | 32,886 | 114,048 | 42,930 |
| Water/def2-SVP | 38,111 | 326,255 | 2,344,466 | 319,254 |
| Formaldehyde/def2-SVP | 199,362 | 2,261,946 | 18,759,192 | 1,995,222 |

These are **source-derived unscreened demand counts**, not measured runtime FLOPs
or speedup factors. The original `coulomb-static-audit.json` arithmetic census is
a **hypothetical DAG design estimate**, not the exact emitted two-schedule
operation count; it is retained with that limitation. H2 has no cross-component
reuse. Prepared geometry, exact tensor size, primitive contraction, and native
SCF work are unchanged.

Scratch is caller-owned: 165 doubles plus the order field, approximately
**1,328 bytes** on this ABI. Only the requested graded prefix is initialized
(orders 0–8 need 1, 4, 10, 20, 35, 56, 84, 120, 165 values). Callers must prepare
it for the exact current primitive geometry; the maximum-order check is **not**
a stale-geometry identity guard. Center/axis permutation and bra/ket odd parity
must follow the compiler-generated maps.

The generated header grows **2,745,678 → 4,842,970 bytes (+76.4%)** because both
schedules remain available. Native `.text` grows **18,843,047 → 19,317,369 bytes
(+474,322 bytes)**. Observed complete builds were 491.1 → 518.7 s, with maximum
child RSS 3,771,236 → 4,460,064 KiB. Qualification concurrency differed between
those build observations, so they do **not** establish a causal compile-time or
peak-memory regression. There is no isolated generated-header build benchmark.

Baseline diagnostic profiles put component algebra near **8.5%** of sampled
endpoint PCs for both def2-SVP cases. Jet storage/lifecycle and native SCF remain
important next investigations. These sampled shares are **not latency benchmarks**,
have no confidence interval, and do not establish a candidate speedup. Unknown
callers remain unknown. The denied `py-spy` attempt, permitted in-process sampler,
all raw/resolved PCs, helper sources, and interpretation limits are retained in
the support capsule; see its `CPU_PROFILE_RESULTS.md` and `CPU_PROFILE_README.md`.

## Identity, reproduction, and retained records

Measured candidate commit: `a7afa3864f12a747b1851a25f97db5500c58c36c`, tree
`17b2951c0a7269c4534a56bb8180d9edaac36baa`. The published-equivalent code commit is
[`dc95a3c`](https://github.com/jinzhezenggroup/generativeqc/commit/dc95a3c742e8ee61b8eebc462867cf9e75e4a247),
with that exact tree and all four changed blobs verified identical; only commit
transport metadata differs. Evidence additions do not change native source
inventory. Baseline is the actual merged #1667
commit `f678b01fff0a7025aa292d3713476402266c2e0f`, freshly built on this host.
Later documentation/evidence commits or unrelated upstream merges must not be
used to relabel these measured binaries. `identity.json` binds source inventory,
commit/tree, loaded library hashes, generated header hashes, and build settings.

- Baseline native SHA256: `ce0a04bd7858512c6090b834da21fe4853864e0f81dae7f5a019f21b9dce8c76`
- Candidate native SHA256: `546dc0f03e2360fa35414c7c1499a20b1fad227171c3f6b3baf2030ed0063a86`
- Baseline source inventory: `8681cab84dc712d57c9a0aa3874b3ede5755e38d701866d7f382ce53556e3d7b`
- Candidate source inventory: `b037d9cc8d496957b9724aacd174ace4f98af52e06c00077e51eeeb1efc55177`

Both builds use GCC 14.2.0, RelWithDebInfo (`-O2 -g -DNDEBUG`), CUDA OFF, the same
OpenBLAS providers, and identical AOT defaults. Configure/build/dependency logs,
complete CMake caches, task-specific dependency versions (`dependencies.json`),
source inventories, loaded paths,
and before/after provenance guards are retained. Keep each native library beside
its BLAS shim with its build's RPATH; copying only the `.so` is not reproduction.
Frozen `benchmark.py`, `run.py`, and `cases.json` are unchanged from
[`../cpu-eri-20261001`](../cpu-eri-20261001), verified by SHA256.

From this directory, with a matching Python environment and completed builds:

```sh
# Offline verification; no native library or external service is needed.
python verify.py
python confidence.py
# Optional lossless restoration of retained original text, logs, and inputs.
python verify.py --restore /path/to/new/evidence

# Normalize source-tree bytecode on BOTH selected trees before cold comparisons.
# Record before/after counts and paths as in the retained bytecode-control.json.
python -m compileall -q "$BASELINE/python" "$CANDIDATE/python"

# Run only after builds, tests, profiling, and other benchmarks have finished.
python matched_cpu_evidence.py run \
  --baseline-source "$BASELINE" --baseline-library "$BASELINE_LIBRARY" \
  --candidate-source "$CANDIDATE" --candidate-library "$CANDIDATE_LIBRARY" \
  --frozen-directory ../cpu-eri-20261001 \
  --output /path/to/new/campaign --repeats 20 --energy-gate 1e-10 \
  --quiet-host-confirmed

# Rebuild complete summaries and the duplicate observations aggregate offline.
python matched_cpu_evidence.py summarize /path/to/restored/matched-endpoints-bytecode-normalized

# Independent supplementary checks; untimed, with oracle self-refinement retained.
python check_hf_pbe_pbe0.py run \
  --baseline-source "$BASELINE" --baseline-library "$BASELINE_LIBRARY" \
  --candidate-source "$CANDIDATE" --candidate-library "$CANDIDATE_LIBRARY" \
  --output /path/to/new/auxiliary
```

The readable runners change only the frozen-input default path, descriptive
docstring, and formatting; the original measured scripts are byte-exact in
`measured-inputs.json.gz`. Supplying `--frozen-directory` explicitly avoids any
path-default ambiguity for the main runner. The restored historical setup README
describes preparation before these runs and is not the current result summary;
this README and the timestamped campaign records take precedence.

The capsules use deterministic gzip (`mtime=0`) around JSON with per-member
SHA256 and byte counts. Canonical JSON members are compactly stored and restore
their original indentation/newline bytes exactly. All observations, outliers,
failures, scientific settings, provenance, and original scripts are retained.
Duplicate `observations.json` aggregates are omitted: their complete content
is reconstructible from the retained process JSON plus status, and all scientific
rows also remain in the original full summary. The ambient `environment-freeze.txt`
is also omitted as non-task setup data because it lists unrelated preinstalled
packages and runtime-specific locations. Its hash, size, omission reason, and
replacement task-specific dependency versions are recorded in the support capsule;
no endpoint, error, timing, scientific configuration, or original measured script
is changed. The dependency record distinguishes initial Ruff 0.16.10 setup from
Ruff 0.16.9 used for final qualification. `manifest.json` binds the compact
publication. No native binary, object/build tree, full generated scientific header,
or reconstruction patch is included; no Release/tag is needed.
