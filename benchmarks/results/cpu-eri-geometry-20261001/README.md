# CPU ERI shell geometry reuse: incremental endpoint qualification

Date: 2026-10-01. This is the **increment after #1662**, not another measurement
of its original interpreter-to-generated-code speedup. Single-threaded FP64 CPU,
exact in-core four-center RHF, energy-only, shared virtualized Intel Xeon Platinum
8573C host. No GPU, density fitting, changed convergence, or density reuse.

## Complete endpoints

Seconds, medians from three independent processes per engine/case. Each process
runs one cold, two fresh-calculator warm, and one changed-geometry endpoint.
Thus cold/changed have n=3 and warm has n=6. Engine order alternates by round.
All samples, including outliers, are retained in `observations.json.gz`.

| Case | Phase | #1662 baseline | Geometry reuse | PySCF in-core | Incremental speedup |
|---|---|---:|---:|---:|---:|
| H2/STO-3G | cold | 0.028674 | 0.024637 | 0.005030 | 1.16 |
| H2/STO-3G | warm | 0.001590 | 0.001822 | 0.002891 | 0.87 |
| H2/STO-3G | changed | 0.001337 | 0.001505 | 0.002367 | 0.89 |
| Water/STO-3G | cold | 0.033687 | 0.031019 | 0.015940 | 1.09 |
| Water/STO-3G | warm | 0.007847 | 0.003836 | 0.026111 | 2.05 |
| Water/STO-3G | changed | 0.007335 | 0.003390 | 0.012044 | 2.16 |
| Water/def2-SVP | cold | 0.126563 | 0.084124 | 0.030912 | 1.50 |
| Water/def2-SVP | warm | 0.093501 | 0.051347 | 0.040333 | 1.82 |
| Water/def2-SVP | changed | 0.084521 | 0.050910 | 0.024383 | 1.66 |
| Formaldehyde/def2-SVP | cold | 0.822972 | 0.429268 | 0.104827 | 1.92 |
| Formaldehyde/def2-SVP | warm | 0.782364 | 0.384788 | 0.101436 | 2.03 |
| Formaldehyde/def2-SVP | changed | 0.735668 | 0.354531 | 0.091501 | 2.08 |

Warm baseline/candidate ranges:

- H2: 1.241–2.362 / 1.331–3.024 ms
- Water/STO-3G: 5.720–13.190 / 2.652–8.562 ms
- Water/def2-SVP: 82.876–103.229 / 40.994–71.249 ms
- Formaldehyde/def2-SVP: 729.766–821.035 / 361.469–393.236 ms

Water/def2-SVP and formaldehyde remain **1.27x and 3.79x slower than PySCF** in
these warm medians. This does not establish general parity, large-system scaling,
force speedups, or a GPU result. The host is noisy, particularly for millisecond
cases and PySCF's small endpoints.

## H2 negative control and bounded confirmation

H2 has one Cartesian component per shell quartet, so it has no geometry reuse.
The first campaign shows a **14.6% warm median regression**. It is retained above;
overlapping ranges alone are not used to dismiss it.

A separate, predeclared **20 alternating process-pair** confirmation used the
unchanged frozen endpoint script: 40 processes, 160 endpoints, and 40 warm samples
per engine. All converged with two native iterations and unchanged energies.

- Pooled warm medians: baseline **1.654780 ms**, candidate **1.681971 ms** (+1.64%)
- Median paired-process candidate/baseline ratio: **0.91335**
- Percentile bootstrap 95% interval for that paired median: **[0.78683, 1.37061]**
- Seven of twenty process pairs had a slower candidate warm median
- Cold pooled medians: 30.047579 / 28.434882 ms
- Changed-geometry pooled medians: 1.709304 / 1.596695 ms

The interval resamples **process pairs**, not individual correlated warm calls:
20,000 resamples, NumPy default RNG seed 20261001. The wide interval makes this
inconclusive. The initial 15% median slowdown did not reproduce, but these data
neither demonstrate an H2 gain nor guarantee non-regression. No timing samples
were excluded, and no H2-specific selector or formula was added after observing
these results. Both campaigns use the same candidate binary.

See `h2-confirmation-summary.json` and `h2-confirmation-observations.json.gz`.
The source arithmetic audit also retains the no-reuse overhead: ssss grows from
86 to 111 scalar operations, excluding Boys evaluation and dispatch. This is a
declared limitation and a candidate for later compiler demand pruning.

## Scientific contract and correctness

The main campaign's **144/144 endpoints converge**. Maximum all-repeat absolute
energy errors are **8.5265e-14 Ha** against #1662 and **4.9738e-13 Ha** against
independent PySCF 2.14.0/libcint. Native iteration counts are unchanged in every
phase: H2 2, water/STO-3G 8, water/def2-SVP 13, formaldehyde 14. PySCF reports
1/8/13/15 plus its extra post-SCF check; its J/K calls are retained.

- Identical spherical BSE basis decimals and coordinates in Bohr
- Core-Hamiltonian initial guess on every fresh object; no converged-density reuse
- DIIS history 8, maximum 100 iterations, |delta E| < 1e-10 Ha, density RMS < 1e-8
- Screening frontend setting 1e-13; equal screening work is not claimed
- Both measured engines materialize exact in-core ERIs; this is not an
  integral-direct scheduling comparison
- Changed geometry moves the last atom's x coordinate by 0.02 Bohr
- Timing includes object setup, integral preparation, SCF, and result collection
- Python/backend import is separately retained; process-cold is not disk-cache-cold
- Every loaded numerical-library pool is recorded as one thread
- No timing processes overlap; local compilation/tests finish before each campaign

The exact script, basis payload, and general runner are reused from
[`../cpu-eri-20261001`](../cpu-eri-20261001). Their hashes are in `identity.json`.

Final qualification:

- 57/57 native tests, including contracted Cartesian/spherical ERIs, derivative
  and f+ fallbacks, and complete 1,296-component dddd quartets in both shell orders
- 14 public HF/hybrid-DFT endpoint/gradient tests; six CUDA-only skips
- 23 compiler tests, including 60,000 independent libcint values through each of
  the prepared and compatibility paths, exact Boys-order gates 0–8, and invalid inputs
- 15 source-identity tests; 59 existing compiler regressions; ten focused
  derivative/IR/CUDA regressions
- Matched HF/PBE/PBE0 in both bases: maximum difference 4.2633e-14 Ha, identical
  SCF counts; these are numerical checks, not a DFT performance claim
- Independent review: thirty shell-pattern orbit checks, 19,200 geometry remaps,
  and 1,440 factored numerical corner comparisons
- Million-Bohr translations retain a conditioning limitation: old/new factored
  expressions differ by up to 1.60e-11 normalized absolute error. This is a
  negative stress boundary, not an ordinary numerical pass
- Pre-commit, ownership/structure, complexity, default-promotion and evidence
  checks pass; type checking adds no diagnostics to eighteen existing warnings

## Work and ownership

The compiler exposes existing component-independent roots from the shared
shell-class DAG, cuts component DAGs at those roots, and emits their geometry
once. Native code owns bounded shell traversal, normalization, contraction and
scatter. No Gaussian-product or recurrence formulas are handwritten again.

| Case | Primitive components, unchanged | Baseline geometry/Boys | Candidate geometry/Boys | Reuse |
|---|---:|---:|---:|---:|
| H2/STO-3G | 486 | 486 | 486 | 1.00 |
| Water/STO-3G | 32,886 | 32,886 | 9,720 | 3.38 |
| Water/def2-SVP | 326,255 | 326,255 | 38,111 | 8.56 |
| Formaldehyde/def2-SVP | 2,261,946 | 2,261,946 | 199,362 | 11.35 |

These are source-derived unscreened work counts, not speedup factors. The native
scratch is bounded by 6^4 component records and is local to one invocation; no
cross-geometry cache or molecule-wide primitive-quartet cache is introduced.
Derivative and f+ systems use the previous traversal/fallback. Range-separated
values and the independent RawSource recurrence are unchanged.

The generated inventory stays at 313 representatives and covers all 10,000
ordered s/p/d components. Header size falls from **3,793,074 to 2,745,678 bytes**;
native text falls by **666,748 bytes**. Existing generated scientific arithmetic
is unchanged outside CPU ERIs: 107 other files are byte-identical, and eight
Libxc files differ only in deliberately source-bound qualification identity
strings. `generated-comparison.json.gz` retains those checks.

The initial complete clean build took 795.9 s with 3,771,032 KiB maximum child RSS;
the final rebased regeneration/rebuild took 522.0 s and 3,771,320 KiB. These are
whole-build costs, not isolated ERI compilation benchmarks or runtime preparation.

## Identity and reproduction

Candidate source: local measured commit `332130e`, published as
[`b18da1e`](https://github.com/jinzhezenggroup/generativeqc/commit/b18da1e2c95d9b2a58b0657dfda5ec8531fd05fc),
with identical tree **01b5153177c608bffb44d70f48b9782f7c2eeafd**. It is based on
the actual #1662 merge **5b61b41**. Publication changes commit transport/author
metadata only. Evidence additions do not change the native source inventory.

Baseline: the qualified #1662 **192829c** tree, local tree-equivalent `c5b67f4`.
The intervening #1649 merge changes CUDA KS execution and a CUDA-guarded admission
predicate; it does not change the measured CPU calculation. The baseline binary
is not relabeled as the later master.

- Baseline native SHA256: `513d98fcfb26df137ab1f478714300e4a6e3e4368e5cbf4c0977aa3da342f2d2`
- Candidate native SHA256: `f726bdcf34f135a9133df269d60b47199ecd4bfcff0f41b0de06808db8f118a5`
- Candidate source inventory: `66329262ef89b87b8b52044696f2c3ab974b2f81a820369a4f32a15778342a03`

Both use GCC 14.2.0, RelWithDebInfo (`-O2 -g -DNDEBUG`), CUDA OFF, the same
OpenBLAS provider, and unchanged AOT defaults. The loaded libraries were checked
against independently recomputed source inventories before timing.

```sh
python ../cpu-eri-20261001/run.py \
  --baseline-source /path/to/baseline --baseline-library /path/to/baseline/libgenerativeqc.so \
  --candidate-source /path/to/candidate --candidate-library /path/to/candidate/libgenerativeqc.so \
  --output attempts/main
python confirm_h2.py \
  --baseline-source /path/to/baseline --baseline-library /path/to/baseline/libgenerativeqc.so \
  --candidate-source /path/to/candidate --candidate-library /path/to/candidate/libgenerativeqc.so \
  --output attempts/h2 --pairs 20
python work_census.py
```

The readable H2 driver is a portable-path/formatting adaptation of the exact
measured wrapper in `measured-inputs.json.gz`; it invokes the identical frozen
endpoint script and changes neither settings nor timers. Summary and validation
JSON retain per-phase samples, every numerical error, work counts and limits.
No binary, generated scientific header, reconstruction patch, Release or tag is
included.
