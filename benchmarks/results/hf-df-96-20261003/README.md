# HF DF occupied symmetry qualification

On the same RTX 5090, the complete 96-atom RHF DF energy-and-force warm
endpoint drops from **5.574 s to 4.488 s (19.5% less time)** relative to
master `b9c626d15b2c19c21a848cb3625932d57f754a71`. Five frozen warm repeats
and five moved-geometry warm repeats retain one SCF iteration each. No timing
is divided by iterations or selected for matching work.

| Complete phase | Baseline, s | Candidate, s | Time reduction | SCF iterations |
| --- | ---: | ---: | ---: | ---: |
| Cold, including prepare | 46.228 | 38.316 | 17.1% | 24 / 24 |
| Frozen warm, median of five | 5.574 | 4.488 | 19.5% | 1 / 1 |
| Changed geometry, reconvergence | 35.814 | 31.197 | 12.9% | 13 / 13 |
| Changed geometry, frozen warm median | 5.576 | 4.500 | 19.3% | 1 / 1 |

The primary comparison is candidate then baseline in n5 Slurm job **1369**.
An earlier baseline in job **1367** gives 5.557 s warm, followed by the
response-only candidate at 5.085 s. [comparison.json](comparison.json) retains
all four runs, every scalar endpoint, complete work counters, identities and
independent original/moved force arrays. That reference was measured on node3;
it is a numerical oracle only for this n5 comparison. Cold and reconvergence
have one observation per arm and are not statistical latency estimates.

## README figure

[hf.svg](hf.svg) is a fresh six-size direct/DF comparison from node3 Slurm
job **12090**, using one final Release sm_120 native library and separate
GPU4PySCF processes. It includes 156 native calls (12 diagnostic DF calls)
and 144 independent reference calls. Every call passes `1e-8 Eh` energy and
`1e-7 Eh/Bohr` force gates; maximum native errors across the entire campaign
are **4.73e-11 Eh** and **1.44e-10 Eh/Bohr**.

At 96 atoms/768 AOs the plotted DF warm medians are **4.575 s** for
GenerativeQC and **11.234 s** for GPU4PySCF. These are complete engine-local
frozen warm endpoints with their actual SCF work: native needs one iteration,
reference repeats need one, two or four. Crosses expose variable-iteration
samples. Node3 and n5 timings are not combined into one speedup estimate.

- [summary.json](summary.json) contains all phase medians and the shared native identity.
- [samples.json](samples.json) retains every scalar native result in named columns;
  recover a row with `dict(zip(table["columns"], row, strict=True))`.
- [references.json](references.json) contains every reference sample and independent arrays.
- [work.json](work.json) retains all grouped diagnostic counters.
- [validation.json](validation.json) records tests, sanitizers, build/source identities,
  larger-size checks and known baseline failures.

The figure uses nested 3/6/12/24/48/96-atom water clusters, spherical def2-SVP,
RHF, full analytic forces and eight OMP/OpenBLAS/MKL threads. DF uses the
retained cc-pVDZ-JKFIT record (3712 auxiliaries at 96 atoms), explicit
`packed-single` values, occupied fitted response, the qualified derivative
schedule and FP64 BLAS. Native energy/density/screening thresholds are
`1e-12`/`1e-10`/`1e-12`; the reference uses energy/gradient thresholds
`1e-12`/`1e-10`, full Fock and direct screening `1e-14`. The second atom moves
by 0.001 Bohr along z before reconvergence and five further frozen replays.
This qualifies these explicit DF settings, not every automatic DF planner.

## Larger size and resource boundary

[larger.json](larger.json) retains all 28 baseline/candidate 99-atom endpoints
and their independent reference. Warm time changes from 6.259 s to 5.690 s;
all numerical gates pass. The existing value planner reduces the auxiliary tile
from 128 at 96 atoms to 12 at 99 atoms. Twenty Gram partials would be required,
so the generated K correctly retains SYRK at this boundary; compact response
still provides a 9.1% warm saving. Cold/moved iterations remain 58/13 and all
warm repeats remain one iteration.

Both versions reject the 108-atom attempt before SCF with out-of-memory status
under the measured value/response budget policy. This is retained as a resource
boundary, not an inaccurate result or a timing sample. Native memory fields are
ledger estimates, not measured whole-process peaks.

## Implementation and provenance

The response roots `r*(r+1)/2` occupied pairs and forms a weighted triangular
metric Gram. Large packed K uses a generated triangular FP64 tiled product
with deterministic reduction slices. Both reuse existing disjoint storage;
no device allocation, precision relaxation, iteration shortcut or oracle work
is added. Exact final-state/resource gates and bounded BLAS/spectral fallbacks
remain explicit. See the [decision note](../../../.agents/notes/implemented/performance/2026-10-03-df-symmetric-occupied-products.md)
and [current contracts](../../../docs/developer/df_occupied_cuda.md).

The figure's library SHA-256 is
`6b216cf564aaf61518f5497ed80071baddb1ffab610d1128a891cfac5b5cb8d4`.
The n5 combined candidate before source formatting has SHA-256
`40a483e0f17d612f4b1bd3be6860af4694b92d0d4597cec1b3d1b72b27fa35a2`;
its exact dirty source is retained in [combined-preformat.patch](combined-preformat.patch).
[source.patch](source.patch) reconstructs the measured final implementation
on the baseline above. [response-only.patch](response-only.patch) reconstructs
the earlier response-only arm. The integration recheck uses master `06459d469`;
[integration.json](integration.json) retains another complete 96-atom DF run
from clean commit `b317e0b7f`, with all gates passing. Subsequent master
`db44f7939` changes only CC arena reuse and has no HF DF overlap. The final [provider compatibility patch](provider-compatibility.patch)
keeps full GEMM available for BLAS interfaces without SYRK; the timed NVIDIA
operation sequence is unchanged. The final compatibility build passes 274 host
tests with both BLAS interfaces and 34 molecular GPU tests; its library hash
and test receipts are retained in [validation.json](validation.json). These reconstruction patches are not needed
to run the current checkout. Raw traces, logs, binaries and exploratory
products remain in ignored
`.artifacts/hf-df-96/` directories.

## Reproduce

Use a Python environment with compiler dependencies, PySCF 2.14.0,
GPU4PySCF 1.8.1, CuPy and NumPy 2.4.6. Matplotlib is needed only for rendering.

```bash
ccache --version
export CCACHE_BASEDIR="$PWD"
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DGENERATIVEQC_BUILD_TESTS=ON \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON \
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
cmake --build --preset cuda-release-sm120 --target generativeqc -j8
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_BENCHMARK_PYTHON="$(command -v python)"
export HF_BENCHMARK_OUTPUT="$PWD/.artifacts/hf-df-reproduction"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:35:00 bash benchmarks/run_hf_acceptance_benchmarks.sh
PYTHONPATH=python:. python -m tools.render_hf_acceptance_benchmarks \
  --raw-directory "$HF_BENCHMARK_OUTPUT" \
  --destination .artifacts/hf-df-figure
```

For an A/B study, build the baseline in a separate checkout with identical
flags and retain a native run from each library against the same independently
computed DF reference. Use `benchmarks.compare_df_direct_endpoint native
--nested-water --aos 768 --route df --repeats 5 --reference <reference/results.json>
--output <new-directory>` in the same finite Slurm allocation. Preserve assigned
`CUDA_VISIBLE_DEVICES`; do not mix measurements from different nodes.

For larger-size qualification only, apply [larger-harness.patch](larger-harness.patch)
in an isolated checkout. It adds deterministic 99/108-atom extensions and retains
failed endpoint status before optional metric diagnostics. Run the same reference
and native commands with `--aos 792` or `864`. These sizes are outside the README
plot. The 108-atom/864-AO case is rejected under the same measured budget policy on
both versions; its failure must not be represented as a performance sample.
