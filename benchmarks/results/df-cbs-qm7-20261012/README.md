# Correlation-only CUDA DF-MP2 and DF-CBS E+F qualification

The production model is conventional exact RHF plus density-fitted correlation:

```text
HF(CBS) + DF-MP2 correlation(CBS) + [DF-CCSD(T) - DF-MP2](cc-pVDZ)
```

The composite uses cc-pVDZ/cc-pVTZ with their explicit cc-pVDZ-RI/cc-pVTZ-RI
auxiliaries, fixed cardinal/extrapolation weights, all electrons and FP64.
This is not a DF-HF composite. PySCF is an independent qualification oracle,
never a production consumer. Native forces include auxiliary-center, metric,
orbital-response and nuclear-repulsion derivatives.

## Source and environment

Raw endpoint JSON records retain the actual native library SHA256 and hashes of
the changed native/compiler/frontend owners, oracle and qualification driver.
Those hashes are checked against the committed file bytes, not the stale Git
HEAD of a separately exported remote checkout. The qualified library SHA256 is
`a1ee0a87e89258b62e072191370070d9659aef9d7d7b3abcaf29c65766b31413`.

Runs use finite Slurm allocations on `node4` (`n4` SSH alias), partition `main`,
one `gpu:5090:1`, preserving Slurm's device visibility. Hardware is RTX 5090,
CUDA toolkit 12.9.86, SM120, Release. Profiling is disabled; source progress
traces remain enabled. Runtime/software and compiler-cache logs are retained.
Cache statistics are process/server observations, not an isolated speedup claim.

This host also has CUDA 12.4 providers. Merely setting `LD_LIBRARY_PATH` does
not prevent a Python dependency from preloading an incompatible absolute-path
library. These runs preload the built native library to pin its CUDA 12.9
dependencies. This environment prerequisite is not a CPU/reference fallback.

## Acceptance

- Integrated real-GPU suite: **8 passed**, including water physical DZ/TZ
  DF-CBS, H2/water/methane DF-MP2, degenerate same-space response, same-source
  CC/MP2/HF companions, source-only auxiliary g handle role rejection,
  failure-publication revocation, and tiny-budget refusal/recovery.
- Relevant host/compiler/API/frontend selection: **145 passed, 48 skipped**;
  skips include explicit GPU and expensive optional endpoint gates.
- Compiler ownership, public API docstrings, native API contracts, Ruff,
  formatting and whitespace checks pass. Host and CUDA libraries build.
- QM7 endpoint acceptance requires energy error at most `5e-8 Eh`, independently
  reconverged directional-force error at most `5e-6 Eh/Bohr` at both `1e-4` and
  `3e-5 Bohr`, translation error at most `1e-6 Eh/Bohr`, convergence, and actual
  CUDA backend provenance. These are directional checks, not a full independent
  Cartesian Hessian/force inventory.

Dataset input is the original `qm7.mat`, SHA256
`fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
Zero-based indices and original Bohr coordinates are retained in the JSON;
stored float32 geometry is promoted to float64 without geometry optimization.

## Complete endpoints

| QM7 case | Native E+F | AO / auxiliary counts | Time (s) | Energy error (Eh) | Maximum directional error (Eh/Bohr) |
| --- | --- | --- | ---: | ---: | ---: |
| Index 1, C2H6, 8 atoms / 18 electrons | Complete DZ/TZ DF-CBS | DZ 58 / 196; TZ 144 / 342 | 39.2277 | 3.68e-12 | 5.05e-9 |
| Index 5604, C7H16, 23 atoms / 58 electrons | DF-MP2/cc-pVDZ | 178 / 616 | 112.3759 | 7.39e-13 | 7.98e-9 |

Each row is one complete first `singlepoint(..., properties=("energy", "forces"))`
call, including SCF, source construction, correlation, response, analytic forces,
publication and group cleanup. Calculator construction, basis/data conversion
and independently reconverged PySCF oracle work are outside these timers.
These are not replay medians or comparisons against conventional/master code;
no speedup is inferred from these two timings. Translation errors are
`1.05e-12` and `1.44e-12 Eh/Bohr`, respectively.

The C7H16 output file retains the label `heptane-mp2`; no specific connectivity
or unbranched isomer is inferred from the molecular formula. Its endpoint is
MP2 E+F, not a full C7H16 CBS qualification. The 58-AO ethane correction is a
genuine complete DF-CBS run beyond the conventional 56-AO force admission.

## Reuse and capacity

Ethane's raw source trace shows two converged target-basis RHF reference owners
and two physical DF source builds. The small correction source produces five
CC core-block GEMMs; its MP2 companion consumes the same source identity without
a new metric/factor/MO build. The high-basis factors-only MP2 source produces
zero CC core-block GEMMs. Preliminary guesses, exact orbital-response actions
and derivative contractions still contribute to endpoint cost; this is not a
claim that all response work or all host orchestration is shared.

There is no inherited conventional 12/56-AO force cutoff in the DF owner.
Admission remains bounded by numerical/resource gates. Three-index storage is
`O(nAO² naux)` and generated MP2 adjoint scratch remains `O(nocc² nvir²)`.
Recorded budgets and `numeric_capacity_bytes` are planned numeric capacity,
not measured whole-process/device allocator peaks. This evidence does not
qualify every QM7 molecule or a complete CBS run at the largest tested MP2 size.
The two endpoint budgets are 4 GiB per ethane group and 8 GiB for C7H16 MP2;
their maximum reported planned capacities are 1,233,237,165 and 1,972,222,310
bytes, respectively.

## Reproduction

Build with the verified compiler cache and CUDA 12.9 compiler. From that source
checkout, preserving scheduler device visibility:

```bash
srun --partition=main --nodelist=node4 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:30:00 bash -lc '
  export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}
  export LD_PRELOAD=$PWD/build/libgenerativeqc.so
  export GENERATIVEQC_LIBRARY=$PWD/build/libgenerativeqc.so
  export PYTHONPATH=python:$PWD GENERATIVEQC_PROFILE=off
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export GENERATIVEQC_DF_PROGRESS_TRACE=$PWD/ethane-cbs.trace.jsonl
  .venv/bin/python benchmarks/df_cbs_qm7_endpoint.py --qm7 /path/to/qm7.mat \
    --index 1 --mode cbs --budget-mib 4096 --output ethane-cbs.json
  export GENERATIVEQC_DF_PROGRESS_TRACE=$PWD/heptane-mp2.trace.jsonl
  .venv/bin/python benchmarks/df_cbs_qm7_endpoint.py --qm7 /path/to/qm7.mat \
    --index 5604 --mode mp2 --budget-mib 8192 --output heptane-mp2.json
  '
```

Run `tests/python/test_df_mp2_native.py` with
`GENERATIVEQC_DF_MP2_CUDA_TEST=1` under the same environment and a finite
`srun --time=00:20:00` allocation. Use fresh trace paths for each attempt;
never merge failed and qualified traces.
