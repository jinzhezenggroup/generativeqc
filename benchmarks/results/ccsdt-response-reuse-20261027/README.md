# CCSD(T) response reuse audit, 2026-10-27

Decision: reject promotion of the scalar response/resolvent composition. Keep
production unchanged. Refs #2516; remaining forward W/panel residency needs the
generic machinery owned by #783/#1889. This publication does not close #2516.

The full rationale and source/physical-binding ledger are in
[the rejected experiment note](../../../.agents/notes/rejected/2026-10-27-ccsdt-response-reuse.md).
The audit uses the existing TensorIR optimizer, not a new CC optimizer.

| AO | Samples per arm | Baseline wall (s) | Experiment wall (s) | Descriptive gain |
| --- | ---: | ---: | ---: | ---: |
| 28 | 4 | 5.045 | 5.025 | 0.40% |
| 56 | 4 | 42.180 | 42.045 | 0.32% |
| 230 | 2 | 215.225 | 214.775 | 0.21% |

These are fresh interleaved complete cold E+analytic-F processes, including
startup and teardown. The repository gate requires at least five paired samples
and improvement beyond max(2%, estimated noise). This exploratory experiment
fails to justify promotion; no formal statistical pass is claimed. Compile
elapsed wall was not measured. The validation envelope deliberately leaves
performance and production as `not-run`, with the missing qualifications explicit.

## Contents

- `audit.json.xz`: runtime-scaled compiler CSE ceilings, exclusive baseline GPU
  ledger, exact physical panel/W identities, final scalar kernel measurements,
  memory inventories, descriptive medians and quantitative numerical errors.
- `samples.json.xz`: every final-source endpoint JSON and whole-process time
  (20 samples). All iteration counts, packing/work metrics, resource refusal
  flags, Q batches, final residuals and force vectors are preserved.
- `receipts.json.xz`: per-arm loaders, binary hashes, Slurm/GPU identities,
  original wall-time receipts, test output/JUnit and compiler-cache receipts.
- `oracles.json.xz`: fresh 28/56-AO independent PySCF directions and energies;
  explicitly historical 2026-10-10 230-AO independent energies and two-component
  force differences. Historical reference timings are not new A/B measurements.
- `validation.json.xz` and `publication.json`: scientific scope, negative
  performance decision, exact source revision/dirty state and file checksums.
- `production-source.patch.xz` and `qualification-tests.patch.xz`: the rejected
  experiment, reconstructed on `fa5a5c5613e816924bfe0b03734ecb3bad2cf5f9`.
  Neither patch is applied to production by this PR.
- Three endpoint inputs, two water metadata records, and `reproduction/` recipes.

Python's `lzma.open(path, "rt")` reads the compressed JSON files directly.
Large Nsight/SQLite traces and build logs remain outside Git; retained hashes
identify the local originals. All durable inputs and reconstruction patches are
in this bundle or the stated Git revision. No external archive is required to
repeat the measurements. Historical raw profiles are not needed to rerun them.

## Reproduce the CPU audit and retained summary

From the repository root, with its normal Python test dependencies installed:

```bash
PYTHONPATH=python:. python -m benchmarks.ccsdt_response_reuse_audit \
  --occupied 9 --virtuals 221 --auxiliaries 488 --output /tmp/2516-audit.json
PYTHONPATH=python:. python -m pytest -q tests/python/test_ccsdt_response_reuse_audit.py
python benchmarks/results/ccsdt-response-reuse-20261027/reproduction/summarize.py
```

The opt-in audit requires a raw output location outside tracked results. It
records source hashes rather than pinning production source spelling in CI.

## Reproduce the rejected GPU experiment

Use separate checkouts of the recorded revision. Apply `production-source.patch.xz`
and `qualification-tests.patch.xz` only to the experiment checkout, decoding each
with `xz -dc PATH.patch.xz | git apply -`. The validation source hash identifies
the decoded production patch; the publication manifest hashes compressed bytes. For the
baseline device ledger use another baseline checkout with `instrument.py ROOT`;
this overlay adds NVTX scopes only. Do not time the instrumented binary as a
clean endpoint. Preserve the uninstrumented baseline library before building
and saving the experiment library.

The retained `.sh` recipes are the actual commands, with original machine paths.
Adapt their `root`, `deps`, Python and CUDA installation paths. They expect
`$root/source`, `$root/evidence`, and saved `control`/`qualified` subdirectories
of the evidence directory, each containing its own `endpoint` and
`libgenerativeqc.so`/`libgenerativeqc.so.0` (the latter may be a local symlink).
Copy the three `.input` files into that evidence directory. Build both source
variants with `build.sh` and save each library and endpoint before changing the
source; use equivalent flags and do not mix objects or runtime libraries.

Recorded toolchain: CUDA 12.9.86, GCC 11.4, CMake Release SM120, portable_cuda AOT,
ccache 4.5.1; CPU linear algebra used the scalar fallback. Preserve existing
cache contents. `build.sh` verifies ccache and records before/after stats.
Link each endpoint with `-Wl,--enable-new-dtags,-rpath,LIBRARY_DIRECTORY` so its
RUNPATH does not override the intended per-arm library. The final recipes
assert `ldd` resolution and record binary hashes before every endpoint run.
Early cohorts with incorrect DT_RPATH resolution were excluded completely.

Every GPU execution must be scheduled, preserving Slurm's CUDA visibility.
SSH alias `n2` has Slurm node name `node2`. The recorded allocation was
`main --gres=gpu:pro6000:1`, RTX PRO 6000 Blackwell, driver
595.99.02, 600 W. For example, on n2 after adapting the recipes:

```bash
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash /path/to/compare-final.sh
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 bash /path/to/clusters-final.sh
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash /path/to/profile-final.sh
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash /path/to/test-final.sh
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash /path/to/memcheck.sh
```

The local RTX 5090 uses `main --gres=gpu:5090:1` instead; select a compatible
available node and exclude n3 while it is under maintenance. Never change
`CUDA_VISIBLE_DEVICES` to bypass scheduling. Primary comparison jobs were 3585
(230 AO, candidate/baseline/baseline/candidate) and 3586 (28/56 AO, four samples
per arm). Profile jobs 3579 and 3590 used Nsight 2025.1.3.140. `profile.sh` is the
original baseline diagnostic recipe; inspect its library paths before replay.
`analyze_profile.py PROFILE.sqlite OUTPUT.json` reproduces the exclusive
baseline GPU ledger; it never adds nested host timers to kernel durations.

For the fresh CPU oracle, decompress the two water metadata files into an
ignored run directory, adapt `root` in `force_oracle.py`, and run with
`PYTHONPATH=python:.:tests/python`. It uses PySCF 2.14.0 and the existing
`test_df_complete_force.independent` original-equation oracle, with directions
seeded by `2516 + nbf` and two finite-difference steps. `cluster_inputs.py`
records how the retained inputs were constructed from standard nested water
coordinates and the existing water fixture's orbital/auxiliary bases.

## Validation limits

The rejected patch passed a scheduled CUDA suite of 99 tests (three skipped),
39 report tests and
12 host tests (one skipped). The GPU skips cover two corrected-Lambda/factor
chain opt-in fixtures and an allocation-refusal interposer case; these remain
unqualified. Sanitizer job 3591 passed one multi-CTA scalar test, zero errors.
The 28/56-AO independent molecular checks are directional, and the historical
230-AO finite differences cover two coordinates, not all components. Complete
cross-arm force parity covers every retained force component (maximum difference
3.29e-14); all tested endpoint energies are identical across arms. No scientific
tolerance was relaxed.

Tracked all-kind live-allocation peak (9,450,642,566 bytes), sampled whole-device
VRAM (27,489 MiB), and modeled endpoint numeric capacity (11,961,676,969 bytes)
are different measures. They agree across variants within their respective
measurement scopes. Copy payloads and compiler work counts are not measured
DRAM traffic. CUPTI reports zero local bytes per thread for these scalar
kernels; no separate hardware spill-traffic measurement was collected.
