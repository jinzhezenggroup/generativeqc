# Production derivative AD retirement qualification

This is the selected evidence for replacing the remaining production nonlinear
derivative programs with differentiation of their actual compiler-owned primals.
The ownership census and rejected implementations are documented in
`.agents/notes/implemented/architecture/2026-11-08-derivative-retirement-closure.md`.
Primitive rules inside the differentiation engines, independently qualified
special-function primitives, linear contractions and implicit solves remain.

## Source and retained records

- Baseline: master `97c34eeb0eb1a0fdabfb24b5436f667149701f32`.
- Candidate: that revision plus the decompressed `source.patch.xz`. It includes new compiler
  owners, generators, generated headers, consumers and tests; this publication is
  excluded, together with its storage-review record, to avoid self-reference.
  Apply it to a clean checkout of the baseline.
- `source-snapshot.json` authenticates the exact original compressed patch retained
  in local Git at `4f555d074700fdb0d340de90f57a9f8d7e4f58cf`. That commit is a
  storage carrier, not a replacement measured producer revision. The checkout
  omits this redundant blob to stay within the shared evidence storage budget;
  no measured samples or source bytes are discarded.
- `samples.json.xz` losslessly retains all selected complete endpoint timings,
  first-use observations, numerical outputs and semantic work counters. Large
  nonlocal CUDA feature vectors are replaced by their element count and measured
  maximum absolute baseline/candidate difference, without dropping any timings.
- `summary.json` records every endpoint ratio, not just aggregate speedups.
  `qualification.json` preserves the executed scientific test output and cache
  receipts. `provenance.json` distinguishes libraries, Python owners, stationary
  DSOs, scheduler allocations and the shared private runtime.
- `reproduction.patch.xz` restores the historical campaign scripts under the ignored
  `.artifacts/classical-derivatives/` directory. They are execution receipts, not
  a new required-CI framework or a replacement for the production APIs.
- `evidence.json` accepts numerical parity. Formal autotuner/default-selector
  performance promotion is **not run**: this cleanup changes no selector and does
  not claim universally significant speedups, compilation-cost or peak-memory
  qualification. Descriptive no-regression measurements remain fully retained.

Replay all retained quantitative and semantic-work checks from the checkout:

```bash
PYTHONPATH=python:. python benchmarks/results/derivative-ad-20261108/replay.py
```

The reader first verifies the shared publication schema, every member checksum,
the numerical acceptance decision and reconstruction provenance, including the
recovered patch's compressed and decompressed identities. Replay needs NumPy
and the recorded Git object locally, but no GPU, native library, private cluster
directory or external archive. A shallow/partial clone missing the carrier must
explicitly fetch its recorded revision first; replay never fetches automatically.
There is no source-fingerprint CI test and no Release/archive dependency.

## Executed scientific gates

- CPU: 122/122 CTest tests; 1194 Python tests and three subtests pass. The host
  suite skips 1601 opt-in GPU or unavailable-toolchain cases; skips are not GPU
  evidence. Pinned GFN1 and D3 reference caches were explicitly synchronized
  after an initial missing-prerequisite failure; pins and gates were unchanged.
- CUDA: 1653 tests pass without skips on n1, including actual compiled scalar
  consumers, independent libcint/Libxc checks, reconverged density response,
  pinned tblite/xTB energy/force comparisons, high-angular-momentum bounded
  forces, nonlocal consumers, nuclear derivatives and COSX scalar summands.
  Existing CuPy ExternalStream deprecation warnings are retained.
- Separate executed native gates cover D3 ATM/zero, D4 reference/EEQ, production
  D4, ragged 4100-reference D4 and gCP. The 80/90-digit primal-only quadrature,
  norm/nonlocal and 450-digit XC tail oracles remain executable tests.
- Expanded response qualification initially lacked stationary artifacts. Six
  source-matched s/p DSOs (LDA/PBE/PBE0, RKS/UKS) were built before rerunning;
  missing artifacts were not hidden by CPU fallback or a relaxed guard.

## Endpoint design and interpretation

Ratios are candidate/baseline, so smaller is faster. Do not offset a regression
in one cohort with a speedup in another. `summary.json` and `replay.py` expose the
individual cohorts, work checks, tolerances and uncertainty.

| Cohort | Complete measured work | Sampling |
| --- | --- | --- |
| GFN2 CUDA | Fresh calculator, cold/warm/displaced E+F and cleanup; H2, water, NH3, CO, HF, HCl, SiH4, water8 | 64 balanced randomized pairs each; independent first-use pair |
| VV10/rVV10 | Plan preparation, cold/warm/displaced value/features/geometry and cleanup; 128/1024/4096 CUDA points | 128 pairs at 128 points, 32 at larger sizes; full N-squared pair counts and tiles |
| CPU nonlocal | 128-point primitive and PBE+VV10/rVV10 NaH-ECP E+F lifecycles | 16 balanced randomized pairs; native-library isolation using current Python for both |
| CPU/CUDA CPKS | Construction, SCF, response-owner construction, solve and cleanup; PBE RKS/UKS, original/displaced geometries | Two reversed-order process blocks, first-use plus eight samples per worker |
| CPU KS E+F | Construction/preparation, cold/warm/displaced force execution and cleanup; PBE/PBE0 RKS/UKS, direct and DF | Two reversed-order blocks, first-use plus eight samples per worker |
| CUDA KS E+F | Same full lifecycle; PBE/PBE0 RKS/UKS direct | Two reversed-order blocks, first-use plus eight samples per worker |
| Classical CUDA | Native prepare/execute/destruction for cold, execute/publish for unchanged/displaced geometry; D3 BJ+ATM, D3 zero, D4 EEQ and gCP, 4/24/64 atoms | Two reversed-order blocks, all seven original samples per row, three executions per sample |

KS and force cohorts use source-matched baseline/candidate Python trees and
native libraries. CPKS uses the same seeded AO fields projected into each
converged MO frame; solved AO density responses, SCF/Fock work, GMRES work and
true-residual convergence are compared. Random vectors in unrelated degenerate
MO gauges are not an independent physical parity oracle.

Randomized paired cohorts report a deterministic 5000-resample paired bootstrap
95% interval for the median ratio. KS/force process blocks and classical blocks
retain their separate descriptive ratios; they are not mislabelled as independent
randomized pairs. Small positive ratios with intervals spanning parity are not
evidence of a speedup or proof of exactly zero overhead. First-use samples are
retained separately, not silently mixed into warmed distributions.

The final rebuilt candidate (`08a7efc4...`, also used in the scientific run) is
timed in n5 job 2020. Complete CUDA CPKS ratios are 0.997844–0.999480 and KS E+F
ratios are 0.999073–1.000181. The second displaced-RKS CPKS block is 1.005052,
so uniformly faster individual blocks are not claimed. GFN2 ratios span
0.988126–1.002572 and nonlocal ratios 0.999996–1.003495; all positive randomized
cohort medians have bootstrap intervals spanning parity. Classical CUDA ratios
span 0.836930–0.932171 (D3 BJ), 0.843924–0.936538 (D3 zero),
0.809519–0.876669 (D4 EEQ) and 0.651426–0.735788 (gCP).

## Hardware, builds and reproduction

All real-device execution uses finite Slurm jobs on **n1/node1 or n5/node5**, not
the n3 maintenance host. Scheduler-assigned device visibility is preserved.
Both baseline and candidate use Release builds, CUDA 12.9.86/GCC 13.3 on sm120
and verified sccache 0.16.0. CPU builds use the same local compiler/cache for both
trees. The exact CMake options and selected receipt identities are retained in
`provenance.json`; cache statistics are in `qualification.json`.

n5 has glibc 2.35 while the n1-built libraries need 2.39. Both selections execute
with the **same private n1 loader/glibc/libstdc++ runtime**, whose hashes are
retained; no system installation is modified. Slurm on n5 does not implement CPU
binding. Final n5 jobs use `taskset` on one CPU from the allocation's inherited
allowed affinity instead; old unpinned experiments are not described as pinned.

For a fresh measurement, recover the exact patch with the shared offline restorer:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/derivative-ad-20261108/source.patch.xz \
  --manifest benchmarks/results/derivative-ad-20261108/source-snapshot.json \
  --output /tmp/derivative-ad-source.patch.xz
```

Decompress and apply that recovered patch to a clean baseline checkout and
`reproduction.patch.xz` to recover the scripts. Build both libraries from the
corresponding trees with the same options and verified compiler cache. Copy the
benchmark harness into the baseline tree too, without copying candidate compiler
or production Python modules. Build the six stationary profiles separately for
each selection. Adapt cluster paths in the historical scripts to the new
checkout/runtime; these paths are not hidden scientific inputs. For example:

```bash
xz -dc /tmp/derivative-ad-source.patch.xz | git apply
xz -dc /path/to/publication/reproduction.patch.xz | git apply
sccache --version
sccache --show-stats
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$(command -v sccache)" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER="$(command -v sccache)"
cmake --build build --parallel 4
sccache --show-stats
ctest --test-dir build --output-on-failure
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --exclusive --time=00:30:00 bash .artifacts/classical-derivatives/run_frontier_n5.sh
```

Use the explicit recorded CUDA/AOT configuration rather than treating the generic
CMake example as that configuration. `run_final_retirement_cuda.sh` contains the
exact selected GPU scientific test list and flags. `run_frontier_n5.sh` contains
the final source-matched GFN2/nonlocal/KS timing sequence; classical timings use
`run_final_classical_n5.sh`. CPU workers are `qualify_ks_response.py --device cpu
--kind cpks`, `qualify_cpu_force.py` and `qualify_nonlocal_endpoints.py --device
cpu --section both --counts 128 --repeats 16`, run against both libraries with
single-thread BLAS. Keep both reversed process orders for the KS workers.

## Explicit limits and rejected experiments

GPU integral AOT shells are **off** in this bounded-path qualification. Stationary
s/p AOT DSOs are enabled independently; this is not a complete default integral
AOT inventory qualification. Existing experimental complete COSX diagnostics
fail identically on baseline and candidate with `native/Python grid plan mismatch`
(the inherited AO-panel budget mismatch). Their guard is not weakened, and no
successful complete COSX-force claim is made. Compiled COSX scalar AD has separate
passing independent-oracle coverage.

Initial nested-Jet XC emission caused a real complete-CPU-CPKS regression.
Sign cleanup, exponent alignment and forced noinline did not fix it. The accepted
flat mixed DAG shares nonlinear intermediates and derives its quotient schedule
from the differentiated residual. Early-return/eager dynamic integral adapters
were also rejected after UKS timing; the final adapter constructs tangent state
only inside the nonempty-channel branch. Earlier b40-based experiments and the
pre-final CPU-adapter GPU binary remain local diagnostics, not relabelled final
master/source-matched evidence. Full retry/build logs remain ignored locally.
