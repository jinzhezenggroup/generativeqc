# Incremental CUDA KS full-energy refinement: PBE0 cold evidence

This experiment diagnoses and repairs a screened full/ΔD energy-qualification
ordering bug after PR #2044. It is **numerical acceptance of a termination
repair**, not a performance/default promotion. The paired repaired cold endpoint
is essentially unchanged with incremental mode enabled: 214.26 s versus
214.07 s median, with overlapping ranges and only two samples per arm.

## Complete endpoint results

| Cohort | Complete cold seconds | Median (s) | SCF iterations |
| --- | --- | ---: | --- |
| Original simulated merge, ΔD on | 425.82939, 416.48573 | 421.15756 | 78, 76 |
| Repaired binary, ΔD off | 218.96381, 209.18039 | 214.07210 | 27, 25 |
| Same repaired binary, ΔD on | 232.12613, 196.38478 | 214.25545 | 32, 26 |
| Prior master, cache-populated diagnostic | 206.68878, 232.29197 | 219.49038 | 25, 29 |

The same-binary primary order is off/on/on/off, with a fresh process and
prepared owner for every sample. Its incremental speed ratio is 0.99914x,
or 0.08565% more time: there is **no established cold performance win**.
Relative to the original broken incremental cohort, the observed repaired
median falls by 49.127%, or 1.96568x speedup. This repairs the long-tail
regression; it is not an incremental speedup over ordinary full-density SCF.
The earlier-master cohort is a cross-binary diagnostic, not the primary paired
comparison. No timing is divided by its SCF iteration count.

All 18 experimental records are retained. Fifteen accepted complete endpoints
pass the existing independent GPU4PySCF gates. `master-off-1`, `merged-off-1`,
`fixed-on-0` and `repaired-on-0` are identified separately as program-cache
first-fill samples, not discarded or mixed into primary medians.

`fixed-*` records belong to a rejected first repair that entered full refinement
but retained redundant RKS corrective closure. `fixed-on-1` exhausted four final
corrections, returned native status 4 and no forces at iteration 37; its 223.90 s
is not a successful E/F endpoint. The failed result's public Fock-build count is
null, while its diagnostic and actual full/delta/final counts retain 37 builds.
`fixed-off-2` was interrupted before a result after rejecting that repair.
`merged-gate4-1` is an interrupted 1e-4 admission-threshold trial, not part of the
zero-threshold repair qualification. Interrupted raw records retain their
original `running` status, with explicit disposition/reason in `summary.json`.

## Science and timing boundary

- Nested water-96, 768 spherical def2-SVP AOs and 48 × 16 × 32 grid points per
  atom, using the existing README PBE0/RKS inputs.
- Exact direct J/K, strict FP64, no density fitting, energy tolerance 1e-12 Ha,
  density tolerance 1e-10, screening tolerance 1e-12 and 100 iterations.
- Incremental density-RMS threshold zero in all qualified repair samples. No
  energy/density/residual tolerance is relaxed to manufacture convergence.
- Complete cold timing is preparation plus the first synchronized energy **and
  analytic-force** execution, including force-owner runtime preparation and host
  force return. Native-library compilation is excluded. No density warm start,
  unchanged-geometry replay or moved-geometry population is measured.
- Independent oracle:
  `../pbe0-composed-baseline-20261005/reference-96.json.gz`, identical protocol,
  1e-8 Ha energy and 1e-7 Ha/Bohr maximum-force gates. This is the existing
  reference, not a newly timed GPU4PySCF run. The repaired maximum errors are
  below 7.8e-11 Ha and 8.3e-11 Ha/Bohr.
- Every raw record preserves its complete iteration history, route/resource
  observations, residuals, force work and actual full/delta/final build counts.
  Quartet-work counters are explicitly invalid; their zero values cannot prove
  an admitted-quartet reduction. Whole-process peak memory and isolated native
  compilation cost are not measured, preventing performance qualification.

## Source identity and validation

The permanent reconstruction base is master
`e6f1a6b0c16b2763ee65e124188fdfbea37a68a5`, composed with #2044 head
`31cbaf03aa699a57aa589f0ed9191c544e0186ea`; the original simulated merge tree
is `34b6c9cc478cf2dd194f0217cb9229b44ed72a0f`.
The production repair is edited on master
`f05015e7c71809155b9be0aa69c3ff62d5d89e30`. The measured library deliberately
uses the frozen simulated composition rather than unrelated intervening
Array API/GFN2 changes; its PBE0 owner matches that master before the repair.

`merged-source.patch` reconstructs the original measured library source from
the permanent base. `rejected-first-repair.patch` applies on top of that original
composition and preserves the failed alternative. `repaired-source.patch`
applies directly to the permanent base and includes all measured library,
compiler and Python owners plus the focused native/controller regression tests.
The reconstructed canonical source identities are independently verified:

| Source | Canonical source SHA-256 |
| --- | --- |
| Prior master | `e50158036b7ec15abaa20ecf505c89c31392f3a341adae5d4f48fb555b3e7307` |
| Original merge | `afaf381722150902ac46b10cb570998061cf020ef74a6b6414f7125e26be436e` |
| Rejected first repair | `19bf682c3dcd8b7d7f4c52bdee66b31905be30693000b08c6489718447c5ce19` |
| Accepted repair | `be6ed903542e00954c5ab8d5b78fd8bc09df5a51028b7cc9ea677ef21aca706e` |

The repaired native library SHA-256 is
`4ddb6e3aa9c920846f8be6b48acb2cd264572a5fc764837f4bb9a25486047411`.
All primary samples use that same binary. Focused host/controller/source/resource
tests pass (79 tests), as do the native incremental buffer/controller test and
the KS exact-exchange RKS/UKS physical-state, final-export and fresh-solve tests.
The monolithic KS suite has an unrelated final-state-validation failure also
reproduced with the original unpatched library; it is not silently reported as
a passing full suite. Summarized test outcomes are in `test-results.json`; raw
build/test logs and binaries remain outside Git.

## Offline verification and reproduction

From the repository root with NumPy and the project Python package available:

```bash
PYTHONPATH=python:. python benchmarks/results/pbe0-incremental-refinement-20261007/verify.py
git worktree add --detach /tmp/pbe0-incremental-repaired e6f1a6b0c16b2763ee65e124188fdfbea37a68a5
git -C /tmp/pbe0-incremental-repaired apply "$PWD/benchmarks/results/pbe0-incremental-refinement-20261007/repaired-source.patch"
python benchmarks/results/pbe0-incremental-refinement-20261007/check-source.py.txt /tmp/pbe0-incremental-repaired
```

The verifier authenticates the publication inventory, recomputes all 15 accepted
E/F comparisons, preserves failed/interrupted classifications, checks actual
build-count conservation and recomputes primary and diagnostic medians. Samples
use deterministic gzip and the repository's shared record reader.

Build with CUDA 12.9.86, GCC 13.3, release `sm_120`, RTX 5090 and the captured
Python environment. Preserve/reuse ccache; the measured CMake CXX and CUDA
launchers are both `ccache` (4.5.1). `cold.py.txt` is the exact frozen harness;
`run-repaired-n1.sh.txt` captures the node-specific cache/compiler/library
environment and sample order. Adapt its local root paths when reconstructing.
Every real-GPU command must run through Slurm, preserving assigned visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodelist=node1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:25:00 bash /path/to/run-repaired-n1.sh
```

The durable decision and rejected alternatives are documented in
`.agents/notes/implemented/numerics/2026-10-07-incremental-ks-full-energy-refinement.md`.
