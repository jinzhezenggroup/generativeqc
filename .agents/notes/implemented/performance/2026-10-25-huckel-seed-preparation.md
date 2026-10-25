# Decision: use native linear algebra and idle CUDA history for Hückel preparation

Status: implemented
Date: 2026-10-25

## Problem and measured causes

For the default Direct HF96 energy-plus-force endpoint, cold Hückel preparation
cost about 0.49 s. The two 224-dimensional molecular subspace eigenproblems
consumed about 0.22 s in the shared scalar provider, and strict target-metric
seed admission consumed another 0.22 s. Preparation is not seven molecular SCFs:
the seven preliminary Fock builds are the existing spherical atomic RHF model.

## Decision and ownership

Keep the atomic model and its strict tolerances on the existing scalar owner.
Use the available one-thread native LAPACK provider for the molecular Hückel
subspace only. Resolve an unavailable provider to scalar before execution;
never retry a failed provider with a different numerical backend. Both routes
retain the original eigen residual and orthogonality gates. Form the molecular
closed-shell density with the shared symmetric rank-k owner, then mirror its
lower triangle before strict admission.

The shared CPU solver adapter owns density recovery as well as preliminary
diagonalization. CUDA products enter through `scf/cuda/matrix_library`, which
owns row-major conversion and tensor-provider status mapping. The initial-guess
modules do not acquire direct dependencies on either tensor implementation.

For CUDA HF at 512 AOs and above, the initial-density validator may borrow three
idle DIIS matrices and the already prepared cuBLAS handle. Two buffers hold
inputs and one holds the product. The second Fock-history matrix exists only
when at least two history entries were reserved; otherwise keep host products.
All items are serialized before DIIS begins, so even batch-neighbor history
storage is idle. This adds no device allocation, handle, graph, stream or owner.
Every product drains asynchronous host reads/writes before return or exception.
The callback is explicit, synchronous and never retained; warm inputs bypass
cold seed preparation. A malformed result or provider/allocation failure cannot
silently switch product backends.

Seven full-target validation products move to the borrowed GPU workspace: two
per symmetric eigenframe and three for metric/occupation admission. Their
mathematical equations, original inputs, finite checks, trace, symmetry,
occupation and final energy/force gates do not change. The shared host frame
validator uses scaled sums of squares instead of repeated `hypot`, retaining
underflow/overflow protection and all acceptance thresholds. Long-double
analytic tests independently cover scales from 1e-290 to 1e150, plus existing
nonfinite/overflow and degenerate-frame tests.

The 512-AO crossover conservatively keeps smaller owners on host products.
Diagnostic 384- and 768-AO device products were beneficial, but no tiny-system
provider promotion is inferred from those measurements.

## Resource boundaries

The preliminary numeric cap remains unchanged. Molecular assembly/admission are
serialized. The LAPACK molecular solve fits inside the existing sixteen-target-
matrix envelope: its input/returned frame, LAPACKE row-major transpose and
DSYEVD work are bounded by the retained molecular allowance; fixed/linear
allowances cover small dimensions and atomic metadata. The native 1088-AO
allocation stress and 48 independent atomic-oracle cases still pass. That
allocation probe observes C++ payloads, not external BLAS allocator RSS.

Large external-provider HF is still outside the public exact-byte inventory:
`estimate_resources(...).require_feasible()` deliberately refuses it. The first
large exact-budget qualification attempt is retained as a failed admission,
not a memory-pass receipt. Existing small-owner tests qualify exact public byte
caps. Separate real 768-AO runs qualify history-one host fallback, history-eight
borrowing and two-item batching, with a one-iteration target limit and the
existing single bounded core retry.

The initial, pre-adapter-refactor sanitizer invocation reports ten expected
CUDA API errors from the
pre-existing negative XsyevBatched graph-capture capability probe (errors 900/901),
not memory access errors. The retained rerun uses `--report-api-errors no` to
check memory accesses and reports zero errors. The final adapter version repeats
that memory check, all resource cases and the full numerical/endpoint suite.
Do not describe this as a clean all-API-error run; both logs are retained.

## Complete endpoint evidence

Base: `98531471e` (full revision and exact measured patch in the capsule).
The candidate changes only seed preparation and the common validation reduction;
there is no experimental cold screening, threshold relaxation or reference seed.
Three alternating rounds on one Slurm RTX 5090 allocation, each with one fresh
cold population and two frozen warm calls:

| Atoms | Control cold (s) | Candidate cold (s) | Control warm (s) | Candidate warm (s) |
| --- | ---: | ---: | ---: | ---: |
| 48 | 6.1495 | 6.0957 | 0.7472 | 0.7480 |
| 96 | 18.5074 | 18.1604 | 2.9258 | 2.9294 |

HF96 median seed preparation falls from 0.4864 to 0.1255 s. Cold target work
remains 10/11 iterations for HF48/96. All 36 paired energy-plus-force calls and
8 original/warm/moved/fresh-moved independent GPU4PySCF gates pass. The latter
maximum force discrepancy is 2.7521e-7 Eh/Bohr. The qualification suite passes
242 tests; standalone analytic validation/provider suites pass another 9 tests. These
measurements use the final SCF adapters after dependency-boundary cleanup,
not the earlier pre-refactor build. Warm variation is below 0.2%; this change
does not establish a warm improvement.
Compiler-cache version, before/after statistics, exact source delta, binary
hashes, allocations, raw samples and replay are in
`benchmarks/results/hf-seed-preparation-20261025/`.

## Rejected experiments and remaining work

A separate bounded screened cold-SCF experiment plus these seed changes reached
11.97 s versus GPU4PySCF's 11.82 s in three alternating HF96 rounds: still about
1.3% slower cold, with warm 2.92 versus 3.36 s. This is not the default behavior
or a result delivered by this PR. It requires a real policy, lifetime and stage
work-accounting qualification before promotion. Two-level screening trials and
native MINAO/incremental alternatives did not establish a better qualified
endpoint. Their raw source, controls and failed/negative receipts remain under
ignored `.artifacts/parity-research/` in the research checkout and
`n1:/data/jzzeng/gpu4-parity-research-20261025/`.

The user requested ending HF tuning here and prioritizing larger DF costs.
Next work should address DF primitive/source repetition and capacity-aware
value/force schedules, with explicit resource limits and independent E/F gates.
Do not spend another tuning campaign chasing this small HF residual before the
larger DF work is addressed. Root README data is not changed by this decision.
