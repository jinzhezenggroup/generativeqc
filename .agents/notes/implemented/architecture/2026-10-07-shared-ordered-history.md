# Decision: compiler-owned ordered Johnson history algebra

Status: implemented; independent full review and real-device qualification pending
Date: 2026-10-07

## Scope and parent

This completes the bounded emission gate documented in
`../../proposed/2026-10-07-ordered-history-emission-gate.md`.
The independently reviewed gate db07e8fa (tree 9c653568) was integrated with
parent 07e7f37a (tree 505fec4a), retaining its weighted-Gram provenance repair
and current master changes. Integration commit: 9000d582.

The common xTB CPU and GFN2 CUDA mixers now consume the shared compiler's
history-window/overlay, full Gram/RHS, lower-Cholesky solve and chronological
seeded correction. Four generated includes replace 134 CPU and 140 CUDA lines;
the eight include directives are the only replacements in those native files.

## Ownership and actual selection

`tensor/ordered_history.py` owns the closed recurrence contract and ordinary
TensorIR scalar stage Programs. Existing add, multiply, divide and sqrt nodes
remain the sole scalar algebra. The source-preserving statement mode is in the
existing scalar C++ lowerer; it neither introduces another scalar IR nor forces
FMA contraction policy. Default scalar helper output is unchanged.

`ordered_history_candidate` constructs real LoweringRequest/LoweringCandidate
records with graph identity, explicit physical layouts, FP64 precision,
finite-failure/publication effects and the retained execution algorithm.
The Gram's two history modes are distinct, describing a full matrix rather than
a diagonal view. CPU leading dimension is h; CUDA leading dimension is capacity.
Native runtime admission still owns actual buffer extents and alias checks;
the default candidate extents identify a shape-independent code template.

`require_history_candidate` verifies the complete selected candidate before
`method/gfn2_history_lowering.py` dispatches every generated phase through its
execution algorithm. An unsupported/forged candidate aborts generation. A
candidate's identity is not just emitted beside an unrelated implementation.

`tensor/ordered_history_gram.py` and `ordered_history_emit.py` share loops,
scalar stage emission and effects across the two schedules. Method bindings
supply physical reads, regularizer/weight inputs and error destinations.
There are no separate copied complete CPU/CUDA algebra bodies.

## Retained semantics

- Chronology: h=min(k,m), tentative slot=(k-1)%m, chronological slot j=(k-h+j)%m
- Tentative normalized df/u/weight overlay reads precede persistent ring reads
- Increasing packed-component dots; weights multiply after the dot
- Full h-by-h Gram, then diagonal omega_zero squared; no symmetry shortcut
- Lower Cholesky, increasing forward and decreasing backward substitution
- Damped seed followed by increasing-history subtraction, never a summed correction
- CPU short-circuit/immediate failures; CUDA error+atomic-invalid and dot-local break
- CUDA matrix/result stores still precede their final checks; CPU retains its
  original check/store distinctions
- Partial scratch writes on failure; persistent history and diagnostics publish
  only through the unchanged native commit and status wrappers

Packing, normalization, omega selection, startup, activity/convergence decisions,
launches, barriers, state ownership, allocations and error publication stay native.
No new provider, BLAS path, allocation, launch, precision mode or performance
claim is introduced. No generic capacity-64 restriction is added below the
existing method admission.

## Source/build identity

Production include expansion retains the complete frozen translation units:

- CPU SHA-256: 844ad997749c33b55bdc8d568c7dd985bfa262f661b1284c69cec8d86c88983d
- CUDA SHA-256: 1e96b2154098685b62b5c7be6659daafacc1e0cc29bfc67f273a1896cffa1b68

The generator emits eight fragments and two identity JSON files, binding each
source digest to its scientific identity and full admitted candidate. CMake
registers every output for CPU and the CUDA archive, with tracked transitive
Python depfiles. The generator itself is added to GenerativeQCSourceIdentity.json;
all compiler modules already belong to its recursive source inventory.
The physical CUDA mixer hash and adaptation description are refreshed in
CUDA_SOURCE_PROVENANCE.json, preserving upstream hashes and prior adaptations.

## Evidence and reproduction

Required cached CPU probes compile the actual common mixer translation unit.
Capacities 1/2/4/64 run beyond startup, through partial/full windows and repeated
wraps (133 steps for capacity 64). An independent chronological NumPy Johnson
oracle constructs its own df/u/weight history. Each transition is checked, then
only committed vector inputs are rebased to the consumer to avoid amplifying
open-loop roundoff; oracle history is never copied from consumer history.

Tests cover ragged dimensions, tentative/inactive history poisoning, scratch
canaries, all persistent outputs/diagnostics, seven failure modes and exact
failure retention, plus the epsilon-normalization/maximum-weight corner.
Separate emitted-helper tests use NumPy and an 80-digit Decimal solve oracle,
padded Cholesky strides, signed zero and seeded-correction cancellation.

Real CUDA fixture code exercises the same production launcher in stream and
captured graph-replay modes, including padding, inactive peers, sticky errors,
failure retention and recovery. This environment has no nvcc: those ten cases
are explicitly not compiled or executed. Existing opt-in CUDA molecular and
electronic fixtures retain their own qualification skips.

The production CPU library and CLI build with verified ccache 4.14.1. The full
selected run passed 296 tests with 22 explicit CUDA/tool/qualification skips,
including CPU molecular goldens, multi-step analytic-force differences and
geometry covariance. The initial molecular run lacked the existing LP64 loader
path; reusing the parent's verified SciPy provider fixed setup without changing
the provider, ABI or production code.

Reproduce the real consumers with:

    CCACHE=/workspace/scratch/1b7630fc9eb4/ccache_tool/ccache-4.14.1-linux-x86_64-musl-static/ccache \
    CCACHE_DIR=/workspace/scratch/1b7630fc9eb4/compiler_cache PYTHONPATH=python:. \
    /tmp/review-queue-venv/bin/python -m pytest -q tests/python/test_ordered_history_consumers.py

For molecular/force tests use this checkout's built libgenerativeqc.so and
LD_LIBRARY_PATH=/tmp/xtb-migration-provider:/tmp/review-queue-venv/lib/python3.12/site-packages/scipy.libs.
The latter is the reviewed parent's existing LP64 provider route, not a new
dependency or changed runtime policy.

Final build/source identity matched exactly:
`cad92cb32cbe7b671546f1da792758c41bbea84cc8dc20e81a929358f39a57a1`.
Generated artifact identities: CPU
`3d484097f5d6c9f0e50fdb3edda5e2039ba3cf5b4e6c7f4ff2dc209a97a7ce1e`,
CUDA `f244c11c54c4124254f69177237a3a1dcccc09580f6a55c7e3f24b707092cf09`.
The physical CUDA mixer hash recorded in its preserved provenance manifest is
`c4443a26ec4b27a77072f4e26e7f325d2e5b29dcd76fe4a348aa4d504fa953c6`.

## Limits and revisit conditions

Byte-preserved CUDA source is not real-device numerical/capture evidence.
Run the optional CUDA consumer and full existing CUDA endpoints on a qualified
device before claiming device execution. Independent full review precedes any
publication. Optimization issues #1882/#1879 remain parked; this migration does
not resolve their visibility/discrepancy prerequisites or promote a faster path.

## Mandatory CUDA compile/link qualification

After exact recovery of tree 901a138c and independent source review, integration
13444a9b (tree c838078d) brings in master 7dc9d79 without changing the 21 migration
files. A bounded qualification repair makes CUDA fixture compilation mandatory
whenever the native NVIDIA archive is present and native CUDA tests are built.

`generativeqc_ordered_history_cuda_compile_check` is a SHARED library containing
the real fixture and production mixer translation units. It owns its generated
include dependency, links CUDA::cudart and, on Linux, rejects undefined symbols.
The native CUDA runtime-test target depends on it. Standard 20, architectures,
RDC/device resolution, compile pool and inherited cache/contraction settings
match the native archive; no FMA override is added.

This target is neither a CTest test nor a dependency linked into/loaded by the
runtime executable. CuMetal, disabled CUDA and configurations without the native
GFN2 archive do not create it. Optional device/capture execution stays optional.

The focused regression runs the actual production CMake helper, inspecting
Ninja edges, source/include/link properties, inherited launcher, architecture,
RDC/device-resolution properties and negative provider/archive guards. Relabeling
CUDA sources as CXX permits graph inspection only; it does not compile them.
Actual NVIDIA compilation/linking remains unrun on this host without nvcc, but
is now a required dependency of the normal NVIDIA CUDA runtime-test build.
