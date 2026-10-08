# Decision: solver-owned CPU Johnson-Broyden state and admission

Status: implemented; CPU qualified, CUDA execution unchanged
Date: 2026-10-08
Base: 99a2a5cac926edc69c28453977d1bc9b70d8bd06
The work began at 3b97c234 and integrates the unrelated #2120/#2122/#2116 changes.

## Problem

The shared compiler already owns ordered Johnson history algebra, but its CPU
consumer's immutable ragged plan, bounded storage binding, failure admission,
initialization, restart and per-system transactions still belonged to the
embedded xTB runtime. The otherwise model-neutral owner depended on xTB status
ABI declarations and method-owned compiler bindings. Moving the directory alone
would leave those dependencies intact.

## Decision and retirement ledger

`src/solver/cpu/johnson_broyden.{hpp,cpp}` owns the complete caller-bound CPU
Johnson-Broyden service. `src/solver/johnson_broyden.hpp` supplies a typed operation
result, explicit numerical policy, and explicit int32 status encoding. The GFN2
adapter is a real consumer: it maps qsh/dipole/quadrupole fields, admits the
wavefunction layout, supplies the existing success/invalid/numerical status
codes, and maps typed operation results to its existing method ABI.

The old `src/xtb/native/src/model/common/scc_mixer.cpp` (1,180 lines) and matching
header (199 lines) are removed from production and build registration. Most of
those 1,379 lines implement retained capability, now solver-owned; this is not a
claim of 1,379 net project lines deleted. The shared provider replaces three
private CPU size-add/alignment/append helpers with the existing transactional
`runtime::WorkspaceLayout` and checked alignment. Signed descriptor arithmetic
and all exact-binding/range checks remain explicit.

A frozen test-only copy of the exact pre-extraction owner and consumer provides
an independent compiled differential reference. It is not a compatibility owner
and has no production build registration. Its manifest records source hashes.

## Compiler ownership

`tensor/ordered_history_artifacts.py` admits the complete lowering candidate and
assembles generated artifacts for both consumers. Generic CPU bindings now live
in `tensor/broyden_cpu_lowering.py`; a CPU-only generator invocation never imports
method modules. The shared CPU provider consumes `generated_broyden_cpu_*`
fragments. Their four mathematical bodies are byte-identical to the preceding
`generated_gfn2_history_cpu_*` bodies. Existing diagnostic strings are retained
for source and failure-message compatibility; they do not create a method
implementation dependency.

The GFN2 compiler binding owns only the unchanged CUDA physical bindings. All
CUDA artifacts, including identity JSON, stay byte-identical. CPU and CUDA
CMake generation are separate commands, with each command's complete transitive
input closure and production consumer dependencies registered.

## Invariants and boundaries

- Caller-owned persistent numerical storage keeps the same sizes, offsets,
  alignment, field packing, iteration/restart counters and int32 status records.
  The additional status encoding lives in immutable metadata and the binding
  descriptor. xTB status is exactly int32_t; the adapter asserts this at compile
  time and introduces no reinterpretation of enum objects.
- Plan copies preserve identity. Failed admission never replaces an existing
  plan or publishes partial bindings. Successful steady transitions allocate
  nothing and use existing caller-owned scratch.
- Initialization is all-or-nothing across the ragged batch. Restart and state
  transactions touch only the selected system. Numerical failure changes only
  that system's status; raw values and persistent numerical history remain
  unchanged. Healthy peers can still commit.
- Johnson ordering, finite checks, epsilon normalization, weights, compact CPU
  Gram layout, generated scalar arithmetic and publication order are retained.
- Full GFN convergence remains in its existing driver. The CPU driver still
  overrides the residual diagnostic byte using energy and RMS convergence and
  still publishes raw terminal multipoles. The solver does not suppress a
  requested transition based on that diagnostic byte.
- CUDA source, host admission, activity masks, launch sequence and kernels are
  untouched. The CUDA provenance manifest is untouched. This slice makes no
  new GPU execution or performance claim and does not remove the whole native
  runtime.

## Generic admission repairs

Independent review exposed three inherited invalid-input gaps that normal GFN
layouts did not exercise. The generic owner now rejects workspace alignment
below alignof(double), misaligned field offsets, and mismatched field element
counts. Canonical state/scratch descriptors cannot reside inside their own
numerical arenas, including copied descriptors that bypass initial binding.
Vector base/range admission precedes exact-field checks, and exact pointers are
compared with overflow-checked integer addresses, avoiding undefined pointer
arithmetic for null or wrapping forged bases. These are intentional stricter
invalid-input contracts; valid layout/state bytes and trajectories remain
compatible with the frozen owner. Dedicated negative tests and UBSan probes
cover these cases. No numerical acceptance policy changed.

## Rejected alternatives

Using the shared DIIS vector-of-vectors owner would change history representation,
allocation behavior and Johnson semantics. Injecting a template implementation
solely to retain xTB typedefs would hide the method dependency rather than remove
it. A caller-selected status encoding plus typed operation results is the bounded
adapter contract instead. Sharing CUDA admission in this slice was deliberately
excluded; it would expand the change beyond the reviewed CPU owner.

## Evidence

- Final focused selection: 267 passed, 23 explicit skips. This includes the
  independent NumPy chronological oracle at history capacities 1/2/4/64;
  frozen-prechange differential complete-state/offset/diagnostic comparisons;
  method-free exact-binding, restart, transaction and zero-allocation probes;
  CPU numerical-body equality; CUDA complete expanded-source equality; and
  ownership/provenance gates. Ten skips require NVCC/device execution and
  thirteen require Ninja for their graph-inspection fixtures. The portable
  production CPU-only and CUDA-present CMake generation tests both passed and
  executed their actual generators using Unix Makefiles.
- Frozen differential coverage compares every persistent byte and 22 resource
  size/offset entries through ragged multifield trajectories, peer-local
  failures, restart/counter overflow, invalid binding, whole-batch initial
  admission and recovery. The reference is independently hash-guarded.
- Independent review recompiled the complete method-free contract under UBSan;
  all allocation/binding/transaction and invalid-address cases passed. The
  selected tests also prove an arbitrary consumer status encoding {7,-3,41},
  while the real GFN adapter retains {0,1,6}.
- Production CPU Release library and CLI built with verified ccache 4.14.1;
  all 78 native CTest tests passed after integrating master 99a2a5ca.
  Unrelated stationary CPU force AOT was disabled for this build.
- Complete GFN molecular tblite goldens, multi-step analytic-force differences,
  geometry covariance, retained-runtime and orbital transaction/public API
  gates: 35 passed, 13 explicit CUDA qualification skips on the integrated tree.
- The native LP64 factory used the existing verified SciPy OpenBLAS provider,
  SHA-256 8fb864c29cac4b25f6e2c139491ea96f2724dde42d51394f84e9c4a622e34790.
  No provider implementation, selection policy or precision setting changed.
- Source inventory and loaded native-library identity matched exactly:
  52e39fb7baab9f1d7f0c0cabc44c4c182108ee95e732f30806f1c0609ab2adf2.
- Compiler structure and electronic-structure boundary checks passed. The exact
  CI command `uvx --with 'numpy>=1.24' --with 'tomli>=2' ty==0.0.82 check`
  passed using task-local writable cache/tool directories. Ruff 0.16.10,
  clang-format 23.1.2 and diff whitespace checks passed. The narrow frozen-fixture
  pre-commit exclusion preserves the independent baseline's exact source bytes.

Reproduction logs, compiler-cache before/after statistics and build identity
are retained locally under `.artifacts/broyden-owner/`. The cache is shared;
its global counter delta is not a task-isolated cache-hit metric.
The CUDA mixer source hash remains
c4443a26ec4b27a77072f4e26e7f325d2e5b29dcd76fe4a348aa4d504fa953c6;
its entire provenance manifest is unchanged. No real-GPU execution or new
performance qualification is claimed.
