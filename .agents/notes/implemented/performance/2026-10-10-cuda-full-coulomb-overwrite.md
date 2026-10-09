# Decision: CUDA Coulomb overwrite and scoped HF device facts

Status: implemented
Date: 2026-10-10

## Problem and scope

CUDA's retained full-range Coulomb auxiliary cleared its entire four-dimensional
simplex before filling it. This occurs inside primitive value/derivative sources,
including d/f Direct-HF consumers. The explicit sweep adds up to 1,820 scalar
assignments at total order twelve without supplying a mathematical boundary
condition. This is a source-level work count, not measured hardware traffic:
the compiler can eliminate stores, and some jet types also initialize members.

The recurrence candidate changes only the full-range initialization sweep. It does not
change recurrence arithmetic, traversal, precision, screening, source admission,
primitive/component counts, shell ordering, graph scheduling or optional
resident allocation. SR/LR's failure/initialization behavior stays unchanged.
The rejected bounded through-f value route remains default-off.

Retain the full-domain overwrite and the scoped shared-device-facts lookup
described below. Neither requires a new resident allocation or a molecule-specific
branch; the existing complete provider and batched grid-limit fallbacks remain.

## Dependency proof

For total order L, the simplex consists of n+t+u+v <= L and contains
choose(L+4,4) states. Full-range filling writes every state before its consumers:

- All R(n,0,0,0) roots are assigned first.
- The z line visits increasing v; each read uses v-1 or v-2 at n+1, already
  covered by the preceding line's larger n extent.
- The y plane visits increasing u for each v; u-1/u-2 dependencies at n+1
  belong to an earlier u plane or the initialized z line.
- The x volume visits increasing t for each u/v pair; t-1/t-2 dependencies at
  n+1 belong to an earlier t volume or the initialized y plane.

Every cell is assigned exactly once by that fill, so an initial zero cannot be a
live recurrence operand. Invalid/unselected component domains still execute the
complete simplex, while the existing valid reachable domain keeps its own
already-qualified dependency closure. No new recurrence or precision policy is
introduced.

## First-reuse setup boundary

The initial recurrence-only candidate is numerically qualified, but its H2
first-reuse endpoint regresses repeatably. Slurm 705 expands the initial four
warm samples per library to eight interleaved processes with thirty repeats
each. Steady warm medians are unchanged (2.6858 versus 2.6848 ms), while the
first warm call is 6.9148 versus 7.9231 ms: a real 14.6% first-reuse regression,
not noise to discard. Process-cold and later reconstructed/moved calls do not
show the same regression. Retain both the original and follow-up receipts.

Static disassembly of the H2 `build_eri_kernel` is byte-for-byte identical
between frozen libraries. Matched Nsight node diagnostics also show the same
approximately 0.675-ms value kernel and 0.169-ms one-electron kernel, but
instrumentation perturbs setup and must not replace unprofiled endpoint timing.
Slurm 708's minimal runtime interposer preserves the first-reuse difference:
the initial asynchronous allocation costs about 0.9 ms in the control versus
1.6-1.9 ms in the candidate, while full device-property queries cost another
approximately 1.4 ms. This locates the observed penalty in workspace setup;
it does not establish a particular driver-internal allocation cause.

Narrowing `OrdinaryStreamEigensolver`'s architecture query is an unsuccessful
intermediate experiment. Slurm 709's 1,584 public samples pass the physical
gates, but H2 first reuse remains 6.8653 versus 7.8489 ms. The constructor is not
the query owner on this RHF route. The change and its new helper-specific tests
are reverted; do not infer a live endpoint owner from a similarly named class.

Slurm 710's runtime backtrace pins the live property query to
`execute_hf_cuda_bucket` in `src/scf/cuda_rhf.cpp`, before its direct schedule
selection. The final candidate delegates that lookup to the existing
`runtime::cuda_device_facts` owner. Its selected attributes provide the same
complete scheduling/resource facts on NVIDIA/Linux; missing attributes or
optional driver metadata and other providers retain the authoritative full
property fallback. No device ordinal or resource result is cached across owners.

The driver's only additional property is `maxGridSize[1]`. It is consumed solely
by multi-system, non-bounded quartet compaction. Keep the original full property
query there, after admitting that branch, rather than querying unrelated limits
on every single-system call. This preserves its exact 2D-grid admission, flat
fallback and CUDA failure mapping. All scientific arithmetic, scheduling policy,
workspace sizing, allocation accounting and source counts remain unchanged.

The existing independent host ABI gates cover NVIDIA and CuMetal, changing
device facts, missing attributes/driver metadata, partial failures and stale
runtime-error preservation. New source contracts protect the single-system
query boundary. The SCF inventory admits exactly `runtime/cuda_device_facts.hpp`,
not a general runtime prefix; negative tests retain rejection of unrelated
architecture and ledger owners.

Qualification must compare the combined candidate against the original
control, explicitly including first reuse rather than hiding it inside a
thirty-repeat steady median. The earlier recurrence-only and eigensolver-only
follow-up candidates are not the candidate proposed for production.

## Independent gates

The host probe poisons all FP64 and Dual3 cells with NaNs and checks every state,
including n>0, against independent 96-point interval quadrature and analytic
Cartesian derivatives for orders zero through twelve. Existing reachable,
invalid-domain, full/SR/LR and Hermite-convolution controls remain unchanged.
The final recurrence, convolution, endpoint-driver, shared-provider, device-fact
and SCF-boundary suite passes 2,003 host tests. Its one real-device fact test is
intentionally excluded from host execution and is separately run through Slurm
using the same native fixture. These host checks are not GPU endpoint
qualification. Configured pre-commit hooks and the 338-file CUDA ownership audit
pass. The reviewed ownership delta is scientific LOC -5, runtime LOC 0, with no
new/removed/reclassified CUDA files or new generated scientific owner.

An additional real-device probe compiles the actual candidate header with NVCC
12.9/O3/sm_120, poisons every FP64/Dual3 workspace, exports all five fields per
cell, and compares them to independent interval quadrature. Its per-thread local
state coverage is separate from Compute Sanitizer initcheck, which checks global
initialization and does not prove every local auxiliary cell was written.
Slurm 702 passes the real-device probe for all 6,188 simplex cells and 30,940
exported fields. The largest raw auxiliary absolute error is 4.2564352e-10;
the fixed 2e-10 relative plus 2e-11 absolute quadrature gate passes. These are
auxiliary values/jets, not energy or physical-force errors. Both memcheck and
initcheck report zero errors. Complete candidate endpoint qualification is
still pending.

## Complete endpoint protocol

`benchmarks/cuda_hf_high_l.py` runs strict CUDA RHF with standard unmodified
orbital bases, no density fitting and no injected reference density. Its
independent PySCF 2.14.0 energy/analytic-force preparation is outside the timer.
Acceptance remains 1e-8 Eh and 1e-7 Eh/Bohr. Backend substitution, nonconvergence
and nonfinite errors fail; partial failures retain valid JSON with null rather
than invented zero errors.

Cold times calculator construction and the public endpoint; warm reuses that
calculator at the same geometry; moved uses that calculator at a displaced
geometry. Only the first cold sample includes process-first CUDA startup.
Individual samples retain iterations, executed backend, errors, actual library
and driver hashes, GPU metadata, Slurm visibility, affinity and policy/runtime
environment. Unscreened Cartesian quartet/primitive domain bounds are explicitly
distinguished from unknown executed primitive counts. Native canonical-work
tests provide separately observed source counts; logical bounds must not be
relabeled as device execution or hardware FLOPs.

Promote only after frozen control/candidate libraries pass independent native
full/range/J/K/spin/representation/derivative gates and interleaved complete
small/control/larger cold/warm/moved endpoints. Energy timings do not establish
force speedups. All real-device work uses finite Slurm jobs on allowed n4, with
the scheduler's CUDA_VISIBLE_DEVICES preserved. n3 is not used.

## Complete endpoint evidence

Slurm 712 compares the original control with the combined candidate on the same
n4 RTX 5090, CUDA 12.9 runtime, driver 13.0, visibility, CPU affinity and scientific
policies. Each library runs in its own process. H2 uses four interleaved
ABBA/BAAB blocks, eight processes per library and thirty repeats per process.
Other energy cases use one ABBA block and three repeats; force cases use two
repeats. First reuse is reported independently from steady reuse.

All 1,584 public cold/warm/moved samples pass with actual backend `cuda`.
Maximum energy error is `1.1321876769443406e-10` Eh and maximum physical-force
error is `3.16349554191353e-10` Eh/Bohr. Every matched SCF iteration list and
logical source domain is unchanged. The independent real-device fact fixture
also exactly matches the full-property oracle, rejects an invalid ordinal and
preserves the selected device.

| Spherical endpoint | Control median | Candidate median | Time reduction | Samples/library |
| --- | ---: | ---: | ---: | ---: |
| H2 first reuse | 6.8592 ms | 6.4629 ms | 5.78% | 8 |
| H2 steady reuse | 2.6800 ms | 1.8081 ms | 32.53% | 232 |
| Water/def2-SVP warm energy | 0.478816 s | 0.475655 s | 0.66% | 6 |
| Water/def2-TZVP warm energy | 1.396150 s | 1.379014 s | 1.23% | 6 |
| Water dimer/def2-TZVP warm energy | 2.537925 s | 2.406574 s | 5.18% | 6 |
| Water dimer/def2-TZVP moved energy | 2.849794 s | 2.698611 s | 5.31% | 6 |
| Water dimer/def2-TZVP process-cold energy | 3.376441 s | 3.213926 s | 4.81% | 2 |
| Water/def2-TZVP warm energy+forces | 1.750489 s | 1.733066 s | 1.00% | 4 |

The dimer has 86 public spherical/96 Cartesian AOs, two f shells and an unchanged
unscreened primitive-component product bound of 41,655,563. Original/moved SCF
iterations are 16/18. Water/TZVP has 43/48 AOs and bound 2,692,535, with fifteen
iterations. Small water and force deltas are modest; do not claim a robust
all-d/f or general force speedup. The two process-cold samples per larger-case
library do not establish a broadly reproducible startup gain. Cartesian water
is physically qualified but has only one independent process per library.

The H2 first-reuse regression of the earlier candidates is not hidden by a
steady median: the combined candidate improves that independently reported
endpoint as well. Public timings retain ordinary production MD-J policy.
Batch/prefix-budget behavior is separately covered by native independent gates;
these singlepoint timings do not establish batch or constrained-memory speedups.
Frozen JSON receipts and all-sample summary are retained as
`qualified-*.json` and `qualification-v3-summary.json` in the artifact directory.

## Build and diagnostic provenance

The isolated source base is master `1db02ce36`. CUDA Release uses verified
ccache 4.5.1, GCC 12, CUDA 12.9 and sm_120, with fast compilation OFF. Frozen
baseline library SHA-256 is
`c9ecdac397816ab0e85b179c5762f2cbe38e25f342a80fb6b5b8f860e400ff6c`.
The frozen candidate library SHA-256 is
`70427ece303e9dcc6a6dffea4f522af50b585d98a157c1201c569d7a60c314cd`;
its candidate header SHA-256 is
`fc00739f896034cfc0728ac6f735dd91bb7664be00f0df212db83ad38c288c46`.
Its original header and native-test binary are retained before copying the
candidate header into the build source. A separate loader directory pins the
control library so its tests cannot silently borrow the candidate after relink.

That hash identifies the retained recurrence-only experiment. The reverted
eigensolver follow-up has SHA-256
`80f2ffb8e3fa02828664426b0a38e7c5ecf84407223d0c1e8ed23a29c59e3e5a`.
The final combined candidate has SHA-256
`48a2c39765661ebc894c20cb9778764d7d863537ad0862022eea74d2c0dcd743`.
Its HF driver source SHA-256 is
`f667c381ada5d1fab53646eaa541239dca865ebd5a2fd34010e77fd085394ae8`;
the recurrence header and endpoint benchmark remain unchanged. The incremental
build uses the same verified compiler cache and configuration, with separate
before/after cache statistics and a separate frozen library/loader directory.

Raw remote evidence is retained under
`/data/jzzeng/qc-cuda-hf-high-l-20261009/{results,source/.artifacts/cuda-hf-high-l}`;
local host/audit evidence is in `.artifacts/cuda-hf-high-l/`. Pilot traces used a
different existing library and are diagnostic only, not matched qualification.

Two initial profiling jobs failed because the Python runtime preloader selected
CUDA 12.4 cuSOLVER before loading a CUDA 12.9 native library, despite an explicit
LD_LIBRARY_PATH. The qualification launcher pins the actual CUDA 12.9 user-space
providers with LD_PRELOAD; device visibility is never overridden. These failures
remain retained and are not numerical passes. This task does not change the
unrelated runtime discovery policy.

Nsight Systems defaults to whole-graph tracing on this driver, omitting graph
node activities from kernel summaries. A node-granularity diagnostic was added;
partial graph/conditional execution traces must not be treated as complete
SCF work counts or proof of CPU substitution. Profiled wall times are not
unprofiled performance evidence.

The first standalone probe unnecessarily requested a 128-KiB per-thread CUDA
stack limit and failed allocation before evaluating a cell (Slurm 700). The
probe removes that unrelated override and uses normal runtime limits. This is
not a production candidate allocation failure or a successful numerical gate.

Slurm 701's initial control native gate fails its generated-J channel census
because the production-default MD-J owner replaces that channel. The numerical
matrix check precedes that owner-count assertion. Slurm 703 explicitly uses the
existing GENERATIVEQC_DISABLE_MD_J=1 qualification setting for native
through-f-response, canonical-values and canonical-work checks; all pass without
changing their independent numerical gates or the census assertion. This pins
the owners that those fixtures count, not a new production default. Public
water/water-dimer pilot endpoints in the same job run with ordinary MD-J policy
and pass independent original/moved energy gates. Candidate native qualification
uses the same owner precondition, while every matched public benchmark retains
the normal production policy. Do not relabel the failed unpinned census as a
default-policy pass.

Slurm 712 requalifies the final combined candidate with all five native modes:
through-f response, range response, canonical values, canonical work and mixed
census. All pass, including full/SR/LR, J/K/spin/representation/derivative and
batch/prefix-budget gates. Its complete public endpoint and real-device-fact
gates also pass. Final source hashes match the frozen build; configured hooks
and the host/ownership gates pass without changing a scientific ledger shard.

## Revisit and disposition

The combined candidate is retained on the evidence above; the earlier
recurrence-only and unused-eigensolver experiments are not retained defaults.

Do not infer an endpoint gain from deleting a source loop or from an isolated
kernel improvement. If complete endpoints regress, restore the production sweep
and preserve this as rejected evidence. If the sweep was already optimized away
or endpoint differences are noise, report that limitation explicitly instead
of claiming a d/f-wide CUDA speedup. Broader source-driven primitive reuse and
register/local-memory work require their own qualified schedule changes.
