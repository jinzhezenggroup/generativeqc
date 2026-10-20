# Decision: borrow Tensor column views for occupied DF response submissions

Status: implemented
Date: 2026-10-21

## Problem and decision

The compiler-owned occupied response emitter still submitted four ordinary and
two strided-batched FP64 GEMMs directly. The existing Tensor runtime already
owns exact column-major borrowed submission adapters for these physical views.
Emit those adapters without introducing another provider, selector, allocation,
execution template, public ABI or mathematical owner. The compiler continues
to own the occupied projection and metric-layout equations; the native Tensor
provider owns vendor submission. This is a bounded ownership slice of #1890,
not a performance or default-promotion change.

## Preserved contracts

Here `n` is the AO dimension, `r` the occupied rank, `a` the auxiliary dimension,
`rr = r*r`, and `ar = a*r`. These are the original physical column-major views:

| Submission | Transposes | m, n, k | Left/right/output leading dimensions |
| --- | --- | --- | --- |
| Panel first projection | N,N | n,r,n | n,n,n |
| Panel second projection | T,N | r,r,n | n,n,r |
| Finish final-K projection | N,N | ar,r,n | ar,n,ar |
| To metric eigenbasis | T,T | a,rr,a | a,rr,a |
| Apply metric root | N,N | rr,a,a | rr,a,rr |
| From metric eigenbasis | T,T | rr,a,a | a,a,rr |

- The first panel submission retains panel stride `n*n`, zero-stride broadcast
  coefficients and temporary stride `n*r`. The second retains zero-stride
  coefficients, temporary stride `n*r`, output stride `rr`, original batch count
  and output offset `begin*rr`. Panels remain caller-advanced pointers.
- All submissions borrow the original handle and host scalar pointers, with
  FP64 alpha=1/beta=0, original operands, order and raw status. Admission,
  zero-rank/no-op behavior and the first-error return remain unchanged.
- Native callers retain allocation, stream, pointer/math modes, workspace,
  lifetime, draining and error mapping. No native caller or shared provider
  header changes. The final-K gather and existing resource/work budgets remain
  authoritative.
- The symmetric metric Gram's two SYRK references, requires detection, lower
  triangle contract and full GEMM fallback remain unchanged. There is no existing
  column-SYRK migration interface for this slice. The inventory therefore remains
  classified `migration`, with one Dgemm and two Dsyrk references.

## Validation and evidence boundaries

The unchanged-production Linux host baseline passed 311 related tests. The new
test-first boundary suite passed 79 behavioral cases and failed precisely its
five ownership-route cases before migration. After migration, the related suite
passed 397 tests, including the original projection SYRK/no-SYRK variants,
retained metric/finite-difference references, streamed caller failure handling,
Gram admission and retirement checks. Compiler structure checked 574 modules
and the vendor audit checked 34 files, both with zero errors.
An additional compiler ownership, packaging and vendor-boundary suite passed
82 tests on the same production source.

The new independent tests compile the entire emitted occupied helper API with
the actual `ColumnGemmView` / `ColumnBatchedGemmView` records and forwarders
extracted from the production header. Only the vendor boundary is a numerical
recording double; a witness at the adapter entry distinguishes ownership bypass.
This is execution of the actual adapter boundary, not compilation of the entire
Tensor CUDA Context or evidence of real device execution. Long-double/NumPy
references use nonsymmetric rectangular inputs, advanced panels, nonzero begin,
broadcast strides, output tails, raw error statuses, null/invalid admission,
empty ranks and borrowed-owner sentinels. A failed first submission cannot reach
the next one; a failed second submission leaves projected output unpublished.

The existing projection fixture needed only actual adapter declarations before
the complete emitted API and the cached native compiler fixture. All original
test bodies, parameters, independent oracles and assertions remain unchanged.
The shared fixture and #2526-owned streamed test/native caller are unchanged.

Eight opt-in controls were run outside required CI. Harmless formatting and
local parameter renaming each passed all 84 new tests. Corrupt transpose,
leading dimension, batch stride, begin offset, raw status and vendor bypass
produced respectively 4, 10, 4, 2, 5 and 1 expected assertion failures. A separate
one-time emitted-byte comparison preserved tile admission, symmetric-pair and
Gram helpers. These controls are qualification artifacts, not permanent source
hashes, AST spelling checks or private-call-count requirements.

Host compilation used verified sccache 0.16.0. The legacy single-source probes
that hardcode `c++` / `ccache c++` used an ignored, explicitly documented local
compatibility launcher that separates compilation through sccache from linking.
No shared fixture, existing toolchain or cache contents were modified to provide
that launch path. Initial local Windows GBK decoding and host preparation
failures (missing ccache, a CRLF shell script and an incorrect script text
replacement) are retained separately; they are not numerical failures or passes.

Source-matched native SDK/header compilation, installed compiler/asset identity,
actual allocated H100/H200 numerical contracts and original complete molecular
energy/force/resource/work gates are recorded separately in delivery evidence.
Host checks and negative controls cannot substitute for those gates, required
CI, latest-head independent formal review or protected merge verification.
Ignored local receipts are under `.artifacts/1890-occupied/`; the independent
remote experiment directory is
`/inspire/hdd/project/chemicalreaction/czxs25220150/1890-occupied-response-1021`.
Historical #2637 libraries and results do not qualify this new source.

## Rejected alternatives and remaining scope

A method-local wrapper would only relocate vendor coupling. A new shared
provider would duplicate the already qualified physical column adapters.
Expanding into SYRK, triangular Gram output or a new selector/LoweringRequest
would change the ownership and qualification scope of this finite slice.

Canonical prepared semantic binding, borrowed-handle debt, the remaining Gram
operations and the broader method/vendor-selector inventory remain #1890 work.
This slice does not close the parent issue or reopen historical architecture
issues. Revisit the retained Gram boundary only when an appropriate shared
semantic interface can preserve its triangle, fallback and resource contracts.
