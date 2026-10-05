# Decision: prepare DF source-response execution through shared providers

Status: implemented
Date: 2026-10-05

## Problem

The streamed DF MO source reverse action used a caller-owned cuBLAS handle and
direct DGEMM callbacks. Its physical source/metric wrapper assumed the inner
operation owned numeric scratch only, so simply adding prepared providers would
undercharge the complete endpoint and break nested ownership accounting.

## Decision

Emit a strict homogeneous region portfolio from the existing reverse program.
Prepare all eight projected contraction sites before the first source read,
then execute the unchanged two-pass traversal through the shared typed table.
Keep scientific equations, orbital frame, precision, source reuse and semantic
work unchanged. The method interface carries device inputs and stream ownership.

The response owns a separate execution context. Admit descriptor host bytes and
simultaneous optional plan reservations before scratch allocation. The physical
wrapper subtracts only its exact borrowed C/W/B overlap and adds binding bytes.
Optional preparation failure releases provisional plans and selects the bounded
generated implementation before any scientific work; execution never retries.

## Rejected alternatives

Borrowing the retained source's raw cuBLAS handle would preserve the vendor
dependency and share mutable stream, workspace and scalar mode without a neutral
lifetime contract. Instead, charge the shared conservative 96 MiB context ceiling
explicitly. This is additional to the physical source's existing handle. A tight
budget admits generated execution, whose binding needs only host descriptors.
A future neutral resource pool may remove duplicate preparation and reservation.

## Invariants

- Retain legal cuBLAS when complete costs are unknown; no production cuTENSOR
  reservation profile is installed by this change.
- Test-only ceilings/ranking exercise providers without a method-level selector.
- No plan search or allocation during contraction replay; graph capture refuses
  preparation/replay until shared replay work accounting supports it.
- Every contraction contributes to the shared sticky finite audit.
- Callbacks remain provisional until success; stream-dependent storage drains
  before release, including exceptions after queued callback downloads.
- Casts, mixed arithmetic, refinement and precision audit obligations require a
  composite executor; this homogeneous region emitter rejects them.

## Evidence and limits

The standalone production consumer passes independent complete-expression gates
for unit/nonunit extents, nonsymmetric inputs, cuBLAS/generated/cuTENSOR execution,
third-plan rejection, nonfinite values and callback failures. Numerical gates
remain atol=3e-11, rtol=3e-13; work remains `3*n+5` contractions and
`6*n**3*q+2*n*n*q*q` summands. Exact minimum budgets fail closed after generated
fallback, rather than treating a larger optional provider ceiling as mandatory.

An exploratory Slurm RTX 5090 probe used the complete validation adapter,
including allocation, H2D, preparation, traversal, downloads and cleanup.
For n=16/q=24 and n=32/q=48, median times over three process-warm calls were
1.48/2.94 ms for cuBLAS and 43.73/44.69 ms for cuTENSOR 20800; cuTENSOR preparation
alone took 42.03/41.88 ms. Each provider's first process call at n=16 also included
CUDA initialization (660/925 ms respectively). This retains a negative result
for per-endpoint cuTENSOR preparation; it is not a controlled speedup experiment,
a claim about persistent cross-call plans, or complete physical-gradient timing.

Qualification used synthetic per-plan ceilings of 64 MiB workspace, 256 MiB
provider and 64 MiB host, across eight plans. Queried workspace was 0/360448 bytes;
observed device growth varied with provider caching and does not establish a
production resource bound. Opaque host/lazy allocation qualification and actual
provider algorithm provenance remain follow-up work in #1887.

## Revisit when

There is a neutral source/response resource lifetime or evidence for retaining
compatible response plans across endpoint calls. Preserve complete simultaneous
resource admission and measure the full physical endpoint before promotion.

References: #1886, #1887, #1890; `test_df_cc_source_program.py` and
`test_df_source_metric_response.py`; the preceding source-row projection note.
