# Decision: share emitted DF-HF charge submissions

Status: implemented
Date: 2026-10-20

## Problem

The compiler's DF-HF response emitter retained two direct cuBLAS submissions
after the native response caller migrated to borrowed Tensor views in #2393.
The panel charge and retained-factor/metric-root vector contractions already
have qualified TensorIR direct-NT and direct-NN layouts, respectively.

## Decision

Emit calls to the existing `generativeqc_tensor::submit_column_gemm` adapter
from `tensor/cuda_runtime.cuh`. Scientific equations and physical views remain
compiler-owned; the existing native Tensor provider owns vendor submission.
Remove only this generator's retired vendor inventory record. The separately
embedded occupied-response emitter retains its own migration record and direct
vendor calls; neither its implementation nor the native response caller changes.

## Preserved contracts

- Panel charge retains physical T,N, `(count, terms, matrix)`, fitted/density
  leading dimensions `matrix`, output `potentials + begin` and full auxiliary
  output stride. Logical TensorIR lowering remains direct-NT.
- Retained charge retains N,N, `(auxiliary, 1, pairs)`, factor leading dimension
  `auxiliary`, density leading dimension `pairs` and output stride `auxiliary`.
  Packed density folding and symmetric metric-root reuse remain caller-owned.
- Both borrow the original handle and retain FP64, host alpha=1/beta=0 and the
  original raw status. The caller keeps its original drain, error mapping,
  allocation, workspace, stream, modes, lifetime and bounded fallback contracts.
- No new selector, mathematical algebra, resource owner, runtime allocation,
  execution template, source pass, precision policy or public ABI is introduced.
  Generation still requires neither an installed runtime nor an oracle/device.

## Rejected alternatives

A new method-local wrapper would merely relocate vendor coupling. A new shared
provider would duplicate the existing exact column-major adapter. Retargeting
the compact row-major executor or introducing a prepared portfolio here would
broaden layout, resource and selection policy beyond these two submissions.

## Evidence and limitations

The clean base `14928e2c39340e04decf1a12ee57e4eac91ef914` passed 39 related
host tests. The migrated slice's initial related run passed 93 tests, including
the existing retained dense/packed independent metric-solve and finite-difference
checks, the unchanged streamed caller suite and stationary equation/generation
checks. Host compilation used verified sccache 0.16.0. Compiler structure and
vendor boundary checks reported zero errors.

After fast-forwarding the base to `1e0aed564c27af8a2fd6b7cb3ee93e71d1691fd7`
(#2621), the final related host and compiler-structure suites passed 119 tests.
The compiler and vendor audits again reported zero errors; the native caller,
shared provider header and #2526-owned streamed test remained unchanged.

`test_df_hf_response_provider.py` executes the actual emitted helpers and shared
record/forwarder with a numerical recording vendor double, nonsymmetric inputs,
multiple terms, tails, offsets, padding, long-double references, failure statuses
and borrowed-owner sentinels. A route witness increments at the actual shared
forwarder entry without rewriting its implementation. Temporary opt-in negative
controls separately corrupted transpose, operand, leading dimension, output
offset, returned status, owner lifetime and vendor ownership: each produced five
expected assertion failures, with the original source restored afterward.
These mutations are qualification artifacts, not permanent source-shape tests.

Host evidence does not qualify actual CUDA resources, numerical endpoints or
science. Source-matched native build, real-device response and complete RHF DF
energy/force results must be retained separately with their source/library/input/
output identities and original acceptance gates. Required CI and exact-head
independent review are separate delivery gates.

The full-API dot/host-ablation fixture explicitly selects the existing `dense`
response route. An unset selector means `auto`, which may legitimately execute
occupied factor-first metric products without metric-dot counters. Binding the
fixture to its asserted operation keeps every RHF/UHF, representation, budget,
dot, numerical, resource and host-switch check unchanged. Production `auto` and
the separate occupied/fitted replay tests remain unchanged. The initial H200
AUTO invocation's work-counter failure is retained separately: its first E/F
comparison passed, but that did not make the complete work gate pass.

## Consequences and revisit conditions

This is one ownership/cleanliness slice of #1890, with no performance or default
promotion. Full canonical prepared LoweringRequest binding, borrowed BLAS-handle
debt, occupied response, CC, API, GFN and remaining provider/selector inventory
remain under #1890/#1886/#1889/#933. Revisit when a broader semantic binding can
preserve the complete response's physical and resource contracts independently.
