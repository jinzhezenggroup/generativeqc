# CUDA occupied DF force response

This is the device-specific response and resource contract for
[occupied RI-K](df_occupied_cuda.md), not another definition
of RI-K mathematics or an independent force implementation.
It consumes the verified [DF final-state identity](df_final_state.md)
and the [generated DF derivative owner](df_derivatives.md).
Borrowed raw/source memory must remain valid for the complete response.

## Resident raw ownership and response storage

A full, single-system host-raw plan retains the original FP64 values in its
former exchange-contribution buffer as `A[Q,mu,nu]`. Setup fills this buffer
while the original raw device input is still live. The new resident K path
fits its temporaries in the two other tensors, so no extra full tensor is
allocated and both the setup and persistent reservations remain unchanged. Transformed B
alone cannot recover discarded metric directions needed by exact forces.

The prepared HF owner binds immutable raw, atom and shell allocations plus
both basis representations. These allocations survive moves into the prepared
cache. A force call must match these bindings before receiving a
`CudaDfRawTensorView`: pointer, dimensions, strides, process-unique owner
identity, and the original metric eigensystem/cutoff. The view always denotes
untruncated raw values, never a streamed panel or transformed B. Rebuilding
geometry, either basis, or the metric owner invalidates the previous view.
Standalone tensor-plan callers have no immutable source binding and upload
through the compatibility adapter.

J/K and response share one stream. Two full buffers become mutable response
scratch; the retained raw buffer remains read-only until the bridge drains.
Matching warm resident calls perform zero raw-tensor H2D copies or transposes.
`GENERATIVEQC_DF_RAW_REUSE=off` retains the upload ablation. An upload revokes raw
validity before submission and restores it only after successful response
from the matching immutable source, including failure/retry handling.

`GENERATIVEQC_DF_RESPONSE_STORAGE=auto` borrows full J/K capacity for singleton RHF
responses with default shell/BLAS controls. Dense response also benefits from
projecting each auxiliary only once, so unavailable occupied factors do not
force repeated panel projections. Occupied algebra additionally requires the
shared work/capacity policy and validated final-state factors. Packed storage
can only lend its smaller scratch to a qualified occupied response.
`panel` preserves bounded execution;
`jk-scratch` requests validated borrowing explicitly. Borrowed capacity is
reported once alongside owned scratch and transfers; reuse is never inferred
from dimensions or a small component-local budget alone.

## Exact occupied force response

`GENERATIVEQC_DF_RESPONSE_SPACE=auto` (also unset) shares SCF's rank/work and resident
capacity selector. Explicit panel storage and diagnostic schedules preserve
their original route. `dense` retains the full-AO comparison; `occupied` requests
factor validation on compatible resident plans, including explicit UHF/batch
experiments. Neither the work model nor a token used as a selection hint
supplies execution authority.

The method passes its verified final-state token. The response owner checks
source identity, solve epoch, system, model, occupations, exact canonical device
density, and each device factor generation before borrowing C. Missing/stale
tokens, corrected determinants, external densities, unreserved plans and
unsupported factors keep dense response under `auto`. UHF additionally verifies
the exact sum of its spin densities and admits both rank-squared projections
together.

On singleton, full-rank streamed RHF plans only, an explicit
`GENERATIVEQC_DF_RESPONSE_SPACE=occupied` request may reconstruct a *new* algebraic
factor from a corrected final density. It requires the same source/model/solve
epoch/occupation, bounded matching density and orbital generation advances,
and the charged occupied value-plan reservation. A GPU eigensolve and full
density reconstruction gate reject indefinite, non-finite, excess-rank or
inexact densities; the existing bounded dense response remains the fallback.
Before overwriting factor scratch, the response revokes the previous SCF
generation, so this factor is never advertised as a canonical SCF factor.
`auto`, UHF, batch and truncated-metric response policies are unchanged.

The response computes `T_Q=C^T A_Q C` and `U_P=sum_Q V_PQ T_Q` from raw
three-center values, preserving finite discarded metric directions. In the
qualified domain it feeds at most 64 auxiliary slices of packed symmetric
AO-pair weights to the existing generated derivative consumers. It does not
retain a full response-weight tensor. Projections, transformed projections and raw
values occupy the three already charged resident J/K tensors; the consumed
projection buffer becomes panel storage. `GENERATIVEQC_DF_RESPONSE_BATCHING=auto`
concatenates Q slices for a large `C^T [A_0 ... A_(a-1)]` GEMM and batches the
second multiplication by C. Previously retained spin factors remain outside
that staging range. Pseudo-density expansion batches `C U_P` and lower
rectangular products across each bounded auxiliary panel. Capacity checks
include weights, projected factors and rectangular outputs simultaneously;
`off` and insufficient capacity preserve serial projections. The algebraic
FLOP count stays fixed while BLAS submissions and repeated factor reads fall.
Additional response workspace contains
four auxiliary matrices, three AO matrices, densities and auxiliary charges.
`DfGradientResources` reports the executed route and borrowed capacity.

`GENERATIVEQC_DF_DERIVATIVE_PAIRS=auto` selects packed weights for the qualified
768/768-AO, rank-160 RHF occupied response on RTX 5090. The previously promoted
resident 192--384-AO shell route instead folds the existing dense weights,
preserving its response producer. Other automatic routes retain their original
execution. Explicit `full` executes the ordered dense shell product;
`symmetric` adds the two dense off-diagonal shell weights and executes each
unordered shell pair once. `packed` requests packed production when a trusted
occupied response and complete generated shell consumer are available, otherwise
retaining the dense producer and symmetric shell consumer. Generic and host
consumers retain their dense contracts. These controls are captured in resource
plan identity and cannot be changed inside a frozen global resource plan.

Packed weights store `W_ii` once and `W_ij+W_ji` at `i*(i+1)/2+j` for `i>j`.
Diagonal-shell AO pairs and normalized spherical/Cartesian expansions preserve
their physical atom derivatives. Auxiliary panels end at whole shells whenever
the cap permits; a smaller explicit cap can still split a shell. Generated
derivatives reduce directly into the atomic gradient, without a derivative
tensor or a second derivative formula implementation.

The producer expands only lower rectangular AO blocks, with 256 rows by
default. `GENERATIVEQC_DF_PACKED_AO_BLOCK_ROWS=64|128|256|384` retains diagnostic
alternatives. Upper off-diagonal blocks are skipped; unused upper entries
inside diagonal blocks are counted as executed work. Counters distinguish
shell pairs/triples, public weight loads including zeros, primitive products,
nonzero Cartesian contractions, rectangular GEMM entries and panel bytes.
`GENERATIVEQC_DF_SHELL_COUNTERS=1` enables diagnostic device atomics and must be
disabled for clean timings.

Packing halves the weight handoff, not the complete resident plan allocation.
Raw uploads, validated raw reuse and borrowed J/K capacity are reported
separately; packed weights do not imply a smaller persistent value plan.
Tiny explicit domains can fit in one panel or one AO block; counters report
their actual dense and packed materialization rather than implying a saving.

The [derivation and lifetime note](../../.agents/notes/implemented/performance/2026-09-15-occupied-df-response.md)
records RHF/UHF coefficients, metric response and rejected schedules.
The [qualification evidence](../../benchmarks/results/issue377-379-df/README.md)
retains frozen-density policy comparisons, independent strict force gates,
component/work counters, reservation and priming costs. These are historical
validation points for the shared work policy, not runtime admission branches.
They establish no universal device latency or COSX crossover. Memory diagnostics
report charged capacity, not a measured global GPU peak. The derivative schedule,
pair-layout and primitive-packet selectors below remain separate policies;
their existing endpoint restrictions do not restrict occupied SCF admission.

The [packed derivative note](../../.agents/notes/implemented/performance/2026-09-15-packed-df-derivative-pairs.md)
documents symmetry, diagonal-shell treatment, the block-size tradeoff and
[current qualification](../../benchmarks/results/issue382-packed-df/README.md).
