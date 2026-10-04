# Decision: share bounded Q tiles between DF primal and Lambda consumers

Status: implemented; GPU and complete endpoint qualification in progress
Date: 2026-10-05

## Problem and existing work

#1903 identifies repeated one-Q dispatch in the DF residual. Lambda already
received batch lifting and ordered fused accumulation in #1829; reproducing
that implementation would create competing mathematical/reduction ownership.
The primal still had six independent accumulation launches after every Q.
The complete #1900 energy experiment confirmed that DIIS bookkeeping is only
about 0.04% of ethane230 total, whereas CCSD takes about 170.6 seconds; further
DIIS tuning cannot fix this phase.

## Decision

Reuse `df_lambda_matrix.batch_auxiliary_program` and its existing packing
transform for the primal auxiliary DAG. Batches expose Q as an explicit IR
axis; shared amplitudes/tau do not acquire Q. A nonunit symbolic representative
produces runtime Q extents, including unequal final tiles. Every Q-dependent
intermediate remains visible to the ordinary liveness/capacity queries.

Move the existing Lambda output-consumer emission to one shared native emitter
helper. The primal supplies its six typed cut outputs to the same helper.
Each lane begins from its retained sum and performs the original serial FP64
Q additions. There is no tile subtotal, atomic reduction, pair projection or
precision change. Every addition retains its sticky nonfinite check.

The shared tensor `gemm` adapter can borrow the already admitted native owner
handle and accept the IR coefficient. Ordinary and strided dispatch stay in
that adapter; no new CC-local vendor call/discovery/auto-tuning is introduced.
This retains the existing compiler callback ABI while the #1889 portfolio
integration proceeds independently. No tensor/context ownership is transferred.

## Admission, fallback and accounting

Start from the proven one-Q matrix plan and test at most the requested limit
(default eight), halving rejected candidates. Every trial includes all resident
state, factors, retained host payloads, provider allowance and scratch. Optional
size overflow rejects a tile without poisoning the one-Q candidate. Matrix
provider dimension limits are derived from the actual packed IR dimensions.
Allocation failure retries a one-Q matrix arena before existing scalar fallback;
other CUDA, provider and arithmetic failures propagate without publication.

Only hoisted CUDA primal execution is tiled. Original expanded physical replay
remains one-Q and its scratch stays admitted. For a tile of b and C retained
cut elements, one accumulation launch touches `(b+2)*C*8` logical bytes instead
of `3*b*C*8` across separate per-Q accumulations. Six launches per Q become one
per tile. Q contraction/packing work is queried for the actual tile extent,
including shared nodes and tails, rather than multiplying every operation by b.
All-Q four-index intermediates remain forbidden; peak batched intermediates
scale as bounded `O(b*o²*v²)` and never have more than two virtual axes.

Diagnostics distinguish actual Q slices, tile count, operation/provider calls,
semantic summands, packing traffic and accumulation traffic. None is a FLOP
estimate or physical bus measurement. CPU execution is unchanged.

## Qualification

n2 Slurm job2282 passed 12 compiler/ownership tests, including existing Lambda
batch algebra and actual native constructor injection for tiled-allocation,
one-Q allocation and provider failures. It also passed 13 CPU solver cases
(five CUDA-only skips), compiler/SCF/cross-method structure, native complexity,
default-promotion inventory, metadata and ownership checks. No node3 build or
compute job was submitted.

The real generated consumer regression uses adversarial cancellation and early
overflow followed by opposite-sign rows; it also guards unused output spans.
CUDA solver comparisons cover Q=5 and limits 1/2/4/8, independent determinant
replay, exact/one-byte-short capacity, scalar/resource fallback and sticky
failures. Complete energy pairs and the same candidate's one-Q endpoint are
queued after CUDA qualification. No phase win or completed large-force
qualification is claimed until those records pass.

## Revisit conditions

Keep a losing/resource-rejected tile on its bounded fallback; launch reduction
alone is insufficient evidence for a new default. Any altered Q reduction tree,
mixed precision, source prefetch/double buffering or provider autotuning requires
separate ownership and scientific/performance qualification. Independent Lambda
audit batching is deliberately not folded into the primary residual change.
