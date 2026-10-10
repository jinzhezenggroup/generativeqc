# Decision: bounded Q batching of fresh expanded physical replay

Status: implemented, scoped CUDA DF energy dispatch
Date: 2026-10-10

## Problem and decision

Incoming-residual mixing leaves about 9.16 seconds in the mandatory independent
expanded physical replay of the ethane230 CUDA DF CCSD(T) endpoint. Its original
virtual graph executes 488 one-Q slices. Q-lift that graph with the canonical
compiler batching/packing helpers, not with hoisted primal cuts. Accepted
amplitudes remain shared; every retained output total visits each Q addition
in original order and checks finiteness before later cancellation.

`df_replay_auxiliary_batch` defaults to false. Only CUDA DF energy owners without
retained response request up to 16 lanes. CPU, conventional, force/Lambda/response
and ordinary iteration fallback remain unbatched. The independent expanded core
is unchanged and fresh. Source inputs are only bov, bvv, t1 and t2.

Admission follows the original primal, single-Q replay and occupied-pair choices.
Existing device scratch is a hard ceiling, not a reason to shrink those choices.
Charge bounded host descriptors to complete numeric capacity. Budget/dimension
or optional binding refusal keeps one-Q replay. Drop batching before existing
OOM fallbacks without retrying an identical device allocation.

## Evidence

`benchmarks/results/df-cc-batched-replay-20261010/` preserves all original
inputs/outputs, samples, failures, compiler/link commands, source/object hashes,
recipes and quantitative acceptance gates in a checksum-bound lossless bundle.

- Original virtual graph hash: `98fc4de32f63064d4a4475b25a1ab65472262cfa9960b4a3e9a2bbd9ea92eab8`.
- Original packed hash: `83897491af50dd86ff4b054efdb67f9f960cea064278030e20f143bcf153f6d6`.
- Batched hash: `d54486f16ed5bca0aa33a35a7d0faf837a45d801560e840d1e8de4e1993a32c3`.
- Accepted incoming-residual library SHA:
  `d851b9c58061a7e99906c8128644e2457fec6a8592357524a4655463ae994fa7`.
- Production library SHA:
  `26321a3275dbed10908688040c29a5f0176aa32bd1cbd4066f94c653ee514f39`.
- Ninja's dependency closure rebuilds 21 affected objects, including all consumers
  of the new SolverOptions field. Do not guess that an appended internal field
  makes an old default-construction object safe to borrow.
- 16 independent determinant-oracle CPU cases; eight prototype and eight actual
  production CUDA actions; 13 prototype and 13 production solver outputs,
  including exact-baseline-budget and original-provider fallbacks. Carried-sum
  overflow is caught; both representative tail memchecks report zero errors.
- Separate prototype pair: 52.150639422 to 51.042734768 seconds, 2.12% shorter.
  Production pair, reverse order: 52.150713116 to 51.149165444 seconds, 1.92%
  shorter; replay 9.163709378 to 8.080924573 seconds. Do not pool these pairs
  or older #2211 samples. No broad statistical/performance/force claim is made.
- Both production arms have 19 observations/evaluations. Original independent
  total/triples/physical R1/R2 gates remain 1e-8/1e-10/1e-10/1e-10. Max errors
  are about 2.43e-12 Eh, 9.27e-14 Eh and 2.94e-12 residual units.
- Device capacity stays 8,583,749,632 bytes. Bounded host descriptors add 113,476
  bytes; endpoint process peak RSS adds 3,239,936 bytes in the production pair.
- Physical tiles 488 to 31, GEMM calls 13,176 to 961, accumulation launches 976
  to 31, packing bytes 684,660,517,632 to 515,801,545,216. Exact contraction
  terms are unchanged. Four scalar contractions become GEMMs, so GEMM summands
  increase slightly. Every endpoint work counter matches its graph prediction.

## Rejected alternatives and pitfalls

Q32 needs more scratch than the admitted original Q32 primal/pair owner. Do not
sacrifice that owner to batch an audit. Keep the earlier core-replay packing and
individual Lt-layout sweeps parked; cached primal values cannot replace replay.

The isolated build must shadow all quoted generated-header dependencies, not
just the extended DF header, or unchanged Lambda consumers can include both old
and new definitions. Resolve Ninja's relative dependency paths before selecting
the rebuild closure. Failure receipts are retained; no numerical tests ran on
either failed build. NVML metadata failed before timing; a direct CUDA runtime
device/driver query succeeded without changing Slurm visibility. Retain that
distinction instead of calling the metadata failure a numerical failure/pass.

Master is monitored at build/publish milestones. The measured library base is
the accepted #2211 integration including RHF #2205/#2206. Later CPU DF-PBE,
host-probe, signed rank-k, mixed-f force and CPU-JIT changes do not alter its
consumed CC/energy paths; do not repeat passing unrelated GPU matrices for them.
The source-matched consumer and whole-library source-base distinction is explicit
in the bundle. Already merged metric-GEMV evidence is compressed byte-identically
to make room; no unmerged receipt, sample, cap or Release is altered.

## Revisit when

Broaden the consumer or allow new scratch only after independent numerical and
complete endpoint gates for that scope. New source/capacity evidence can justify
another tile size; fewer launches alone cannot justify production promotion.
