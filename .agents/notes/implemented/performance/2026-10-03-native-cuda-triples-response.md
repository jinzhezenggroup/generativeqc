# Decision: execute the native standard-(T) response on CUDA

Status: implemented
Date: 2026-10-03

## Problem

After bounded ERI derivative batching, the 28-AO complete force endpoint still
spent about 2.67 seconds evaluating its triples response on CPU. CUDA forces
already executed the energy, Lambda and Hamiltonian/orbital graphs on device.

## Decision

Reuse the runtime-indexed triples TensorIR VJP and its generated CPU arena plan.
The CUDA generator verifies that backend preparation has identical logical
identity and symbolic liveness slots before sharing the admission function.
Emit a separate CUDA translation unit so this new evaluator does not alter the
existing large CCSD CPU/CUDA generated files. Both remain byte-identical to the
parent. Reuse the compiler's range-safe scaled-bilinear arithmetic helper.

The native CUDA owner uploads the eight physical inputs once, retains an arena
and eight accumulators, uploads only maps/active masks/degeneracies per triangular
virtual-triple page, and downloads completed projected cotangents once. A page
has at most 16 lanes by default. Preserve last-two ovvv/ovoo and composite-pair
ovov symmetry projection. Inactive lanes keep valid maps and denominators.

Use output-driven indexed scatter with one writer per destination and ascending
lane summation. This preserves deterministic source-order accumulation without
atomic FP adds. Each output examines at most q lanes; this is an explicit work
tradeoff, not a claim that memory bounds alone bound computational cost. Other
reductions also retain source-major order. Finite/error gates remain active.

## Capacity and lifetime

Charge host outputs and page controls plus device inputs, accumulators, arena,
page controls, seed, error flag and final allocation padding. Borrowed reference,
problem, amplitudes and orbital energies belong to the enclosing force phase.
Compose both sides into the complete force budget, and include triples device
capacity in the public device high-water mark. Native owner admission reduces
q when necessary; reject explicitly if one lane cannot fit. The complete force
planner retains its existing conservative default-page admission.

Each page fences its error flag before host controls are reused. The storage
owner also drains pending transfers on unwinding and is destroyed before any
borrowed host transfer buffer. Release device storage before diagnostic string
publication. Numeric payload accounting follows existing post-HF conventions:
allocator metadata, CUDA context state and implicit kernel stacks are excluded.

## Qualification

Native fixtures compare all eight blocks to the independent full-six-index
triples VJP for (o,v)=(1,2),(2,3),(3,2), with page capacities 1,2,3,16 and exact
budgets. A separate runtime TensorIR reference exercises arbitrary repeated
maps, a non-unit energy seed and an inactive final lane. Malformed inactive maps
must report an error without an out-of-range access. Tests also cover reduced
page admission, one-byte-short rejection, invalid devices, nonfinite amplitudes
and noncanonical denominators.

Initial n1 qualification passes the complete 14/28-AO energy/triples/force and
response-residual gates against the retained independent PySCF gradient oracle,
40 public CPU/CUDA CC tests, and the native probe. Native memcheck on n2 reports
zero errors. Paired complete-endpoint and profiler evidence is being collected
before publishing final performance claims.

## Alternatives and limits

Calling the development Python/CuPy triples response from production would add
an undeclared runtime dependency. Reimplementing triples algebra in CUDA would
create a second scientific owner. Atomic scatter would change accumulation
order. None is needed for this native TensorIR lowering.

This change does not enlarge the 28-AO force qualification boundary or resolve
the 56-AO same-space degeneracy/gauge problem. Revisit page scheduling or kernel
fusion only if complete endpoint profiles show that this phase is again material.
