# Decision: bounded native CPU forward selection

Status: implemented
Date: 2026-10-14

## Problem

Canonical TensorIR already owns finite generic `where`, and the materializing
CPU owner already stores Boolean values in a checked one-byte ABI. Selection
still rejected before native compilation. A new emitter or float-valued mask
would duplicate semantics or corrupt the Boolean storage contract.

## Decision

Admit `where` only in `tensor/cpu.py`, through a separate preparation opt-in
requiring the materializing Boolean owner. Read the already materialized mask
and FP64 branches at each output index. TensorIR validates generic domains and
explicit broadcasting; the emitter rejects every live non-FP64 real value.
The existing byte layout, budget checks, transactional native runtime and
detached publication remain the execution owners.

## Invariants

Every live input and branch is evaluated and checked before selection. A
conditional storage read must not introduce lazy evaluation or hide invalid
unselected arithmetic. Boolean values retain one-byte storage and never become
floating masks. Pure-real graphs retain their prior source and ABI. Native AD,
float32/mixed branches, CUDA/portable/scalar/bundle lowering and scientific
selection domains remain unqualified and fail closed. External fixed-mask
runtime reference AD retains its separate existing contract.

## Rejected alternatives

- Extending shared CPU admission would expose unqualified emitters and bundles.
- Embedding branch expressions inside a C++ conditional would skip validation.
- Reusing FP64 0/1 storage would lose Boolean semantics and byte accounting.
- Implementing a second emitter would split deterministic source ownership.

## Evidence

`tests/python/test_tensor_cpu_where.py` executes compiled native libraries against
NumPy and a per-index Decimal expression. It covers selectors, broadcasting,
empty axes, repeated gather, signed zeros, detached multi-output replay,
invalid unselected arithmetic, hostile Boolean bytes, nonfinite raw ABI feeds,
alignment, transactional failure, resource limits and dtype rejection.
Existing Boolean, pure-real CPU, preparation ABI and reference-mask AD tests
remain applicable. One-off source/identity controls and raw execution receipts
belong in ignored qualification artifacts, rather than required fingerprint tests.

## Consequences and revisit conditions

This materializing path retains both branches and their work; selection alone
does not establish a performance gain or complete Array API conformance.
Wider dtype/backend/AD support requires its own source-matched independent
qualification. Refs #2137 and #2111; neither parent scope is completed here.
