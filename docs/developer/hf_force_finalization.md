# Direct CUDA HF force finalization

This page documents implementation semantics for direct CUDA RHF/UHF analytic
forces. It is not a public support matrix; user-facing availability remains
execution-context dependent.

## Physical convergence criterion

Direct RHF/UHF force solves require the maximum physical AO commutator

```text
|F P S - S P F|
```

to satisfy `min(1e-8, density_tolerance)` in addition to the ordinary energy
and density-update criteria. The maximum includes both UHF spin channels.
Non-finite residuals cannot pass.

Additional SCF updates remain within the requested iteration limit and are
included in the public iteration count. A mixed-precision coarse stage retains
its own stopping contract; exact FP64 refinement and exact-precision neighbors
apply the physical force criterion.

## Final physical state

Before publishing forces, finalization projects the final physical-Fock
orbitals to a determinant, rebuilds the physical Fock for that determinant, and
rechecks its commutator. Energy, forces, and the returned warm state therefore
share the same physical `P/F(P)` state.

The Pulay weight is

```text
P F(P) P / 2
```

for RHF and

```text
P_sigma F_sigma P_sigma
```

for UHF. A failed final residual reports non-convergence. Energy-only execution
and detached physical-reference execution retain their separate finalization
contracts.

## Work and diagnostics

Force finalization adds one density projection, one physical Fock build, four
matrix products for residual validation, and two matrix products for the Pulay
weight. Existing device scratch owns those products; the work counter is copied
at the existing completion fence.

`GENERATIVEQC_DF_PROGRESS_TRACE` records final updates, physical Fock builds,
residual checks, and rejections separately from iterative SCF updates. Complete
endpoint performance measurements must include this finalization work instead
of timing only the iterative Fock path.

The durable rationale and qualification boundary are retained in
`.agents/notes/proposed/2026-09-17-consistent-direct-pulay-weight.md`.
