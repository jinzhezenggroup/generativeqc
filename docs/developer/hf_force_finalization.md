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

Large cold restricted owners can first use a bounded screened FP64 operator.
Admission requires at least 512 AOs, one system, shells through d, at least 24
allowed iterations, and the bounded quartet provider. Warm/imported seeds,
physical-reference export, resident ERIs/values, mixed precision, incremental
J/K, and UHF retain their existing routes. A requested screening tolerance
at or above `8e-8` also retains the ordinary route.

The proposal uses a screening tolerance of `8e-8` for at most eight iterations.
The driver then resets operator-dependent DIIS and energy state and converges
the requested target operator under the original acceptance criteria. Both
stages consume the same user iteration allowance. The proposal graph is local
to the cold execution; retained warm graphs always represent the target.
`GENERATIVEQC_HF_COLD_SCREENING=0` or `none` disables this diagnostic policy.

Screened proposals still use FP64 arithmetic: their work contributes to
`strict_stage_fock_builds` and the public iteration total, never to
`mixed_stage_fock_builds`. `refinement_iterations` counts only the target tail.
Different converged trajectories can have different observable errors within
the same requested tolerances; a more accurate force target should request
tighter convergence, independently of the preliminary operator.

The bounded task page is independent of fixed-topology storage. Targets with
at least 24 GiB may use up to one quarter of device memory, capped at 32M tasks
and 6 GiB. Smaller targets retain the one-sixteenth memory budget and unknown
targets use 256 MiB. If the larger arena allocation is refused, the owner
recomputes its layout and queue offsets and retries at most once with the old
8M-task ceiling before any upload or graph capture. Physical-reference export
retains the old 8M-task/1.5-GiB profile.

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
