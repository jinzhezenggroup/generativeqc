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

Eligible large cold restricted owners automatically use a bounded screened FP64
proposal when `GENERATIVEQC_HF_COLD_SCREENING` is unset. Exact `1` also admits
the proposal; `0`, `none`, empty and invalid explicit values preserve the ordinary
operator and conservative page policy. This is not mixed-precision arithmetic.

Admission requires at least 512 AOs, one system, shells through d, at least 24
allowed iterations, and the bounded quartet provider. Warm/imported seeds,
physical-reference export, resident ERIs/values, mixed precision, incremental
J/K, and UHF retain their existing routes. A requested screening tolerance
at or above `8e-8` also retains the ordinary route.

The proposal uses a screening tolerance of `8e-8` for at most eight iterations.
The driver then resets operator-dependent DIIS and energy state and converges
the requested target operator. Its target tail does not use the ordinary
Standard-owner residual relaxation: the physical AO-maximum residual must meet
`min(density_tolerance, 1e-8)`. Public controls and observable acceptance gates
are not relaxed. Both stages consume the same user iteration allowance. The proposal graph is local
to the cold execution; retained warm graphs always represent the target.
The switch is part of prepared-HF resource identity; changing it does not reuse
an owner prepared under the other policy.

Screened proposals still use FP64 arithmetic: their work contributes to
`strict_stage_fock_builds` and the public iteration total, never to
`mixed_stage_fock_builds`. `refinement_iterations` counts only the target tail.

The bounded task page is independent of fixed-topology storage and proposal seed
history. Automatic larger-page admission covers singleton FP64 RHF through-d
bounded owners with at least 512 AOs and 24 allowed iterations. Small, batch,
UHF, higher-l, mixed, incremental and physical-reference owners keep their
conservative defaults. Generic Direct/KS resource policy is unchanged.
The existing explicit `1` diagnostic retains its broader page-selection behavior.

Admitted targets with at least 24 GiB may use up to one quarter of device memory,
capped at 32M tasks and 6 GiB. Smaller targets retain the one-sixteenth memory
budget and unknown targets use 256 MiB. If the larger arena allocation is refused,
the owner
recomputes its layout and queue offsets and retries at most once with the old
8M-task ceiling before any upload or graph capture. Physical-reference export
retains the old 8M-task/1.5-GiB profile even when opted in.

The [strict-tail decision](../../.agents/notes/implemented/performance/2026-11-07-cold-hf-strict-screened-tail.md)
and [source-bound qualification capsule](../../benchmarks/results/hf-cold-defaults-20261107/README.md)
preserve the independent default-request gates, endpoint/work receipts, bounded
page-refusal tests and superseded negative evidence.

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
