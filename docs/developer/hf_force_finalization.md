# Direct CUDA HF force finalization

This page documents implementation semantics for direct CUDA RHF/UHF analytic
forces. It is not a public support matrix; user-facing availability remains
execution-context dependent.

## Physical convergence criterion

Direct RHF/UHF force solves require the maximum physical AO commutator

```text
|F P S - S P F|
```

to satisfy the owner-qualified HF stationarity limit in addition to the energy
and density-update criteria. All-electron restricted FP64 ordinary owners use
`min(1e-5, density_tolerance)` when both public controls are Standard or looser
(`energy_tolerance >= 1e-9`, `density_tolerance >= 1e-5`). Tighter requests and
unqualified owners retain `min(1e-8, density_tolerance)`. Selecting incremental
J/K does not change this limit. The maximum includes both UHF spin channels;
UHF remains on the strict limit. Non-finite residuals cannot pass.

Additional SCF updates remain within the requested iteration limit and are
included in the public iteration count. A mixed-precision coarse stage retains
its own stopping contract; exact FP64 refinement and exact-precision neighbors
apply the physical force criterion.

## Final physical state

An SCF-owned determinant with valid orbital-frame provenance retains the
converged physical `P/F(P)` pair without projecting another density. Otherwise,
finalization projects the final physical-Fock orbitals to a determinant,
rebuilds its physical Fock, and rechecks its commutator. Arbitrary external
seeds, changed-geometry seeds and the explicit final-rebuild diagnostic retain
this bounded canonical fallback. Energy, forces, and the returned warm state
share the same physical `P/F(P)` state in either route.

Incremental J/K uses the same retention and finalization policy. Once density
RMS and the physical residual are both within ten times their ordinary limits,
the solver latches full-only refinement. This scheduling margin does not relax
any convergence gate: it overlaps full-energy confirmation with the remaining
SCF tail instead of waiting for stationarity to start confirmation. After any
delta work, a reconstructed current energy baseline is discarded. At least two
fresh full-density Fock evaluations must establish ordinary energy closure before
convergence; a reconstructed Fock alone cannot license retained publication.
An already full evaluation can supply the first usable baseline. Stationary
no-delta warm solves keep ordinary convergence and stay full-only if an
energy-only tail needs another step, rather than unnecessarily starting deltas.
Screened reconstruction
retains a fixed full-density anchor between refreshes; neither screening nor
floating-point reduction order promises a bitwise-identical SCF trajectory.

The experimental selector is documented in
[batched HF](../user/batched_hf.md#incremental-direct-jk). It does not alter
the closure or canonical fallback described here.

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

The shared Rys x/y companion changes recurrence storage, not convergence or
the force determinant. An internal automatic request selects the exact SM80
profile; exact SM90 remains opt-in and other profiles retain their incumbent.
`GENERATIVEQC_CUDA_HF_SHARED_RYS_XY=0` disables it; exact `1` requests it on
supported profiles. Unset/empty selects the internal automatic policy, while
invalid explicit values fail before graph preparation. Weighted Z remains a
separate SM90-only control. Existing restricted-FP64, generated capability,
materialized-page, no-incremental/separate-JK, no-relevant-streaming,
no-persistent-ERI and ordinary-output admission gates still apply. The plan
freezes the request, admission and class mask; unsupported requests use the
bounded incumbent before enqueue, never a retry after a candidate error.

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

The canonical fallback adds a density projection, a physical Fock build and
four matrix products for residual validation. Both routes use two matrix
products for the Pulay weight. Existing device scratch owns those products;
the work counter is copied at the existing completion fence.

`GENERATIVEQC_DF_PROGRESS_TRACE` records final updates, physical Fock builds,
residual checks, and rejections separately from iterative SCF updates. Complete
endpoint performance measurements must include this finalization work instead
of timing only the iterative Fock path.

The durable rationale and qualification boundary are retained in
`.agents/notes/proposed/2026-09-17-consistent-direct-pulay-weight.md`.
