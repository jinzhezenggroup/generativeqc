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

Only `GENERATIVEQC_HF_COLD_SCREENING=1` selects the optional cold-HF experiment:
a bounded screened FP64 operator and larger reusable task pages. The unset
default, `0`, `none`, empty and invalid values preserve the incumbent operator
and 8M-task / 1.5-GiB HF page policy.

Opted-in large cold restricted owners can use a bounded screened FP64 operator.
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
The switch is part of prepared-HF resource identity; changing it does not reuse
an owner prepared under the other policy.

Screened proposals still use FP64 arithmetic: their work contributes to
`strict_stage_fock_builds` and the public iteration total, never to
`mixed_stage_fock_builds`. `refinement_iterations` counts only the target tail.
The retained default-tolerance experiment failed the additional `1e-7` force
diagnostic (3.02e-7 versus the incumbent 5.10e-8 Eh/Bohr). Explicitly tighter
requests qualified the optional route; they did not resolve that default failure.
The measured speedup is therefore not a claim about the restored default.

The bounded task page is independent of fixed-topology storage. When opted in, targets with
at least 24 GiB may use up to one quarter of device memory, capped at 32M tasks
and 6 GiB. Smaller targets retain the one-sixteenth memory budget and unknown
targets use 256 MiB. If the larger arena allocation is refused, the owner
recomputes its layout and queue offsets and retries at most once with the old
8M-task ceiling before any upload or graph capture. Physical-reference export
retains the old 8M-task/1.5-GiB profile even when opted in.

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
