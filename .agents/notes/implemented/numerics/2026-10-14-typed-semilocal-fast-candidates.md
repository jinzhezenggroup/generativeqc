# Decision: shared typed semilocal candidates with strict boundary recovery

Status: implemented (compiler candidates only; public admission unchanged)
Date: 2026-10-14

## Problem

The README Fast objective includes PBE/PBE0, B3LYP, r2SCAN and its 3c
composition, M06-2X and WB97M-V as well as HF and supported DF endpoints.
Casting completed double XC outputs does not reduce graph arithmetic. Copying
each functional into a separate handwritten float implementation would lose
the shared scientific owner and its work-domain/derivative contract.

## Decision

Select float at the shared scalar emitter and semilocal/split-hybrid graph
owners. Parameters, graph constants, intermediates, elementary functions and
result storage use float; mathematical expression identities are unchanged.
Emitted source/artifact identity distinguishes arithmetic. Float elementary
functions use their explicit `*f` entry points, including fused multiply-add.

Reuse the existing stationary Cartesian adapter and component work-input
conditioning rather than adding different clipping or derivative equations.
Work-input floors, physical-density energy rescaling and Cartesian conversion
remain FP64. PBE keeps its separately normalized shared point implementation.
The guarded generic candidate retains FP64 for tails, unsupported input ranges,
strong spin imbalance, or a nonfinite candidate. Equal-spin sharing requires
exactly equal produced density/gradient/kinetic inputs; a method name alone is
not proof. An optional execution-owned flag reports whether the point actually
returned the typed graph, not just whether it was requested.

## Rejected alternatives and numerical evidence

The first generic guard admitted every finite density between `1e-6` and `1e3`
with bounded gradients and kinetic density. Independent Libxc checks rejected
that approach: minority-spin derivatives for M06-2X and WB97M-V amplified float
rounding near full polarization. One WB97M-V kinetic derivative differed by
about `1.689e-4` (relative `2.147e-3`) despite all outputs being finite. These
failures remain in the local diagnosis logs. The point gate was **not** relaxed.
The generic candidate now additionally requires a spin-density ratio at most
16, avoiding the observed `1-zeta` cancellation regime; stronger polarization
uses the exact strict result. This is conservative empirical admission, not a
proof of a molecular force bound.

One-off migration comparison confirms byte-identical default FP64 raw and
Cartesian point source for all six electronic functionals. This receipt is an
ignored artifact rather than a permanent source-hash-pinning CI test.

Host independent point/ownership checks before equal-spin dispatch refinement:
119 passed, 43 scheduled/intentional skips. Scheduled CUDA point checks, Slurm
8234: 41 passed, one intentional PBE-specific guard skip. They compile the
actual generated functions and validate energies plus all first partials
against independent PySCF/Libxc; host compilation also rejects implicit double
promotion in raw float graph arithmetic. The first CUDA test invocation failed
on a fixture flag: nvcc owns `-Werror`, so GCC's double-promotion option must be
passed explicitly through `-Xcompiler`. Its failed receipt is retained.

The all-ten-profile native build succeeded. Slurm 8237, candidate library
`9df476e6d38635a7630a014f5942e17ca449d9486411529ae203ea1b92facc53`:
18 complete same-state gradient tests passed, including PBE, PBE0, B3LYP and
r2SCAN at 24/96 AOs, existing PBE0 Cartesian/ammonia cases, nonfinite weighted
task retry and pending-geometry reset. SCF stayed FP64. Independent moving-grid
energy/force gates, positive mixed work, translational balance and strict
replay are required. These results precede the final equal-spin dispatcher
refinement; requalification is separate and must not reuse this binary identity.

The final equal-spin dispatch was independently rebuilt and requalified in
Slurm **8242**: **41 CUDA point tests passed** (one intentional guard skip) and
**18 complete gradient/recovery tests passed**. Final candidate library:
`b18af797ca2a4070af4daea9d4a0cd7f4ccbe71c5b59a697f902f5a33f429d1a`.
The latter tests took 474.50 seconds; that suite duration is not a performance
comparison. Separate final host point checks passed 45 tests, and shared
scalar/semilocal/r2SCAN/weighted-force regressions passed 94 tests (one skip).
Compiler dependency checks and the relevant pre-commit hooks pass. This
closes the equal-spin candidate's small-system numerical requalification,
not the outstanding public method/batch/budget/endpoint matrix.

## Public boundary and next owners

No new public method is admitted by this change. Compiler capability is not
the public native model gate. HF has no XC and requires a shared HF/KS precision
request and actual integral/contraction work. DF metric response, exact/RSH
exchange, VV10 and 3c D4/gCP corrections cannot be omitted or relabeled float.
Their native/phased/composite owners require explicit transport and bounded
recovery. Public PBE0 admission and the README endpoint protocol remain unchanged.

## References

The follow-up transport and bounded public-consumer integration are recorded in
`../architecture/2026-10-14-mean-field-application-budget-transport.md`. This
note preserves the original compiler-candidate boundary; the user precision
guide owns the current public matrix.

- Proposal: `../../proposed/2026-10-14-readme-budget-fast.md`.
- PR #2452 is the separate PBE0 milestone, not delivery of this objective.
- Ignored source-matched receipts: `.artifacts/readme-budget-fast-20261013/`
  and `n1:/data/jzzeng/readme-budget-fast-20261013/`.
