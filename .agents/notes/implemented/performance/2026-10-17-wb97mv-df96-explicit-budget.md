# Decision: retain completed DF/96 endpoints with explicit resource budgets

Status: implemented
Date: 2026-10-17

## Context

The user explicitly requested another native ωB97M-V DF/96 measurement with
a longer deadline, without GPU4PySCF. This supersedes the earlier user-requested
stop recorded in the
[force-budget decision](2026-10-14-wb97mv-readme-force-budget.md), not its
scientific gates or diagnosis. The publication still describes frozen source
`1593d3b2fea9ed3e9232efc49eedede1beb1787b`, not current-master performance.

## Resource policy and evidence

Slurm n1/9073 repeated the default-policy failure: SCF converged, but force
admission returned `stationary CUDA byte budget is not representable`.
Its 1027.926-second failed call is negative evidence, never an E+F timing.

Slurm n1/9108 uses a 12 GiB public DF workspace hint, explicit 4 GiB device
and 4 GiB host force allowances, and a 2 GiB long-range resident-cache ceiling.
The inner deadline is 14400 seconds inside a finite 04:10:00 allocation.
These are additional force allowances, not a 4 GiB total-device-memory cap.
The production native library, Python implementation, FP64 policy, public
initialization, basis, auxiliary basis, grid and numerical tolerances are
unchanged. Only the benchmark consumer gains a resource option; preserve its
exact executed bytes separately from later formatting/docstring edits.

The first complete cold E+F call took 4717.471 seconds, with 37 SCF iterations
and 37 Fock builds. Its finite 96-atom host force array was checked before
scalar reduction. DF telemetry reports a streamed value path; integral
derivatives account for about 1967.84 of the 2098.58 force seconds. The workspace
hint admits bounded memory, not bounded work or a performance improvement.

The four-hour deadline expired during warm repeat index 4 (exit 124). Four
warm calls completed in 2145.750, 2148.055, 2146.223 and 2146.284 seconds, each
with one SCF iteration/Fock build. Publish cold only; retain the incomplete
warm set without inventing its fifth sample. There are now 63 complete
populations plus one partial population: 744 checked and 17 unchecked calls.

## Publication invariants

Keep the original 63 native populations, historical reference data, Direct
figure's original bytes and sixteen original allocation receipts unchanged. Retain default
failure and explicit-budget observations separately. No reference calculation
is run, and no absent DF/96 accuracy oracle is inferred from convergence or
execution success: energy/force errors and the accuracy gate remain null.

A later timeout must not erase a completed cold phase or a complete five-call
warm phase. Conversely, fewer than five warm calls must never become a plotted
warm median. Retain their successful endpoint records as partial-population
evidence, preserve the actual timeout exit code, and audit the ordered phase
prefix and phase-completeness labels. Never label a partial run as a complete
twelve-call population. Revisit work complexity or capacity planning in a new,
independently qualified campaign rather than silently retiming this cohort.
