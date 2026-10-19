# Decision: amortize native WB97M-V Hessian dispatch within existing budgets

Status: implemented
Date: 2026-10-17

## Problem

The water-dimer/STO-3G complete endpoint exposed a scaling regression hidden by
tiny-water observations. A finite Slurm RTX 5090 profile of the previous linked
CLI executed over three million launches. Fixed-grid VV10 pairs, repeated exact
induced-Fock recurrences, and individually projected/reduced nuclear shell groups
dominated. Merely retaining derivative jets across the two uses of a right
direction did not remove this work. Memory-bounded was not work-bounded.

## Decision

- Raise the fixed/moving VV10 pair tile cap from 32 to 256, clamping storage to
  the grid extent. The reference-screened ordered-pair domain is unchanged.
- Retain optional strict FP64 full-range and LR induced-Fock integral values
  using the existing canonical Direct source API. Full and LR operators have
  distinct immutable owners. DF retains its fitted primary operator and caches
  only exact LR. Composite admission waits until every mandatory owner has
  reserved its bytes; sources and additional provider metadata are charged.
- When derivative residency, small public shell products and linked compiler
  batch callbacks permit it, batch primitive chunks sharing an exact selector.
  Store only exponents and scale; the compiler-emitted adapter reconstructs
  the ordinary primitive record from immutable coordinates and atom incidences.
  The scalar primitive, selector inventory and AD algebra remain authoritative.
- Project independent shell groups concurrently. A CSR list preserves the old
  group/incidence order for each public output. A warp gathers independent entry
  contributions, but lane zero sums them sequentially in that original order;
  there is no floating-point atomic or tree reassociation. Primal and directional
  outputs share incidence metadata, not numerical accumulators.
- Reuse a bounded projected shell-group tile rather than requiring the entire
  public quartet tensor. The intermediate full-projection plan fit water dimer
  but rejected ethane at the same 64 MiB integral budget, falling back to a
  111.66-second endpoint (109.39 seconds in Hessian execution). This is a
  resource-dependent dispatch cliff, not an acceptable scaling solution.
  Admission reserves immutable sources and metadata, then selects a projection
  tile from the remaining arena and its independent 16 MiB cap. CSR entries are
  sorted by group, so binary searches restrict each owner's traversal to the
  current interval instead of rescanning the full list per tile. Lane zero
  carries the exact existing sum across intervals; no interval subtotal is
  formed and added with a different association.
- Increase moving Becke geometry's point tile cap from 64 to 1024. The old tile
  repeatedly launched one partially occupied CTA. Only independent points are
  regrouped; per-point scaled-product/AD order is unchanged. All scratch remains
  included in checked workspace planning.

## Rejected alternatives

Uncompressed 17-double primitive records caused optional batching to miss the
existing 64 MiB CLI integral budget for the dimer. That intermediate preserved
correctness but left the endpoint around 46 seconds. Raising the budget to make
the benchmark fit would conceal the resource problem; compact immutable records
instead enabled the schedule at the original budget. A larger batch is not
unconditionally better: public shell products above 128 retain the existing
multi-CTA projection rather than serializing them inside one batch CTA.

Do not use cache keys based only on geometry for directional derivative jets:
the retained right direction is part of their lifetime contract. Do not reuse
one full-range Direct owner for LR values. Do not equate zero action recurrence
evaluations with zero work: source construction and replayed quartets remain
observable. Do not silently suppress numerical/device failures as optional OOM.

## Invariants

The runtime remains linked AOT CUDA and Python-free. There is no host numerical
fallback, displaced SCF or change in precision, functional, screening, CPKS gates
or mathematical work admission. Missing batch callbacks, large public shell
products, zero optional caps and insufficient workspace retain bounded ordered
replay. Optional allocation refusal is allowed; other CUDA/numerical failures
propagate. Actions allocate nothing. The source/library identity and complete
endpoint work counters accompany performance observations.

## Evidence

The retained campaign is `benchmarks/results/wb97mv-hessian-dispatch-20261017/`.
It separates clean fresh-process endpoint samples, instrumented profiles,
intermediate attempts, host/build qualification and coupled RKS/UKS/DF/ECP/
spherical parity. Full independent GPU4PySCF matrices use the unchanged
`3e-6 Eh/bohr^2` gate. Component comparisons use independent libcint/LibXC/VV10
oracles and batched/unbatched/resource-fallback controls. A non-divisible
1035-point geometry fixture checks bitwise tile equivalence plus independent
Gaussian and displaced-value derivatives.

One qualification harness assumption was wrong: its forced 512-byte projected
tile fit the all-s direct fixtures, but not even one p-shell group in the
DF/ECP fixtures. Correct bounded fallback produced identical numerical outputs,
while two positive-batch-admission assertions failed. The probe-only cap was
corrected to 32 KiB for those fixtures and both DF/ECP cases rerun separately.
Production CLI/library bytes were unchanged. Preserve that failure and the
followup's distinct PRO 6000 placement, not an invented all-pass original run.

## Consequences and revisit conditions

This removes avoidable dispatch/recurrence work, not asymptotic pair or shell
work. The diagnostic 8/4/8 grid and minimal basis do not establish chemical grid
convergence or practical large-molecule scaling. Complete endpoint improvement
does not imply Hessian-only parity: independently report SCF, preparation and
Hessian phases, because reference SCF iteration counts vary between campaigns.
Sampled total-device memory is not the owner arena and is not an exact peak.

Revisit with a new complete-endpoint profile when optional storage no longer
fits, larger shell products select fallback, or repeated nonlinear VV10 pair
actions dominate. An invariant-space VV10 response cache would need compiler AD,
separate bounded optional storage, complete work accounting and independent
spin-coupled validation; it must not be assumed equivalent merely because its
fixed-geometry reference fields are immutable.
