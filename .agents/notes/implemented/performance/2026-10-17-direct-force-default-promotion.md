# Decision: use qualified Direct recurrence and density reuse by default

Status: implemented; unset-selector endpoint qualification in progress
Date: 2026-10-17

## Problem

The generated relative-response and density-first consumers already pass
independent scientific gates and reproducibly improve complete Direct endpoints.
Leaving both behind an explicit environment setting prevents eligible users from
receiving those gains. The user explicitly requested default enablement of
performance-qualified options, not promotion of every experimental selector.

## Decision

Resolve the captured automatic force schedule to `DensityReuse` when the method
authenticates the existing eligible Direct RKS source. Do not widen the existing
method hint or introduce a functional, molecule, basis, AO-count or GPU-model
special case. An absent or empty environment selector is automatic; explicit
`relative-reuse`, `spatial-reuse`/`1`, and `0` preserve their historical meanings.
An unknown nonempty selector still fails closed to the incumbent.

This composes order-six relative-response reuse with density-first contraction
for pure-J orders six/seven. Existing guards retain component contraction for
nonzero exchange, derivative seeds, incomplete masks and unsafe reorderings,
and preserve the original bounded launch paths without coherent resident
topology or compatible derivative caches, angular domains and launch shapes.
Generic HF/RSH callers and callers without a qualified method hint keep their
previous automatic schedule. Explicit UKS qualification remains available; this
change does not infer an unrestricted default from restricted endpoint data.

The resolver changes no allocation, generated scientific kernel, threshold,
precision, force sign, orbit symmetry or SCF policy. Workspace and fallback
qualification from the retained implementation still apply; fixture resource
receipts must not be described as universal production occupancy measurements.

## Evidence and source boundary

The [earlier qualification](2026-10-17-direct-force-endpoint-qualification.md)
retains full source-matched forward/reverse populations, independent oracles,
actual SCF/Fock counts, counted and clean profiles, and sanitizer limitations.
Its approximately 29.6% 96-atom warm and moved-warm gains are measurements of
explicit selectors on its frozen source, not additional gains from this policy
change. It does not include occupied-XC #2577, guarded FP32 or the cold-planning
stack; none is silently composed or promoted here.

The new benchmark candidate defaults to `automatic` and removes an inherited
selector rather than setting an equivalent manual value. Executable host tests
cover absent/empty selection, explicit rollback, eligibility, force/value owner
separation, and preparation-time capture. Existing compiled dispatch tests retain
resource/failure injection and all spin/output/source fallback combinations.

Review follow-ups also restore the inherited all-order canonical FP32 fixture
(orders zero through twelve, including asymmetric/f-containing and sparse
exchange-only cases) without reverting the derivative work upper bounds. The
benchmark retires owned descendants even after leader exit, preserves primary
failures and receipts, and refuses reused/symlinked worker output before writes.
The historical reader retains authenticated, read-only Git recovery while
checking population-to-case protocols and available peer/profile scalar gates.
Those scalar checks still do not replay missing force arrays. These corrections
address reviews 5555852350 and 5556330619 and adapt the bounded behavioral
controls proposed in issue comment 6230099163; they add no production GPU work.

The default-policy host cohort passes 232 tests; documentation checks pass 19
tests and 61 subtests. On node2, finite Slurm3394 runs the restored complete
native fixture, the explicit mixed-JK driver and density contraction, followed
by density memcheck, synccheck and racecheck with zero reported errors/hazards.
All fourteen generated fixture headers are byte-identical to the previously
qualified density source. This does not turn the older unfiltered linked
racecheck timeout into a passing result.

## Rejected alternatives and remaining work

Do not enable degree pruning, separate-cache XY CSE or borrowed-frame prefixes:
their complete-endpoint evidence did not justify retained production complexity.
The [negative experiments](../../rejected/2026-10-17-direct-density-followup-trials.md)
explain the measured limitations. Do not enable approximate precision merely to
claim parity, or normalize away different convergence trajectories.

Default promotion is not GPU4PySCF parity. Continue reducing the remaining
low-order force, stationary grid/response and cold SCF costs using complete
source-matched qualification. This decision does not authorize merging PRs.
