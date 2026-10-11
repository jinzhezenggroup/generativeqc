# Decision: provider-consistent warm energy after Hybrid phase reuse

Status: implemented
Date: 2026-10-11

## Problem

The opt-in cold-auto PBE0 water-96 endpoint in job 7242 converged using a
canonical resident J/K source and then retired it. Its last-good warm density
was consistent with its stored compensated electronic energy, but the next
warm solve used incumbent generated J/K. Different FP64 evaluation schedules
can shift the energy at identical density beyond the unchanged 1e-12 Eh gate.
Both retained auto warm histories failed that first energy gate (5.7892e-12
and 4.4055e-12 Eh), while density and residual gates already passed. Warm
endpoints needed three/four physical Focks instead of the disabled arm's one.
Later phases performed no source preparation, so rebuilding the source was
not the cause. The prior measurements remain historical evidence in the
[cold-auto decision](2026-10-11-screened-hybrid-cold-auto.md).

## Decision

After successful source retirement/audit, and only when updating the warm
cache, evaluate incumbent strict-FP64 J/K once at the unchanged terminal D.
Use the existing compensated diagnostics reduction with the retained XC at
that same D to replace only the cache's high/low electronic-energy baseline.
Do not recompute XC, solve orbitals, change D, or modify the returned resident
physical Fock, energy, derivative snapshot or accepted convergence history.
The next warm solve still evaluates and passes all original physical gates.

The additional operator application belongs to the cold endpoint. Count it
in total Focks, post-SCF Focks, a finalization event, J/K and diagnostic
precision census, scalar transport and a dedicated handoff trace/counter.
It is not an SCF iteration or source replay. Retire graph borrowers and source
storage before this application, avoiding a new storage allowance. Frozen
warm updates leave the last-good density/energy untouched and skip the handoff.
A handoff error drains the stream and invalidates the new cache/final state.
The cache baseline is explicitly incumbent-only. If forced selection admits
another resident source on a warm phase, do not use that baseline; establish
resident energy history with the existing first-iteration exclusion instead.

## Rejected alternatives

- Relaxing convergence, screening or precision would change the endpoint.
- Replacing the warm energy with its first current energy would manufacture
  an energy-history pass without an independent prior evaluation.
- Keeping the quartic source across gradients/warm solves breaks phase-local
  ownership and the measured resource fallback contract.
- Recomputing a complete SCF proposal or XC solely for the energy anchor adds
  unnecessary work and may change the accepted terminal state.
- Moving work into the cold endpoint without counting or measuring it would
  misrepresent complete-sequence performance.

## Invariants

- The cache baseline and cache density use the next warm provider at the same
  geometry. A geometry change clears warm energy validity as before.
- The returned physical state and every numerical acceptance gate stay intact.
- A warm attempt still needs a fresh incumbent physical Fock and diagnostics.
- Source admission, precision, memory reserve and defaults remain unchanged.

## Evidence

Host guards protect handoff ordering, accounting and final-state immutability.
The native Hybrid probe checks PBE0/B3LYP, resident-to-incumbent warm solves,
frozen cache behavior, explicit seeds, chunk widths and precision provenance.
Fresh source-matched GPU endpoint evidence is retained in the ignored
`.artifacts/hybrid-warm-fix-20261011/` directory, separate from job 7242.

Final source identity is
`b7d3310b6beb0856d6e08ff385c930f0ee3789a4a1817a6ceb3025288f57c3e2`.
Job 7254 runs one fresh-process ABBA block on node1 RTX 5090, at the unchanged
96-atom PBE0/def2-SVP grid, screening and convergence protocol. All 16 large
E+F gates and eight generated 3-atom E+F gates pass. Maximum large energy and
force errors are 9.5497e-12 Eh and 3.7938e-11 Eh/Bohr, below 1e-8/1e-7 gates.

| Complete E+F endpoint median | Disabled (s) | Auto (s) |
| --- | ---: | ---: |
| Cold including preparation and handoff | 85.842973 | 77.385836 |
| Warm | 16.010984 | 16.012590 |
| Moved | 70.224702 | 63.088928 |
| Moved-warm | 16.027505 | 15.993477 |
| Four-phase total | 188.106165 | 172.480831 |

Physical Focks are A1 `17,1,12,1`; B1/B2 `18,1,12,1`; A2 `17,1,16,1`.
Each auto cold count includes 17 resident replays plus one incumbent handoff.
Every later auto phase skips source preparation and handoff. Warm first
energy changes are 9.0649e-14 and 1.6222e-14 Eh; the terminal fixed-D handoff
observes energy shifts 5.5143e-12/5.4074e-12 Eh, exact-exchange shifts
-5.5138e-12 Eh and zero XC shift. This directly identifies the evaluation
schedule mismatch that the cache refresh removes. No warm gate is waived.

The observed whole-sequence ratio is 1.0906, but A2's moved trajectory has
four extra physical Focks. This is not a fixed-work kernel speedup or a
statistical default-promotion result. Cold fixed-work (17 SCF iterations in
both arms, plus the charged auto handoff) has an observed ratio of 1.1093.

Host validation passes 945 tests, with 89 domain/GPU skips and four known
missing-offline-asset tests deselected. Job 7255 passes four independent
water f/SPD probes (chunk 1/2, both PBE0/B3LYP); job 7256 completes f-shell
memcheck with zero errors, including frozen forced-warm cache coverage.
Job 7264 separately verifies the optional public E-only probe counts its
handoff (14 SCF iterations, 15 physical Fock actions). That reporter-only
update leaves the raw J/K and warm probe source unchanged; original matching
binaries and the new reporter binary/hash are retained separately. This
E-only probe is not represented as force evidence.

## Consequences

A source-backed cold solve pays one incumbent J/K action to prepare its next
warm baseline. This is not a promise of one-iteration warm convergence for
arbitrary densities or of universal speedup. Promotion still requires full
cold/warm/moved/force qualification, complete work counts and broader resource
and profitability evidence. The environment control remains opt-in.
