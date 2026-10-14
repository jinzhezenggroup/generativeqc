# Experiment: retain audited range values across fixed-geometry replay

Status: opt-in experiment, not default or a completion claim
Date: 2026-10-14

The range-source experiment rebuilds fixed-geometry ERIs only for cold and
transported geometry, but retires them before every warm call. This follow-up
allows `GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY=1` together with explicit range
residency to retain the audited lease inside the same immutable geometry owner.
It stores the accepted source-schedule warm energy separately from the incumbent
energy rebase. Simply reusing the incumbent baseline would compare different
FP64 accumulation schedules at a sub-ulp convergence gate.

Replay requires the same bucket and capacity policies, an active lease, a valid
accepted warm density, strict FP64 single-step control and no incremental or
external resource-ledger policy. Force-memory eviction, failures, explicit seeds,
changed geometry and policy changes retire the source and invalidate its energy.
The incumbent baseline remains available for the ordinary provider. All physical
Fock, convergence, energy and force work still executes; zero-order ERIs never
stand in for derivatives. No new allocation is performed for warm replay.

Qualification must show real warm source consumption with zero new ERI builds,
full/partial residency, policy-change fallback, independent molecular E/F gates
and original complete endpoint populations before any default promotion.
