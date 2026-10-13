# Experiment: hold range exchange with the provisional VV10 anchor

Status: default-off prototype; qualification pending
Date: 2026-10-14

The 96-atom one-iteration diagnostic on source `7369698bc` locates an 83-second
VV10 pair sweep and roughly 6.8 seconds of semilocal/AO work. Its primary
full-range J/K takes only 1.6--2.3 seconds. The diagnostic deliberately caps each
attempt at one iteration: the existing seeded-failure fallback makes two
attempts, so the full 214.24-second run must not be described as one Fock.
The complete 96-atom initializer still pays for range exchange on every
provisional inner step, and both retained initializer populations time out.

`GENERATIVEQC_VV10_FROZEN_RANGE_SEED=1`, together with the existing frozen-VV10
initializer, holds the anchor's range-exchange matrix during each provisional
inner solve. Current-density full-range J/K and semilocal XC still run. A new
physical anchor recomputes both VV10 and range exchange; ordinary final target
confirmation and acceptance retain the original gates and total solver budget.
The prepared range owner and its bound remain allocated. There is no new
storage, CPU solve, reference density, approximate integral or discarded work.

The existing owner-local range matrix is read-only after its anchor completes.
The normal stream/event dependency and scalar error check already finish that
anchor before any inner step. Incremental J/K, mixed precision and unsupported
initialization domains remain excluded. Scope 3 in the private history receipt
explicitly names both frozen components; these rows are never physical target
evaluations and cannot publish forces or warm state. This enables actual fresh
range applications to be counted from completed target and semilocal-seed rows.

Independent molecular E/F, changed geometry, finite-budget rejection,
memory/synchronization checks and complete endpoint trials are required before
any automatic selection or performance claim. Holding this stronger operator
may require more anchors or fail to converge; the retained full-target tail
and bounded failure are mandatory. This does not close #2426.
