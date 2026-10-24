# Decision: retain complete method-owned occupied projections when they fit

Status: implemented in candidate; final integration qualification pending
Date: 2026-10-23

## Problem

The [through-d LR call-graph bound](2026-10-23-through-d-range-force-call-graph.md)
restores enough memory for WB97M-V DF96 to keep packed B across a geometry
rebuild. The optional-U planner can nevertheless report successful packed
storage with rank capacity zero. The method budget resolver treated that as
successful admission of the requested complete occupied projection. The final
force then used the bounded general-density algorithm, reconstructing fitted
panels repeatedly. Bounded scratch alone did not bound endpoint work.

In the traced candidate, cold/warm complete in 365.264/45.344 seconds, but the
same owner's moved/moved-warm calls take 583.975/384.657 seconds. They execute
2,888 fitted-panel projections in total. All four calls pass independent
static or same-geometry moved energy/force gates; the issue is redundant work.

## Decision and boundaries

Inspect the planner's returned rank capacity before accepting a requested
complete U. Preserve a successful rank-zero plan as the explicit bounded
fallback. If the preferred response margin cannot admit U, use the existing
shape-derived response floor and find the smallest sufficient value allowance
by byte bisection through the actual planner. Complete B/U admission is the
predicate, so an optional-U refusal cannot masquerade as success.

Only automatic live budgets with an authenticated restricted occupation or
physical pure-J reservation participate. Positive public requests, response
overrides, failed live probes and unavailable complete storage retain their
existing bounded paths. The absolute observed free-memory bound minus reserved
headroom and the complete response floor remain mandatory. No allocation probe,
CPU integral oracle, precision change or extra mathematical source is added.
The selected value/response split remains part of prepared-owner identity.

## Evidence

The implementation candidate is based on 376821d5c plus the retained source
patch and a 5,984-input manifest. All five stationary manifests were rebuilt
with verified sccache 0.16.0; original build/cache logs and immutable package
hashes remain in the stage-nine artifacts. Tests cover exact and one-byte-short
complete-U admission, explicit total/response limits and constrained fallback.
309 host cases pass (one optional skip); 24 scheduled Direct/high-angular GPU
cases, independent CPU fitted-WB composition, cold/warm factor proofs, complete
Fock composition and current-memory force/replay checks pass. Complete Fock
composition under Compute Sanitizer reports zero errors.

The original test run failed because a host density-upload probe depended on
an obsolete workspace variable. Its existing repair from 795f5d998 and matching
C++ extraction helper are retained as a separate test-input delta. A local run
without a native library and the intermediate missing-helper run are also
retained; neither is counted as a numerical failure or passing qualification.

The new traced trajectory takes 372.649/45.461/279.526/45.504 seconds for
cold/warm/moved/moved-warm with 28/1/13/1 iterations. Fitted-panel projections
fall from 2,888 to zero, and each endpoint uses one occupied-product response.
All four calls pass unchanged independent gates; moved maximum force error
against fresh same-geometry GPU4 DF is 2.238e-7 Eh/Bohr. The moved plan charges
16,910,106,050 total bytes, including 2,549,049,300 response bytes, against
26,919,436,288 observed free bytes. These single traced observations diagnose
work; they are not repeated cold performance claims.

## Consequences and revisit

Retaining complete U trades optional fractional slack for a bounded final-force
algorithm. Low-memory owners remain correct through the saved fallback. Review
new response temporaries against the floor whenever that algorithm changes.
Final source integration, clean repeated native/GPU4 comparisons, Fast and
additional moved endpoints must retain their own source identities.
