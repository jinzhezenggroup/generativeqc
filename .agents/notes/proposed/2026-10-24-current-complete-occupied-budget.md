# Candidate: preserve complete occupied projections across geometry rebuilds

Status: proposed; current-source qualification in progress
Date: 2026-10-24

## Decision

Restore the complete-U budget admission investigated in historical commit
`c7eec7ed780e2a72b449edf995c8041ec30806f7`, after the current held-factor and
through-d range-call-graph candidates. The packed planner may legally return
resident B with rank capacity zero. That is a bounded fallback, but it does
not satisfy a method owner's requested complete occupied projection U.

Check the returned rank capacity. Preserve any successful bounded fallback;
when the automatic live budget can accommodate complete B/U within observed
free bytes minus reserved headroom, use the shape-derived response floor and
byte bisection through the actual planner to charge the minimum sufficient
value allowance. Explicit total/response caps, failed live probes, unproved
occupations and unavailable complete storage retain their existing paths.
No device allocation probes, source recomputation, precision or screening
changes are introduced. The selected split remains part of owner identity.

## Rationale and qualification boundary

The older integrated source retained packed B after reducing LR stack backing,
but lost optional U after a geometry move. Its diagnostic executed 2,888
repeated fitted-panel projections; retaining complete U reduced those to zero.
Those historical timings are not current-source performance claims. Current
cold/warm/moved/moved-warm energy and force gates and semantic work receipts
are required before promotion. The separate stack candidate must retain its
own source identity so the mechanisms can be assessed independently.

Executable planner tests cover complete-U admission, the one-byte boundary,
explicit total/response caps and constrained fallback. Real-device method
composition, sanitizer checks and complete endpoints remain required. Revisit
the response floor when response temporaries change; never infer bounded work
from bounded scratch or successful B admission alone.
