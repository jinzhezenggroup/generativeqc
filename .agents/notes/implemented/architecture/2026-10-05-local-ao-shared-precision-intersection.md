# Decision: retain the shared precision schedule through local-AO admission

Status: implemented
Date: 2026-10-05

PR #1865 makes XC layout capability independent of the whole KS precision
request. Its first implementation projected that result into a DFT-specific
pair of booleans, losing the shared region/directive representation introduced
by #1838. That would separate later provider selection from the qualification
and arithmetic metadata it needs (#1886/#1889).

`ExecutionPrecisionSchedule::filter_lower_precision` now copies the complete
schedule and replaces only rejected lower-precision directives with the common
strict FP64 fallback. It preserves region inventory, surviving directives,
qualification, arithmetic mode and audit ownership. A capability predicate
cannot introduce lower precision or modify the requested schedule.

The KS iteration resolver returns that shared schedule. A local density layout
can decline mixed density while retaining qualified Direct J. Strict refinement
filters every region, including future composition regions, to FP64. The final
pending-work flags are derived at dispatch, as before; they are execution census
state rather than another precision decision contract. No provider names or
scientific qualification rules are added.

Host regressions execute the production header and verify preserved directive
metadata, input immutability, no promotion from FP64 and refinement of an extra
component outside the current J/density pair. These representation checks do
not replace the PR's independent GPU AUTO/local-AO chemistry qualification or
the composed #1895 endpoint protocol. No speedup or default promotion is claimed.
