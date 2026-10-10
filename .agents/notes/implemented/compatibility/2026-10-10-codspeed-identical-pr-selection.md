# Decision: compare identical PR/master CodSpeed benchmark selections

Status: implemented
Date: 2026-10-10

## Problem

The [version 2 selection receipt](2026-10-08-codspeed-selection-receipts.md)
proved that master covered each requested endpoint, but allowed a strict subset
of its benchmark sequence. In PR #2232, master `2b68d10a9` ran the five-case PR
tier including WB97M-V, while candidate `f7f35bf45` ran four cases. CodSpeed
reported the changed-geometry RHF pair at 26.3 versus 27.8 simulated ms, a 5.37%
efficiency regression. The candidate job actually built merge tree `21dc69f27`.

The pinned pytest-codspeed 5.0.3 integration preserves collection order and calls
each benchmark in-process. WB97M-V precedes PBE-force and the RHF pair. Callgrind
START/STOP resets simulated cache state, but does not reset application state or
allocator history. Different preceding selections therefore introduce a real
comparison confound; this does not establish the cause or magnitude of the
reported regression. The existing result must not be dismissed or acknowledged
away merely because the production change concerns CUDA.

## Decision

Executed PR and master-push jobs use the same bounded five-case sequence: warm
RHF, warm PBE, small-grid WB97M-V, PBE-force, then changed-geometry RHF. Preserve
change-aware skipping for known irrelevant-only PR changes. Scheduled/manual
runs keep the full tier, including the larger formaldehyde RHF endpoint.

Qualification requires identical normalized, allowlisted extra-case selections,
not a subset relation. Reject unknown, duplicate and malformed receipt entries.
Do not execute arbitrary selectors supplied by a downloaded receipt. Keep the
version 2 schema because its existing fields already bind the identical
benchmark source, tier and extras; an authentic matching old receipt remains
usable without republishing or rewriting evidence.

## Invariants

Keep the exact master SHA, master artifact provenance, benchmark source hash,
CPU/runtime fingerprint and qualified-only upload guard. Keep scientific checks,
the 5% regression threshold, complete measured targets and allocation costs.
Matching selectors removes one confound; it does not promise identical process
state or prove equivalent performance. The new matched comparison must still
resolve the reported regression before performance clearance is inferred.

## Rejected alternatives

- A blind job rerun retains the different four-case selection
- Manual dispatch selects full tier and does not use PR baseline qualification
- A PR-specific exception or fabricated changed path would hide the policy issue
- Cache-reset reasoning alone cannot exclude allocator/application-state effects
- Excluding allocations would change the complete-endpoint measurement

## Evidence and consequences

Focused tests reject five-versus-four selection and matching unknown/malformed
extras, preserve valid v2 receipts and fail-closed source/runtime checks, verify
the actual benchmark definition order without running endpoints, and execute the
workflow's run/skip selector against small Git histories. No benchmark definition,
production implementation, numerical tolerance or retained raw receipt changes.
Relevant PRs now pay for one additional existing small-grid sentinel when it was
previously excluded. This CI repair itself makes no new performance claim.

## Follow-through: authenticate the actually tested base

The first repaired PR job, [114290059407](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38078393147/job/114290059407),
built synthetic merge `440e32daa956d864a73abde7f10994ca1b1cd3db`, whose first
parent was then-current master `dec25f5922f8a63c56ecd9f6f89eb345595f3755` and
second parent was PR head `5b583f52a291042f08db148a0e2d26c627731b70`. Its event
base still supplied `2b68d10a9f43175d796500d7853ed90245f24315` to qualification.
Seven intervening master commits included CPU preparation/layout changes, so
comparing that tested merge to the stale base could not isolate the PR delta.
The CPU fingerprint differed independently, and qualification correctly skipped
all benchmark execution/upload. Job success was not performance clearance.

The qualifier now derives the base only after matching checked-out HEAD to the
event merge SHA, proving exactly two distinct valid parents, and matching the
second parent to the expected PR head. The first parent is then the sole SHA
used for authenticated master-artifact lookup and receipt validation. Require
the target branch to be `master`; stacked targets, malformed identities,
unrelated/head-only/shallow checkouts and unproven ancestry remain advisory and
cannot upload. Never fall back to the payload's stale base or the latest branch
tip, which may have moved beyond the actually tested merge.

Small local Git-history tests model master advancing beyond a stale payload,
and negative tests cover malformed/swapped/unrelated parents and non-master
targets. Existing schema-v2, exact selector, source and CPU/runtime requirements
remain unchanged. Missing actual-base receipts or a different runner fingerprint
still prevent comparison; neither condition warrants relaxing those requirements
or blindly rerunning. The final CodSpeed analysis must also identify the intended
baseline before it can resolve the earlier regression. This fix changes no
benchmark endpoint, production source, raw receipt or measured result.

## References

- [PR #2232 report](https://github.com/jinzhezenggroup/generativeqc/pull/2232#issuecomment-6100855187)
- [Pinned Python instrument](https://github.com/CodSpeedHQ/pytest-codspeed/blob/v5.0.3/src/pytest_codspeed/instruments/analysis.py)
- [Pinned native hooks](https://github.com/CodSpeedHQ/instrument-hooks/blob/b9ddb5bc654b2e6fa13eb18efcd3a45e7ecda0bb/src/instruments/valgrind.zig)
- [Callgrind instrumentation resets](https://valgrind.org/docs/manual/cl-manual.html)
- [CodSpeed state-dependent variance](https://codspeed.io/docs/instruments/cpu/reducing-variance)
