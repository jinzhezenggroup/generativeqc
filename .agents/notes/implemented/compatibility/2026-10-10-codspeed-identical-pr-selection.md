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

## References

- [PR #2232 report](https://github.com/jinzhezenggroup/generativeqc/pull/2232#issuecomment-6100855187)
- [Pinned Python instrument](https://github.com/CodSpeedHQ/pytest-codspeed/blob/v5.0.3/src/pytest_codspeed/instruments/analysis.py)
- [Pinned native hooks](https://github.com/CodSpeedHQ/instrument-hooks/blob/b9ddb5bc654b2e6fa13eb18efcd3a45e7ecda0bb/src/instruments/valgrind.zig)
- [Callgrind instrumentation resets](https://valgrind.org/docs/manual/cl-manual.html)
- [CodSpeed state-dependent variance](https://codspeed.io/docs/instruments/cpu/reducing-variance)
