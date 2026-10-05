# Decision: bound retained correlated RHF plans by complete numeric capacity

Status: implemented
Date: 2026-10-04
Agent: dot

## Problem

The initial correlated CUDA plan retention in #1869 used the plan's directly
owned device bytes as its entire downstream reservation. Host topology, solver
host workspace, and provider-retained allocations remain live too. Retaining
even the small mandatory arena can also displace a previously admitted MP2/CC
endpoint. The existing warm-state reservation changes the available reference
budget between cold, warm, cleared, and frozen execution, so exact budget
identity prevented the intended first replay from reusing its executable.

This note supersedes the device-only budget and exact-budget reuse portions of
[the original retention decision](2026-10-04-correlated-cuda-rhf-plan-reuse.md).

## Decision

Keep full retained numeric capacity separate from directly owned device
telemetry. Count live host vector capacities even after clear, nested ECP
payloads, solver host workspace, and recorded provider retention. Container
headers, graph objects, allocator rounding, and CUDA context/stack are outside
the established numeric-buffer convention.

For exported references, a changed budget can retain an otherwise identical
plan only after checking its previous complete reference admission, positive
host-capacity growth, and present retained capacity against the new allowance.
The cold admission formula stays unchanged: its existing candidate/topology
allowance must not be charged twice. An insufficient budget rebuilds the owner,
preserving the optional ERI cache's bounded reference fallback. Scientific
option, topology, provider, precision, and device checks remain in force.

Each downstream provider, solver, triples, or response phase may retire the
optional RHF executable once on a memory-admission or allocation failure and
retry with the restored phase budget. It keeps the detached physical reference
and completed scientific work. Failed phase-local owners unwind before retry;
numerical and validation errors propagate. Provider work counters remain outside
the retry callback, while unsuccessful output owners are replaced.

Opaque MP2/response result counters still describe the successful phase. They
are not complete attempted-endpoint work counters after an allocation retry;
those APIs have no partial-progress sink. Whole-call elapsed time includes
retries. This repair does not use successful-phase counts to claim a speedup.

Complete endpoint diagnostics preserve the simultaneous capacity of each
successful earlier phase before later phases retire storage. An admitted source
constructor's bound is also retained if partial allocation fails; rejected
pre-allocation estimates contribute no peak. A measured MP2 force endpoint stays
unknown if the force phase began with an external plan, including when a failed
attempt later retires it. The reference reuse flag still describes the successful
RHF execution if a later phase retires its plan; the retained device-byte field
then reports zero.

## Evidence

The host tests execute production capacity, bucket admission/lifecycle, and
method retry code without CUDA execution or chemistry. They cover host/provider
reservations, budget-only replay transitions, exact and one-byte-short admission,
optional ERI rebuild, callback unwinding, restored downstream budgets, and
non-memory failure propagation. The #1812 frozen-policy preflight and rejection
counters remain present in the composed RCCSD fixture.

The production MP2 planner for a two-AO, one-occupied H2/STO-3G shape admits
113,386,440 bytes with the current sm_90 tile-one kernel capacity. Its RHF
reference fits below 8.4 MiB. Retaining only the 2,456-byte RHF arena and 128-byte
ERI cache already makes the exact MP2 allowance fail; plan retirement restores
admission. Full host retention further strengthens this lower-bound example.

No GPU numerical qualification or timing campaign was performed for this repair,
and no speedup is claimed.

## References

- #1869 and #1856: correlated executable reuse
- #1812: frozen CUDA DF source-policy preflight
- `tests/python/test_correlated_reference_plan_capacity.py`
- `tests/python/test_correlated_reference_retirement.py`
