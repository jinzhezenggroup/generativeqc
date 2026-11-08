# Decision: admit shared MP2 primal work before energy

Status: implemented
Date: 2026-11-08

## Problem

The iodine-ECP CBS inputs from #3034 exposed a work-amplification boundary in
the existing optional CUDA MP2 force cache. Its six compatible MO blocks were
constructed only after energy completed. Energy batched tiny occupied/virtual
tiles within a numeric budget, but each batch scanned the complete AO source.
The 97-AO TZ case repeatedly traversed 88,529,281 AO values to return only
125,952 MO bytes per batch. Retained incomplete runs spent 829–840 seconds per
scan. Additional GPU memory and the later force cache did not remove this work.

This supersedes the construction order, sequential producer schedule and
energy-only exclusion in
[the original force-cache decision](2026-10-12-mp2-cuda-force-mo-cache.md).
The original measurements remain historical evidence, not qualification of
this changed schedule.

## Decision

Construct optional compatible primal blocks before energy, in one heterogeneous
provider batch sharing its AO input and exact matching transformation prefixes.
Use the existing six-block symmetry cover for force endpoints, or OVOV alone
for energy-only endpoints and when the larger store cannot be admitted.
Choose the widest basis-axis source tile whose complete heterogeneous batch
peak fits the remaining phase allowance. Charge common host/device ownership
once, each request's numeric buffers individually, and retained output vector
capacities to all subsequent consumers. Keep the quarter-budget storage limit.

Borrowed energy consumes OVOV with one reduction job at a time. It does not
construct an unused AO provider, reserve a budget-sized tile-job collection,
or charge nonexistent transformation staging. Its own reduction workspace and
the still-live reference/source remain admitted separately from caller-owned
OVOV storage. Charge all actual reference vector capacities, including the
weighted density, rather than assuming a fixed matrix count after removing
the provider's conservative scratch allowance. This distinction is necessary
even when enough storage exists for the complete force cover.

## Invariants and fallback

- Exact conventional AO integrals, canonical orbitals, FP64 contractions,
  denominators, convergence tolerances, iteration caps and force gates do not
  change. No density fitting or production CPU/reference oracle is substituted.
- The cache is local to one endpoint and immutable canonical frame. A source
  memory fallback cannot outlive or change that reference's orbital ownership.
- Construction retries only bounded resource failures, first with OVOV alone
  if appropriate and then through the existing streamed consumer. Non-memory
  backend and numerical failures propagate.
- Independent energy/force cache disables retain diagnostic replay. Work
  journals distinguish primal AO scans from borrowed energy/force consumption.
- Quartic VVVV storage remains optional. This removes repeated source scans
  only on admitted domains, not a universal arbitrary-basis work guarantee.

## Validation

Executable native tests cover heterogeneous returned-block parity, exact and
one-byte-short admission before source traversal, borrowed versus independently
transformed MP2 energy, and borrowed reduction admission without AO staging.
Public CUDA tests compare independent fixture energies and complete forces
across source, energy-cache and force-cache selections. The original #3034
component/composite two-step finite-difference gates remain unchanged; complete
case qualification and exact library identity must be recorded separately from
these focused tests.

## Rejected alternatives

Raising the memory budget or merely increasing the source tile leaves the
energy's repeated AO scans intact. Moving six sequential scans before energy
removes that first amplification but needlessly regenerates compatible AO
inputs. An unconditional dense MO store would remove bounded admission and is
not acceptable. Relaxing reference/force gates cannot fix this scheduling issue.

## Revisit when

The admitted six-block store becomes dominated by VVVV capacity or host block
gathering. Prefer a response-aware bounded representation and complete endpoint
qualification to raising the storage fraction or advertising kernel-only wins.
