# Decision: native corrected KS programs own their resource lifetimes

Status: implemented
Date: 2026-10-28

## Problem

The complete native named KS program was deliberately disabled whenever Python
supplied a global ResourcePlan. Its electronic owner retained one D3 plan per
system, whereas the standalone correction request priced one ragged D3 fleet.
CUDA tables, CPU execution allowances, and the equal-share local correction cap
therefore differed. Simply removing that dispatch guard would undercount storage.
D3 CUDA allocations also bypassed the shared device ledger entirely.

## Decision

Reuse `d3_resource_usage` to query the actual KS collection of single-system
correction owners. The public query reports the minimum aggregate local cap
required by the existing equal-per-item policy, separately from simultaneous
retained capacities. Native named preparation uses that same query before any
numerical owner is allocated. No dispersion equations or force signs change.

Add named scalar/batch preparation with the existing public device-resource
handle. The ordinary native prepared lifecycle retains the electronic and
correction charges together; closing a frontend handle does not detach them.
D3 allocates and frees through the shared CUDA ledger, including rollback after
partial setup. A custom-deleter construction guard also covers host exceptions
while staging uploads or formatting an error.

Python keeps separate inspectable KS and D3 requests but supplies their selected
capacities to one native ledger. It copies completed correction receipts instead
of executing a second correction. Native query or preparation failures never
retry the standalone owner. Missing additive symbols retain the explicit older
library route.

CUDA cutover requires the incumbent native stationary-capacity receipt. Otherwise
an electronic force compatibility consumer could create a second unplanned
correction owner. CPU force admission additionally includes a prepared-item work
bound; CPU plans conservatively reserve the standalone correction-gradient layout
alongside native energy owners. Such a consumer never adds the correction energy
again. This allowance is capacity, not a claim that compatibility work ran.

## Rejected alternatives

- Pricing a native KS batch as a single ragged D3 owner hides replicated tables
  and changes the meaning of an exact local budget.
- Binding correction allocations only around the Python call loses ownership
  during native geometry rebuild and after frontend handle destruction.
- Falling back after native OOM could silently change the selected topology and
  discard the refusal receipt.
- Replacing the existing KS correction scheduling with a new ragged controller
  would broaden a resource cutover into a scientific publication/lifetime
  refactor. Sharing immutable tables or grouping correction launches may be
  revisited with its own endpoint qualification.

## Evidence and limits

[Source-matched evidence](../../../../benchmarks/results/ks-d3-resources-20261028/README.md)
retains independent CPU/CUDA builds, numerical and complete endpoint gates,
installed SDK checks, actual CUDA transfers/launches/peak storage, and explicit
exact/below-capacity and cleanup tests. The per-item topology increases table
uploads and correction launches; it is not described as less GPU work.

This advances #2227 R2/R5/R6 for admitted named D3 programs. D4/gCP global planning,
general resource authority, precision-audit/fallback ownership, and other current
production cutovers remain separate. R3 response/HVP is unchanged. Historical
ownership inventory source anchors and unrelated PR evidence are not relabelled.
