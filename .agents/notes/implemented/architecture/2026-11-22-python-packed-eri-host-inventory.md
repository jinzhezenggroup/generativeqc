# Decision: Bind Python packed-ERI plans to the native retained fleet

Status: implemented
Date: 2026-11-22

The energy-only contract here remains unchanged. The later
[native force plan](2026-11-23-python-packed-eri-force-plans.md) adds a separate
inventory and explicit admission; it does not upgrade these energy plans.

## Decision

Previously the CPU calculator's `ResourcePlan` sized ordinary dense ERIs
even when native preparation was asked to retain a packed array. Bare
programs could also lose named-provider selection when a resource plan was
present. Route explicit packed plans to the same `KsHostFleetResources` owner
used by the [native CLI](2026-11-22-cli-packed-eri-host-inventory.md), retaining
the ordinary dense host query and its portable metadata fallback for callers
who did not opt into packed storage.

The native packed query validates the whole ragged fleet transactionally.
It uses the provider's packed extent, charges each retained array once and
keeps the conservative ordinary setup workspace. It classifies an insufficient
provider cap separately from an invalid query, without publishing partial
capacity. The global host plan composes that native primal inventory with
other owners such as native D3. Changed geometry releases the old item plan
before installing its replacement, so two copies of its packed array are
not retained during a rebuild. The packed mode and requested cap are part of
the plan identity, even when loosening that cap would not alter the peak.

This is an energy/primal resource contract, not forces, a new derivative
algorithm, or an RSS limit. Packed budgeted batches default to energy and do
not upgrade to forces. An explicit packed request cannot use the dense
compatibility path if the packed query or named program is missing. Unbudgeted
packed execution and ordinary dense behavior retain their prior controls.

## Evidence

The native query tests exact fit, one byte short, larger caller ceiling, and
transactional refusal of an invalid later item. Python exercises ragged PBE0
RKS and corrected B3LYP UKS: exact provider and global caps, one byte short
before native work, external plan identity, replay without a geometry change
and after rebuilding, sampled host capacity, and dense-provider energy parity.
The CPU resource/CLI suites passed 4 CTests and 192 Python tests, with 11
out-of-scope skips. The focused packed tests also passed against the installed
library. The preceding independent hybrid/native Hessian suite completed
separately: 57 passed.

For CsH/def2-SVP with scalar ECP, corrected B3LYP RKS admitted at
121,456,900 modeled host bytes with a 758,640-byte packed cap. Either cap
short by one byte refused before native preparation. The observed numeric
host samples during the geometry rebuild reached 2,662,744 bytes; sampling
does not cover every transient allocation. The complete cold, warm and
moved-geometry sequence reported respectively 16, 4 and 11 Fock builds.

Seven interleaved pairs per case on one EPYC 7K62 core used the same native
library and pre-change versus patched Python adapters (base `cae2959e1`). All
five complete sequences returned identical energies, SCF iterations, Fock
builds and warm-state selections. The maximum measured median slowdown was
0.23% on ordinary dense paths; none of the three packed cases regressed.
The old bare PBE0 packed case was unbudgeted because it had no equivalent
named budgeted path. Timings and counters are retained under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/python-packed-bench`.
