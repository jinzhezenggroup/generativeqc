# Decision: retain the existing qualified Fast anchor default

Status: implemented
Date: 2026-11-12

The [shared-default follow-up](../../rejected/2026-11-13-shared-fp64-anchor-default.md) evaluates
extending the same default to strict FP64 owners without withdrawing Fast reuse.

The subsequent [target-selected policy](../architecture/2026-11-13-target-selected-precision-presets.md)
keeps the relaxed-target default but removes frontend-profile privilege: equal
resolved budgets and solver goals select equal execution in Standard and Fast.

This repair supersedes only the default-withdrawal proposal in
[common construction](../architecture/2026-11-13-standard-fast-common-construction-and-anchor-selection.md).
Constructor reuse and selected-force resource admission remain unchanged.

Preserve the pre-#3075 unset policy: already eligible budget-Fock plus
budget-stationarity owners at 192 or more AOs select the existing anchor.
Strict Standard and smaller owners stay opt-in. Explicit zero disables reuse;
one retains all scientific, provider, lifecycle and resource eligibility checks;
invalid settings reject even ineligible owners. No numerical threshold changes.

The retained fast-anchor-domain-20261019 packet reports independently qualified
complete water-24 E+F reverse-order improvements of 16.47% and 16.49%; the older
fast-anchor-default-20261018 packet reports a single water-96 pair improving
20.45%. These historical measurements are not current-source timings or general
speed guarantees. Their domain nevertheless makes blanket withdrawal a credible
performance risk, not merely an absence of supplementary common-default proof.
The new water-tetramer selector GPU test is below the existing automatic floor.

Known #3064 moved/fresh force discrepancies concern DF PBE0 spatial scheduling;
that scheduling excludes range/nonlocal consumers. #3065 concerns Direct RHF
orbital stopping, and #3067 retains DF HF numerical/fitting and performance
failures. This anchor admits only restricted exact-Direct WB97M-V VV10 cold
owners; it excludes DF and other models. No evidence reviewed connects those
specific failures to this anchor or shows withdrawing it repairs them. This is
not proof that all anchor workloads are failure-free. None of those historical
failures is solved or waived by this candidate or its old performance receipts.

Host behavioral regressions cover both sides of the 192-AO floor, all eligibility
and budget-control combinations, explicit overrides and malformed values. The
candidate changes selection only, not the shared operator controller, capacity,
force precision, final-state proofs or owner/checkpoint/catalog consumers.
A future shared automatic default still needs relevant endpoint qualification.
A common opt-in policy would deliberately withdraw the previously qualified Fast
default and requires an explicit decision accepting that performance tradeoff.
