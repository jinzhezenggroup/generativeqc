# Decision: Share the prepared DF Fock device inventory with the KS planner

Status: implemented
Date: 2026-11-21

## Decision

The private `generativeqc_resource_prepared_df_fock_v1` query accepts compact
orbital and auxiliary topology for one full-range or erf owner. It shares the
prepared Fock owner's source metadata bound, fitted policy, and optional
restricted resident promotion. The response envelope includes any exact
provider in this owner; the initial metric upload can exceed that retained
envelope. `generativeqc_resource_ks_cuda_fleet_v2` carries the construction
peak and separately admits deferred setup that overlaps all retained items.
The original Direct fleet wire layout is unchanged.

## Invariants and next composition

Do not reuse the batched HF DF inventory: it sizes a different SCF and DIIS
owner. Do not infer an explicit fitted-capacity guarantee from a successful
shape query. Authenticate the provider, basis role, storage controls and
native derivative reservation from the same descriptor used by preparation.
In a range-separated KS plan, budget independent radial sources and compose
the late secondary source construction with the already-live primary owner.

The device leaves alone do not authorize public fitted budgets. On the host,
`df_source_capacity::plan` covers source packing, not the whole fitted value
owner. Metric setup now applies the shape-derived eigensolver bound to both
returned workspace sizes before allocating either one, using the same policy
as the ordinary DF AO eigensolver. Compose that allowance with source staging,
optional MD/host panels and retained metadata before enabling the complete KS
request. Do not substitute the existing Direct host inventory or treat a
post-hoc diagnostic as a bound.

## Evidence

The native bridge tests cover a source metric larger than the retained budget,
exact-provider coexistence, automatic capacity, invalid-input nonpublication,
and separate fleet construction/deferred maxima. Five host suites passed 284
tests (11 skipped), including execution against both mock eigensolver ABIs;
an oversized host-only metric request does not launch a solve or allocate
either workspace. Slurm `2230` completed six real-device endpoint regressions
on RTX 5090 against the build including the metric workspace check. Three
cases exercise fitted LR, but only the Direct PBE case uses a public global
budget; these regressions do not qualify a fitted public plan. See
`build/hessian-gpu-sm120/r2scan-orbital-f/hessian-fitted-metric-regression-n5.log`.

## References

- [Public Hessian composition](2026-11-21-cuda-hessian-public-budget.md)
- [Public fitted PBE composition](2026-11-21-cuda-pbe-fitted-hessian-public-budget.md)
- [Resource planning contract](../../../../docs/maintainer/resource_planning.md)
