# Decision: Native CPU KS fleet capacity authority

Status: implemented
Date: 2026-10-15

## Problem

The per-item CPU KS host query already shared its sizing equations with the
installed CLI, but Python still summed twelve retained components and selected
the maximum serialized setup/SCF workspace. The CLI separately multiplied its
homogeneous retained bound. This duplicated lifetime composition and left
aggregate overflow outside the native sizing boundary.

## Decision

`scf::CpuKsHostFleetResources` sums retained component capacities, maximizes
serialized workspace and validates the complete signed-int64 peak. Append is
transactional, including failure after a valid prefix; a repetition count
represents homogeneous CLI batches without repeated descriptors. The AO cache
stays transient even when its underlying per-item capacity is retained in the
diagnostic vocabulary.

The private additive `generativeqc_resource_ks_cpu_fleet_v1` query receives the
existing seventeen scalar shape fields per item and an explicit workspace
floor. It returns twelve retained-owner totals, retained/workspace/peak and
sixteen diagnostics per item. Two scalar passes validate all items before
publication, using O(1) native scratch, no provider discovery and no successful
query heap allocation. Inputs must remain immutable and disjoint from output;
range/length overflow and malformed later items preserve all output.

Python makes one fleet query and one existing scalar ECP query per ECP-bearing
item. It presents capacities through unchanged resource labels/lifetimes and
retains the nonlocal provider's independent sub-budget gate. This is ownership
work, not an endpoint performance claim or scientific capability promotion.

## Consumer allowance and compatibility

The force-workspace floor is a caller-supplied capacity input. Its policy and
consumer eligibility remain in the existing frontend owner; the C++ query
validates its numerical domain and combines it with retained/workspace storage,
but does not claim to own force execution or admission. Global resource budgets,
candidate selection and leases remain outside this increment. CUDA KS inventory
composition is unchanged. Capacity bounds are not complete allocator/RSS
receipts and do not complete #2227 R2.

Native rejection cannot retry the per-item query or portable math. Missing
fleet symbols retain native per-item sizing plus portable composition; missing
both queries retain metadata-only dry runs. Item and fleet authority have
separate schedule-identity fields, independent of ECP authority. No library is
implicitly loaded for a CPU dry run.

## Evidence

Native tests cover retained components, repetition equivalence, serialized
workspace/floor lifetimes, transactional invalid append, exact/overflowing
complete peaks, invalid later items, canonical booleans, malformed lengths,
overlapping storage and huge byte ranges. Python tests compare independent
portable ragged bounds across spin, public/Cartesian dimensions, AO cache
thresholds, ECP/nonlocal capacity and workspace floors. Public plans cover
per-item-only/old-library provenance and exact/below-budget boundaries. Native
CLI tests compare homogeneous LDA/PBE RKS/UKS batches against the fleet ABI.

Existing CPU resource endpoints retain exact/below-budget preparation,
cold/warm execution, geometry rebuild and cleanup gates. The four public ECP
ragged-budget endpoints also pass against PySCF 2.14.0 for PBE RKS/UKS and
Cartesian/spherical representations with packaged CPU force AOT enabled.
Real-GPU resource tests require an allocated Slurm job and are not run as part
of this CPU sizing increment.

## Rejected alternatives

Keeping Python aggregate arithmetic after native per-item queries would not
move the ownership boundary. Allocating a native per-item result vector would
add O(items) scratch solely to avoid a second bounded scalar pass. Requiring the
new symbol would break provider-free and old-library planning; using those
compatibility paths after an actual native rejection would mask invalid shapes
or incomplete peak capacity.

## Revisit when

Prepared native owners acquire complete admission/lease and force-consumer
policy authority, or typed descriptor publication can avoid the second scalar
pass without extra native storage or weaker transactional guarantees.

## References

#2227 R2; #933/#934; `docs/maintainer/resource_planning.md`;
`src/scf/ks_host_resources.hpp`; `python/generativeqc/resources_ks.py`;
`.agents/notes/implemented/architecture/2026-10-15-native-ecp-workspace-authority.md`.
The preceding per-item decision is recorded in
`.agents/notes/implemented/architecture/2026-10-12-shared-cpu-ks-host-sizing.md`.
