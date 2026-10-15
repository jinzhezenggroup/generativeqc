# Decision: Shared native ECP workspace authority for HF and KS

Status: implemented
Date: 2026-10-15

## Problem

The installed CLI already used the generated-policy-backed
`generativeqc_resource_ecp_workspace_v1` shape query, but Python HF/KS plans
continued evaluating the ECP polynomial and CUDA radial schedule in Python.
Even the newer native HF fleet and KS host queries received a Python-owned
ECP capacity. That left a production sizing dependency on Python compiler
policy modules despite an existing native authority.

## Decision

The shared Python adapter forwards public/Cartesian AO counts, atoms,
primitives, term counts and the CPU/CUDA scheduling selector to the already
loaded library's ECP query. CPU HF phase/fleet and KS host inventories, their
portable enclosing-inventory compatibility paths, and CUDA HF/KS adapters
all use it independently of their other sizing-query versions. No extra
library selection or GPU probing is permitted. All-electron inputs return
zero without querying ECP or importing its portable compiler policy.

CPU HF now makes one scalar ECP query per ECP-bearing item in addition to
the existing fleet query. CUDA HF reuses one ECP query for its host and device
allowance; CUDA KS uses separate CPU-host and CUDA-device selectors. This
deliberately keeps the existing private ABIs stable rather than adding another
fleet schema merely to fold these scalar inputs into its descriptor.

Dimensions are checked before ctypes conversion, including the host size_t
range. Native rejection is authoritative and cannot fall back to Python.
Returned capacities retain the signed-int64 resource-planner gate. The native
ABI remains unchanged and publishes output only after a successful bound.

## Compatibility and provenance

Library-free dry runs and older libraries missing the ECP symbol retain the
portable formula. Its imports are isolated to that explicit compatibility
path. This fallback is necessary for provider-free planning, not an alternate
production authority when the native query rejects a request.

ECP authority has its own schedule-identity field rather than inheriting the
HF/KS phase-query owner. KS distinguishes host and device ECP owners: a CUDA
dry run can finish portable host sizing before explicitly loading its native
device-layout library. Recording just the eventual device library would
incorrectly label the earlier host sizing as native.

## Boundaries

Generated policy, scientific equations, quadrature execution, CUDA execution
eligibility and the ECP radial tile gates are unchanged. A CUDA selector in
this query performs only scalar arithmetic and does not qualify a real-GPU
endpoint. Global admission, leases and complete endpoint allocation receipts
are still outside this migration increment; #2227 R2 remains open. No
performance improvement is claimed.

## Evidence

`tests/python/test_ecp_resources.py` compares native bounds with independent
portable arithmetic across CPU/CUDA selectors and the 16/17-public-AO radial
gate, including distinct public/Cartesian sizes. It covers absent/old-library
fallbacks, failure injection, invalid inputs/outputs, native transactional
publication, arithmetic overflow and the largest admissible term count.
Public ragged RHF/UHF/RKS/UKS plans cover exact/below capacity budgets,
independent ECP/enclosing-inventory provenance and mixed-age libraries. CUDA
adapter wiring uses stubbed CUDA layouts and the real CPU-build ECP scalar
query, with portable ECP sizing forbidden; it does not run GPU science.

The four existing public CPU ECP ragged-budget endpoints pass for PBE RKS/UKS
and Cartesian/spherical representations against PySCF 2.14.0. They retain
below-budget rejection, exact-budget cold/warm execution, geometry replay,
failure recovery and independent energy/force checks. Qualification enables
`GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT`: the all-electron fragment's
native force-admission assertion is not valid for a build with that option
disabled. These endpoint gates are not complete allocator/RSS receipts.

## Revisit when

Native prepared-resource descriptors own ECP sizing directly, allowing the
private scalar adapter to retire, or the old-library/provider-free planning
contract is explicitly removed.

## References

#2227 R2; #933/#934; `src/api/c_api_resources.cpp`;
`python/generativeqc/resources_hf.py`; `python/generativeqc/resources_ks.py`;
`docs/maintainer/resource_planning.md`.
The preceding fleet decision is recorded in
`.agents/notes/implemented/architecture/2026-10-15-native-cpu-hf-fleet-sizing.md`.
