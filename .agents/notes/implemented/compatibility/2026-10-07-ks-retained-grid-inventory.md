# Decision: charge the complete retained molecular grid in KS inventory

Status: implemented
Date: 2026-10-07

## Problem

The three-output native KS resource bridge reused a standalone XC layout as a
combined XC-plus-grid bound. Standalone XC owns four doubles per point (xyz and
partition weights). Generated CUDA quadrature now retains five doubles per
point, including the pre-partition atomic measure used by force consumers, in
one shared `OwnedCudaBuffer`. The borrowed XC plan keeps that entire owner alive.
The old bound omitted eight bytes per point for every retained KS item. A
serialized setup excess cannot cover this persistent fleet-wide difference.

## Decision and boundaries

Keep the existing ABI and compute its XC slot from the borrowed-grid FP64 arena
plus `cuda_resident_grid_bytes`, using checked addition before publishing any
outputs. Python sums the combined slot once per retained owner and continues to
charge only setup excess as transient storage. No new Python grid formula or
CPU/GPU admission policy is introduced. Explicit caps remain binding; a request
that exceeded an old underestimated cap must become infeasible before execution.

Public device-fused KS constructs a CUDA grid and borrows it. Host-unfused KS
retains no CUDA grid and already clears the combined XC slot. Direct XC callers
with a host grid and response callers retain their separate four-array private
layout. Mixed AO computation still uses FP64 storage. Those layouts and the
optional point-panel fallback guard are unchanged.

This correction covers the dense incumbent. It does not certify optional AO-map
or lowering-provider reservations, physical GPU free memory, or arbitrary force
workspaces. Do not treat the host ledger tests as a GPU molecular endpoint run.

## Evidence

`tests/python/test_ks_grid_inventory.py` compiles the production shape functions,
program metadata, allocator ledger and shared owner with CUDA allocation test
doubles. It checks LDA/PBE and both spins, tile tails, independent direct/response/
mGGA private-array counts, atomic-weight retention through the last shared lease,
FP64/AUTO public composition, host-unfused clearing, explicit-cap rejection and
both multiplication/addition overflow without partial outputs.

The dense 2,048-owner H2/PBE case reserves the complete retained storage and then
the existing full 512-MiB force allowance, with zero optional point-panel bytes
and complete release. The old 3,860,795,004-byte plan is no longer admitted.
`tests/native/test_cuda_quadrature.cpp` also compares the native XC slot against
private arena plus measured retained-grid allocation in the real-device gate.

## References

- [PR #2089 review](https://github.com/jinzhezenggroup/generativeqc/pull/2089#pullrequestreview-5444750725)
- [Current grid ownership](../../../../../docs/developer/dft_grid.md)
