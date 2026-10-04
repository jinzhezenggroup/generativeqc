# Decision: Admit DF nuclear sink construction overlap

Status: implemented
Date: 2026-10-04

## Problem

The sink's former twice-logical host allowance did not bound construction.
Growing primitive vectors retain old and new allocations simultaneously. A
257-primitive orbital s shell plus a one-primitive auxiliary s shell reached
10,252 host bytes against an 8,584-byte allowance on the reviewed Linux host.
Final capacity checks run too late to prevent this peak.

Exact packed-array reservations alone are insufficient: spherical d/f expansion
initializer lists briefly duplicate their term arrays, and the g generator has
additional Cartesian, polynomial and vector workspace. The Arena pointer table
also overlaps host metadata and device uploads.

## Decision

- Validate counts and reserve all eight packed arrays before filling them
- Reserve the g expansion's known AO and per-AO term bounds without changing
  its polynomial or normalization arithmetic
- Replace the sink's generic factor of two with packed payload sizes, positions,
  result, the largest single-shell typed expansion workspace, and 18 pointers
- Reserve and charge that pointer table only for the nuclear sink

The single-shell bound is paired with `molecule::ao_expansions`: Cartesian
components, heap-resident AO vectors and term arrays; duplicate d/f terms; and
the g polynomial plus reserved term arrays. Revisit it if that implementation or
the admitted angular range changes. Allocator bookkeeping, fixed owner objects,
caller systems and CUDA runtime overhead remain outside the numeric-payload
contract.

## Evidence and limits

`tests/python/test_df_nuclear_sink_admission.py` extracts the production packer,
Arena and constructor, links real `basis.cpp`, and substitutes only CUDA runtime
calls. Its 96 cases cover every admitted shell-role/representation combination,
primitive/shell growth boundaries, byte-equivalent packed arrays, exact and
one-byte-short budgets, and cleanup after rejection. Undefined-behavior
sanitization is enabled. The 257-primitive case now peaks at 4,448 host bytes
under a 4,496-byte allowance; the observed host/device overlap is 8,728 bytes
under an 8,788-byte combined allowance.

The original `0aa34782` constructor fails the same peak checks for that case
and minimal spherical auxiliary f/g cases. These host tests establish storage
admission, not GPU execution, synchronization correctness, or kernel numerics;
the existing real-device and independent derivative tests retain those roles.
