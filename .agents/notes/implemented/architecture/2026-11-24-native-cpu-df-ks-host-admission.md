# Decision: Bind CPU DF-KS host capacity to native preparation

Status: implemented
Date: 2026-11-24

## Problem

A matching Python or CLI quote previously admitted the request only at the
frontend. The SDK could prepare the same CPU DF-J graph without receiving its
effective limit, and an energy-only quote did not define the native handle's
replay boundary.

## Decision

The named KS descriptor carries an optional effective host limit and admitted
property set. The single-system and fleet entry points reconstruct the
existing native capacity inventory from the actual system handles and parsed
execution controls before creating a scientific owner. A fleet adds retained
providers but takes the largest serialized workspace. On success the handle
retains the allowed properties; a force or materialized-provider upgrade is
not a replay of an energy-only or streaming reservation. Earlier descriptor
prefixes retain ordinary unbudgeted semantics.

## Invariants

- This is the existing all-electron, FP64, named LDA/PBE DF-J domain, not
  admission for guesses, hybrid DF, ECPs, density-fitted second derivatives or
  hidden allocator/vendor/process memory.
- An effective budget comes after frontend reserve/headroom composition.
  The native gate does not trust a frontend's topology or reuse it as a
  scientific identity. The prepared force owner still authenticates support.
- Explicit SCF and force tiles must be represented in the planned bound.
  An energy bound does not have to reserve a force-only tile.

## Rejected alternatives

Copying the frontend quote verbatim into the handle would freeze a claim
without authenticating the consuming topology. A new allocator-wide CPU
ledger would overstate the current ability to account for opaque allocations.
The existing checked resource model supplies the narrower truthful gate.

## Evidence and follow-up

`tests/native/test_named_ks_catalog.cpp` exercises exact and one-byte-short
single and fleet budgets, energy-only force refusal, and unchanged outputs
against ordinary native execution. It also passes both earlier wire sizes
without allocating the new descriptor suffix. The CLI and Python public DF
tests cover explicit auxiliary topology and prepared force parity. The installed
CLI, C++ SDK and Python binding agreed exactly on the H₂/STO-3G PBE-RKS force
endpoint at 203,824 modeled bytes and each refused a cap one byte smaller.
An installed Python replay check also matched ordinary energy, forces, SCF
iterations and Fock-build counts across original, moved, and restored water
geometries for PBE-RKS and LDA-UKS.

General provider leases and cross-owner host/resource accounting remain #2227
work; the inventory reserves capacity for one owner, not across independently
prepared handles.
