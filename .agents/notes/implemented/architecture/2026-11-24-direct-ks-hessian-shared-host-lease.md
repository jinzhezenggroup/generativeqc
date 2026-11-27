# Decision: Lease Direct CPU KS Hessian workspace with its prepared fleet

Status: implemented
Date: 2026-11-24

## Problem

The CPU Hessian resource composer modeled a combined peak and handed a
numerical cap to the SDK, but did not bind that peak to any shared native
reservation. Two independently feasible prepared fleets could therefore
exceed the same intended host capacity.

## Decision

Provide an opt-in native preparation entry for linked all-electron Direct FP64
fleets that retains the existing host ledger. Size the primal providers from
the named native system, correction, grid and actual SCF options before making
provider copies. Charge retained bytes across all live fleets and charge the
largest serialized workspace, conservatively including the full primal scratch
plus the explicit derivative capacity for that batch. Energy replay, prepared
HVP sizing and derivatives serialize on the same lease. Clip derivative execution
to its lease without borrowing unreserved numerical capacity. Scoped failure
leaves the provider and its accepted SCF state intact. A pool handle is a
metadata view; releasing it does not revoke a live provider's lease.

## Boundaries

This admission accepts energy replay and linked second derivatives; it does
not admit force replay, orbital export, a preliminary or approximate provider,
or a global process-memory limit. A conservative Python endpoint estimate may
still include frontend staging or a wider scope than the native lease. Preserve
the existing capacity-only route for contexts that do not request the lease.
Do not treat this as qualification for other methods, CUDA or all of #2227 R2.

## Evidence

The native SDK and Python PBE/D3 tests cover exact and short capacity, refusal of
a second owner, preserved accepted state, and a retry after budget refusal.
Both check that a live batch works after the metadata handle is destroyed.
