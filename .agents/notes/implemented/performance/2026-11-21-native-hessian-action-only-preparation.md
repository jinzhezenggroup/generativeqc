# Decision: Prepare standalone native HVPs without dense staging

Status: implemented
Date: 2026-11-21

## Problem

The C API and C++ CLI knew whether the request was for a vector or a matrix,
but always prepared an owner with a private `(3N)^2` FP64 matrix and optional
repeated-RHS caches. The one-vector path never read that storage. Paying for
it under the aggregate budget could reject a feasible one-vector request.

## Decision and invariants

Select an internal preparation mode at each frontend. Both modes retain the
same private HVP staging, physical CPKS solve, authentication and publication
gates. Only dense preparation admits the private matrix and optional CPKS and
frozen exact J/K caches. Dense publication still requires the complete matrix
to converge and pass symmetry and live-state checks. Do not allocate a dense
matrix lazily: that would defer capacity failure until the action and make
prepared memory usage incomplete. A dense call on an action-only owner fails
before attempting even one response direction or touching output.

There is no public ABI change. Omitting the dense candidate does not imply
linear total HVP memory: provider workspaces still account for AO, grid, and nuclear
pair resources separately. Freed capacity may be used by other optional
provider workspaces under the same bound.

## Evidence

The host resource fixture fits action-only preparation below the mandatory
dense footprint with repeated-RHS caps enabled. It retains the dense one-byte
rejection and allocation/provider failure cleanup controls across spin and
fitting shapes. The linked RTX 5090 probe verifies unchanged output and zero
attempted directions on wrong-mode calls, then compares the standalone HVP to
both the dense owner's HVP and its assembled matrix times the same direction.
On job 2065, Direct RKS and fitted-primary/fitted-LR UKS give identical
standalone and dense-owner HVPs; the maximum errors against the assembled
matrix action are `3.99e-17` and `4.59e-15 Ha/Bohr²`, respectively. Both
one-shot calls report one completed direction and no frozen-exact storage.
