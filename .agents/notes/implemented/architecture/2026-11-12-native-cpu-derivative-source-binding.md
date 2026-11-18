# Decision: borrow CPU derivative sources from their authenticated KS owner

Status: implemented
Date: 2026-11-12

## Problem

The native stationary column assembler needs the actual prepared Fock, normalized
AO basis and moving grid from the same converged KS owner. The existing snapshot
bridge exports detached geometry and frame arrays, while the integral-system
borrow exposes only the normalized system. Reconstructing the other providers
from a snapshot can silently change grid, basis, provider or lifetime identity.

## Decision

`borrow_dft_host_derivative_sources` uses the existing private batch owner port
to authenticate the current token before borrowing all CPU source owners. It
returns immutable pointers to the actual system, prepared Fock, normalized AO
basis, grid and frozen native dispersion metadata. It performs no numerical
evaluation or shape-dependent numerical allocation. Unsupported backends,
missing states, forged or cross-item tokens, invalid indices and revoked states
clear the entire binding. Existing ports without this capability refuse it.

The owning context must be serialized over the whole synchronous borrow and
consumer lifetime. Returning a binding does not extend the owning batch lifetime
or authorize retaining pointers across replay, geometry rebuilding, destruction
or releasing the context lock. The consumer still needs an independently verified
stationary frame and all its model, capacity and derivative admission gates.

The dispersion pointer describes only what the native KS owner actually owns.
Some frontend total methods compose an additional geometry-only correction
outside that owner. Such a correction cannot be inferred from the electronic
token or advertised as covered by this binding; a complete endpoint must bind
and authenticate the requested total composition explicitly.

## Validation and scope

The compiled `generativeqc_ks_host_source_binding_tests` executes actual native
B3LYP RKS/UKS and canonical PBE0-D3(BJ) RKS/UKS batches. It checks matching
verified-frame/Fock/basis/grid and integral-system owners, actual native
correction identity, pre-SCF refusal, forged and cross-item tokens, invalid
indices, recovery, same-geometry replay, displaced geometry and explicit state
revocation. These are ownership and lifecycle controls, not new scientific
qualification of PBE0/B3LYP/D3 Hessians or a public CPU endpoint.

Installed native CPU Hessian/TS composition, real heavy-element electronic
qualification, complete electronic scaling and requested final saddle/frequency
acceptance remain separate requirements. Revisit the raw-pointer handoff if a
future consumer needs asynchronous execution or a lifetime beyond the serialized
native context; do not silently weaken the current synchronous lifetime contract.
