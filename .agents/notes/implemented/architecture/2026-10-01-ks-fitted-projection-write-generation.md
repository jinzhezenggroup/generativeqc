# Decision: Bind KS fitted-projection leases to provider write generations

Status: implemented
Date: 2026-10-01

## Problem

A KS final-state token establishes density identity, but the prepared DF provider
can serve another KS owner while that token remains current. A later occupied-K
build can overwrite the same U allocation with the same rank. Pointer, device,
stream and rank checks alone then revive an old Cocc/U lease with unrelated U.

## Decision

Every existing projection-scratch revocation advances a provider-local monotonic
write generation before submission. The occupied-projection facade returns it;
KS captures the exact generation immediately after its occupied-K submission,
requires it when retaining its density-generating Cocc, and checks it again when
querying the final lease. Interleaved writers before completion and failed writes also invalidate
the old generation. UINT64_MAX saturates and disables this optional reuse rather
than allowing an old generation to repeat. This adds no allocation, transfer or
synchronization and does not change the SCF Hamiltonian or derivative equations.

## Invariants and evidence

The source-executing host probe reproduces same-rank lease revival on the prior
implementation and covers unchanged, revoked, same-rank, different-rank,
new-state, interleaved-before-finalization and saturated-generation cases. Existing warm-publication failure and
corrected-final lifetime probes remain separate gates. Native CUDA qualification
also checks D=2*Cocc*Cocc^T and U=B_CPU*Cocc against the independently prepared
CPU whitened tensor, query transfer counts, cache-update-disabled operation and
replacement-owner invalidation. Those GPU cases require real-device execution;
host probes do not qualify their numerical values.

## Revisit when

A future owning immutable projection buffer removes the shared scratch lifetime,
or a concurrent provider API introduces an explicit asynchronous borrow protocol.

References: PR #1661, src/scf/cuda/df_plan_internal.hpp,
src/scf/cuda_fock_execution.cpp, src/dft/cuda_ks.cpp.
