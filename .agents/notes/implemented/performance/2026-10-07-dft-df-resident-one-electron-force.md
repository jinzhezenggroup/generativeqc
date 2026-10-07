# Decision: reuse final CUDA KS D/W for density-fitted one-electron force

Status: implemented; endpoint performance not yet qualified
Date: 2026-10-07

## Problem

The density-fitted KS force bridge already reuses the prepared DF provider and,
for restricted fitted hybrids, the final occupied Cocc/U = B*Cocc projection.
However, its stationary one-electron H' and Pulay S' sources still contracted
the snapshot-exported host D/W with retained host derivative matrices. CUDA KS
had already staged the exact token-bound final stationary D/W on device, so the
force path repeated an O(3*Natom*NAO^2) host contraction after convergence.

## Decision

For CUDA density-fitted stationary derivatives, first borrow
CudaKsPlan::resident_final_stationary_weights() under the exact final-state
token. When the binding matches device, spin and AO shape, execute the existing
bounded paired CUDA one-electron consumer with empty host weight spans and the
resident D/W pointers.

This does not upload D/W again. The generic paired consumer still prepares its
own one-electron shell metadata and returns only the two 3*Natom source vectors.
The fitted J/K response is unchanged, including the existing restricted-hybrid
final fitted-projection lease.

If the optional paired device consumer is unavailable or cannot fit the caller's
budget, preserve the established exact host contraction. Numerical/device
failures other than admission/availability do not silently fall back.

## Resource accounting

The existing nine-slot DF stationary source record remains ABI-stable:

- slot 2 records the paired one-electron device peak when the resident CUDA path executes;
- slots 4/5 record its actual H2D/D2H movement;
- the compact publication host peak remains slot 3;
- final-state export counters remain separate;
- DF-provider response scratch/transfers remain explicitly outside this partial scope.

Python publishes density_fitted_one_electron_resident_cuda and keeps
density_fitted_one_electron_host_contraction as the complementary fallback flag.

## Invariants

- The final-state token remains authoritative; detached snapshot matrices alone
  do not authorize resident execution.
- Resident D/W must match the final KS device, spin count and AO shape.
- No new density or weighted-density H2D copy is introduced by this path.
- CPU behavior is unchanged.
- The DF J/K Hamiltonian and final Cocc/U reuse are unchanged.
- Budget failure retains the prior exact host result instead of changing the
  Hamiltonian or relaxing a resource gate.

## Qualification

Host/source-contract coverage verifies the resident binding is selected before
the fitted-projection response, the paired CUDA call receives empty host D/W
spans plus resident device pointers, the host fallback remains present, and
resource metadata distinguishes the two paths.

No GPU endpoint timing is claimed by this change. A matched PBE0-DF force
benchmark should measure the one-electron source and complete endpoint before
making a speedup claim.

References: #1478, #1567.

Agent: ChatGPT
Model: GPT-5.6 Sol
