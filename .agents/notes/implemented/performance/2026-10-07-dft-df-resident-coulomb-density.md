# Decision: borrow the final CUDA RKS density in DF Coulomb force response

Status: implemented; endpoint performance not yet qualified
Date: 2026-10-07

## Problem

After the prepared CUDA KS solve, the exact accepted restricted density remains
resident on the same stream used by the prepared DF Fock provider. The
density-fitted stationary response nevertheless allocated a fresh response
density buffer and copied the detached host density back to the device for each
component call.

For PBE0-DF this meant that even after final Cocc/U reuse removed one major K
reconstruction, the Coulomb J' component still paid another full N_AO^2 H2D
copy of the already resident final density.

## Decision

Add a narrow method-owned CudaDfBorrowedResponseDensity lease. It contains the
device, exact density pointer, AO-matrix extent and owning stream. The DFT owner
constructs it only from CudaKsPlan::resident_final_density() under the same live
final-state token used by the stationary force snapshot.

The first production consumer is deliberately restricted to one restricted
Coulomb response term:

- one term only;
- nonzero Coulomb coefficient;
- zero exchange coefficient;
- one-system prepared DF owner;
- matching device, stream and AO matrix extent.

The detached host density remains present and is still validated for shape,
finiteness and symmetry. The lease changes placement only: the CUDA response
algebra reads the already resident device density rather than uploading the same
matrix again.

Exchange response does not consume this lease in this change. The existing
final fitted-projection path remains independent and executes before J when its
one-shot projection is available. UHF and multi-term response retain their
ordinary density upload.

## Resource and schedule invariants

The response keeps the existing conservative private-budget arithmetic, including
the density-sized reservation used in auxiliary-tile selection. Removing the
allocation/copy therefore cannot widen the selected tile or silently change the
response schedule in this first integration.

When the resident density is used:

- density_host_to_device_bytes is zero for the J-only response;
- the borrowed density bytes are reported through borrowed_device_bytes;
- response_borrowed_density and response_borrowed_density_bytes trace counters
  identify actual execution;
- all existing raw-value, fitted-projection, shell, screening and metric
  response policies are unchanged.

Explicit host-response diagnostics remain authoritative and do not select the
resident density lease.

## Validation

Native CUDA coverage compares the ordinary host-density Coulomb response with
the resident-density response on the same prepared plan and requires identical
derivatives, a full density H2D on the baseline, zero density H2D on the borrowed
path, and explicit borrowed-byte accounting.

Host/source-contract tests verify the full handoff:

CudaKsPlan final density -> DFT method -> PreparedFockPlan -> CUDA DF provider ->
generated force response -> bounded CUDA response bridge.

They also enforce the J-only admission boundary so exchange or multi-term
responses cannot accidentally consume the lease.

No complete endpoint speedup is claimed before matched GPU measurement.

Stacked on PR #2046.

Agent: ChatGPT
Model: GPT-5.6 Sol
