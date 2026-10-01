# Decision: Report only measured DF stationary resource scope

Status: implemented
Date: 2026-10-01

The stationary DF bridge contracts retained host H'/S' derivatives for its
one-electron and Pulay sources. Their force-time CUDA transfers are zero, but
this says nothing about the separate DF J/K response. The CUDA provider builds
J and K response terms separately: RKS uses one density for each, UKS uses one
total density for J and two spin densities for K. The response also has
policy-dependent metadata, tensor, weight and workspace movement. Deriving a
whole-response upload count from spin count is therefore invalid.

Keep the private nine-slot wire layout. Slot 3 reports the actual simultaneous
capacities of compact H'/Pulay/J'/K' vectors and their four-source candidate;
pre-admission reserves at least eight coordinate vectors and capacity validation
runs before publication. Slots 4/5 report only one-electron execution movement.
Python names the compact-publication host peak explicitly and exposes
`density_fitted_response_resources_included=0` both in native-source metadata
and the outward stationary work record. The CUDA transfer subrecord and device
bound also state their incomplete DF scope. Whole-KS DF resource-plan admission
remains rejected.

A source-executing host regression covers exact and over-reserved capacities,
budget boundaries, all nine wire slots, outward scope markers and the production
CUDA response term/upload-loop counts. These are structural/host proofs, not
measurements of a GPU execution or a complete DF response inventory.

Revisit only when the common Fock derivative seam propagates actual
DfGradientResources for every response invocation, preserving sequential peak
versus cumulative transfer semantics and all policy-dependent paths.

References: PR #1654; src/scf/cuda_fock_provider.cpp;
src/scf/cuda/df_gradient_bridge.cu; src/methods/dft_method.cpp.
