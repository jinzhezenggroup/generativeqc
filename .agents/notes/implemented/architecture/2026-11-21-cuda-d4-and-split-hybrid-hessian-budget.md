# Decision: Separate CUDA KS energy capacity from force capacity for Hessians

Status: implemented
Date: 2026-11-21

## Problem

A linked stationary Hessian prepares an accepted SCF energy, then opens its
own response arena. The named D4 resource gate also demanded an inventory for
a standalone analytic force owner, which rejected matched DF and some Direct
contexts even when they could run the Hessian. The fitted planner identified
PBE0-D4 and B3LYP-D4 by their `pbe-rks` transport carrier rather than their
physical electronic models. The opt-in M06-2X/MN15 fitted routes also lacked an
admitted state/Fock inventory.

## Decision

Use the resolved electronic identity for fitted energy planning, and continue
to validate the complete D4 identity against the native catalog. Energy-only
requests can use a native named D4 owner without a native force receipt. An
energy-only prepared batch cannot be promoted to forces: doing so could open
a second correction or generated derivative owner outside the prepared
ledger. Existing force planning requirements are unchanged.

The same energy-only admission applies to catalog D3. Shape-only standalone
D3 remains a valid first-order compatibility path, but Hessian planning must
reject it before allocating a batch because its response is not in the KS owner.

For canonical Direct hybrids, the native rebuild-state query counts KS state
with K matrices and XC/grid storage; its `fitted_v1` name does not mean it
counts a fitted source. The native radial inventory independently reserves the
Direct Fock owner and initial setup. Do not use the old J-only fallback state
query for these hybrids or count the Fock source twice. Fitted hybrids use the
same state query with their own DF capacity envelopes. This also admits bare
PBE0/B3LYP with f orbitals, where the native stationary force receipt is absent.
M06-2X/MN15 require the
exact graph and linked stationary domain, strict FP64, the fused order-3 grid,
all-electron orbitals, and explicit auxiliary sources when fitted.

The intrinsic `pbe-d4-rks` ABI has a fixed native correction allowance. For
budgeted execution only, resolve its exact catalog counterpart, plan the PBE
projection and D4 separately, and pass the caller's dispersion cap to the
catalog ABI. Ordinary unbudgeted intrinsic execution stays unchanged.

## Evidence

`test_named_d4_cuda_complete_hessian` checks corrected matrices and actions
against the native CLI and an independent charged `dftd4` gradient stencil.
`test_cpp_cli_and_python_cuda_split_hybrid_hessian_and_hvp` checks the linked
point domain, Direct and explicit DF, RKS and UKS, and s/d/f orbitals with the
appropriate auxiliary inventory. Both check device-ledger peaks, allocation
refusals, and release of response storage. The named D4 energy-only replay test
checks that forces fail before any compatibility correction is launched.
`test_d3_cuda_complete_hessian` isolates the D3 response via independent
`dftd3` gradients, and the missing-inventory test verifies early refusal.
On the Slurm SM120 node, jobs 2257, 2259, and 2260 covered 24 split-hybrid
contexts, 16 named D4 contexts, and 10 named D3 contexts, respectively. The
extended intrinsic PBE-D4 test compares both budgeted endpoints to the
unbudgeted physical model and the native CLI.
