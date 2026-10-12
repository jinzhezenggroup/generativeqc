# Decision: native resident scalar CPKS reuses the orbital/vector engine

Status: implemented
Date: 2026-10-12

## Problem

Device XC/J actions alone left installed CPKS with Python AO/MO transforms and
Krylov vectors. Removing those seams must not introduce another solver, another
XC algebra, a functional-name whitelist, or an unsupported full HVP claim.

## Decision

The current-state native RKS object's scalar solve uses `solve_gmres_resident`.
Adapt the existing resident RHF orbital/vector engine with the actual Fock K
coefficient and a method-owned native XC density action. The shared engine
produces the signed AO direction; the XC plan consumes it directly, borrowing
the same authenticated final D/grid. Stream ordering is explicit, with scalar
domain/status validation before the potential is used. Python transports the
request and final solution but never orchestrates iterations or scientific work.

Admit only generated response-capable kernels with complete native providers.
Component scaling also requires its generated capability. Fixed MethodIR X/C
weights pass through the shared point traversal and existing response jet;
missing components are not evaluated, avoiding `0 * unused singular jet`.
Full-range exchange uses the authenticated restricted Fock coefficient, not a
named hybrid constant. Labels do not decide admission.

The first real-device qualification exposed the inherited RHF engine's always-K
output pointer on J-only requests. The common device seam requires absent output
pointers to be null. A subsequent hybrid test exposed the unweighted response
traversal and its blanket scaling refusal; passing fixed graph weights through
the existing differential, rather than removing the refusal alone, is the fix.

## Invariants

- Validate current state and exact provider system/spec/device before zero-RHS
  acceptance, host-workspace refusal or numerical work.
- Keep orbital/AO/Krylov vectors resident; only setup, scalar records and final
  solution publication may cross the host/device boundary.
- Charge retained J/K, XC and the resident arena to one explicit device allowance.
  Native host scalar/result payload has its own finite solver budget; transport
  arrays, borrowed state, setup and context/library storage are separate scopes.
- Recheck freshness before atomic publication. Typed provider/allocation errors
  propagate, and RAII drains/frees the arena on every exit.
- A CPKS density callback cannot borrow RHF weighted-density reconstruction:
  its missing XC response sources must remain an explicit unsupported boundary.

## Retained boundaries

The generic host-vector `solve` adapter, multi-RHS blocked/recycled strategies,
UKS resident CPKS, nuclear RHS/reconstruction and molecular HVP are not retired
by this scalar RKS increment. No production CPU/reference fallback is introduced
in the new endpoint. ECP, DF, tau, range/nonlocal response remain unsupported.
In particular, a converged WB97M-V SCF does not supply its missing tau, range-K
and VV10 response sources. Do not add its name to an admission list.

## Evidence

- Independent libcint/Libxc complete scalar solves, including full-range hybrid
  and renamed semilocal graphs; initial guess, zero RHS, stale state, host/device
  budget refusals and valid replay.
- Independent Libxc potential differences for fixed X/C weights in restricted
  and unrestricted point conventions, including exchange/correlation omission.
- Executable ctypes/native control-size, status propagation and lifecycle gates;
  unchanged legacy CUDA RKS/UKS response gates.
- Finite Slurm execution and complete native endpoint memcheck; raw receipts,
  source/binary identity and compiler-cache statistics are retained in ignored
  `.artifacts/2227/resident-cpks/` and the separate qualification worker tree.

## Rejected alternatives

Do not wrap a Python GMRES loop around resident kernels and call it native.
Do not substitute a dense CPU Jacobian or PySCF action in production. Do not
differentiate a second XC formula or infer response support from SCF/force
capability. Keep current explicit compatibility consumers rather than silently
replacing their blocked/recycling algorithm with independent scalar solves.

## Revisit when

Add method-owned resident UKS/multi-RHS/reconstruction bindings through the same
solver/backend only after independent endpoint and lifetime/resource gates.
Broaden tau/range/nonlocal graphs only when every response source is complete.
