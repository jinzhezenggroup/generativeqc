# Decision: move closed-shell response equations to installed owners

Status: implemented
Date: 2026-09-29

## Problem

After #1412 and #1572, Hessian orchestration and closed-shell nuclear-response
reconstruction were installed owners, but the shared RHF/CPKS problem contract,
matrix-free response equations and fixed-density semilocal XC response kernel
still lived under repository-only `tools/generativeqc_response`.

Moving the live NativeRKS Hessian adapter directly into the public runtime at
that point would either create a production-to-tools dependency or duplicate
the response equations.

## Decision

Make these installed modules canonical scientific owners:

- `generativeqc.response_problem`: RotationLayout, ResponseProblem and response
  compatibility/error semantics;
- `generativeqc.response_operator`: RHF/CPKS matrix-free operator equations,
  operator identities, transpose semantics and logical resource contract;
- `generativeqc.response_xc`: fixed-density semilocal XC feature-Hessian
  response kernel.

The corresponding `tools/generativeqc_response/{problem,operators,xc}.py`
modules are now compatibility re-exports only. Existing solver, backend,
live-state and spin adapters keep their current imports and therefore consume
the exact same installed class/function objects.

The RKS native adapter is also bound explicitly to the installed
`CPKSResponseOperator`, `ResponseUnsupported` and
`FixedDensityXCDerivativeKernel` objects.

## Preserved boundaries

- No GMRES/Krylov implementation is moved or duplicated in this slice.
- No public Calculator Hessian/HVP property is added.
- No C property ABI changes.
- No new method/backend scientific capability is inferred.
- NativeRKS/UKS live-state leases, direct/CUDA J providers, reference/source
  adapters and solver bindings remain separately owned and must migrate or be
  explicitly integrated before a public Hessian endpoint is promoted.
- Existing tools imports remain source-compatible.

## Evidence

Ownership tests require object identity between tools compatibility imports and
installed owners, require NativeRKSResponse to subclass the installed
CPKSResponseOperator, reject any production import of `tools.*`, and reject
scientific definitions in the three tools compatibility shims.

Existing response/Hessian test suites remain the numerical acceptance gates;
this change intentionally alters ownership, not equations.

Refs #180, #179, #932, #1412, #1572.

Agent: ChatGPT
Model: GPT-5.6 Sol
