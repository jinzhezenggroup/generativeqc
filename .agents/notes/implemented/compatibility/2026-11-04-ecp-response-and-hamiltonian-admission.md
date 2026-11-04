# Decision: admit the native scalar-ECP Hamiltonian and response workspace

Status: implemented
Date: 2026-11-04

## Problem and decision

CBS compared decoded BSE JSON records, although the native owner receives typed
floating-point residual terms. Equivalent decimal strings could therefore be
rejected as different Hamiltonians. Reuse the existing native-input ECP resolver
and compare nuclear charges, core counts and atom-mapped typed residual terms.
Unused element records and formatting do not change this identity; genuinely
different ionic operators still fail before endpoint execution.

The CUDA one-electron derivative bridge also invokes the ECP quadrature owner.
Add its existing complete host/device workspace bound to one-electron response
staging, rather than relying on unreserved endpoint headroom. Keep the common
single-owner ECP source-capacity accounting, including spare vector capacity.

## Evidence and boundaries

Orchestration tests compare decimal/exponent/numeric encodings in extrapolation
and correction bases, while retaining core/potential/all-electron mismatch
refusals. A native shape-only contract independently composes ECP quadrature
workspace and the two packed metadata copies. Existing physical CPU ECP,
unrestricted response, exact-budget, compact-source and lifetime contracts stay
required. No ECP quadrature, derivative equation, CUDA kernel, precision policy
or scientific acceptance tolerance changes, and host admission is not new GPU
execution evidence or performance qualification.
