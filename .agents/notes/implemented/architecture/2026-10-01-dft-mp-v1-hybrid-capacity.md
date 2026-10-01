# Decision: include global hybrids in the DFT-MP-v1 capacity preflight

Status: implemented
Date: 2026-10-01

## Problem

The frozen DFT-MP-v1 manifest already requires direct-CUDA FP64 PBE0 and
versioned B3LYP energy-plus-force rows, but the static stationary-capacity
preflight filtered its report to LDA/PBE/r2SCAN. That left hybrid production
rows outside the same basis, resource, packaged-AOT, and public-route audit even
after the shared CUDA hybrid force path and hybrid stationary AOT profiles had
landed.

## Decision

The capacity preflight now includes required PBE0 and B3LYP
`fp64_energy_forces` rows alongside the existing semilocal rows.

Hybrid rows:

- derive the stationary plan from the resolved MethodIR rather than substituting
  a semilocal plan;
- retain the native functional code and exact full-range exchange coefficient in
  the selector contract;
- bind the public hybrid selector to its stable native DFT carrier while keeping
  the hybrid MethodIR identity authoritative;
- select and verify the exact PBE0/B3LYP packaged stationary AOT profile;
- consume the same frozen def2-SVP/grid/resource limits as the semilocal rows.

The report remains a static preflight. It does not claim CUDA execution,
numerical correctness, convergence, or speed.

## Rejected alternatives

Keeping a second hybrid-only capacity script was rejected because it would
duplicate the frozen workload, basis/grid accounting, resource gates, and AOT
verification. Treating PBE0 as plain PBE was also rejected: the semilocal
functional code may be shared, but the stationary plan and AOT identity include
the exact-exchange primitive. B3LYP likewise requires its own semilocal
functional code and hybrid plan identity.

## Invariants

- Method names do not select a separate scientific force implementation.
- Hybrid capacity rows must preserve the resolved MethodIR exact-exchange
  coefficient and full-range operator identity.
- PBE and PBE0 may share a native semilocal functional code without sharing a
  stationary-plan/AOT identity.
- Missing or mismatched hybrid AOT artifacts fail closed.
- Static capacity acceptance is not scientific or performance qualification.

## Evidence

The capacity test suite covers all 35 required FP64 force rows in the frozen
manifest, including required PBE0 and B3LYP rows. It verifies B3LYP/PBE0 AOT
profile names, carrier identities, coefficients, exact-exchange payloads, and
the existing static resource blockers.

## Consequences

The #1187 production work package can now use the same machine-readable
preflight as #1186 instead of treating its hybrid rows as out-of-band. Final
closure still requires the real-device numerical, lifecycle, no-runtime-compile,
resource, and endpoint evidence defined by #1187.

## Revisit when

Revisit if the native DFT carrier model changes, the hybrid stationary AOT
profiles are replaced by a different packaging identity, or the DFT-MP-v1
frozen workload contract is versioned.

## References

- #1187
- #1185
- #1191
- #1483

Agent: ChatGPT
Model: GPT-5.6 Sol
