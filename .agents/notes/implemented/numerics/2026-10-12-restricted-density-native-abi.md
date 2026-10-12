# Decision: preserve original total density through the native restricted ABI

Status: implemented
Date: 2026-10-12

## Problem

The explicit density-work Graph distinguishes unpolarized total-density inputs
from polarized per-spin inputs. For the pinned Libxc 7.0.0 worker, folding an
RKS total density in `[floor, 2*floor)` into two spins floors both spin inputs.
It produces a different potential from the unpolarized worker. Independent
PySCF 2.14.0 values at rho=1.1e-15 differ by roughly 1.6e-6 to 2.0e-6 for
Chachiyo, VWN4 and modified PZ. Routine molecular geometry does not expose this
error. The predecessor correctly blocked molecular qualification rather than
changing the restricted oracle.

## Decision

Add an independent restricted value/evaluator/descriptor and native integration
and SCF overloads. Retain the public polarized descriptor's layout and existing
mangled symbols. The new overloads have distinct symbols, so an old library
cannot silently claim to implement the new semantics. A restricted evaluator
receives original total density, gradient and tau and returns scalar vrho,
Cartesian gradient coefficients and the native half-vtau coefficient.

Density-only LDA automatic lowering uses the explicit projected Graph on both
CPU and CUDA instead of reconstructing its worker projection in the adapter.
The Graph owns floors and raw-density energy weighting; the adapter only
screens total vacuum before evaluating it. CPU registry generation adds a
separate restricted lookup without extending its existing entry layout. Real
CPU method dispatch and the molecular qualification runner use this registered
descriptor; qualification is not a stand-alone correctly behaving callback.
Public plan and diagnostic domain selection follow the same structural family
policy. Merely accepting both versions in one generic diagnostic reader would
hide a stale native implementation, so each expected domain keeps an exact gate.

## Rejected alternatives

- Expand the polarized descriptor: incompatible with already built libraries.
- Qualify a private unpolarized callback without changing production dispatch:
  fixes the evidence fixture while retaining the production error.
- Use a polarized oracle, omit RKS, or relax the tail threshold: changes the
  scientific requirement rather than implementing it.
- Feed floored inputs into the projected Graph: applies energy reweighting twice.

## Invariants

- Construct and optimize interior derivatives before projecting worker inputs.
- No derivative through clipping and no hidden removal of minority coefficients.
- Domain-5 RKS rejects the polarized overload; legacy domains retain fallback.
- The binding v5 identity includes the exact restricted source/expression and
  normalization, and smoke v3 requires all five native cases (40 values).
- The molecular runner verifies the actual native restricted registry identity.
- Work-domain parity is not physical birth, force, precision or cache closure.

## Evidence

The retained independent diagnosis is the predecessor's `restricted-tail.json`.
`tests/native/test_dft_restricted_point.cpp` checks the pinned complete restricted
E/vrho values, vacuum and invalid inputs, the actual native registry, independent
total-density AO contraction across tile sizes, and rejection of a polarized
domain-5 RKS descriptor. Compiler tests require the companion's distinct
normalization and reject the obsolete two-case compilation smoke. Qualification
uses the existing pinned production-domain and both-spin molecular gates.

## Consequences

An additional U1 artifact is emitted only for the structurally admitted
density-only family. Generic restricted gradient/tau ABI fields remain available
but do not admit other ingredient families. Compiler-source identities
conservatively invalidate old receipts; source compatibility is not scientific
qualification.

## Revisit when

Other ingredient families gain independently qualified total-input worker
projections, or a versioned native ABI replaces the legacy descriptors.

## References

- #1118, #1119, #1120, #1121, #1124.
- Supersedes the restricted blocker rationale in
  `2026-10-12-explicit-density-work-qualification.md` without reinterpreting its
  earlier ordinary-geometry evidence.
- `docs/developer/libxc_bulk_capabilities.md`.
