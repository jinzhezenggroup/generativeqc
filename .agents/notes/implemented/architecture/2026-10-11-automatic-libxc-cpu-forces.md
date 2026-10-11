# Decision: Bind automatic CPU Libxc forces to the shared stationary owner

Status: implemented
Date: 2026-10-11

## Problem

The automatic semilocal CPU SCF registry and its native E/vrho/Cartesian-gradient/
vtau provider already feed the shared stationary diagnostic. Public force
admission nevertheless rejects every automatic selector before that owner can
run. Using the LDA/PBE/r2SCAN carrier to decide force science would repeat the
named-method bottleneck in #1118/#1122 and could discard imported provenance.

## Decision

Extend the existing output-aware, bounded direct all-electron CPU force context
to a single, exact automatic semilocal component with rho, rho/sigma, or
rho/sigma/tau ingredients. Require FP64, no exchange, no density fitting, no
correction/nonlocal source, and an all-electron s/p/d basis. The public Calculator
context may advertise optional forces; the backend-neutral registry and default
energy-only output stay conservative.

No force algebra, source regularization, generated expression, or native method
identity changes. The live imported point provider remains the differential
owner. The existing stationary pipeline supplies AO-center and point motion,
Becke partition response, overlap/Pulay, one-electron and J-only integral sources,
and nuclear repulsion. Existing resource, final-state and changed-model guards
remain mandatory.

The shared KS selector resolver must also recognize automatic registrations for
resource planning and live derivative identity validation. The imported CPU tau
executor uses the same four AO jets and dense potential matrices as GGA, with
only fixed-size stack tau/point values. Its existing host inventory therefore
applies without a new resource formula. Keep the older unsupported guard for
other tau-planning domains and explicitly reject automatic CUDA planning; do not
turn successful CPU selector resolution into an implicit backend admission.

## Rejected alternatives

- A positive per-functional force list would reproduce the issue's bottleneck.
- Force computation under the curated carrier's point model would differentiate
  different mathematics from the converged imported SCF.
- Calling PySCF in production would introduce a prohibited reference fallback.
- Advertising CUDA, ECP, density-fitted, response or second-order capability from
  the CPU semilocal force seam would inflate unmeasured product coverage.

## Evidence

The numerical campaign uses independent pinned PySCF 2.14.0 / Libxc 7.0.0 SCF
and separately reconverged native energy finite differences at every Cartesian
coordinate, using the actual moving native quadrature. The maintained matrix
contains three non-curated LDA, five GGA and three tau-MGGA registrations, genuine
open-shell UKS and RKS, cold/replay/changed-geometry force endpoints, and separate
fully polarized zero-beta-density checks. Acceptance requires energy agreement
within 2e-7 Hartree, force agreement with both numerical references within
3e-5 Hartree/Bohr, and translational residual below 2e-8 Hartree/Bohr.

`tools/qualify_libxc_forces.py` emits hash-bound force receipts containing exact
capability, library, MethodIR/plan, grid, functional, solve-generation and full
displacement-stencil provenance. Receipt regression tests reject rehashed wrong
backend/spin/method/functional, incomplete matrices, substituted finite
differences, stale hashes and changed-geometry aliasing. Control-flow tests
exercise energy/force budget transitions and preserve excluded contexts.

Normalize snapshot ingredient sequences at the receipt boundary: native detached
identities use tuples while JSON uses lists. Checking list equality against the
in-memory tuple incorrectly rejects a live campaign that validates after JSON
round-trip. A regression protects identical admission for both representations.
The breadth fixtures must themselves be imported/representable; for example,
the current TPSS exchange registration is structurally blocked by a non-scalar
Maple exponent and cannot be counted toward tau-MGGA endpoint coverage. This is
not a functional-specific force blacklist.

Validation source base: `637466c624c4c4c4dd4eb7c511a2c88675744325` plus this
focused Python change; the native CPU library is built from that source using
the existing compiler cache. Raw qualification outputs stay in ignored local
`.cache/` artifacts, not a positive runtime admission inventory.

## Invariants and revisit conditions

Do not remove the exact component/ingredient/provider checks or infer public
second derivatives from this path. Libxc work floors/clamps retain their pinned
E/vxc convention; this change does not differentiate the clamps. A reproducible
functional-specific defect must fail numerical qualification and be diagnosed
under the existing structural/negative-list policy, not hidden by an oracle.

Revisit when CUDA imported point execution or another derivative ingredient has
its own complete-endpoint qualification. This CPU force slice does not close the
remaining CUDA execution and packaging deliverables of the parent tracker.

## References

- #1118 and #1122.
- `docs/user/libxc.md`.
- `tests/python/test_libxc_automatic_forces.py`.
- `tests/python/test_cpu_semilocal_force_contract.py`.
- `tests/python/test_libxc_force_evidence.py`.
