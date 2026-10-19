# Decision: seal bounded XC declarations outside production admission

Status: implemented
Date: 2026-10-19

## Problem

Issue #2330 needs an auditable arbitrary-formula precursor without granting a new
formula the native SCF or nuclear-force qualifications of existing compositions.
The scalar Graph and `FunctionalSpec` already have separate mathematical and
scientific responsibilities. Capturing arbitrary Python would require proving
purity, bounded execution and closure stability, beyond a small public-contract
slice.

## Decision

Introduce a compiler-owned string-to-Graph adapter for bounded dual-spin LDA
polynomials. Export the immutable value through `extensions.xc.expression` and
`xc.inspect`. Keep it distinct from `FunctionalSpec`, so existing method admission
continues rejecting it. No runtime, integral algebra, TensorIR AD, XC production
domain, CMake, or method manifest changes are necessary.

Reuse canonical Graph construction, associative canonicalization, evaluation and
symbolic AD. Serialize reachable nodes with dense remapped indices and typed
operation payload spelling as existing XC graph provenance does; hash through the
canonical provenance owner. Preserve original source separately from semantic
identity and revalidate that relationship on replay. Graph and inspection exports
are detached, preventing mutation of a sealed formula through retained handles.
Replay compares canonical JSON hashes of the complete envelope, preserving JSON
index types rather than Python's coercive `True == 1 == 1.0` equality. Decoder or
encoder recursion failures inside the length budget become the public validation
exception; they do not escape through a separate runtime-error contract.

## Rejected alternatives

- Python decorators/callback tracing: cannot reliably reject hidden global state,
  control flow or unbounded loops without a substantially larger capture contract.
- Accepting caller Graph handles: exposes mutable ownership, mixed graphs and
  undeclared variables at the public boundary.
- Fractional powers and density denominators in v1: require explicit boundary and
  derivative-domain policy, particularly at zero minority spin and vacuum.
- A parallel polynomial/AD engine: violates the existing scalar mathematical owner.
- Reusing `FunctionalSpec` or naming a built-in: would imply scientific admission
  that neither the contract nor pointwise tests establish.

## Invariants

V1 accepts only the documented bounded syntax, rejects sigma/tau, and defines
per-volume atomic units. No native kernel, SCF callback, oracle fallback or
production entitlement follows from construction or evaluation. Both CPU and
CUDA compilation, SCF and forces remain UNQUALIFIED. Semantic identity includes
the versioned domain/spin/unit contract, excluding descriptive labels and paths.

## Evidence

`tests/python/test_xc_user_expression.py` checks independent 60-digit Decimal
energies, hand-derived gradients/Hessians through canonical AD, central
differences, vacuum/minority spin, declaration rejection, budgets, domain failure,
copy/process replay, tampered envelopes, detached exports and method refusal.
Existing extension, scalar expression, capability and compiler-structure tests
remain the relevant regression controls. See the PR for exact source-matched
commands/results; these tests are pointwise contract evidence, not endpoint science.

## Consequences and revisit conditions

This intentionally excludes realistic fractional-power LDA exchange as well as
GGA/meta-GGA. The small polynomial vocabulary permits explicit zero-density
behavior and independent checks. Extend with a new schema/domain policy only
after feature derivatives, boundaries and numerical acceptance are specified.
Native MethodIR/SCF/force/CUDA integration remains in later #2330 slices; this
does not close #2330, #558 or #1118.
