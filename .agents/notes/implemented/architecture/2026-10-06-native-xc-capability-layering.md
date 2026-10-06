# Decision: Keep native grid capability lowering below XC algebra

Status: implemented
Date: 2026-10-06

## Problem

The generated-capability refactor in PR #2010 made DFT grid consumers import
higher-level XC modules, creating four prohibited DFT-to-XC edges. The new
Python capability census also omitted fields read by source generation, and
new native header includes were absent from wheel/JIT input inventories.

## Decision

The manifest remains the only capability source. Its canonical generated
Python execution records now live in DFT, alongside the native grid ABI
lowerers. The former XC metadata module re-exports the exact same record
objects for compatibility. FunctionalSpec construction and XC expression
resolution remain in XC. DFT consumes native family selectors or read-only
resolved components, range parameter and ingredients without importing XC.
Compatibility spellings are explicit manifest aliases. Legacy grid admission
still checks exact native component/range composition and the existing
LDA/PBE point-kernel lowering, rather than assuming all GGA features mean PBE.

## Rejected alternatives

Broadening the compiler dependency allowlist would hide the inversion.
Moving FunctionalSpec construction into DFT would preserve the same problem.
Maintaining a second handwritten family or ingredient inventory would permit
policy drift. Generated compatibility imports avoid a second mutable registry.

## Invariants

- Preserve the DFT-to-XC dependency prohibition and canonical registry identity
- Keep unsupported compositions and ingredients fail-closed
- Include both semilocal_family.hpp and xc_capabilities.hpp in wheel assets and
  grid JIT native-header hashing inputs
- Keep scientific source/asset hashes based on the actual compiler inputs
- Device-free registry and admission checks do not qualify GPU numerics

## Evidence

The compiler structure check passes with zero dependency errors after the
change. Focused generation, legacy-alias, actual dispatch emission, native-free
force/Hessian admission and compiled-resource tests cover the data path. The
final-state admission probe consumes the real generated C++ family metadata
through the shared ccache fixture, rather than a stale handwritten family enum.

Agent: dot
