# Decision: one native owner for compiled named KS programs

Status: implemented
Date: 2026-10-16

## Problem

The native CLI already compiled canonical MethodIR compositions at build time,
but kept the data in CLI translation units. Python reconstructed those same
semilocal/exchange wire arrays on every native descriptor lowering. Native method
ID/capability queries did not transfer ownership of these scientific parameters.
This advances R5/R6 of #2227 independently of the deferred R3 response/HVP work.

## Decision

Move catalog registration out of the CLI-only CMake branch and compile its data
once in the native method owner, including library-only builds. CLI discovery,
singlepoint and geometry composition consume borrowed views from that owner.
The existing generator remains the mathematical compiler: in particular it
emits each exchange primitive's already-derived spin-resolved Fock coefficient,
rather than introducing a new native exchange equation.

Python submits the complete immutable MethodIR identity and explicit numerical
domain to an optional native descriptor query. Matching represented graphs use
library-owned static component/exchange arrays directly. Grid/radii, tile,
schedule and size-tagged application precision controls remain caller-owned;
preparation still copies pointees and owns all scientific admission.

## Compatibility and invariants

- No library/provider/device discovery occurs during selection. Query only the
  calculator's already-selected library, including resource preflight.
- An explicit named SDK query may omit the identity; an identity-based Python
  query may omit the name. Counted strings reject embedded NULs, malformed
  hashes, invalid backends and aliased output/input storage transactionally.
- A matching name with a different full identity never substitutes the compiled
  default for a custom graph. A different numerical domain is also explicit.
- Successful zero selection leaves the descriptor unchanged and declares the
  existing compiler/custom/noncatalog route. This is not a numerical failure
  fallback. Actual native refusal or inconsistent publication is authoritative.
- Missing optional symbol and library-free metadata retain the existing portable
  compiler lowering. Selected descriptors retain the library until native
  preparation has copied its borrowed constant arrays.
- Backend flags describe the existing catalog's representability, not universal
  force/HVP support. RSH/nonlocal/correction domains unrepresented by this
  catalog retain their existing explicit graph owners and qualification gates.

## Verification and remaining work

Native tests cover the complete catalog on both shape-only backend selectors,
transactional strings/identity/alias boundaries, execution-control preservation,
same-name custom models and recovery. Python compares native scientific wires
with independently compiled MethodIR programs, exercises native-only routing,
portable compatibility, refusal, application extensions and actual prepared
endpoints. Actual catalog/API-object sanitizer coverage is separate from full
scientific-provider and CPU/CUDA endpoint qualification.

This is not complete R5/R6 or #2227 acceptance. Typed frontend resolution,
unrepresented complete graphs, basis/custom-model parity and independent
numerical/resource/endpoint gates remain. R3 stays deferred. Per-head raw build,
oracle and timing receipts must distinguish completed qualification from pending
work; moving catalog data is not a numerical or performance claim by itself.

## Rejected alternatives

Keeping a second CLI table or re-lowering Python arrays then overwriting them
does not establish one native runtime owner. Matching only public names loses
custom scientific parameters. Treating a selected native failure as an old-library
retry would hide ABI/admission bugs. Moving compiler algebra into C++ would create
the duplicate scientific implementation forbidden by the parent tracker.

## Revisit when

All named scientific graphs have one qualified native program representation, or
a deliberate public catalog API supersedes this additive private transport.
Compatibility retirement must retain explicit custom compilation and independent
oracles, not infer support from a method-name inventory.
