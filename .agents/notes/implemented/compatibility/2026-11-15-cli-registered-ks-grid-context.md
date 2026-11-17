# Decision: explicit CLI KS grids resolve the complete native composition

Status: implemented
Date: 2026-11-15

## Problem

The installed SDK/CLI/Python force comparison exposed a CLI-only routing gap.
The native ABI registry wins name lookup before the generated composition
catalog. Consequently PBE and r²SCAN RKS/UKS accepted their ordinary native
default-grid endpoints but rejected explicit Treutler levels and tensor grids,
even though the identical composition and complete prepared force owner were
already available. Changing only the force flag would not resolve this input
contract, and substituting another functional would change the science.

LDA also illustrated a distinct, valid naming boundary: `lda-rks` is a native
ABI selector, while its existing generated complete composition is
`lda_xc_pw-rks`. The installed-product test uses that canonical composition
selector rather than inventing an alias or copying component equations into
its C++ consumer.

## Decision

After parsing all controls, an explicit-grid request for a registered DFT
method may select its exact same-name generated composition and carrier.
Backend/model admission and prepared force properties remain authoritative.
Requests without explicit grids retain the native default route. Missing
compositions and non-DFT grid requests retain their explicit refusal.

The installed-product test compiles against installed public headers and links
the exact installed library also loaded by Python. Native SDK and CLI children
have neither Python nor compilers on PATH. Cold/warm complete SDK and Python
forces, fresh CLI forces and energy-only replay are compared with independently
reconverged PySCF energies and moving-grid analytic gradients. This is not a
second force owner or a new scientific admission list.

## Rejected alternatives and invariants

- Do not globally prefer composition names: that could change native defaults,
  initial guesses, resource accounting and ordinary energy-only callers.
- Do not silently drop explicit grids or replace them with native defaults.
- Do not waive the prepared capability query or remove energy-only DF resource
  promise guards to make product tests pass.
- Do not treat the small-system product comparison as complete R1, molecular
  HVP, arbitrary custom-program, or endpoint-performance qualification.

## Evidence and remaining qualification

The new installed-product CPU matrix passes 56 cases with 16 explicit
unavailable-SCF-provider skips. Its pre-fix failures retain the CLI's typed
grid-control refusal. The required native CLI regression covers explicit
level-zero, native-default-equivalent level-three and tensor-grid requests for
both PBE/r²SCAN spins; non-DFT requests still fail without output.

Full CUDA, source-matched endpoint performance and the remaining cross-cutting
R1 gates are separate requirements. No R1 completion, PR or parent closure is
implied by this repair.
