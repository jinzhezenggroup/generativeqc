# Decision: keep generated Libxc source metadata in the XC layer

Status: implemented
Date: 2026-10-16

## Problem

Source-derived nonlocal component discovery made XC import the generated Libxc
metadata module from MethodIR. This inverted the established dependency direction:
method composition may consume XC, while XC must not import method policy.

## Decision

The existing generator now emits its single metadata product as
`python/generativeqc_compiler/xc/_generated_libxc_methods.py`, beside the XC-owned
Libxc source extractor. XC, MethodIR and native-family generation consume that
same product. Maintained imports, CMake dependencies, generated-file attributes
and source-registry output ownership move together. The old private module is
removed rather than retaining a second generated authority.

The generation and mathematical contracts from
[the original metadata decision](2026-09-22-libxc-method-metadata.md) remain in force.

## Rejected alternatives

- Whitelisting the upward dependency would allow method policy into XC
- A second component catalog would duplicate source ownership and drift
- Moving these Libxc-specific facts into generic common services would obscure
  their scientific owner

## Invariants and evidence

The generated module bytes are preserved by this relocation. The compiler
structure guard and its allowed dependency graph are unchanged. Negative controls
reject both XC-to-method and DFT-to-method imports; an isolated XC import rejects
any attempted method-layer load. Both generators, source-registry verification,
compiler type checking and focused method/metadata tests cover the relocation.
This changes no equations, coefficients, native capability, or runtime admission.
