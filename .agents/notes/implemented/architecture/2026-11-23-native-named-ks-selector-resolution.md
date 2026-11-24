# Decision: Resolve named CPU KS compositions in the selected native catalog

Status: implemented
Date: 2026-11-23

## Decision

Use the existing compiled named KS selector table to resolve CPU composition
strings in `Calculator`, including aliases and spin. This is a synchronous
metadata query on the same library the calculation will use. It opens no
context and cannot authorize a method, property, basis, or provider. The SDK
and CLI already bind selectors through this table. Python still presents a
typed MethodIR view and verifies that its reconstructed graph has the exact
native digest before applying the existing execution controls.

Do not rediscover a native-selected spelling through the Python alias index or
independently choose a different MethodIR. Missing bindings and older libraries
retain the existing resolver, while malformed native metadata and digest
mismatches fail before preparation. Stable legacy carrier spellings, explicit
custom graphs, automatic Libxc resolution, and CUDA constructor/profile
selection are unchanged. This does not make Python construction free of
compiler metadata imports or change the generated science.

## Evidence

The whole native catalog test matches name metadata to the prepared program
for each backend without executing CUDA. Python tests block the old alias
resolver for two hybrid aliases and a spin-polarized D3 composition, test a
native identity mismatch, and check compatibility with the old ABI. The
installed CLI and Python PBEH-RKS Hessians agree on the exact MethodIR, six
raw columns and completed-direction count. All 152 CPU CTest cases pass.
Ten post-warm-up paired samples on the same HeH⁺ endpoint give complete
constructor-to-Hessian medians of 0.294 s before and 0.293 s after. Both
variants perform seven SCF Fock builds, six directions, 24 induced Fock
actions, and 18,432 response XC point visits. The installed comparison,
raw timings and work receipts are retained under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/native-named-ks-resolution`.
