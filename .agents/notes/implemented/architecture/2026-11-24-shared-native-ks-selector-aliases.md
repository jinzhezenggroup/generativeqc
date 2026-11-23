# Decision: Share named CPU KS selector resolution

Status: implemented
Date: 2026-11-24

The public method manifest maps exceptional KS spellings onto existing catalog
selectors. Compile these aliases into lookup, not into duplicate program rows:
exact MethodIR lookup, provider capabilities and basis ownership must not change
with the spelling. Composite aliases derive their target from the manifest's
identifier and spin. The installed catalog does not need a compiler at runtime.

CPU `Calculator` looks up its default KS graph even when the name already has
a stable ABI carrier. It authenticates the detached Python graph against the
native identity, but does not replace explicit science on carriers that allow
custom options. Missing rows or old libraries retain their existing metadata
route. Native refusals and identity mismatches are terminal. The SDK and CLI
query the same programs; CLI carrier routes retain their existing controls.

This change moves name selection, not compiler IR construction for custom
methods, basis/resource admission or GPU execution policy. The compatibility
parser reads the shared alias manifest rather than defining another alias table.
