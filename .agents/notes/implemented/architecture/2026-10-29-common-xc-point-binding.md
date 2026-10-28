# Decision: prepare all CUDA XC providers through one execution binding

Status: implemented
Date: 2026-10-29

## Problem

Curated, composed and automatic providers already shared physical feature
traversal, but ordinary launch, batch launch and automatic identity were resolved
separately. Native batching also used automatic-registry membership as its gate.
Moving curated methods into that route would therefore disable their qualified
batching before any change to the actual point kernel.

## Decision

Resolve a CudaXcPointBinding once during preparation. It owns the selected
ordinary callback, the qualified batch callback and optional borrowed automatic
scientific descriptor. The compiler emits both callbacks from the same provider
census. Delete the separate late batch resolver and consume the retained binding
in both execution schedules. A null batch callback means bounded tile execution.

Keep all existing kernels, thread choices, equal-spin selection, component
scales, precision paths and scientific domain identities. Automatic raw batch
entries remain available to their existing independent probes but are not
advertised as production-qualified. Response retains its unbatched contract.
No host/device allocation, feature repacking or per-point dispatch is added.

This is the execution-binding step toward provider unification. It does not
claim that curated numerical policies can already be replaced by raw Libxc
policies, or promote automatic mixed precision, graph replay or hybrid methods.

## Rejected alternatives

- Reclassifying curated functional codes as automatic would inherit narrower
  preparation and execution gates, and could substitute different low-density
  and empty-spin rules.
- Enabling every descriptor's raw batch callback would confuse executable
  availability with complete endpoint/resource qualification.
- A second per-point functional switch would add device work to a host-only
  preparation concern.

## Validation

Compiled host probes execute the emitted selector and actual LDA/PBE point
arithmetic against independent fixtures. They cover ordinary/batch pairing,
policy changes after binding, restricted/unrestricted and response execution,
split hybrids, automatic descriptor retention, missing entries and unqualified
automatic batching. A one-off comparison confirms the complete production
point program is byte-identical; no permanent source fingerprint is introduced.

GPU qualification and matched complete-endpoint measurements are recorded with
the source and library identities in the PR and ignored local qualification
artifacts. This refactor makes no compile-time or library-size speedup claim.
