# Decision: Keep residency execution aligned with its allocation-capture inputs

Status: implemented
Date: 2026-10-10

## Problem

The residency observer stack imported the allocation producer's workload. After
the latter strengthened its geometry gate to an explicit internal displacement,
the residency producer still rebuilt a rigid translation for execution. Its pin
and observed/reference endpoints therefore described different coordinates.
Historical observed-work profiles also no longer matched the new workload, and
the residency producer retained the earlier NVML-ordinal identity shortcut.

## Decision

Execute the pinned moved coordinates in all three residency histories: ordinary
CUDA, independent CPU, and observed CUDA. Reuse the allocation producer's minimal
runtime-visible full-GPU UUID resolver, preserving failed-probe and MIG rejection
without a visibility-token/NVML fallback. Require the shared source owner in the
residency manifest.

Preserve the retained work-ratchet policy bytes and historical measurement
identities. Tests reconstruct the exact earlier workload literally and check that
current workloads are rejected. Current diagnostic capture remains supported
without a configured ratchet; a current within-ratchet claim requires its own
independently qualified and reviewed matching policy. Do not relabel historical
measurements or silently widen selector matching.

## Verification and limits

CPU doubles check runtime-device selection, failed probes, numeric MIG rejection,
and exact pinned moved coordinates. Existing allocation identity and numerical
protocol regressions remain applicable. Synthetic policy tests identify their
baseline as synthetic, never as a renamed retained GPU run. Host source tests are
not new source-matched GPU/CUPTI qualification or a residency PASS.
