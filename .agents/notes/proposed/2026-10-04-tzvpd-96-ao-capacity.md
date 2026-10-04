# Bounded stationary capacity for full 96-atom def2-TZVPD

Status: proposed; capacity implemented, independent GPU qualification pending
Date: 2026-10-04

## Problem

The requested HF-style series is 3/6/12/24/48/96 atoms with full spherical
def2-TZVPD, cold and warm energy plus analytic forces, and displaced geometry.
The last case has 1856 AOs, 1184 packed primitives, 768 shells, and 2359296
points on the matched 48 x 16 x 32 per-atom grid. Its three 1024-AO force
admission barriers are unrelated to the existing through-f integral coverage.
Small accepted endpoints do not qualify the larger domain or its performance.

## Decision and resource audit

Extend the compiler and native stationary allocation bound to 2048 AOs; the
composite driver uses the compiler's existing shape validator instead of a
third independent literal. Preserve the 128-atom/16384-primitive bounds, the
4096-point tile bound, and the requirement for native integrals beyond the
small AO-descriptor diagnostic domain. No enlarged AO^4 descriptor path is
admitted. D/W arrays and AO metadata are dynamic; generated geometry uses
size_t local/global AO indices. Native grid allocation uses checked products,
and its GEMM dimensions remain below INT_MAX. Neither the one-electron provider
nor prepared Direct binding has a 1024-AO array.

The existing exact arena formula, conservative native reserve, optional cache
fallback, and complete host/device admission stay unchanged. At 1856 AOs the
native reserve alone is 885850112 bytes. The native dry nonlocal query for the
full grid and a 256-point pair tile returns 434257940 bytes. With seven runtime
sources and a 1024-point AO tile, the complete planner bounds are 3263916916
device bytes and 3965317140 host bytes before the optional 64 MiB AO-map cache.
These are inventory bounds, not measured peaks. The pre-existing SCF owner and
driver/compiler objects are excluded and must still be provisioned separately.

Keep the production 1 GiB device / 2 GiB host incremental allowance unchanged.
The comparator accepts explicit `--force-max-device-bytes` and
`--force-max-host-bytes`, records both in `native_experiment`, and applies them
only to the scoped WB97M-V force consumer. Four GiB per domain admits this
inventory without reducing the grid or replacing the full basis. Reducing
tile size does not eliminate the native reserve or full-grid nonlocal storage.

## Validation and evidence boundaries

Host tests compile the actual native allocation/create bodies, compare exact
compiler/native bytes through 2048 AOs, check the 1856-AO last density address,
reject 2049 and SIZE_MAX, and reject a one-byte-under mandatory capacity before
allocation. Planner tests retain optional center-cache fallback and full-grid
storage. The real-device geometry gate uses the unmodified 96-atom snapshot,
both spins, dense and noncontiguous maps across index 1024, and independent
PySCF/Libxc finite differences. This isolated slice is not complete-endpoint
qualification: retain separate full cold/warm/displaced E/F gates.

The running frozen pre-extension 12-atom campaign has observed about 63 s
warm versus 18 s reference; one native warm includes 33.16 s integral
derivatives and 3.31 s combined geometry/pair drain. These partial observations
do not constitute accepted complete-series timings or a VV10-only measurement.
They disprove extrapolating the 3/6-atom advantage to this size. Native cold
also pays many repeated SCF Fock builds. Raising capacity is not a speedup.

## Rejected shortcuts

Do not strip diffuse/f functions, thin the grid, loosen gates, report missing
sizes as zero, lower the conservative reserve without provider evidence, or
interpret task/AO counts as FLOPs. Do not mix the frozen small-size binary with
the capacity extension without a source-equivalence or new qualification record.

## References

- PR #1761; benchmark controls: [same-basis cold note](2026-10-04-tzvpd-cold-density-controls.md).
- Supersedes the admission ceiling, subject to independent resource/numerical
  gates, in [through-f scheduling](../implemented/performance/2026-10-01-default-screened-through-f.md).
