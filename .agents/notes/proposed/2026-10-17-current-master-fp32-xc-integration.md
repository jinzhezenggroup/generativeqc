# Decision: qualify FP32 XC products on the current shared runtime

Status: proposed
Date: 2026-10-17

## Scope

Compose the shared FP32 matrix admission and guarded float diagonal schedule
on master `b8102246862f68603adae0394dd41f2205d87425`. Preserve its basis,
device-fleet, and existing FP64 diagonal implementation. This composition is a
new numerical candidate; qualification of an earlier source is not transferred.

Budget Fast selects FP32 density/potential inner products by shared capability
and shape, not by functional name. Density AO columns use binary normalization
to preserve physical tails in FP64 storage without moving their dot products
back to FP64. Both standalone and compact potential consumers can use the
already-existing diagonal schedule with float arithmetic. Its float-only range
guard falls back to the original float two-legged contraction, never a hidden
strict solve. Strict point algebra, compensated totals, global storage and
inter-tile accumulation retain their current contracts.

The composition also includes the separately submitted guarded FP32 Boys owner
and its geometry-test dependency correction from PR #2574. It does not include
the separate pair-FMA experiment or rejected chained-FMA candidates. No new
algorithm, cache, or scheduling optimization is introduced for FP64.

## Validation boundary

The compact host simulator implements actual width-16 shuffle semantics with
full-warp rendezvous so the normalized float density body executes unchanged.
This is host evidence, not evidence of GPU compact execution. Initial host
qualification passes 68 tests with 164 explicitly opt-in CUDA skips; the
compiler dependency audit passes 565 modules with zero errors. Formatting
checks pass. None of these observations qualifies complete E/F or performance.

Freeze the composed source, authenticate the native binary, and run the full
mixed-matrix E/V population, all 108 XC and 84 Boys cases, and materialized
mixed J/K under finite Slurm jobs. Memcheck and independent references remain
required, with no tolerance changes. Only subsequent independent physical-tail,
family and six 96-atom E/F returns can qualify complete Fast behavior. Fast
requires one solve and zero retries; a strict recovery is not a successful
Fast result. Existing Fast E/maxF/RMS limits remain 1e-5/1e-4/3e-5 and strict
limits remain 1e-7/1e-6/3e-6 in their established units.

Performance must use newly identified complete endpoints against the same
master plus the same Boys changes, excluding the new XC float work from the
denominator. Use a clean matched RTX 5090 CUDA/NVML device and retain complete
first-use and regression samples. Node2's known NVML mismatch admits numerical
evidence only. Historical v19/v22 timings and isolated v27 CUDA tests are not
performance or complete-endpoint evidence for this composition.

## References

- `2026-10-17-shared-fp32-xc-matrix-products.md`
- `2026-10-17-fp32-diagonal-xc-products.md`
- `../implemented/performance/2026-10-16-guarded-fp32-boys-shared-owner.md`
