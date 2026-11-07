# Decision: reject relative-channel force reassociation without endpoint benefit

Status: rejected
Date: 2026-11-07

## Problem

The recovered A100 diagnostic (LSF 227335) attributes 1.787 s of the
r2SCAN-3c cold stationary-force call to the order-six relative force consumer.
This is nested within 3.6547 s of Direct J/K force work and 5.6331 s of the
stationary-force endpoint, not an additional independent timing. The current
density/Hermite contraction reapplies the same primitive-relative-to-atomic
linear map to every Hermite product and axis.

## Candidate

Accumulate the three relative response channels per axis and CUDA lane, then
apply the primitive's state-independent atomic map once per lane. Keep the
common primitive prefactor inside the accumulations. This changes FP64 summation
order but not the mathematical derivative, shell mask, Hermite product count,
source channels, or scatter. It adds nine lane-local doubles without a new
shared/global allocation, persistent cache, device-name selector, or environment
knob. Existing density-first admission and finite-result refusal still apply;
hybrid exchange consumers retain their current fallback.

## Qualification and decision

The emitted contraction executes against a separate long-double, per-state
atomic-map oracle across order-six/seven domains, center counts, lane tails,
signed inputs, and primitive exponent scales. That and adjacent radial/response
tests pass (107 cases). The normal dual-architecture Release image subsequently
passes six independent HF/default/lifetime/allowance tests and sixteen Libcint
signed J'/K' tests on each GPU. The native raw-AD/fallback probe passes on both;
5090 memcheck and racecheck report zero errors/hazards.

Nevertheless, the complete cold energy + analytic force + host export does not
improve. With ten unchanged r2SCAN-3c Focks, A100 is 17.373840 to 17.371610 s
(-0.01%); 5090 is 28.225757 to 28.317683 s (+0.33%). Warm medians are
6.943555 to 6.917361 s and 10.423331 to 10.434337 s, respectively. PBE0 and
WB97M-V are also essentially flat. Each arm has one cold and three own-density
warm endpoints; these are not statistical cold rankings.

Reject and remove the compiler change. Keep the independent emitted-contraction
oracle test, not a new opt-in. Nine more lane-local doubles and a different
summation order are not justified by fewer source-level operations when the
complete endpoint is unchanged. The retained HF policy is a separate qualified
change and must use its own post-revert source/ELF, never this combined image's
qualification identity.

## Evidence boundaries

The original diagnostic electronic-structure calls finished, but its compute-to-
login SSH callback failed (LSF exit 74). A CPU-only LSF read recovered its original
archive without repeating GPU measurements. The recovered archive SHA256 is
`42caade012490fbc83383582490d42af4de6e484a4a024dfc5a3b924c1ad9bbd`.
PBE0 tracing perturbs the optional MD route substantially, so its traced cold
time cannot identify the clean PBE0 bottleneck or support a speedup claim.

The completed combined-image qualification uses source
`b2fed9a7ec1a77a4003c363b243c692d914154e3986118af597959f17c21cc66`, ELF
`dc97e53ff6c82e2cd2aa7cc51d97c0d755a290308ddd58e9865d89e46f7bd931`, actual
LSF 227637 and Slurm 16671. The A100 original result archive SHA256 is
`b736c27ff2cd3916cf21bf267be10ab37a495a497f29889ae0488e462b460ca3`.
Source-bound compact evidence belongs in
`benchmarks/results/hf-cold-defaults-20261107/`, separately labeled as rejected
DFT history, not as the accepted HF-only image.

## Revisit when

A generated schedule changes the real register/occupancy or memory bottleneck,
and independent complete A100 endpoints demonstrate a material benefit without
5090 regression. Do not repeat this algebra-only change or the rejected AO/jet
cache based solely on source-level operation counts.

## References

- Issue #2976.
- `docs/maintainer/performance_engineering.md`.
- `tests/python/test_direct_density_gradient.py`.
- `tests/python/test_direct_density_gradient_cuda.py`.
