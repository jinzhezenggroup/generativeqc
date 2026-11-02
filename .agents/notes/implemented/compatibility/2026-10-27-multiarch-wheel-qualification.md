# Decision: qualify multi-architecture wheels separately from runtime fixes

Status: implemented in draft; full wheel qualification pending
Date: 2026-10-27

## Problem

The existing wheel only contains an SM 120 device image. Wider image coverage
can remove that binary compatibility limit, but compilation cost must be measured
before enabling it for recurring master builds. That packaging decision must not
hold up independent device-selection and source-build bug fixes.

## Decision

Keep wheel changes in a separate draft PR. It depends on the target/profile and
derivative registry fixes in [#2831](https://github.com/jinzhezenggroup/generativeqc/pull/2831).
Land those fixes first, then rebase this packaging change before running a full
wheel qualification. Cancel the premature wheel run from the combined PR and
defer the split draft's full build until its source dependency is present.

The candidate wheel selects native images for SM 80/86/89/90/100/120 and PTX for
120. Only 120 uses the measured shell profile; other targets use portable
profiles. The compiler snapshot identity includes the architecture list, and
shared caches are retained. Binary coverage is not scientific or performance
qualification on those GPUs.

## Qualification budget and cost gate

The successful single-target master wheel run
[40435225653](https://github.com/jinzhezenggroup/generativeqc/actions/runs/40435225653)
spent 107 minutes 20 seconds in its build step (01:54:01–03:41:21 UTC on
2026-10-27). Its 150-minute build limit does not establish a sufficient budget
for six targets.

This candidate allows 330 minutes for compilation inside a 360-minute job,
reserving 30 minutes for setup, cache retention and required checks. Preserve
the 10-minute cache-save bound, failure propagation and required wheel payload,
linkage and smoke tests. These limits are qualification ceilings, not expected
runtime or evidence that the recurring cost is acceptable.

Before promotion, retain the complete CI build/job timings, cache statistics,
cache size and repaired wheel size for a successful run against the fixed
sources. Compare the observed cost with the single-target baseline; reduce or
rework the image matrix if the cost is unacceptable. A failed or timed-out run
does not qualify the wheel. Repeated expensive attempts need a diagnosis or a
specific configuration change, not just another timeout increase.

## Evidence and limits

The combined change's local wheel/cache/budget tests passed. CUDA 12.9.86 with
verified sccache 0.16.0 compiled a TensorIR example separately for all six
targets, and a stub-arithmetic derivative-header probe produced all six cubins.
These probes do not qualify a complete native build or wheel distribution.

The additional six-target compilation of `direct_bounded_fallback.cu` reached
its 600-second limit (exit 124) without a finished object. No full local wheel
build or non-5090 runtime/performance qualification has been completed. The
cancelled combined-PR run is not passing wheel evidence. Existing target-specific
source builds remain available while packaging cost is evaluated.

## Revisit when

A complete wheel CI receipt demonstrates acceptable time, cache and artifact
cost, or deployment requirements justify changing the architecture/PTX matrix.
Do not remove portable fallback admission or promote 5090 schedules to other
devices merely to reduce compilation work.

## Prerequisite integration, 2026-10-31

PR #2831 merged as `31bfe905ffbee77fe77cfeb8313a880a56041ded` on
2026-10-27. The packaging branch now includes master
`0388bcfbb42834e20fad577ac237d573772e9266` through a history-preserving merge;
the wheel architecture/profile selection and all qualification gates remain
unchanged. The 70 wheel snapshot/cache/budget checks pass after integration.

The earlier wheel job was explicitly cancelled while starting its container,
before compilation, so it provides neither build-failure nor timing evidence.
A full wheel run against the integrated source is now appropriate. Its successful
build/job timing, cache statistics/size and repaired artifact size are still
required before this packaging PR is ready.

## Superseding matrix decision

The owner-selected 80/90/120 native matrix and compute-80 PTX supersedes the
six-native-target policy for new wheel builds; historical results above remain
bound to their original inputs. See [the reduced-matrix decision](2026-11-01-three-native-wheel-targets.md).
