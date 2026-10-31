# Decision: qualify a portable force seed independently of value lowering

Status: implemented
Date: 2026-10-30

## Decision and scope

Use existing legal force schedules, subsequently qualified on A100, as the
portable seed. These schedules were not originally autotuned on A100. Preserve
the twenty original portable value kernels through independent consumer IRs;
add the already implemented joint `dddd` Rys-force/tiled-value companion.
Value-only `dddd` Rys is not implemented for a fourth d center: relaxing its
legality check would misrepresent compiler coverage.

The profile remains portable/untuned. SM120 auto still selects its measured
profile. Target validation applies to the actual architecture, including block
and shared-resource limits; no device-name whitelist selects these schedules.
Uncovered classes and optional-resource refusals retain exact bounded fallbacks.
The tuner must retain the measured consumer IR through manifest publication.

For small materialized portable topologies, reuse the existing angular force
queue when total shell quartets cannot fill one persistent worker round per
active generated force class on average. The worker count derives from actual
SM/resource capacity. This is a conservative work heuristic, not screening or
a claim of optimal scheduling. Empty metadata does not authorize the preference.
Larger materialized, bounded, and already tuned profiles retain their existing
routing. Fock selection and scientific force coverage are unchanged.

## Evidence and rejected alternatives

The 21-class seed completed all 64 README method/mode/size combinations on
A100, RTX 5090 and RTX PRO 6000: cold plus three complete energy/force warm
calls from the engine's frozen cold density, one math thread. Each device
passes the same 58/64 independent default-grid gates. HF DF fitting and small
M06-2X default-grid failures remain visible. WB97M-V DF's long-range fitting
contract differs between the two engines.

The seed changes A100 HF Direct 96 from 11.678 to 2.417 seconds, compared with
4.156 seconds before adding `dddd`. PBE0 Direct 96 stays near 6.52 seconds.
On RTX 5090 HF Direct 96 is 2.990 seconds versus the measured auto profile's
2.903 seconds and GPU4PySCF's 3.336 seconds. A100 GPU4PySCF HF remains faster
at 1.466 seconds; several DFT endpoints also remain slower. These static,
default-grid endpoints establish neither universal parity nor all-GPU safety.

Both SM120 devices showed a reproducible small-HF regression with the seed:
PRO 6000 old/new paired warm medians were about 43/60 ms. Reusing the angular
queue restores about 42 ms. Reject using that queue for every materialized
portable topology: it gives up the 12-atom improvement (threefold on A100, smaller on SM120).
The work-dependent crossover preserves the large-work route without a benchmark
size or GPU-model allowlist. Final dispatch qualification is a separate cohort,
not a relabelling of the preceding 64-case binary.

Final work-guard qualification on all three devices includes ten complete
endpoints (HF Direct/DF at four sizes, PBE0 Direct at 3/12 atoms), each with
cold plus three warm calls, and 23 device tests. No warm median regresses more
than 5% against the corrected original portable baseline in this cohort.
Maximum paired force differences are below 2.3e-12 Eh/Bohr. Independent gates
pass 6/10; the four unchanged HF DF fitting failures remain visible. A100 HF96
is 2.384 seconds, RTX 5090 2.980 seconds, and PRO 6000 2.767 seconds. Native
host policy tests cover capacity-dependent crossover, missing metadata and
integer overflow. This limited sample cannot guarantee unmeasured-GPU behavior.

An instrumented broad-angular-fallback trial failed at HF12 with a CUDA runtime
error; its clean endpoint rerun completed. The failed diagnostic is retained
locally and is not counted as a completed timing. No universal trace guarantee
is inferred from the clean run.

An A100 PBE0 DF3 population was noisy. Three interleaved old/new repeats retain
all observations: warm medians were 69.71/56.46, 56.57/102.62, and 56.62/56.90
ms. This does not establish a stable 24% regression or justify dropping outliers.

The compact evidence at `benchmarks/results/ecnu-portable-force-20261030/`
retains all seed populations, work counts, failed gates, paired errors,
producer identities and source-preservation checks. Earlier DF evidence is
recoverable byte-for-byte from its pinned Git snapshot. Full raw observations,
rejected experiments and compiler logs remain in ignored task artifacts.

## Revisit when

Use per-device calibration/autotuning when it has independently qualified
complete-endpoint evidence. Revisit the small-work crossover with a broader
basis/geometry/device set, retaining complete populations and explicit native
fallbacks. An A100-tested starting point is not a reason to freeze batch sizes
or to claim unmeasured architectures are optimal.
