# Decision: retain distinct Becke work and intrusive phase observations

Status: implemented
Date: 2026-10-06
Related: #1894, #1830, #1950; draft PR #1996

## Problem

Requested primitive selection and a smaller reverse-pair payload do not establish
less total work, lower hardware traffic, or a faster complete energy/force
endpoint. Previous native-owner assertions also did not retain maximum errors.
The ordinary and nonlocal composite consumers have two independent retained
owners, so reporting only one owner hides work and concurrent storage.

## Decision

The shared native owner exports optional version-one ABIs for 17 counters and
seven CUDA-event intervals: point-center distance, pair primal/switch-log, atom
log reduction, normalization, reverse derivative, ordered atom gather, and point
motion/source publication. Counts describe launched dense domains. They do not
describe accepted completed work after a failed force, conditional log reads, or
the number of scalar transcendental evaluations.

The traffic model counts distinct logical pair-panel values and additional
cached center-direction values. It does not count executed loads or hardware
transactions. In particular, per pair, ordinary reverse writes and unique gather
values require 32 + 64 logical bytes. The coefficient route requires 16 + 32
bytes, but its two incident gathers additionally read 48 cached-direction bytes.
These modeled totals are equal. Claiming half the total traffic from the smaller
reverse panel alone would omit the direction reads; actual transaction/cache
behavior remains unmeasured.

Profiling records eight boundaries after AO/XC seed production on the same
borrowed stream. Event reuse requires synchronizing the last event before the
next tile. Each profiled tile therefore adds eight records and one explicit
fence. Profiling is intentionally intrusive; disabled profiling records no events
and adds no fence. Event creation is transactional, and destruction retains the
existing device/context discipline. Neither these event durations nor durations
from concurrent owners are added to clean host-wall endpoint components.

Runtime deltas difference counters and nested event durations per force.
Composite telemetry preserves the semilocal and nonlocal owner deltas separately.
The benchmark normalizer retains requested/selected routes, counters, logical
bytes, and profiling scope. Unsupported, disabled, and unsplit generic timings
are null, even when the native ABI returns zero-filled arrays. Capacity checks
bind the changed owner/launch blocks and the new ABI/profile functions; mutation
tests retain fail-closed checks rather than weakening existing admission.

## Evidence

`benchmarks/results/becke-native-phases-20261006/publication.json` retains the
frozen dirty-source patch, compiler/native input checksums, compact receipts, all
192 route observations, and a finite Slurm reproduction recipe.

n1 job 6148 uses finite `main/gpu:5090:1`, preserves assigned device visibility,
passes 96 GPU tests with no skips, and passes one profiled external-source
primitive case under each of four sanitizers. Racecheck reports zero hazards and
warnings. The 96 phased observations cover 48 and 96 atoms, three geometries, 256
plus one-point tails, explicit/implicit owners, AO selections, external seeds,
primitive on/off, and profiling on/off. The 48 profiled phased observations retain
seven actual positive event intervals and their event/fence counts. Recorded
maximum errors relative to the generic native route are zero for this cohort;
this is not an independent physical RKS/UKS force oracle.

The mathematical generated `.cu` is byte-identical to job 6139. The included
native header changes, and the compiled library identity differs. The full
compiler/header input manifest, not the `.cu` checksum alone, binds the new
cohort. Ccache is invoked and before/after statistics are retained. Concurrent
builds share that cache; their aggregate deltas cannot attribute individual
hits/misses to this owner compilation.

## Limits and rejected interpretations

The publication decision remains inconclusive for full numerical acceptance.
Synthetic owner gates and intrusive intervals do not replace independent
physical RKS/UKS/moved forces or paired complete 48/96-atom PBE0 energy/force
endpoints with actual solver histories. Hardware traffic is not measured, and
no privileged profiler workaround is introduced. The experiment remains opt-in;
default selection stays off and #1894 remains open. The separate physical gate
uses 36-atom closed-shell and 35-atom doublet hydrogen clusters inside the existing
greater-than-32 phased-cache domain. Lowering the automatic 48-atom policy
threshold would not bypass the independent compiler/native cache guards and
cannot select this primitive on H2/H3. Neither guard nor production threshold is
changed. Host eligibility assertions are not executed GPU selection evidence.

The profiling allocation-fault probe must mock all twelve native events: four
existing source events and eight Becke phase boundaries. Its old four-event mock
failed compilation in PR #1996's core-b CI cohort after phase telemetry was added.
The corrected ccache-backed host probe injects every event-creation failure,
checks that no partial event handles are published, retries successfully, and
checks idempotent activation. This probe does not require or simulate physical
GPU numerical acceptance.

Do not reinterpret these observations as promotion of the unchanged indexed or
dense-coefficient losing schedules, or conflate them with #1893 AO/XC gains.
