# Decision: qualify existing mixer receipts through the complete endpoint

Status: implemented; restricted diagnostic qualification retained
Date: 2026-10-14

## Problem

PR #2157 delivered default-off mixer instrumentation inside the real SCC graph.
Its bootstrap checks complete energy/host-force equality and independent ring
work, but aggregate stdout alone does not retain per-invocation clock intervals,
complete outputs or alternating instrumentation samples at moved geometries.
Whole-graph NCU statistics cannot attribute this work to modified Broyden.

## Decision

Extend the existing bootstrap with an explicitly selected bounded journal mode.
It reuses `enable_mixer_diagnostics`, synchronous restricted production execution
and settled readback. Separate OFF/ON caches execute alternating cold, cache-warm
fresh SCC, moved and moved-replay calls. Independent tblite input/reference
preparation stays outside production. A small Python validator enumerates history
slots and component visits chronologically without using the generated history
window implementation. It retains malformed, missing and negative evidence.

## Rejected alternatives

- A new public diagnostic ABI or production timer: unnecessary expansion of this
  readout slice, and a conflict risk at common runtime/public schema owners.
- A new CMake target: the existing bootstrap is already available, while another
  active PR owns shared CMake test registration.
- Nominal GPU clock conversion or cross-SM cycle sums: local `clock64` intervals
  do not supply a complete SCC wall-time denominator.
- Provider/GEMV optimization or a relaxed 768-AO gate: neither follows from
  delivering measurements, and both exceed this diagnostic slice.

## Invariants

Production arithmetic, SCC convergence/mixer/history/ring/omega precision and
default routing remain unchanged. Journal writes follow endpoint settlement and
are bounded to 64 MiB separately from the runtime's existing receipt allocation
bound. No host science is introduced into the endpoint. Original independent
`5e-7` E/F gates and the unresolved historical 768-AO FAIL are retained.

Each stage share uses the measured stages of one invocation on one CTA; it is
not a fraction of the full mixer or SCC. Complete endpoint timing includes ON
readback and charges cold constructor time. Missing SCC timing, graph submission
counts and allocator peaks are explicitly unavailable, not zero.

## Evidence

Local report behavior tests cover chronology, history fill/wrap, malformed
dimensions/visits, call/plan mismatch, actual arm order, partial/empty/truncated
receipts, unavailable clocks, independent numerical failure and E/F disturbance.
These synthetic inputs are host behavior checks, not NVIDIA qualification.
The correct-LF Linux baseline passed 28 tests with one expected real-device skip.
The final readout passed 68 focused host tests with one such skip, including 40
report/lifecycle behavior cases. Release sm90 native library/probe compilation and
linking completed through verified sccache 0.16.0 and CUDA 12.9.86. The independent
CPU preparation resource uses its own environment/cache/output directory; oracle
preparation ran under Python 3.10.12 and resumed code generation under 3.12.15.
The subsequent H200 run retained 216 controlled complete E+host-force samples
and six lifecycle calls. Independent unchanged tblite gates, OFF/ON E/F/work
parity and failure/recovery validation passed; canonical preflight/bootstrap/
probe/reducer exits were zero. Maximum E/F errors were `1.963371687452309e-9` /
`3.960852935913639e-8` against the `5e-7` gates. No speedup or 768-AO resolution
is inferred from this restricted diagnostic qualification.

Two retained preparation failures constrain reproducibility. First, ordinary
Windows `git archive` applied `core.autocrlf` conversion to exported native source:
the atomics header's committed LF SHA-256 was `d51ed8756d358fb6d206709875b306fdc7f6e6fe5dea7a8ebf7dd670ad83e174`,
while the exported CRLF SHA-256 was `670134cbc7983ab706d79c287a12ad6ed93ac49f8459556bca46201e8985bf68`.
The initial 27-pass/one-skip/one-failure log and bad export remain retained. Use
`git -c core.autocrlf=false archive`, then verify every exported blob against the
Git tree. This checked all 10,756 baseline and 10,761 initial candidate blobs;
it did not modify production source or the vendor manifest to bypass the failure.
Second, Python 3.10 code generation stopped at an unrelated `typing.Self` import.
The failure log remains retained; selecting a task-owned Python 3.12 environment
repaired the build environment without changing MP2/codegen source or clearing
the compiler cache.

The library's production inputs were frozen at base `2946eed598c6ea54c0b493c60f695c0efad7fc50`
/ initial candidate tree `6315087045b87a207b763822f47a789cb5b92f11`. Subsequent deltas
added only test readout/reporting and documentation, including actual CUDA device
identity and raw lifecycle calls. Native library SHA-256:
`6fcef302226d8c72da0d5b02b53ad633fd0833475395948a62cf526f8ab25e31`.
Final linked readout SHA-256:
`28aec2c69864d275b26031b7535164b4849f94ef030c5ce3cf19d65dbbc0f671`.

Complete raw vectors/receipts and source/resource/command identities are retained
in [the compact qualification publication](../../../../benchmarks/results/gfn2-mixer-endpoint-20261014/README.md).
Per-invocation full-history history-stage shares were about 79.33%/86.48% of the
measured stages at 48/192 AOs, while actual warm paired ON overhead medians were
about +0.19%/+0.07%. These local cycle ratios do not supply a mixer/SCC denominator.
Requested fair HIGH/4 became platform NORMAL/20 for an unconfirmed reason; no
resource was restarted or relabeled. Cache statistics show counter discontinuity,
so no whole-build hit/miss attribution is made. Parent #1882/#560 remain open.
The retained build log explicitly reports `sccache: warning: The server looks like
it shut down unexpectedly, compiling locally instead`; therefore launcher use
does not establish uninterrupted cache service for every compilation.

Independent replay found a consumer-only type-validation gap in lifecycle rows:
Python equality accepted float call IDs and bool/int status/snapshot substitutions.
Behavioral red cases reproduced it, then both readout paths were routed through
one strict metadata validator. Replaying the unchanged real raw journals retained
the same semantic report. Native/probe/scientific inputs were unchanged, so no
duplicate GPU experiment was submitted.

## Consequences and revisit

The subsequent host-consumer review reproduced four false qualifications on
unchanged real journals: zero executed timings, terminal SCC science substituted
for pre-reset rejection, and completion flipped on each controlled NaN failure.
The consumer now requires phase-appropriate timing and the observed controlled
failure status/sentinel/completion combinations. Unknown device-restoration
failure domains remain unqualified: mixer completion belongs to the transaction
before `finish()` may replace its status while restoring the caller's device.
A generic equivalence between completion and final API success was rejected.

The independent work oracle still enumerates chronological occupied slots, but
counts component Cartesian products by cardinality. Dimension-sized counting
could perform billions of host increments within an admitted 64 MiB journal.
Small explicit component enumeration checks the independent counts, while a
bounded subprocess covers a maximum 300-receipt/100000-element report and rejects
a corrupted final Gram count. This repair does not change production math, ABI,
provider selection, frozen native inputs, or the original GPU measurements.
Final consumer identity and replay evidence are appended in publication provenance
without relabeling earlier consumer repairs or the 45-case historical record.

After PR #2470 merged, a source observer identified three additional review
threads created before that merge. The delivery SHA, CI, and delivered blobs were
real, but the all-findings-closed completion statement was withdrawn publicly:
the queue watcher had checked CI without rechecking later review threads. The
bounded follow-up rejects each missing executed stage interval, default owner
identity on available failed snapshots, and fallback/submission contradictions.
The original 84-case record remains historical rather than being relabeled.

Only the first measured residual stage runs on the initial invocation; the other
three measured intervals must remain zero there. Available failed snapshots copy
the concrete candidate plan and cache device, while pre-reset rejection has no
snapshot and retains its default identity. Failed calls in the authentic journal
can have family 1 with submission true, so neither graph submission nor mixer
transaction completion is equated to final API success. The raw journals and GPU
build are unchanged. Current source-bound host validation is appended separately;
every later substantive thread must be checked while the follow-up waits in queue,
even after an exact-head approval has been recorded.

Raw endpoint data can be independently revalidated and diagnostic overhead stays
visible. Restricted single-system qualification leaves ragged/unrestricted
semantics and profitable provider schedules open. Revisit the timing boundary
only if an independently measured comparable SCC denominator becomes available;
revisit a provider schedule only after source-matched complete endpoints justify
it. See [the runnable procedure](../../../../docs/maintainer/gfn2_mixer_endpoint_qualification.md).

Refs: #1882, #560, PR #2157.
