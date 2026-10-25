# Decision: preserve complete and partial Direct WB97M-V returns

Status: proposed publication; task adapters executed
Date: 2026-10-25

## Problem

Issue #2732 needs a bounded H100 replay of the repaired Direct WB97M-V
water-48/96 experiment. Measured scientific source is `cbd4d807…`, not archive
commit `7522994e…` or current master. Its original driver performs convergence
and optional metadata operations before the retain callback. A dummy replay of
the unchanged archived function loses an already returned E/F pair when either
operation fails. Using the callback alone therefore cannot satisfy raw-first
retention.

The scientific source and existing SDK must remain immutable. Editing the
original Slurm CLI, fabricating an allocation, rebuilding the SDK or switching
to a newer production driver would change the experiment's source boundary.

## Proposed adapter

Derive the two archived endpoint functions in the task namespace, adding a
post-timer return observer and an exception observer. Before each original
cold/warm timer, assign an unavailable sentinel to phase-local E/F/item values.
The original assignments naturally establish which values actually returned.
This prevents pairing a newly returned warm energy with the previous warm
force vector, including kernel, gradient and synchronization failures. The
first prototype used flags inside the timed loop; that checkpoint is retained
locally and superseded by the sentinel proposal.

Removing the observer additions must recover the original statement AST.
Original calculator/owner/seed, convergence, grids, response and timer statements
are retained. IEEE bytes and their index are exclusively written and fsynced
before metadata, finite checks or oracle comparison. Partial E or E/F has its
own partial schema; absent timing or force values are not synthesized.

The adapter is separately identified from the source driver. Python minor
versions can affect generated wrapper text: Windows HOST derivation hashes
must not automatically qualify Python 3.11 execution. A same-binary Linux HOST
receipt must supply the execution derivation before a scientific release.

## Runtime admission and lifecycle

A stdlib-only isolated bootstrap authenticates the full retained environment,
source and DSO before site activation. It admits the original 40,661-file
manifest plus exactly one independently proved cache derivative, preserving the
original manifest and failure history. No cache cleanup or broad pyc exemption
is permitted. The scientific source inventory is independently matched to all
12,130 canonical Git LF source files; the original measured 5,852-file manifest
remains unavailable.

Task-local copies of the retained identity and process-guard utilities preserve
their source lineage without inheriting the sibling packet's execution or
release authority. Each actual guard now also proves visibility of a known-live
owned child before acknowledging its command. `/proc` and pidfd availability
alone are insufficient. A live child missing from the descendant view is a
CONTROL failure; the scientific command must not start. A prematurely exited
control is distinguished from evidence of hidden live children.

The separate Linux HOST controls retain their original logs, lifecycle records,
PID identities, bytes, modes and hashes. Windows mocked visibility tests cannot
substitute for these controls. Actual target-Job guards repeat the bounded
zero-science visibility check because CPU and GPU executors can expose different
process views.

## Evidence and limits

The HOST reader recomputes both selected independent reference grid gates,
the original water-3 CPU/GPU quarter-margin check, and all 24 selected historical
samples. It executes no oracle or scientific solver. Reference geometry, basis,
full moving-grid response and per-engine convergence differences remain explicit.

Existing HF build/profile receipts independently establish portable sm90 HOST
provenance, not WB device qualification. Missing work, cache-hit or operator
precision telemetry remains unavailable. Fock stage, audit, initialization and
skipped-work counters must not be summed into an invented primary count.

At the initial HOST checkpoint this was not a completed benchmark or a resource
grant. The first
proposed diagnostic pilot is three fresh serial processes / 12 complete E/F,
with no automatic retries. Larger cells and formal repetitions need their own
bounded acceptance and release. Source, HOST qualification, device/numerical
acceptance and publication remain separate until their actual evidence exists.

## Executed outcomes and publication decision

The subsequent five finite executions retain 98 actual complete E/F returns.
The first diagnostic returned two numerically qualified pairs but failed a
validator that conflated the raw loop ordinal (0,1,2,3) with the unchanged public
warm metadata ordinal (0,0,1,2). Ten calls were NOT_RUN. The fix validates both
canonical integer namespaces and phase separately without modifying original
metadata, scientific statements or timers. The failed worker/guard statuses and
raw values remain evidence; the corrected reader does not retroactively turn
that control failure into a successful batch.

Two successful diagnostic cohorts each provide only one independent process per
arm. Separate water48 and water96 formal cohorts provide three fresh processes
per arm, each cold plus three warms: 72 formal complete E/F returns from 18
processes. All original gates pass, but physical GPUs and cache cohorts differ
between sizes. Cold process medians and the median/range of process warm medians
remain separate; pooling nine warm calls or combining the two sizes would
misrepresent independence. GPU4PySCF was faster in these cohorts; no repeat was
requested to chase a preferred outcome. Missing force-operator precision, peak
memory, GPU4PySCF fallback/work and billing evidence remains unavailable.

The source/environment and original execution layouts remain immutable. The
current publication format stores identical original bytes once by SHA, with
explicit per-job path/size/hash/POSIX-mode inventories and every original
nonmember envelope field. The 1,101 selected terminal members are byte-restorable.
Only three first-pilot generated native binary contents are inventory-only;
their full original local/remote captures remain retained. Public reconstruction
or download of those contents is explicitly unavailable. Initial HOST files and
the failed V2 receipt remain reconstructible as a separate historical namespace.

This storage choice fits the existing single-file, complete-change and aggregate
retention bounds without raising them, relabelling failures or newly nesting
compressed run dumps. The standard-library reader checks bounded XZ decoding,
original full IEEE E/F frames, independent arrays, settings, guard birth/ACK/argv,
archived allocation and source/environment/device witnesses. It never executes
archived source or loads an SDK. Restore requires a new directory and exclusive
creation; mode witnesses do not claim Windows permission equivalence.

Rejected alternatives are staging every duplicate terminal capsule (over the
change budget), dropping negative/control/source records, normalizing original
CRLF byte evidence, clearing caches, increasing retention caps, or treating
private aliases as public availability. The six artifact bytes and exact storage
metadata scope were independently reviewed as
`root-2732-two-size-lossless-storage-20261025`. Storage classification and passing
bounded E/F gates do not close #2724 or substitute for latest-head independent
review, CI, protected merge and delivered-blob verification.

## References

- `benchmarks/results/h100-direct-wb97mv-20261024/`
- Issues #2732 and #2724; delivered protocol #2742 and environment #2750.
