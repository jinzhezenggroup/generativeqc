# Decision: observe DF compatibility event fences without inventing lineage

Status: implemented
Date: 2026-10-10

## Problem

The generated CUDA DF host-tile adapter waits on its timing event before
reading generation timing and again after the actual raw-tile D2H copy. These
are existing host-blocking event calls, not stream synchronizations. The source
and CPU DF provider also reported a literal zero for subsequent H2D bytes even
though neither owner observes downstream GPU consumers.

Tracing the current tooling establishes `CudaDFSource` raw reads followed by
NumPy orbital projection and metric whitening in `DFProvider.three_index`,
and NumPy block reconstruction in `DFProvider.get`. The ordinary small MP2
bridge consumes those host blocks on CPU. The dense DF CC oracle also explicitly
selects CPU. These facts do not establish an executed GPU-to-host-to-GPU chain.
The separate native CUDA DF CC source and resident J/K handoff must not be
conflated with this host-transform adapter.

## Decision

Append source operation kind 3 for event synchronization without renumbering
existing tags or changing the 14-word boundary layout. Append the
`posthf_df_source` owner and dedicated generation, raw publication, publication
fence, release and handoff sites. Observe exactly the existing CUDA calls;
preserve event handles, stream, zero-size early return, numerical results,
timing boundaries and original status-checking behavior. In particular the
event calls retain `cuda_resource_check`, including its bad-allocation mapping.

Keep compatibility, lifetime and preparation declarations distinct. A
compatibility label does not exempt work in a selected public replay from
independently reviewed limits. The counter `source_event_fences` is separate
from stream fences and actual CUPTI synchronization activity. Retain the
actual synchronization subtype so a device-side stream wait cannot verify a
declared host-blocking event fence. Queries cannot verify it either.

Schema-v2 work policies must name all counters. Existing frozen v1 policies
have zero event-fence allowance and retain their original source/binary evidence
and bytes. Do not rewrite the independent HF baseline to accept new work.

Report downstream H2D as unknown (`None` / JSON `null`) until a qualified real
consumer observes it. No source payload generation, pointer, byte count, name,
or cache identity establishes a scientific dependency. No dependency IDs are
added and residency remains `INCOMPLETE`.

## Rejected alternatives

Calling the existing waits stream fences would corrupt source/API provenance.
Inserting extra CUDA fences or changing event timing would measure different
work. Reporting unobserved H2D as zero would turn partial owner coverage into
a false zero-round-trip assertion. Wiring a synthetic GPU consumer just to
produce a chain would not qualify the retained real consumer.

## Evidence

CPU-double tests compile the actual observer and wrapper against the pinned
CUDA/CUPTI headers. They verify exactly-once event calls, event identity,
failure status, disabled observation and callback exceptions. Synthetic joins
check API kind, synchronization subtype, separated counters and explicit
event budgets, including frozen-v1 rejection. These are not real-GPU evidence.
Scheduled device qualification and independent numerical checks are recorded
separately when completed.
