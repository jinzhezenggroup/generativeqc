# Decision: bind observed CUDA work to source-owned boundary roles

Status: implemented
Date: 2026-10-10

## Problem

The CUPTI capture for #1629 observes legal final result downloads inside public
warm `execute` intervals. Classifying every such copy as a hot replay regression
would mislabel publication. Conversely, allowing an entire source file or public
endpoint to publish would hide real host iteration/control work. Graph inspection
does not supply these transfer roles or scientific payload dependencies.

## Decision

Extend the method-neutral optional source callback with a separate fixed layout
for execution lifetimes and boundary begin/end records. Native immutable tags
name the source owner, role, site and field. The source artifact, not a shadow
Python registry, is the authority for their names. Invocation and submission
generations identify named source payload instances without retaining handles
or treating address equality as scientific identity.

The HF bucket driver declares result fields/forces as publication; active,
refinement and final-Fock control reads/fences as iteration; input validation
as preparation; and explicit debug/profiling diagnostics as oracle work. Each
wrapper calls its original CUDA operation once with unchanged arguments/order,
including optional zero-byte calls. The observer neither synchronizes extra work
nor changes errors, captures, solver routes or equations.

The optional sidecar preallocates a bounded scalar history and uses CUSTOM2 to
correlate source operations, independent from lifecycle phases (CUSTOM0) and
host graph submissions (CUSTOM1). Runtime/driver API envelopes establish the
actual invoked API but cannot both count as copies. Observed transfer activity
must agree on direction, bytes and qualified multiplicity. Successful source
fence calls and observed CUPTI synchronization records remain distinct metrics.
Queries cannot become verified fences merely because CUPTI uses a sync subtype.

## Rejected alternatives

- Public phase names do not distinguish publication from internal control loops.
- Filenames, byte sizes, addresses and adjacent transfers cannot establish roles
  or host transforms.
- Calling an async copy twice to measure or warm it changes scientific work.
- A zero-byte source call is not a real zero-byte GPU transfer event.
- An API call count is not the same quantity as observed transfer activity or
  physical blocking duration.
- Wrapping CUDA calls must not make them disappear from static boundary scans.

## Invariants

- Source storage is allocated before execution and never grows during capture.
- Observational exceptions/recursion, exhausted generations, scope gaps,
  unknown native tags, bad correlation, loss and incomplete API records fail closed.
- Source callback detachment precedes reading either graph or boundary history.
- Unknown or unannotated production activity remains visible.
- Named field instances are not full tensor/provider dependency identities.
  No D2H/H2D link is inferred, even when bytes, labels or addresses match.
- All dependency/implicit/graph coverage flags remain incomplete until their
  owners supply independently qualified evidence. No zero-round-trip PASS is
  authorized by this slice.

## Evidence

CPU probes compile the actual SDK headers, wrapper/header implementation and
optional collector, but link only CUDA/CUPTI doubles. They verify exactly-once
calls with preserved arguments/statuses, disabled observation, zero bytes,
callback exceptions/recursion, overflow, correlation failures and ID exhaustion.
Protocol tests check separate roles/counters, higher-risk tile/iteration work,
explicit legal oracle/compatibility roles, query rejection, exact activity joins,
unknown/unannotated work and that equal-sized transfers cannot fabricate links.
Static scanner tests preserve loop boundary visibility for qualified wrapper
names and do not resolve unrelated namespaces. These are not GPU qualification.

### Scheduled qualification, 2026-10-10

Slurm job **6923** completed with `ExitCode=0:0` on `node1`, partition `main`,
`gpu:5090:1`, with an explicit ten-minute limit. The controller state was inspected
before its short retention expired; accounting storage is disabled on this cluster.
The retained command log and both numerical histories identify the same job.
The final CPU suite passed **297 tests and 57 subtests**; the explicitly opted-in
GPU resource regression passed **2 tests**.

Qualification uses source tree `98d9dc4f2e5631c0c88d3f03986eb10e0f04841c` over base
commit `f87d51ab616cc9aa774bd344a15e268cef7eaf05`. This is the tree **before this
evidence append**, not a claim that later documentation or implementation changes
were executed. Its full source manifest, independently pinned RHF/UHF contracts,
frozen source archive, raw activity/source/graph records, numerical results,
collector/calibration/native binaries, build configuration and compiler-cache
evidence remain in ignored `.artifacts/issue1629-source-boundaries/`. The frozen
archive SHA-256 is
`2ba9ae4944a0fe729fa7fcd6b91fe3667125fb5e3e3f3fae797ae99e0901feb6`.

The Release `sm_120`, AOT-disabled native build retained these binary identities:

| Artifact | SHA-256 |
| --- | --- |
| Native library | `2882c7c715df489bccc14c7b780f6d0fb2f2307a6c4a8c107a1d6955fba79f33` |
| Combined collector | `d16005a073f9d7e46cf4fa5086bb0990a3c9bcd70b69091a283a99fa8b169bbf` |
| Independent calibration helper | `29386cbbf64acc95b95c9a355e63adb49187425704e339d7a028de040087c305` |

The actual CUDA 12.9 Update 1 / CUPTI API 28 headers and runtime were used. CMake
and generated compiler commands select the verified existing **ccache 4.5.1**;
the retained before/after statistics show two added hits and two added misses,
without clearing the shared cache. No build default is promoted by this evidence.

The controlled wrapper calibration reproduces an independent 16-double result
exactly. Publication has 128 actual D2H bytes, two source/observed fences and one
zero-byte source copy; iteration and tile roles each have 256 actual D2H bytes and
two fences. The zero-byte call creates no invented transfer activity, and the
iteration/tile work remains higher risk instead of being publication-exempted.

Each RHF/UHF history has 10 annotated bucket executions, 204 source operations,
428 boundary records, and intact correlation with no dispatch errors, dropped
records or outstanding operations. Both activity streams and graph inventories
are also intact. The matched public energy/force/moved-geometry histories preserve
all iteration counts. Maximum absolute energy/force errors are respectively
`1.4210854715202004e-14` / `3.497202527569243e-15` for RHF and
`5.684341886080802e-14` / `5.773159728050814e-15` for UHF, inside the independently
pinned `1e-10` / `1e-9` acceptance gates.

Warm endpoint D2H bytes split by the declared source role as follows. Each row
reports **actual observed bytes** and separately verified source fence calls;
the two quantities are not interchangeable.

| Endpoint | Role | RHF D2H bytes | UHF D2H bytes | Fences per method |
| --- | --- | ---: | ---: | ---: |
| Warm energy | iteration | 8 | 8 | 2 |
| Warm energy | publication | 549 | 1005 | 2 |
| Warm forces | publication | 714 | 1170 | 2 |

Thus warm energy is **not publication-only**: its totals of 557/1013 bytes include
control readbacks. Warm force publication totals remain 714/1170 bytes. Each
history still contains **165 unannotated production activity records**: 147 H2D
records and 18 synchronization records. They remain visible rather than being
implicitly classified by endpoint or helper name. Dependency links, graph wait
execution counts and implicit blocking are not qualified; both receipts remain
**INCOMPLETE** with `zero_round_trip_assertion` unset. These observations make no
complete-endpoint performance or repository-wide zero-round-trip claim.

The rebuilt native library also passed the separate #1630 RHF/UHF allocation
receipts in Slurm job **6922**, on the earlier qualification tree
`e34a0519a78eeaa3ac60e0cae10e56ea2901c35b`. Those receipts are retained separately;
their source identity is not retroactively replaced by job 6923's tree.

### Explicit HF input extension

The driver now carries a native payload tag beside every existing static/dynamic
upload descriptor. Topology, geometry, initial warm-state inputs and previous
energy seeds are declared preparation at their actual upload sites. Reordered
shell-pair inputs keep their own field tags. This is not a preparation exemption
for every H2D operation in an enclosing public endpoint; operations in other
owners remain unannotated until those owners provide declarations.

The former HF `copy_to_device` helper deliberately skips empty uploads. The new
`residency_upload` adapter preserves that early return and emits neither a CUDA
call nor a source transfer boundary for an empty helper. Conversely, existing
raw publication copies still call CUDA exactly once even with zero bytes. Both
semantics need to remain distinct; wrapping every helper with an unconditional
copy would change real work and then invent evidence for that changed workload.

Field descriptors retain the same borrowed buffers, extents, conditions and
ordering, adding only scalar tags on the stack. CUDA status mapping remains the
original `cuda_status` mapping. No host transform, numerical storage, fence or
scientific fallback is added. Input seed tags deliberately differ from similarly
named returned results; they still supply no cross-invocation scientific tensor
or producer/transform/consumer identity. The previously qualified tree does not
cover this later implementation extension; it requires separate qualification.

#### Input extension qualification, 2026-10-10

Slurm job **6925** completed with `ExitCode=0:0` on `node1`, `main`,
`gpu:5090:1`, with a ten-minute limit. The controller output, raw records,
numerical histories, independently pinned contracts, build/cache provenance,
actual binaries and frozen source remain in ignored
`.artifacts/issue1629-hf-inputs/`. The source tree is
`b7b69d90d6ec2f14b2e9a9d10bbfd4ac87d3a24d`, before this evidence append. The archive
SHA-256 is `f13a80158be952321fa711c21f44d7b8a0bae52eb364b46efda6b612615a7230`;
the native library SHA-256 is
`381cb8168b0fca05a4652c571fe69b8b86c013cb98e8cb2ca9e537055ebabcad`.
The independently built input calibration helper is
`220dc5728aee490ed1d81b9988f3024fbfa8bb24b753af1e596e451492dc6bd2`, and the
combined collector remains
`d16005a073f9d7e46cf4fa5086bb0990a3c9bcd70b69091a283a99fa8b169bbf`.
The assigned GPU UUID is `GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`; it differs
from the earlier qualification's device, so this is not a paired timing or
performance comparison. Release `sm_120`, AOT-disabled settings and the existing
verified ccache 4.5.1 are retained; no compiler cache is cleared.

The controlled input calibration reproduces an independent 16-double result
exactly, with one observed 128-byte H2D transfer and no source fence. The empty
input helper emits no transfer boundary or CUDA API envelope in its phase;
CPU doubles separately verify zero actual calls, exactly-once nonempty copies,
unchanged streams/statuses, failure propagation and disabled observation.

Each public RHF/UHF history now has **351 source operations and 722 boundary
records**, with zero dropped records, dispatch errors or outstanding operations.
All **147 observed H2D records** join their declared input fields: 6649 total
H2D bytes for RHF and 8041 for UHF. The remaining unannotated production history
is **18 stream synchronization records**, rather than the prior 165 transfer/sync
records. Intact activity and graph stores still do not prove implicit waits,
device-tail graph execution or scientific dependency links.

Warm energy preparation uploads **24 bytes** of `input.previous_energy_seed`;
warm forces upload the same 24-byte seed plus **16 bytes** of
`input.generated_fock_shell_class_mask`, for both methods. The previously
measured source iteration/publication D2H totals remain 8/549 bytes for RHF warm
energy, 8/1005 for UHF warm energy, and 714/1170 for warm force publication.
These are named source fields, not a claimed D2H-to-H2D dependency receipt.

All matched iteration counts agree. Maximum absolute energy error is
`1.4210854715202004e-14` for each method; maximum force errors are
`2.1094237467877974e-15` for RHF and `6.5503158452884236e-15` for UHF, inside the
pinned `1e-10` / `1e-9` gates. The two explicitly opted-in GPU resource regressions
pass. Both #1630 allocation receipts pass on this same source/library identity;
warm owned-device allocation count and requested bytes remain zero, without
claiming zero intercepted host allocations or zero total driver allocations.

The relevant unit/resource scope is validated in its required environments:
**266 local tests plus 57 subtests**, including the real-index freeze checker,
and **37 native resource tests** in Slurm job **6928**. A first local resource run
lacked the native DSO; a broad remote attempt passed 302 tests but could not run
the single Git-dependent freeze check in the non-Git source copy. These raw
prerequisite failures remain retained. The exact Git check passes locally, and
all 37 native resource cases pass remotely with the qualified DSO; neither test
logic nor production behavior is weakened to hide the environmental mismatch.
Ruff, header formatting and whitespace checks pass. The static scanner retains
the qualified upload wrapper as an advisory loop boundary, not a runtime proof.

The remaining sync records correlate to `cudaStreamSynchronize_v3020`: four in
first energy, eight in first forces and six in close. Graph capture/upload,
eigensolver/resource lifetime helpers are concrete source candidates, not
attribution proven merely by matching counts or API names. Their declarations,
the other scientific owners and dependency/wait coverage still need work. Both
residency diagnostics remain **INCOMPLETE**, with the zero-round-trip assertion
unset and no complete-endpoint speedup claim.

## Remaining scope

Source role/payload annotation for other owners, cross-owner scientific dependency
links (including the explicit #369 compatibility provider), graph/device-tail
execution and implicit blocking remain outstanding for full #1629. A bounded
source history is not a repository-wide residency proof or endpoint speedup.

The next #369 owner must be selected by its real consumer, not by the name
`CudaDFSource`. In this source snapshot, `tools/generativeqc_posthf/sources.py`
routes raw three-center host reads through `src/posthf/df_bridge.cu` into NumPy;
`tools/generativeqc_posthf/df.py` then performs the MO transforms, metric whitening
and contractions on the CPU. The native metric-read branch serves already retained
host values, whereas the raw three-center branch generates and downloads a CUDA
tile. The staging acceptance in `tests/python/test_posthf_cuda.py` deliberately
uses this compatibility contract and reports no subsequent H2D in its CPU
provider. It is not, by itself, a D2H-to-H2D round-trip dependency receipt.

By contrast, `tools/generativeqc_response/backends.py` selects a one-way native
generated-source handoff, or an explicitly re-prepared device-resident source,
for `CudaDFJKBackend`. These are not consumers of the CPU provider's transformed
raw tiles. Their separate response tests require raw DF D2H/H2D counters to stay
zero. A future annotation must preserve these distinct contracts and actual
producer/transform/consumer identities. Neither blanket compatibility-tagging
the generated-source class nor confusing ordinary HF metric staging with this
provider supplies the missing cross-owner scientific links. The current usage
contract is `docs/developer/posthf.md`.

## References

The isolated PR integration explicitly registers the observation runtime as
leaf ownership in the SCF structure checker. Eigensolver, resource, graph and
driver owners admit only their named observation adapters, not all of
`runtime/`. Regression gates reject unrelated runtime dependencies and any
scientific dependency acquired by the observation leaves. The CPU graph
lifetime fixture declares the event API required by the shared CUDA adapter;
it still executes only fake CUDA calls and is not a real-device qualification.
These integration tests and ownership registration postdate the historical
source-matched GPU evidence and do not alter that evidence's source identity.

- [Observed CUPTI activity](2026-10-10-cupti-observed-residency-activity.md)
- [Source-owned graph inventory](2026-10-10-source-owned-cuda-graph-inventory.md)
- Current usage: `docs/maintainer/residency_receipts.md`
