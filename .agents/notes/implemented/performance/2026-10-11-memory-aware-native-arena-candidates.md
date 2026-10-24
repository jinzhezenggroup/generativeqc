# Decision: conservative native memory candidates before default promotion

Status: implemented (qualified cross-size CMake default; serial scratch default-off)
Date: 2026-10-11

## Problem

The requested baseline is upstream master, fetched at `637466c62` (2026-10-10).
It already owns equal-symbolic-capacity arenas and bounded DF Lambda matrix,
replay and audit storage. Extending these mechanisms must preserve FP64 equations,
operation order, optional-cache value and complete-endpoint policy.

## Decision

Extract `common.storage.analyze_lifetimes` from the existing concrete analysis.
Concrete coloring and native symbolic planning share alias, opaque-effect,
external-owner and donation legality. Materialized native FP64 supplies unit
capacities solely for lifetime analysis, never for symbolic capacity assignment.
The symbolic planner does not need quadratic interference diagnostics.

Keep equal-product coloring for standalone/fallback and retained plans. The cross-size planner merges
whole slots only when every pair of owner intervals is disjoint. Its checked
runtime maximum cannot exceed the old sum for any shape. Storage identity
advances to v2, but scientific equations are unchanged. Retained plans keep
their established coloring. Candidate DF matrix-core capacity queries retain the
frontier scratch floor: reducing uncached scratch must not refuse the useful
retained path. Both requirements remain bounded by the old full matrix arena.

Add a separate default-off Lambda owner candidate. Expanded/scalar, staged/matrix,
fresh replay and independent audit share scratch on the existing single stream.
Persistent cuts, tau, seeds, sums and invariant frontier remain separate. Compiler
capacity queries are unchanged; the native owner supplies the serial consumption
proof. The existing full DF force benchmark can select this candidate.

## Rejected alternatives

- Representative numeric bytes confuse unrelated runtime symbols.
- Unconditionally growing greedy slots can exceed the old total for other runtime
  shapes. Whole-slot coalescing has a simple shape-independent bound.
- Appending shared scratch after persistent storage can add trailing alignment
  padding and increase a single-arena layout. Reserve an aligned prefix after
  inputs instead, then shift the persistent suffix by rounded scratch bytes.
- Changing retained coloring can refuse a useful cache; qualify that separately.
- Phase names do not prove serial consumption across independently owned streams,
  SCF, triples or orbital response.
- Static capacity diagnostics do not justify default promotion or speedup claims.

## Invariants

- Producers never overlap live inputs; borrowed outputs survive return.
- Runtime maximum terms are checked for overflow, including scalar size types.
- Retained storage is exclusive across epochs.
- CPU admission and CUDA pointer assignment use the same candidate selection.
- Existing copy-elision proofs and scientific/kernel work are unchanged.
- Admission, allocation attempts and shifted pointers describe the same layout
  after every retention/audit/provider/cut fallback.
- No new fence, CPU scientific work, Q traversal or precision change is introduced.

## Evidence and qualification boundary

Tests cover concrete-storage compatibility, unequal/inverted runtime dimensions,
all production graph lifetimes and CPU/CUDA plan agreement, retained-frontier
compatibility and runtime-shaped CPU Lambda numerical oracles. Native DF planner
tests compile the actual owner code with query stubs and check optional-stage
combinations, alignment tails, exact admission and rollback snapshots. These
stubs are storage tests, not numerical or real-GPU evidence.

Focused validation finished with 587 passing tests across the storage, candidate,
frontier, generation and native CPU-owner suites. Sixteen real-device cases and
two unavailable Clang configurations were skipped. Compiler structure checked
513 modules with zero dependency errors; Ruff and the changed compiler's type
checks passed. The three modified CUDA translation units compiled for `sm_120`
using verified ccache and CUDA 12.9.1. Compilation is not real-device acceptance.
Local reproduction outputs are under ignored `.cache/memory-planning/`.

At symbolic `o=9, v=221, q=16`, static DF iteration arena capacity changes from
416,916,232 to 412,810,424 bytes (53 to 19 slots). DF Lambda matrix-core capacity
changes from 712,793,032 to 706,697,880 bytes (65 to 47 slots); its retained scratch
floor is 578,431,456 bytes and remains admitted. These are compiler capacity
requests in this worktree, not complete-endpoint peaks, measured traffic or
physical VRAM savings. Large slot-count reductions alone need not imply large
byte reductions, which is another reason not to promote from coloring alone.

Real-GPU tests reuse the native DF Lambda probe and compare bitwise outputs,
independent residuals, action counts, launches, transfers, synchronizations and
allocation bounds. They require a rebuilt library and finite Slurm allocation.

The first local `sinfo` query exposed only the drained node3 controller. This
was not a fleet-wide availability query: n1/n2/n4/n5 have separate Slurm
controllers and must be queried over their SSH host aliases. Follow-up queries
found available GPUs on all four allowed nodes. n2 has a kernel/userspace
NVIDIA driver mismatch (595.91.07 versus 595.99.02), so only host compilation
runs there. Finite Slurm admission checks passed on n1 (RTX 5090, driver
580.95.05) and n5 (RTX 5090, driver 570.86.16); CUDA 12.9.1 device allocation
also passed on n5. These are environment checks, not scientific qualification.

CMake now has an explicit default-off paired generation switch, covering all
shared RCCSD/DF/RHF native emitters. Seventeen CLI/build-plumbing cases passed;
the refreshed host safety suite passed 492 cases with 18 unavailable-device or
compiler configurations skipped. The complete force benchmark can additionally
run cold, same-process warm and displaced-geometry moved-warm calls. All three
rebuild scientific owners; this is not prepared force replay.

### Initial master-snapshot device evidence

The following measurements use the initial master snapshot `637466c624c4c4c4dd4eb7c511a2c88675744325`,
with cross-size arena and serial scratch selected together. They do not qualify
either candidate on a later source revision. All device work used finite Slurm
jobs. The nodes have independent controllers: querying the local controller does
not describe the other nodes. n1 and n5 supplied RTX 5090 devices with drivers
580.95.05 and 570.86.16 respectively. n2 supplied host compilation only because
its kernel driver and userspace libraries did not match; n3 was not used.

Both baseline and candidate passed 28 native numerical/response/failure cases on
each of n1 and n5. Candidate memcheck and initcheck each passed 16 cases with zero
reported errors on n1. Three interleaved water24 complete E+F lifecycle pairs on
each node passed the original independent numerical gates and a 2% timing-noise
screen for cold, warm and moved-warm. This screen is not permission to retain a
known systematic slowdown. Whole-endpoint force assembly is not bitwise
deterministic even between baseline calls: baseline cold/warm differed by
`1.33e-15`; differential forces therefore use `atol=1e-10, rtol=0` in addition to
the independent energy, finite-difference and residual gates. The small native
Lambda bit-pattern gate remains unchanged.

One complete 230-AO ethane lifecycle pair passed numerical gates, including
maximum differential force `6.42e-14`. Baseline total-energy error was
`2.188e-12`, and independent directional finite-difference errors were
`7.216e-9` and `3.300e-9`. The cold calls took 303.552079/303.908327 seconds;
warm calls took 394.162000/311.389611 seconds; moved-warm calls took
394.109046/311.491761 seconds (baseline/candidate). One pair is not statistical
complete-endpoint performance qualification.

The matched cold CUPTI allocation trace observed requested-live-device peaks of
15,571,039,334/9,450,425,446 bytes and 337,393 kernel launches for each variant.
Observed H2D, D2H and D2D counts and bytes matched. Complete static kernel
register, stack, shared and local-resource dumps matched after normalizing only
anonymous-namespace hashes. This is not a hardware HBM-traffic or dynamic-spill
measurement. The existing ledger journal covers ledger-bound owners, not raw CC
CUDA allocations or opaque context/provider retention; it cannot be substituted
for the CUPTI trace. A complete cold one-second NVML child-process residency
sample observed maxima of 31,506/27,522 MiB. These are sampled maxima, not a
continuous allocator high-water or the global cold/warm/moved-warm peak.

### Why combined default promotion is withheld

Warm resource admission changed. The baseline retained its 396,517,056-byte
invariant frontier and used scalar audit/replay; the candidate admitted matrix
audit/replay but refused that frontier. Lambda owned requests increased from
7,286,698,944 to 9,025,428,776 bytes. Accounted packing output increased from
1,328,460,014,096 to 1,879,245,622,048 bytes, while total accounted work decreased
and generated-kernel count fell. Faster warm timing does not turn this into a
fixed-work, pointer-only storage optimization. It also does not justify deleting
a useful invariant cache.

The original large collector stopped on an unnecessarily bitwise force
comparison after writing both lifecycles; this is a collector failure, not a
solver failure. Its records remain retained. Partial lifecycle NVML samples had
different coverage (baseline displaced interval versus candidate warm/displaced
intervals), so they establish neither a global VRAM increase nor reduction.
Together these admission and observation gaps prohibit combined default
promotion. Serial scratch remains default-off pending a cache/provider-aware
policy and full-lifecycle qualification.

Cross-size-only qualification is separate: serial scratch stays disabled in both
variants, so the comparator requires identical semantic work, cache/provider
selections and nonincreasing numeric requests. It samples the exact child PID
through every lifecycle, rather than relying on free-memory before/after
differences. After upstream advanced during qualification, the working source
was refreshed to master `a70ebfc61`; earlier receipts remain explicitly scoped
to their original source. CUDA libraries and numerical/performance gates must be
refreshed before changing any default on the updated source.

On the initial snapshot, one full cross-size-only lifecycle pair on each node
preserved semantic counts, frontier retention and audit/replay selection.
Lambda requested bytes fell from 15,542,559,936 to 15,401,062,848 in cold calls
and from 7,286,698,944 to 7,183,750,848 in warm/displaced calls. Complete lifecycle
one-second sampled NVML maxima were 31,506/31,370 MiB on n1 and
31,504/31,368 MiB on n5. These are complete coverage of the three-call processes,
not continuous high-water measurements. n1's remaining old-source repetitions
were canceled after the first passing pair because the updated-source run
supersedes them; no statistical promotion claim uses that canceled run.

The refreshed master host suite passed 683 cases with 18 device/compiler skips;
an additional delayed-copy Lambda host-borrow matrix covers sharing on and off.
Compiler structure checked 514 modules with zero errors. Refreshed CUDA libraries
were built on n2 with verified ccache 4.5.1, GCC 14.3.0 and CUDA 12.9.86, Release
and `120-real`; ordinary shell AOT was enabled, while unrelated stationary CPU
and CUDA force AOT were disabled. Both builds use the same updated source and
explicit OFF/ON CMake selections. Their SHA-256 values are
`66843ce98be450c75bbedef0d1d2a9581209116e9b7808ba7debcb4f77225c73` (baseline) and
`323aeff5fddbe57abed44e09cb2202bb2399ebbc14dd8ce65bdb9e89e1388b7d` (candidate).
The benchmark and its header ABI closure did not change in the upstream refresh;
each node reuses its previously linked, platform-compatible driver against the
new library. No executable is transferred from newer-glibc n1 to older-glibc n5.

Remote runtime compatibility is part of receipt provenance: n5 requires the
build toolchain's libstdc++/libgcc and a locally linked benchmark because its
glibc is older than n1's. Preserve the library SONAME symlink, and do not alter
Slurm device visibility to work around runtime problems. PBE0 48/96-atom
optimization and automatic global cross-method planning are outside this slice.

### Qualified cross-size-only promotion

Updated-source baseline and candidate passed 28 native cases each on both nodes;
n1 candidate memcheck/initcheck each passed 16 cases with zero errors. Three
interleaved 24-AO water lifecycle pairs passed on each node. n1 also passed three
230-AO ethane lifecycle pairs; n5 passed one independent hardware/driver pair.
All comparisons preserved FP64 numerical gates, semantic work, cache retention,
audit/replay selection and nonincreasing numeric requests. The n1 large-case
median candidate/baseline ratios were 0.9997020203 cold, 0.9998520047 warm and
1.0004993585 moved-warm; n5's one-pair ratios were 0.9998110704, 0.9996481501 and
1.0005277257. These pass the 2% noise screen, not a speedup or formal statistical
noninferiority claim. Timing runs are unprofiled; Nsight runs use a separate job
and GPU and do not supply timing acceptance.

Every complete large lifecycle observed a lower sampled NVML process maximum:
31,506/31,370 MiB on n1 and 31,504/31,368 MiB on n5. Small-case sampled maxima
were unchanged. Matched updated-source cold CUPTI traces observed requested-live
device peaks of 15,571,039,334/15,429,542,246 bytes, with identical 418 allocations,
337,393 launches, memcpy counts/bytes and traced register/shared/local resource
histograms. Traced local-memory reservation per thread was zero in both variants;
this is not a dynamic spill-transaction measurement. Static resource dumps match
after anonymous-namespace hash normalization. Both observed pool-event reserved
maxima were 704,643,072 bytes; opaque context/provider/Graph retention remains a
separate, incompletely attributed scope. Actual HBM traffic is not measured.

Enable only `GENERATIVEQC_NATIVE_CROSS_SIZE_ARENA` by default in CMake. Its explicit
OFF fallback and standalone CLI baseline remain available. This default change
does not alter the qualified explicit-ON generated code. The serial-scratch
runtime and endpoint selectors remain OFF: its earlier warm admission/cache
trade-off is not waived by cross-size-only qualification. No general cross-method
peak or universal speedup claim follows from these two inputs.

Compact source-bound raw receipts, scripts, library hashes, tests and profiler
summaries are retained in `benchmarks/results/native-memory-arena-20261011/`.
Large local profiler binaries retain hashes rather than being committed or
published through a Release. The measured patch predates the default flip and
reconstructs explicit OFF/ON comparison builds from the full master base.

## Revisit when

Whole-lifecycle physical-memory and cache/provider-preservation gates are met
for the combined candidate. Broader reuse needs an explicit phase dependency graph
with persistent-output and asynchronous-reader lifetimes. More aggressive
per-value coloring needs a runtime-shape capacity fallback rather than assuming
fewer slots always means fewer bytes.

## References

- `docs/developer/iteration_reuse.md`
- `.agents/notes/implemented/architecture/2026-10-10-compiler-symbolic-native-arena.md`
- `tests/python/test_tensor_native_arena.py`
- `tests/python/test_rccsd_arena_liveness.py`
- `tests/python/test_df_lambda_serial_scratch.py`
- `benchmarks/df_ccsdt_force_endpoint.cpp`
