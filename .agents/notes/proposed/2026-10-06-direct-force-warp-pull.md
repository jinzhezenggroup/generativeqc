# Experiment: bounded class-bucket warp pull for Direct forces

Status: proposed production qualification; experimental implementation present
Date: 2026-10-06
Owner: #1892 P0-B; related evidence-only PR #1978

## Problem and scope

The retained bounded consumer already assigns one quartet to a warp. Its static
strided drain can nevertheless leave short-task warps waiting at packet
retirement while another warp finishes its assigned long tail. Hardware warp
scheduling does not redistribute these application-level task assignments.

This candidate tests dynamic consumption of already admitted tasks. It does not
claim that all observed Barrier stalls come from this tail, that a smaller CTA
necessarily increases occupancy, or that previously measured timings apply to
this source. Only an actual matched GPU experiment can establish profitability.

This is a **packet-local persistent warp drain under the existing persistent
CTA producer**, not a global cross-CTA scheduler and not completion of P0-B.
The P0-A independent generated-source queues remain a separate composition gate.

## Implementation

- Preserve the existing single block/page claim, screening predicate, canonical
  quartet orientation, generated-class/overflow exclusions and force profiling.
- Record the already computed exact shell class beside each admitted slot.
- Build a stable counting-sort index of slot handles, not scientific task data.
  Source identity is uniform for the enclosing full-range force launch.
- Each warp leader atomically claims the next handle in a class bucket and
  broadcasts that handle to all 32 lanes. A completed warp can immediately take
  another task, without a CTA rendezvous between tasks.
- Each warp owns the complete quartet, including every update of its mutable
  tile field. Retain the existing warp synchronizations around those updates.
- Retain the CTA publication and packet-retirement barriers. They protect shared
  storage and cannot simply be deleted. Low-order scalar consumers are unchanged.
- An invalid class index falls back to the original static strided drain. Capacity
  remains an upstream admission invariant: at most one accepted task per lane
  in a packet no wider than the existing 256-entry task storage.

The queue is a native scheduling leaf, not a second scientific/codegen owner.
The SCF structure checker registers it with the existing device queue owners and
allows the exact consumer-to-queue dependency, without broadening numerical
module access. No integral recurrence, source coefficient, density contraction,
precision policy or atomic scatter formula is replaced.

Grouping by class does not make primitive lengths identical and does not by
itself remove intra-warp divergence. The first experiment is FIFO within each
class; cost prediction, primitive bucketing, subtile splitting and global work
stealing are deliberately outside this change.

## Diagnostic controls and fallback

Set `GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_SCHEDULE` **before preparation/capture**
and keep it fixed for that process:

| Setting | Full-range Force-purpose boundary |
| --- | --- |
| unset, empty, `0`, invalid | Existing static 256-thread schedule |
| `static128` | Static strided drain with 128 threads |
| `warp128` | Class-bucket dynamic pull with 128 threads |
| `warp256` | Class-bucket dynamic pull with 256 threads |

Nonstandard incoming block shapes retain the incumbent. Fock-purpose/value,
angular diagnostic and SR/LR/RSH launch boundaries retain their existing route.
The control is a private qualification seam, not public automatic selection.
A captured graph keeps the launch selected during capture; changing the
process environment is not a graph-reconfiguration API. Record the setting
explicitly in every qualification receipt and verify the actually launched
kernel. Passing a test that never reaches this boundary does not qualify it.

The 128-thread static arm and the two dynamic arms separate block-size effects
from queue-consumption effects. Packet admission and drain strides both follow
the actual block size so shrinking a block does not drop scientific work.
The global page cursor and its terminal claims remain CTA-owned and unchanged.

## Cost and numerical risks

The class index adds bounded integer shared storage to dynamic instantiations:
for capacity 256 and 55 classes its C++ layout is 2496 bytes. Actual linked and
application/driver shared reservations must still be measured. Index preparation
costs O(packet capacity + class count) on the leader; atomic contention and
small/empty buckets may erase a load-balancing benefit. There is no new device
allocation or retained scientific cache.

Static instantiations have no class-index arrays, but template emission and
changes to the shared translation unit may still change the final binary's
register/stack/code footprint. A disabled rebuild must be compared with the
sealed original; an unchanged default selector is not proof of zero overhead.

Within-task arithmetic is preserved. Cross-task floating atomic accumulation
order can change, so bitwise equality of final forces is **not** promised.
Independent energy/force gates, signed source checks and solver histories are
required. Raw-page warp claiming from the rejected #1978 experiment is not
reintroduced; this candidate consumes existing admitted handles only.

## Host evidence in this change

The local environment has no CUDA toolkit/GPU or ccache. No uncached project
build, native link, sanitizer, full-repository suite or GPU benchmark is claimed.
Fetched original source bytes were checked against Git blob identities before
patching; the checkout used for local tests contains only relevant files.

`python -m pytest -q tests/python/test_direct_warp_queue.py` passes with GCC and
with `CXX=clang++`: **8 passed, 1 explicitly skipped GPU test** each. The actual
C++ queue has 65 constexpr ownership/rejection assertions plus 15 selector
assertions, evaluated with `-fsyntax-only` (no object or executable emitted).
The tests cover tails, empty and skewed buckets, uneven worker progress, index
reuse, invalid-class recovery, 128/256-thread candidate coverage for all sizes
0..1024, and a negative dependency-injection control. These are host proofs and
models, not CUDA synchronization evidence.

## Required GPU validation (all outstanding)

1. **Build and provenance.** Verify ccache, compile/link the full release CUDA
   source at the target architecture, retain source/library/toolchain hashes,
   and run the normal project gates. Compare original versus disabled rebuild,
   `static128`, `warp128` and `warp256`. Confirm actual kernel identity/block size,
   including the dynamic template discriminator; retain registers, stack, local,
   linked shared and application shared separately. Do not infer occupancy or
   executed traffic from those reservations.
2. **Queue correctness/synchronization.** Run the included real CUDA probe and
   memcheck, initcheck, racecheck and synccheck. It exercises 180 launches with
   three packet lifetimes each, multiple CTAs, empty/tail/full packets, uneven
   task lengths, exact-once ownership, mutable tile isolation, and invalid-index
   fallback. A no-device run fails, rather than becoming a pass.
3. **Scientific correctness.** Exercise actual dynamic Full/FullSources force
   dispatch in RHF/UHF and RKS/UKS; independent fixed-density J'/K' and signed
   Combined/Separate outputs; zero coefficients; through-f, generated disable/
   overflow recovery, indexed/unindexed domains, inactive systems, batched and
   changed geometry. Retain existing independent E/F gates on every repeat
   (PBE0 E <= 1e-8 Eh, F <= 1e-7 Eh/Bohr). Run full force paths under sanitizers,
   not just the queue-only probe. Check untouched SR/LR/RSH/value compatibility.
4. **Work and endpoints.** On the same allocation/GPU, run interleaved arms at
   48/96 atoms for HF and PBE0: cold, five warm, moved, five moved-warm. Keep actual
   SCF/Fock histories and all reference pairings; do not normalize away a changed
   trajectory. Count admitted and executed per-class/source quartet/tile work,
   including fallback and queue overhead. Preserve stream/capture/replay and
   bounded-storage behavior. Keep profiler replays out of clean timings.
5. **Mechanism and composition.** Collect candidate-specific NCU occupancy,
   eligible/issued warps, Barrier/Wait/Scoreboard stalls, active lanes and local
   load/store *requests*. Measure index-build/claim overhead and contention.
   Compare standalone and qualified P0-A composition; do not add speedups from
   unrelated campaigns. Reject if counters improve without complete endpoint
   benefit. Neither kernel timing nor this PR promotes a production default.

### Runnable queue probe

Within a finite user-approved Slurm allocation, preserve scheduler device
visibility; do not change driver policy or use privileged profiling implicitly.
The architecture below is an example for the RTX 5090 qualification target:

```sh
ccache --version
GENERATIVEQC_RUN_WARP_QUEUE_GPU_TESTS=1 \
GENERATIVEQC_WARP_QUEUE_TEST_ARCH=sm_120 \
python -m pytest -q tests/python/test_direct_warp_queue.py

ccache nvcc -std=c++17 -arch=sm_120 -I src \
  tests/cuda/direct_warp_queue_probe.cu -o /tmp/direct-warp-queue-probe
for tool in memcheck initcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 1 /tmp/direct-warp-queue-probe || exit 1
done
```

The probe does not replace independent chemical-force or full-project tests.
Promotion and broader scheduling are revisited only after the gates above.

Agent: ChatGPT
Model: GPT-6 Astra Pro
