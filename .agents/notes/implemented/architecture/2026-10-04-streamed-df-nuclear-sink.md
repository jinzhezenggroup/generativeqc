# Decision: consume device DF cotangents in one persistent nuclear sink

Status: implemented
Date: 2026-10-04

## Problem

The physical DF source reverse map delivers provisional full A rows and one
metric cotangent on its owning stream. The existing standalone gradient bridge
accepts host spans, repacks basis metadata per call, and synchronizes every call.
Downloading all weights or requesting every nuclear-coordinate derivative would
lose the bounded producer/consumer structure needed by hundreds-AO forces.

## Decision

`scf::CudaDfNuclearSink` uploads normalized basis/geometry metadata once and
borrows successive device weight tiles. The first tile binds a nonnull producer
stream; subsequent calls must use exactly that stream. The existing generic
weighted kernel visits all center channels together, including auxiliary-only
atoms and six-term spherical g expansions. It does not materialize any nuclear
coordinate derivative tensor or copy response weights through the host.

The setup stream is explicitly drained before the first producer callback can
use its metadata/output. During consumption, the producer may reuse a weight
buffer in stream order after enqueueing its read. The producer's stream owner
must outlive the sink. Destruction drains outstanding reads and publishes no
result. `finish` is single-use and must only be invoked after the entire producer
successfully returns; it downloads and validates the compact nuclear gradient.
Failed enqueue/stream validation permanently closes that sink.

Complete numeric admission reserves a conservative host setup-capacity bound
before packing or allocating device buffers. The remaining allowance bounds
all device allocations. `numeric_capacity_bytes()` charges both overlapping
owners to the outer source reverse call. This differs deliberately from the
legacy standalone bridge's independent host/device maxima. No full-weight cache
is introduced, and the ordinary host-span bridge remains available.

## Invariants

- A weights address every full `[mu,nu,P]` entry once; M weights are full
  `[P,Q]` Frobenius coordinates. There is no triangular doubling.
- One all-row physical source reverse call consumes exactly `N^2 Q + Q^2`
  weights, independent of the number of nuclear coordinates. The primitive
  work within each weighted integral still depends on angular degree and
  contraction lengths; this count is not a claim of constant scalar cost.
- Consume does no allocation, host weight copy or synchronization.
- Borrowed weights remain valid until their queued reads complete, and no output
  may be published merely because one callback completed.
- Neither this sink nor a successful fixed-orbital source derivative certifies
  orbital stationarity, Pulay response or a complete CCSD(T) force.

## Evidence

The validation probe composes the actual retained physical-source VJP with this
sink. The host RawSource is used only for normalized geometry/basis metadata;
its CPU integral read interface is never called. Independent libcint values,
NumPy metric roots and two-step nuclear finite differences define the oracle.
Tests cover orbital f, auxiliary f/g, both public representations, independent
frame cotangents, translation cancellation, both callback failure locations,
stream rejection and single publication. They also check the exact weight count,
zero response H2D bytes, compact gradient D2H, and combined reverse admission.

The initial eight cases passed on Slurm job 12213 in 6.42 seconds with library
SHA256 `6e2ca1b0e9103c79442d7f8eed211d2245624bee983186553b90904026fb8c03`.
The expanded ten-case tier passed under memcheck on Slurm job 12216 in 68.26
seconds, with zero reported errors. This additionally checks a different valid
producer stream and exact combined reverse-budget admission (one byte less is
refused transactionally). These are component qualification tests, not cold
force endpoint benchmarks. Artifacts are ignored under `.artifacts/nuclear-sink/`.

## Rejected alternatives and remaining work

- Download all cotangents: restores unnecessary `N^2 Q` host storage/traffic.
- Call the synchronous tile bridge once per source row: repeats metadata work
  and drains the producer for each row.
- Loop over nuclear coordinates: bounded storage would still replay integral
  work in proportion to atom count.
- Add a second recurrence or force formula: reuse the already qualified
  compiler policy and weighted traversal.

The method must still bind the sink to the exact source geometry/frame and
compose conventional-RHF orbital/Z and overlap/Pulay response. Complete forces
remain gated. Combined endpoint work, numerical stationarity, independent force
references and hundreds-AO timing are required before any public promotion.
