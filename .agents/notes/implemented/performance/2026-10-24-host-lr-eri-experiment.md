# Decision: opt-in pageable backing for omitted long-range ERI buckets

Status: implemented opt-in experiment; same-source endpoint screen qualified
Date: 2026-10-24
Base: PR #2744, `bd2fed8ee4709192a580361c97ac14ce40a4930f`

## Problem

The fitted WB97M-V primary leaves exact LR exchange to a separate provider.
GPU partial residency repeats omitted canonical recurrences at each Fock.
The retained #2744 bridge has 28 iterations and 29 Fock builds for **both**
water96 and water108, with cold E+F of 317.736 and 673.204 seconds. The larger
case's extra time therefore is not explained by more SCF iterations in these
specific records. These single historical observations do not establish that
the entire difference comes from LR recurrences or predict a host-cache win.

## Decision

An explicitly bounded pageable FP64 tier may retain omitted whole LR buckets
after the GPU selection has been published. Two pinned/device tile pairs use
copy-ready and consumer-done events to protect independent host/device reuse.
The compute stream retains the existing compensated scatter and projection.
Construction uses the same canonical recurrence with a bounded work offset;
screened row prefixes must clamp, never expand, a tile's submitted extent.

The host byte ceiling includes its pageable values, pinned tiles and numeric
descriptors. The complete consumer budget also charges both device tiles.
The host tier never displaces a GPU bucket. The tile setting is a maximum;
remaining device capacity can shrink it. An initial water108 diagnostic exposed
why fixed 256 MiB tiles were insufficient: the retained GPU selection left less
than the required 512 MiB pair, so the entire host tier refused. Its completed
LR actions still evaluated 986,804,376 omitted values (about 13 s/action) beside
1,169,998,706 GPU-cached values. These are profiled prototype observations, not
a qualified endpoint speedup. Adaptive staging preserves the GPU selection.
Optional allocation refusal retains GPU/Direct behavior. Driver/numerical failures
propagate. The owning geometry,
operator, omega, screening and precision are immutable, and changed host policy
retires the source before warm replay. Both streams drain before destruction,
including preparation failure and partially submitted replay.

The initial tier requires an admitted strict GPU source. It does not add a
host-only admission strategy, graph capture, mixed storage, SSD persistence or
CPU integral production. GPU source refusal therefore still means Direct.

## Rejected alternatives and limits

Pinning all ERIs would unnecessarily lock tens of GiB. Mapped random GPU reads
would turn canonical scatter into fine-grained PCIe traffic. Copying the integral
recurrence into a separate host-cache kernel would create a second scientific
authority. None is needed for sequential tile reuse.

Descending angular order is only a cost proxy. A user mask can leave unprofitable
buckets on Direct; no measured per-bucket crossover or automatic default
promotion is claimed. A full GPU cache takes priority and adds no host storage.
Preparation intentionally uses a simple fenced tile download; steady replay is
the double-buffered experiment. Construction cost remains in cold timing.

## Validation contract

Independent PySCF/libcint SR/LR matrices exercise original/displaced spherical
water systems, signed/zero density, restricted/unrestricted contraction, partial
and full GPU residency, screened/unscreened rows and stricter replay predicates.
The host variant uses 4 KiB tiles to expose nonzero offsets and short tails.
Complete cold/warm/moved E+F comparisons exercise host admission and refusal.

The A/B/C driver uses one binary and matched input identity. Journals retain
submitted per-action coverage and H2D bytes; final census audits establish
completion. Journal enqueue duration is not GPU recurrence timing. Kernel/DMA
profiling must be a separate Slurm run, and profiled wall times must not enter
the unprofiled endpoint comparison. Raw build, source, tests and endpoint
receipts are retained in ignored `.artifacts/host-eri/` during qualification.

The final library passed 51 Slurm GPU tests (four independent host matrix cases,
seven host E+F/lifetime/policy cases, and 40 existing range/mixed/replay cases).
Compute Sanitizer found zero errors and zero leaked bytes. Each host matrix case
also exercises an 8192-byte total device staging allowance and zero-budget
refusal followed by successful retry, preserving the existing GPU values.

The separate n4 CUDA-event campaign confirms that GPU partial residency repeats
986,804,376 recurrences per water108 SCF Fock (median 12.775 s). Host admission
keeps the same 1,169,998,706 GPU values and replays all omitted values, with zero
SCF recurrences and a 0.704 s median host replay region. That region includes CPU
staging, waits and contraction; it is not a DMA duration. Actual n4 device
staging is 138,084,352 bytes, while the n5 primer admits 235,864,064 bytes,
demonstrating why the configured 256 MiB per tile must remain a ceiling.

A separate first-action Nsight capture on n4 records 7,894,435,008 H2D bytes and
321.091 ms DMA activity (24.586 GB/s while active). Only 8.795 ms overlaps resident
contraction, 2.739% of DMA time. Double buffering permits overlap, but this
measurement does not justify crediting most of the benefit to overlap. Avoiding
recurrences is the main mechanism; pageable staging and pipeline gaps remain
potential later optimizations. Disable Nsight CUDA event completion tracing for
this diagnostic: its default warns of possible artificial dependencies between
streams. The initial capture with that default was stopped and excluded.

Keep Fock counts explicit. The current Direct arm has 28 cold Focks; the GPU and
GPU+RAM arms have 29, including one uncached post-SCF action. Frozen-density warm
is one strict Direct action. The cache comparison has identical SCF/Fock counts,
but treating all recorded actions as cached replays would overstate reuse.

Complete receipts and the offline oracle/work-accounting reader are retained in
[`benchmarks/results/host-lr-eri-20261025/`](../../../../benchmarks/results/host-lr-eri-20261025/README.md).

The unprofiled n5 same-binary cold E+F screen completes at water96 in 617.876 s
(Direct), 310.453 s (GPU), and 310.445 s (GPU+RAM); water108 takes 768.181 s,
686.400 s, and 354.803 s. The 108-atom host arm saves 331.598 s (48.31%) against
GPU partial residency. Construction cost is included. All 12 formal E+F calls
pass independent reference gates; total-energy cache differences are at most
3.638e-12 Eh and force-max differences at most 3.180e-13 Eh/bohr. The separate
primer/profile receipts also qualify. This is one observation per arm, with
no order-variation estimate, and does not promote a default.

The unchanged n5 water96 latency is an admission limit. A same-node supplemental
profile keeps 1,727,360,068 GPU values, admits no host values or staging, and
still recomputes 145,861,648 values per SCF action. On n4, different available
capacity produces a different GPU selection and successful host admission at
the same geometry. Do not infer full GPU coverage from the absence of a host
speedup, or pool those nodes' timings. The exact physical-VRAM versus complete
numeric-budget refusal was not instrumented. A future policy might reserve
staging before choosing GPU buckets; that changes the incumbent selection and
needs a separately qualified performance/resource tradeoff.

## Revisit when

Same-source independent E+F gates and repeated complete endpoints establish a
benefit at water96/108, with actual transfer/recurrence time and semantic counts.
Only then consider measured bucket selection or a default policy. Host-only
admission would require its own capacity, lifetime and endpoint qualification.
