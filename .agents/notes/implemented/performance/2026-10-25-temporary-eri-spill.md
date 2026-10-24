# Decision: Spill omitted FP64 ERIs to bounded local temporary files

Status: implemented
Date: 2026-10-25

## Decision and lifetime

The requested hierarchy extends GPU and pageable RAM with temporary file backing
before Direct recomputation. Cold strict full/LR KS sources first preserve GPU
buckets, then fill RAM, then reserve whole omitted buckets on local disk. No
integral equation or precision changes. Both backings share the same two pinned
and device tiles and compensated contraction. Device and pinned allocations
remain charged to the complete numeric budget and available job RAM. File
bytes have a separate capacity bound and never masquerade as resident RAM.

The default directory is `/tmp`; `GENERATIVEQC_ERI_TMPDIR` can select another
local directory. `GENERATIVEQC_KS_DISK_MAX_BYTES` and its RANGE counterpart
allow explicit ceilings or zero to disable. Default space is at most 64 GiB and
one quarter of available local filesystem capacity. tmpfs/ramfs and common
network filesystems are refused. `posix_fallocate` reserves actual capacity;
unsupported reservation refuses instead of promising space through a sparse
file. Temporary files are created privately and immediately unlinked. Their
owning descriptor closes on normal destruction, exceptions or process exit.
Geometry, basis, operator, screening and precision remain bound to one owner.
Changing range disk controls invalidates a retained warm lease.

## Failure and data movement

Construction uses the native CUDA recurrence and audits every retained value.
Reservation/write refusal discards only the disk tier, preserving GPU/RAM and
exact Direct. A numerical or CUDA failure propagates. After publication, failed
reads fail the whole action; partial accumulations are never certified. The
provider still permits an explicit full Direct retry, which resets outputs.

Construction flushes data and advises page-cache eviction. Reads copy bounded
file tiles into pinned buffers and advise eviction of consumed ranges. Eviction
is an OS hint, not a claim that all I/O reached physical media. Diagnostics
separate RAM/disk values and logical file/H2D bytes; endpoint receipts also
retain process-wide physical I/O, including other process traffic. File I/O is
sequential and currently synchronous with host staging; it can overlap already
queued device work but is not an asynchronous storage pipeline.

## Qualification and follow-up

Executable local-file tests exercise reservation, byte bounds, short reads and
unlinked lifetime. Independent full/LR signed-density, UKS and tiny-staging
probes exercise disk-only and combined RAM/disk sources; a real truncated file
must fail replay and preserve independent parity on explicit Direct retry.
Complete cold/warm/moved E+F and complete endpoint timings are recorded with
the built source before any performance claim. A small/full-GPU cache never
opens a file. A bucket's angular degree remains a cost proxy; timing evidence
must determine where disk transfer actually beats recurrence regeneration.

## Measured boundary for automatic spill

Automatic spill is restricted to LR exchange. The source-matched 96-atom PBE0
nonincremental cohort on n5 takes 94.784 s with GPU plus Direct versus 143.826 s
with GPU plus local NVMe: 51.74% slower, even with real physical I/O and identical
nine iterations/ten cold Focks. Explicit full-range disk ceilings remain for
controlled experiments; unset full-range disk retains Direct. The faster RAM
tier and the natural incremental/generated schedule are unaffected. This is an
empirical method boundary, not a universal disk-bandwidth or per-bucket cost
model. Broadening the automatic disk tier needs new complete endpoint evidence.

The [retained LR cohort](../../../../benchmarks/results/disk-eri-20261025/README.md)
uses identical 108-atom source/binary and GPU selection. Complete cold E+F is
681.193 s with Direct for omitted values, 363.250 s with RAM, 493.574 s with
NVMe, and 462.319 s with a 2 GiB RAM ceiling plus NVMe. Disk profiles verify
physical reads alongside logical work; gains are not attributed to page cache.
Each arm has 28 SCF iterations/29 Focks. The single observations support this
LR fallback hierarchy, not a general per-bucket latency model.
