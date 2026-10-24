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
