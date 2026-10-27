# Compiled CUDA RHF provider-guard diagnostic

This fresh engineering observation exercises the changed production
`reference_detail::retained_bytes` with real cuBLAS/cuSOLVER handles on one
H100 80GB HBM3, driver 595.58.03 and CUDA runtime 12.9. It is separate from
the earlier standalone vendor-initialization observation associated with #2810.

The unchanged 192 MiB combined allowance correctly refused the observed
375,390,208-byte difference (358 MiB). The preserved `std::length_error`
contained the exact successful CUDA snapshot consumed by the guard:

```text
CUDA RHF provider exceeds retained allocation allowance: free_before_bytes=84467187712 free_after_bytes=84091797504 observed_delta_bytes=375390208 allowance_bytes=201326592
```

All four parameter/handle/stream cleanup calls returned zero. This is a passing
diagnostic and cleanup check of a refusal, not successful provider admission.
No GenerativeQC SDK, geometry, SCF, matrix operation or scientific endpoint ran.
The free-memory difference includes driver/allocator effects; it is not an exact
allocation ledger, process peak or billing measurement. The per-provider policy
remains 96 MiB, and no historical CBS observation is filled in by this result.

`receipt.json` retains the actual bridge stdout normalized from transport CRCRLF
to LF, the four measured values, loaded-library sizes/hashes, device identity
hash, cache statistics, compile/link commands and exact source identities.
`gpu_guard.cpp.txt` is the measured bridge source, preserved byte for byte as
an evidence artifact. Its wrapper observes the real `cudaMemGetInfo` result;
the production guard still performs its actual stream synchronization and
runtime error checking. The bridge independently checks all reported fields
against that successful snapshot and destroys vendor resources on refusal.

An earlier invocation stopped before compilation because the existing compiler
cache executable was absent from PATH. The successful independent invocation
exposed the verified existing launcher on its process PATH; it changed no source,
shared cache, cache key or budget. The failed prerequisite is retained in the
receipt and is not counted as a code or device-guard failure.

To reproduce on an explicitly allocated compatible CUDA device, use the measured
base commit in the receipt plus the changed production header from this PR.
Verify the header and bridge LF hashes in the receipt. Copy the bridge to a
scratch `gpu_guard.cpp`, use a verified `sccache` (or `ccache`) launcher for the
C++20 compilation, and link against CUDA runtime, cuBLAS, cuSOLVER and `dl` as
shown in the retained commands, adapting only repository/toolkit paths. Preserve
cache statistics before and after compilation. Run one bridge process with a
finite timeout and retain its full output. A different device/library snapshot
may accept or refuse; acceptance must return the observed delta within the
unchanged allowance, while a refusal must report all four exact byte fields.
The committed host regression covers deterministic equality, one-byte-over,
nondecreasing-free-memory, maximum-byte-count and typed-error cases without a GPU.
