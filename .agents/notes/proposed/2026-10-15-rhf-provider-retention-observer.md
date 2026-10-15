# Proposal: diagnose RHF provider-retained memory outside the SCF owner

Status: proposed
Date: 2026-10-15

## Problem

The source-bound issue #1765 ethane230 energy-only campaign reached native CUDA
RHF setup on an H200 but stopped before SCF/CCSD work with
`CUDA RHF provider exceeds retained allocation allowance`. The failed receipt is
bound to source `a3088c5d2825de719123682facc36e8327a155e9`, library
`e39cbf740cd81ace24c56fd0c1e7bdb2bcaeb8dc93aca02c7f37c70d20c0e001`,
endpoint `19b34468267ef52174e14afb950cb80e08c33fd3d861b89b068819886120cc65`,
input `9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`
and oracle `ba40ed0b759eb27a9f3909377a6da0795fd292019a9b079e58c82929941a059b`.
It retains neither the two `cudaMemGetInfo` samples nor actual provider flags,
so it cannot establish the measured delta or effective allowance.

PR #2287 at exact head `babba1d6d6a09e9b130d01655d2a8f1bf8ba85fd`
currently owns `src/scf/cuda_rhf.cpp`. This diagnostic must not create a second
writer in that owner or weaken its fail-closed resource guard.

## Existing observability boundary

- `DeviceResourceLedger` and `DeviceAllocationJournal` cover owned
  `resource_cuda_*` allocations and shape-only provider reservations. Their
  contract explicitly excludes CUDA context, graph, pool and library-internal
  allocations. v3 `provider_reserved_bytes` is not a measured provider delta.
- `GENERATIVEQC_DF_PROGRESS_TRACE` is behavior-neutral when disabled, but the
  failing guard emits no before/after/delta/allowance values. The existing
  `cuda_rhf_reference_completed` values execute only after successful reference
  publication.
- The public SCF diagnostic exposes density-update and physical-residual RMS and
  is unavailable after backend failure. The eigensolver diagnostic has a later
  available-memory snapshot and workspace query, not the combined RHF provider
  interval.
- `MatrixLibraryOwner::retained_bytes()` privately measures one cuBLAS owner,
  excludes cuSOLVER/the RHF sequence, and clears its value on allowance refusal.
  `hf_cuda_retained_numeric_bytes()` also receives `provider_retained_bytes_`
  only after the failing check returns successfully.
- The private residency observer describes graph/source lifetime boundaries,
  not library-internal retained memory.

No existing public or diagnostic entrypoint can recover the missing values from
the failed a308 binary.

## ELF prerequisite

The actual a308 `libgenerativeqc.so` was inspected with a stdlib ELF64 parser.
All seven target functions are undefined `GLOBAL`/`FUNC` symbols with `DEFAULT`
visibility and `R_X86_64_JUMP_SLOT` relocations in `.rela.plt`; the DSO has no
`DT_SYMBOLIC`/`DF_SYMBOLIC` tag:

| Provider | Imported symbols | Required symbol version |
| --- | --- | --- |
| CUDA runtime | `cudaMemGetInfo` | `libcudart.so.12` |
| cuBLAS | `cublasCreate_v2`, `cublasSetStream_v2`, `cublasSetPointerMode_v2` | `libcublas.so.12` |
| cuSOLVER | `cusolverDnCreate`, `cusolverDnSetStream`, `cusolverDnCreateParams` | `libcusolver.so.11` |

The library directly needs `libcudart.so.12`, `libcublas.so.12` and
`libcusolver.so.11`. The endpoint itself needs `libgenerativeqc.so.0` and has no
target imports. These facts make dynamic interposition structurally plausible;
they do not prove that an untested preload DSO will satisfy the versioned symbols
or observe the real process. A fake Linux DSO test and actual-device capture are
still required.

## Proposed diagnostic

`benchmarks/rhf_provider_retention_probe.cpp` is a diagnostic-only ELF
interposer. It forwards every input pointer/value and provider return code with
no extra CUDA call, allocation, synchronization, context initialization, warmup
or retry. It journals a fixed maximum of 512 host event rows to a fresh file:

- application and provider-internal `cudaMemGetInfo` calls are distinguished by
  thread-local provider-call depth;
- cuBLAS/cuSOLVER create, stream binding and parameter/pointer setup have paired
  begin/end records and retain real status codes;
- PID/TID and a monotonic sequence disambiguate threads and nested calls;
- overflow, dynamic-symbol resolution errors and write errors remain separate;
- a normal-process destructor writes the terminal record. Missing terminal is
  incomplete evidence.

The version script exports the exact versions found in the measured ELF. Symbol
resolution uses `dlsym(RTLD_NEXT, ...)`; resolution failure returns a provider-
appropriate error and marks the capture unavailable rather than calling a
different route.

`benchmarks/rhf_provider_retention_report.py` is stdlib-only. It accepts a
separate source-policy receipt containing provider allowance values and required
events, plus source/library/endpoint/shim/event-log identities. It selects
exactly one same-thread pair of successful application-origin memory queries.
Vendor-internal queries are explanatory and cannot become pair endpoints.
The source policy supplies an ordered list of required calls for each provider.
Effective allowance is derived only when every required call has one successful,
non-overlapping begin/end pair in that exact order. Missing, duplicate, nested,
reordered, ambiguous, failed or lossy evidence produces `INCOMPLETE`; absent
providers never become false/zero. A successful provider create observed outside
the source policy is also `INCOMPLETE` and remains visible in the observed flags,
rather than being silently omitted from the effective allowance.

This observation covers one provider-retained free-memory delta. It is not an
allocation ownership census, clean endpoint timing, scientific qualification or
permission to raise/disable the guard.

## Host evidence

The host suite uses independent hand-authored events for:

- ordered cuBLAS/cuSOLVER setup with a vendor-internal memory query;
- `after > before` without unsigned underflow;
- missing terminal/before/after/provider/configuration events;
- duplicate/reordered/nested/ambiguous begin/end pairs;
- actual providers omitted by the source policy;
- provider errors with original status;
- capture loss, write errors and sequence gaps;
- source/event-log identity mismatch; and
- malformed/blank/non-object JSONL rows.

No production source fingerprint, AST shape or internal call count is pinned.

## Prepared fake ELF qualification

`benchmarks/rhf_provider_retention_fake/` contains a self-contained Linux-only
qualification that has not yet been executed. Its fake provider exports each of
the seven targets under the production import version and under a deliberately
different default `FAKE_PROVIDER_9.9` version. A version-pinned driver runs both
without and with the observer; byte-identical result JSON plus provider-side
expected/default call counters is the forwarding oracle. The fake cuBLAS create
path makes a nested exact-version memory query so the trace must also prove the
thread-local `provider_internal` boundary.

The hand-authored verifier separately checks provider and resolution failures,
fresh-file `O_EXCL` behavior and mode `0600`, the 512-event bound and loss
counters, one injected host write failure followed by a terminal record, exact
symbol versions, and a complete fake 200-byte report. Compiler, SDK-header,
launcher, source, DSO, driver and case-exit evidence is retained in the result
directory. The fixed a308 production source/library/endpoint hashes remain
provenance only: neither the fake library nor a CPU PASS is a production, CUDA-
device or scientific oracle.

## Missing qualification

This Windows host has no available Linux distro, ELF compiler, `sccache`,
`readelf`, `nm` or `objdump`; no tool was installed. The shim is therefore
uncompiled and its forwarding/version protocol is unverified. The prepared
bounded qualification must still prove that each versioned import reaches the
intended shim and that `RTLD_NEXT` resolves the corresponding real implementation
while preserving arguments, output pointers, status and nested-call tagging.
Compile/readelf success alone is insufficient. Then inspect the shim with:

```bash
readelf -WsW libgenerativeqc.so | grep -E 'cudaMemGetInfo|cublas(Create|Set)|cusolverDn(Create|Set)'
readelf -rW libgenerativeqc.so | grep -E 'cudaMemGetInfo|cublas(Create|Set)|cusolverDn(Create|Set)'
readelf -dW libgenerativeqc.so
```

Compile the one-file shim only after verifying `sccache >= 0.16.0`, for example:

```bash
sccache g++ -std=c++20 -O2 -fPIC -shared \
  -I/usr/local/cuda/include benchmarks/rhf_provider_retention_probe.cpp \
  -Wl,--version-script=benchmarks/rhf_provider_retention_probe.map \
  -ldl -o rhf-provider-retention-probe.so
```

The minimum future preparation is one 1-CPU/4-GiB CPU instance for at most 15
minutes at the observed 0 points/hour. The diagnostic-only device attempt needs
one compatible sm90 H100/H200 for at most 0.1 hour, no restart, with an observed
upper bound of 0.1 point at 1 point/hour. A clean scientific retry is a separate,
currently unapproved gate.

## Owner-coordinated alternative

After #2287 releases ownership, its owner could instead add a failure-path
progress scope or internal POD immediately before the throw, retaining
before/after/delta/allowance and provider flags while keeping the guard and
disabled-trace no-op behavior. That is a core SCF change and must not be authored
concurrently by this #1765 writer.

## References

- Issue #1765 comment `6178545903`.
- PR #2287, exact head `babba1d6d6a09e9b130d01655d2a8f1bf8ba85fd`.
- `src/scf/cuda/reference_export.cuh` and `src/scf/cuda_rhf.cpp` at a308.
- `tests/native/test_contraction_workspace_cuda.cu` and
  `tests/native/test_ks_matrix_provider_cuda.cu` for existing link-wrap patterns.
