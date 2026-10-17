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

The version script exports the exact versions found in the measured ELF. Each
wrapper resolves its corresponding real implementation with
`dlvsym(RTLD_NEXT, name, exact_version)`; resolution failure returns a provider-
appropriate error and marks the capture unavailable rather than calling a
different version or route.

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

## Fake ELF qualification

`benchmarks/rhf_provider_retention_fake/` contains a self-contained Linux-only
qualification. Its fake provider exports each of the seven targets under the
production import version and under a deliberately different default
`FAKE_PROVIDER_9.9` version. A version-pinned driver runs both without and with
the observer; byte-identical stdout plus provider-side expected/default call
counters is the forwarding oracle. The fake cuBLAS create path makes a nested
exact-version memory query so the trace must also prove the thread-local
`provider_internal` boundary.

The hand-authored verifier separately checks provider and resolution failures,
fresh-file `O_EXCL` behavior and mode `0600`, the 512-event bound and loss
counters, one injected host write failure followed by a terminal record, exact
symbol versions, and a complete fake 200-byte report. Compiler, SDK-header,
launcher, source, DSO, driver and case-exit evidence is retained in the result
directory. The fixed a308 production source/library/endpoint hashes remain
provenance only: neither the fake library nor a CPU PASS is a production, CUDA-
device or scientific oracle.

The first Linux run at observer commit `6662a2a553ed23eb641819debca56531d1e33aa7`
failed as intended: unversioned `dlsym(RTLD_NEXT, name)` reached every
`FAKE_PROVIDER_9.9` default implementation, changing the memory outputs from
`1000:2000` to `111:222`. That negative run is retained under `run-1`. Commit
`5f65cd687518b06d0aebeda74fd4776abe8052b8` changed only the diagnostic shim to
exact-version `dlvsym`. The independent `run-2` then passed all nine cases, both
positive and provider-failure stdout comparisons were byte-identical, and the
fake report was `COMPLETE` with delta 200 and allowance 250. Its retained result
manifest SHA-256 is
`92986f4a9a80165d5bcde69b48dae9ca9da80bf421b4f7d2c614e999768c7f96`.

## Qualification sequence

At the end of the fake-only stage, the Linux ABI/forwarding protocol was
qualified, but the shim had not run in the actual a308 CUDA endpoint process.
A fake PASS does not prove real CUDA
library loading, actual retained-memory values, complete endpoint execution,
scientific acceptance or performance. Actual-device use must still inspect the
production library with:

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

The completed fake qualification used one 1-CPU/4-GiB CPU instance for 756
seconds at the observed 0 points/hour, effective HIGH priority, and stopped 144
seconds before its platform timer. The later diagnostic-only device attempt was
planned for one compatible sm90 H100/H200 for at most 0.1 hour, no restart. At
the observed quote of 1 point/hour, 0.1 point was a nominal planning estimate;
the billing minimum and actual allocation/billing boundaries were not verified,
so this is not an observed cost upper bound. A clean scientific retry remains a
separate, currently unapproved gate.

The first bounded Linux preflight on `pytorch:25.06-py3:25.06` found no
`sccache` or `ccache` through that shell's `PATH` and stopped before archive
upload or compilation. That is not an exhaustive image/fileset absence proof:
the earlier source build named persistent candidates under
`issue1765-attribution-20261014/runtime/{bin,local/bin}/sccache`, but their current
existence, version, SHA and ABI were not rechecked before the one-window timer
expired. A future separately authorized window must inspect those absolute paths
first. The second bounded window found the retained
`runtime/local/bin/sccache`, verified version 0.16.0 plus the target compiler and
CUDA headers, and used that absolute launcher for all five compile/link commands.
The offline bootstrap remained unused.

## Real H200 observation: retain the ambiguity

One instrumented invocation of the frozen a308 endpoint ran on H200 driver
570.124.06. It used the qualified shim SHA-256
`c0081483958d442ec5a5f9caaccfddc2dbd2b807a67171a8d0e1919e60dccbba` after
rehashing every entry in the retained fake-result manifest and independently
recomputing both raw stdout comparisons. The endpoint returned 1, and the
111-byte native log matches the earlier provider-allowance failure exactly
(SHA-256 `72944503040cc4bc621bb7444672470a5969f376953473369bab50b236ba5468`).
No clean timing or scientific result was obtained.

The actual event journal was retrieved read-only and verified as 18,227 bytes,
SHA-256 `84e8d3d11dec6b0faa924be88daaeb1f6b27e63d6642ad6cdc4725d5d3f21e45`.
It contains 87 ordinary records plus terminal sequence 87, one process/thread,
balanced provider begin/end nesting, and zero dropped, resolution-error and
write-error counters. Nevertheless, three adjacent application-query intervals
contain provider creation:

| Query sequences | Before free bytes | After free bytes | Raw delta bytes |
| --- | --- | --- | --- |
| 0 to 39 | 149570977792 | 131340435456 | 18230542336 |
| 39 to 62 | 131340435456 | 130969239552 | 371195904 |
| 63 to 86 | 131340435456 | 130969239552 | 371195904 |

The first interval contains two top-level cuSOLVER setup sequences, so it does
not match one source-required RHF setup. Each later interval has the complete
ordered top-level cuBLAS and cuSOLVER setup; each also contains four nested
cuBLAS creates and one nested stream binding within cuSOLVER creation. The shim
currently marks provider-call records `application` even when a provider wrapper
is already active; only memory queries receive the depth-based origin tag.
Consequently, those nested calls must not be mistaken for additional independent
application providers. The existing parser still refuses the three intervals
and emits `INCOMPLETE` with a null primary measurement and flags.

Both later raw deltas are 354 MiB. **Conditionally**, if either is the RHF guard's
interval, the observed top-level providers imply the unchanged 96+96 MiB source
allowance, leaving 162 MiB above it. That does not identify a unique guard
interval or authorize changing the allowance. The a308
`df_ccsdt_force.cu` first prepares a DF guess; `rccsd_method.cpp` then retries a
failed seeded reference once with a cold reference. This supports an explanation
for the repeated setup, but the journal records no caller PC or explicit stage
that proves the correspondence. Do not select the last interval, discard the
initial interval, or merge equal deltas to manufacture a complete observation.

The host reporter now reconstructs balanced provider-call nesting separately
for each PID/TID. A unique interval may contain vendor-internal provider calls
without counting them as duplicate external setup. Candidate diagnostics expose
the reconstructed nesting, original query sequence ranges and raw deltas, but
do not assign a stage or an effective allowance to an ambiguous interval. Invalid
or unclosed nesting remains incomplete. In particular, no compatibility score,
record origin text, ordinal position or equality of deltas selects a primary.
Independent hand-authored tests cover a nested cuBLAS provider call inside
cuSOLVER and repeated identical intervals. An offline replay of the actual
88-row trace keeps all three candidates, top-level call counts `[9,6,6]`, nested
counts `[10,5,5]`, and the original `INCOMPLETE` result with null primary fields.
This is host analysis of old bound device evidence; the executed shim remains
the old `c008...` binary, with no new device execution implied.

The GPU Job ended `FAILED` after diagnostic exit 2. Requested priority 4 became
the observed platform value 20/`NORMAL`; no running-priority snapshot was
obtained during the 45-second grace period. Scheduled-to-terminal was 67 seconds,
container-start-to-terminal 58 seconds, and creation-to-terminal 139 seconds;
the last includes queue time. Billing is unverified. A separate free 1-CPU/4-GiB
reader retrieved only the journal and stopped after 90 seconds, within its
10-minute platform timer. These are diagnostic and evidence-recovery results,
not current-upstream endpoint or parent-issue completion.

## Owner-coordinated alternative

The external owner dependency is recorded at
https://github.com/jinzhezenggroup/generativeqc/pull/2287#issuecomment-6185131043.
That evidence/ownership comment is not an approval or an ownership transfer.

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
