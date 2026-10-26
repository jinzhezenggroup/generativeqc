# H100 standalone provider-initialization diagnostic

Refs #2809 and #2727. This family retains one completed standalone vendor-handle
initialization diagnostic from 2026-10-26. It contains no GenerativeQC execution,
geometry, SCF, matrix operation, or scientific request. It changes no production
allowance and grants no numerical, device-endpoint, performance, or capacity
acceptance.

## Observation

The selected device was an NVIDIA H100 80GB HBM3, driver 595.58.03, PCI
`00000000:3B:00.0`, UUID SHA-256
`977070d342fac93a326dafacea887da87b42ff849e2c413a449f065e0c4ed666`.
The before/after device records agree. CUDA runtime version was 12090;
`CUDA_VISIBLE_DEVICES` was unset. Explicit library paths, sizes, modes and hashes
are retained inside the observation.

| Synchronized snapshot | Free bytes | Signed baseline minus free bytes |
| --- | ---: | ---: |
| Before providers | 84,467,187,712 | baseline |
| After cuBLAS initialization | 84,395,884,544 | 71,303,168 (68 MiB) |
| After cuBLAS and cuSOLVER initialization | 84,091,797,504 | 375,390,208 (358 MiB) |

All 15 recorded setup/synchronization/query API statuses and all four handle
cleanup statuses were zero. The probe clock was 1.1172564253211021 seconds;
the enclosing controller clock was 5.632252200040966 seconds. These are nested
scopes, not additive intervals or complete endpoint/billing timings.

At frozen comparison revision
`e1e244197411b58da42aa8a2cbce49014d7ed3ff`, `src/scf/cuda_rhf.cpp`
assigns 96 MiB per enabled provider, or 201,326,592 bytes (192 MiB) for both.
`src/scf/cuda/reference_export.cuh` refuses when the nonnegative synchronized
free-byte difference exceeds that allowance. The standalone combined difference
is 166 MiB above that source allowance. This is a comparison with inspected
source; the diagnostic did not call that RHF path or reproduce its refusal.

The three explicitly hashed libraries are `libcudart.so.12.9.79`,
`libcublas.so.12.9.2.10`, and `libcusolver.so.11.7.5.82`, under
`/opt/generativeqc-reuse/cuda-12.9/targets/x86_64-linux/lib/`. The retained
locator stdout also lists transitive dependencies. Their byte hashes remain
`UNAVAILABLE_NOT_HASHED`; explicit-library hashes do not identify that entire
dependency closure.

## Retained bytes and provenance

- `observation.json.xz` losslessly stores the original 4,886-byte observation.
- `provider-stdout.utf8.xz` stores the exact UTF-8 encoding of the actual
  transport stdout string, including CRCRLF and the trailing `OK`. Its decoded
  observation equals the separately saved JSON; the two serializations have
  different bytes and hashes.
- `library-locator-stdout.utf8.xz` stores the exact locator stdout string.
- `provenance.json` binds these raw bytes, original controller-container hashes,
  actual inline-source identity, command identity, comparison source, and the
  public reproduction derivative. Original controller stderr and private
  coordination metadata are omitted explicitly, with source hashes retained.
- `platform-observation.json` is a declared engineering-field extract rather
  than the full platform response. `preparation-history.json` retains the
  preparatory failures and their original source identities.
- `retention-manifest.json` binds every other public member by bytes and SHA-256.
  It is an engineering retention manifest, not a scientific publication gate.

The actual inline Python source SHA-256 is
`d470eb22bf3a7ecfed2078cded86bb2abdcf3a8cec0bcbbc98217d3ada99ffc5`.
The decoded command's `-c` argument matched that saved source exactly. Its
embedded `permit.script_sha256` still referred to the older prepared-file SHA
`be32f1394864115a4cf5c43877bdde7b3f48a92cec7be62cb7dd4877f199883a`;
that stale pointer is not the identity of the executed inline source.
The saved command has CRLF; provenance separately names its raw hash and the
explicit read-only LF view matching the transport command hash.

An initial CPU shared-HDD staging attempt failed with `EDQUOT` / errno 122 and
a zero-byte staged file, before any provider process. A local CRLF source-hash
refusal also preceded any remote GPU attempt. These failures were retained;
the quota is not declared repaired. The single subsequent actual provider
process used inline `python -c`, with no stdin or shared-file write. There was
no provider retry.

## Offline replay

From the repository root, run only:

Use committed LF bytes, for example a `git -c core.autocrlf=false archive` export
or a checkout with `core.autocrlf=false`. A Windows CRLF checkout changes the manifest-bound text
bytes and must fail integrity verification rather than silently normalize them.

```console
python -I -S -B benchmarks/results/h100-provider-retention-20261026/replay.py
```

This reader uses only standard-library data reads. It verifies retained-member
hashes, bounded XZ decoding, the observation/stdout relationship, the actual
event and cleanup records, device identity, byte arithmetic, frozen allowance
comparison and preserved unknowns. Its success status is
`PASS_RETAINED_STANDALONE_ENGINEERING_REPLAY_ONLY`, with scientific/device
endpoint acceptance `NOT_GRANTED`. This standalone wire format is different
from the interposer begin/end schema consumed by
`benchmarks/rhf_provider_retention_report.py`; no SDK/endpoint report is inferred.

`reproduce.py` is an unexecuted public derivative, not the byte-identical actual
source. It removes private permit fields, retains the consumed 512 MiB
engineering stop cap and API sequence, adds refusal of optimized Python, and
applies public source formatting. Its exact changes and hash are declared in
provenance. Executing or importing it starts a real GPU diagnostic and requires
separate resource authorization. It expects Linux, one visible H100, the listed
library installation, and an outer finite timeout; the original launcher used
`timeout --signal=TERM --kill-after=5s 160s` and Python `-I -S -B`.
The 512 MiB diagnostic stop cap is not the 192 MiB RHF source allowance.

## Limits

The new 358 MiB observation does not recover historical #2727 CBS
before/after/delta/allowance fields. Those remain unavailable. It does not prove
physical OOM, exhaustion of a 32 GiB numeric budget, causal per-provider
allocations, internal work, peak usage, billing, or a universal H100 capacity
requirement. Concurrent or opaque device activity is not causally attributed.
It does not qualify an increased production allowance. #2727 and the production
diagnostic work in #2809 retain their remaining scope.
