# Historical benchmark retention at consumer granularity

Source snapshot: `8e74c66de97e2bfacd655976888c0e0abb252ace` (published master).

This migration keeps numerical evidence in the checkout while retiring duplicate
provenance and old experiment machinery. It does not change a scientific gate,
promote a benchmark result, rewrite Git history or publish an external archive.

## Retained measurements

26 completed campaigns consolidate 245 historical JSON records. Only JSON
whitespace is removed: original number/string tokens and array order are compared
exactly before retiring the original files. All paired timings, negative runs,
work counts, endpoint vectors, settings and numerical/reference identities remain.
Small decisions and the historical Markdown reports remain readable.

The [inspection tool](../../README.md) exposes each original logical record
without a Git checkout, network, native library or GPU. The original path may
include `.gz`; extracted data is JSON, not a reconstructed gzip stream.

| Campaign | Original bytes | Retained JSON records |
| --- | ---: | ---: |
| `issue308-shell-schedules` | 1,045,316 | 25 |
| `issue382-packed-df` | 828,125 | 21 |
| `issue408-device-validation` | 638,308 | 15 |
| `issue206-current-df` | 622,354 | 34 |
| `issues388-391-df` | 551,881 | 18 |
| `issue308-shell-block-prototype` | 334,515 | 11 |
| `issue308-response-timeline` | 302,380 | 11 |
| `issue399-density-seed` | 288,784 | 31 |
| `issue206-aot-and-stock` | 252,060 | 10 |
| `issue392-393-rejected` | 242,062 | 16 |
| `issue308-768ao-response` | 196,576 | 9 |
| `issue206-direct-diagnosis` | 159,166 | 16 |
| `issue206-stable-response` | 223,116 | 3 |
| `scf-direct-native-rtx5090` | 200,008 | 3 |
| `issue283-response-panels` | 65,864 | 2 |
| `issue308-stage-ledger` | 132,216 | 3 |
| `issue308-streamed-k` | 43,163 | 1 |
| `issue308-force-state` | 38,951 | 1 |
| `issue308-retained-b` | 26,484 | 1 |
| `issue206-device-diis-baseline` | 75,599 | 1 |
| `issue206-device-response` | 74,630 | 1 |
| `issue206-final-state-96` | 56,370 | 1 |
| `issue206-final-state-192` | 49,461 | 1 |
| `issue206-completion` | 43,637 | 7 |
| `issue282-streamed-panels` | 37,043 | 2 |
| `issue282-generated-resident` | 23,970 | 1 |

The Direct phase/moved publication retains all numerical/protocol records, its
driver reconstruction patch and compact source/SDK/environment provenance. Its
immutable numerical audit produces the same report without full source/SDK or
package-file inventories. The PBE0 CUDA schedule publication retains every
original numerical/failed run and independently recomputes both qualification
reports; source verification now retains counts, failed lines and original hashes
instead of repeated full path lists. Raw build/source logs remain recoverable.

## Exact recovery

`snapshot.manifest.json` binds every original file with its path, byte count and
SHA-256. It includes the pre-migration readers and manifests, so full historical
verification is possible without interpreting new compact metadata as old bytes.
A local restore of all 464 entries was byte-verified before publication.

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/retention-granularity-20261024/snapshot.manifest.json \
  --output .artifacts/retention-granularity-original
```

The destination must be new. This reads local Git objects, never fetches or
executes anything, and restores bytes rather than platform-specific file modes.
In a shallow clone, explicitly fetch the recorded revision first. Reconstruct the
recorded source checkout and dependencies before executing old scripts; scripts
may reference additional campaigns already retired in earlier migrations. Use
those campaigns' recovery inventories too. A normal numerical replay needs none
of this historical source or environment material.

## Retired experiment entry points

These standalone issue-specific drivers have no maintained runner, CI, test or
current-documentation caller. Their exact original bytes are in the same snapshot;
their numerical measurements and independent scientific tests remain retained.
Use the [current benchmark guide](../../README.md) to select new measurements.

- `benchmarks/issue172_r2scan3c_endpoint.py`
- `benchmarks/issue351_cpu_st_retirement.py`
- `benchmarks/issue41_component_ledger.py`
- `benchmarks/issue471_cpu_autotune.py`
- `benchmarks/issue52_current_fock_profile.py`
- `benchmarks/tensor_streaming_783.py`
- `benchmarks/ri_mp2_cuda_endpoint.py`
- `benchmarks/readme_pbe0_phased.py`
- `benchmarks/becke_phased_probe.py`
- `benchmarks/becke_phased_probe.cu`

## Validation scope

- All 245 consolidated records match the originals after whitespace-only removal.
- All 464 original snapshot files restore with the recorded SHA-256 and length.
- Both CUDA schedule qualification reports recompute from retained arrays and timings.
- Direct phase/moved replay produces the identical original observable-admission report.
- Ordinary scientific fixtures and binary array inputs are unchanged.

Full historical-clone storage is unchanged. These savings concern current tracked
checkout bytes and file count only.
