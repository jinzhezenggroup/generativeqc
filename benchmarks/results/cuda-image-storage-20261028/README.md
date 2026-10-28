# CUDA image storage qualification

This change prunes resolved relocatable inputs from native stationary archives
and gives standalone stationary DSOs private device-link owners. Public shared
libraries retain whole host archives, exports and executable images. Original
archives remain available. Additional fatbinary compression is **opt-in**,
`GENERATIVEQC_COMPRESS_CUDA_FATBINS=OFF` by default, because measured cold DF
endpoints regressed.

## Source and measurement boundaries

The initial campaign uses `b61dd15e8d8e7d4f98fff57b23419d087050d94f` plus its
retained overlay. During qualification, #2835 and #2852 merged. The final
integration uses `99c1193b4365fe80e63cdeb49441373ab7e7bb56` plus this change; do
not pool measurements across those source identities. Both native configurations
use SM120, CUDA 12.9.1/GCC 11.4, native stationary enabled and 73 selected automatic
XC registrations, with optional shell/Hessian/standalone bundles disabled in the
main library. A separate build qualifies all 32 curated standalone libraries.
This is a complete loadable native library in that configuration, not every
possible architecture/functional/optional release bundle.

Builds use verified sccache 0.16.0 and retain cache statistics. GPU work uses
finite Slurm allocations on node2's PRO6000, preserving assigned visibility.
No release, tag or distribution was published.

## Size

Historical source, decimal bytes:

| Artifact | Original | Pruning only | Pruning + compression |
| --- | ---: | ---: | ---: |
| Complete native library, including native stationary | 761,557,656 | 660,603,544 | 410,591,896 |
| 32 standalone stationary libraries | 557,808,568 | 280,618,064 | 280,618,064 |
| 80 DF objects | 293,984,736 | 293,984,736 | 86,426,016 |

Standalone pruning saves **49.7%**. Compression alone did not shrink the original
standalone libraries. For the native library the historical combined saving is
46.1%, but that is an **opt-in** result. It is not directly comparable to #2852's
537.7 MB configuration, which excluded native stationary.

The retained attestations match decoded executable text sections for all 80 DF
objects and all 32 standalone libraries across their respective comparisons.
All **18,666** native sections match between the historical pruned and compressed
libraries. The historical original retains only its size and binary hash, so
replay cannot establish its instruction match against the pruned library.
The final mainline comparison below retains both
instruction attestations. These comparisons check instruction bytes and
multiplicity, not just kernel totals. Public standalone symbols match, all 32
manifests pass the native cubin audit, and all 32 libraries load/register with
`CUDA_MODULE_LOADING=EAGER`.

On the final mainline source, the matched **default** comparison is
**589,663,288 -> 566,705,208 bytes**, saving **22,958,080 bytes (3.9%)**.
All **13,941** executable text sections match. The baseline is a host relink of
the exact same objects with the original stationary archive; the candidate uses
its pruned host copy. #2835 already reduced the native stationary code inventory,
so neither its gains nor the older baseline are attributed to this change.

## Scientific and timing gates

Each process includes Calculator/preparation plus first E+F in the cold timer,
three frozen-seed warm calls, and one moved-geometry call. Independent PySCF
references run in separate processes. Cohorts cover Direct FP64, DF FP64, public
DF Fast, PBE Direct and PBE DF, each at 24/96 AOs. Module loading is LAZY.
Per-cohort cold/warm/moved acceptance is candidate/baseline median-of-process-
medians <= 1.02. Paired E/F gates are 1e-9 Eh / 1e-8 Eh/bohr; independent E is
1e-7 Eh and strict/Fast force is 1e-6/1e-5 Eh/bohr. Iterations, Fock counts and
reported mixed/retry work must match. Missing counters remain missing, not zero.

The first **500 calls** fail the cold timing gate for df-96 (1.0367), dffast-96
(1.0470) and pbedf-96 (1.0239). All numerical/work gates pass. Compression is
therefore retained opt-in even though all warm ratios passed.

The pruning-only follow-up retains **575 calls**, including a third compressed
arm for all repetitions of those three cohorts. Compression/pruning-only cold
ratios reproduce at 1.0414/1.0367/1.0252. The pruning-only comparison itself has
two timing failures: direct-24 cold 1.0232 and pbe-96 warm 1.0414. The first cohort
overlaps the separate standalone load allocation; that is recorded interference,
not justification to discard samples. Both failures remain in the replay output.
The final mainline campaign uses ten balanced randomized process pairs per cohort
and an exclusive Slurm allocation, after this task's compilation/other GPU work
ends, with unchanged numerical/work and 1.02 timing gates.

All **1,000 final calls pass**, with maximum endpoint ratio **1.0169129585**.
Maximum paired energy/force differences are **4.55e-13 Eh / 5.41e-11 Eh/bohr**;
maximum independent errors are **7.74e-12 Eh / 7.34e-8 Eh/bohr**. Reported work
matches. The mainline scientific suite also repeats the three native, 25
numerical/lifecycle and 12 curated stationary tests successfully (four existing
PBE50 skips). The historical standalone build with compression OFF produces
exactly the same 32 binary hashes as the qualified pruned/ON build and passes
its own EAGER registration witness.

## Other validation and rejected experiments

The pre-mainline candidate passes three native tests (runtime, independent DF
derivatives, stationary binding), 25 HF/DF/Direct/GFN2/FCI numerical/lifecycle tests,
and 12 curated stationary PySCF routes (four existing unsupported PBE50 CUDA
cases skip). The final host/compiler suite passes 878 tests; the earlier exact-
internal-target-name assertion failure is retained and replaced by executable
Ninja pool ownership checks. Final CUDA archive contracts pass 15 tests including
real/PTX/combined/multi-SM/multi-config, incremental rebuild, OFF, forced PTX JIT,
shared exports/execution and provider-free wheel imports. Documentation builds
with Sphinx warnings treated as errors.

DF noinline samples grew; modern compression-size samples were smaller but
require the newer driver format. Speed/balance samples did not improve storage.
All negative probes and failed prerequisite/test diagnostics are retained in
the original Git receipt; recover it as described below. The failed timing
observations are replayed from that same authenticated original receipt.
There is **no controlled whole-project compilation-time improvement claim**;
concurrent cache activity makes the recorded build durations unsuitable for one.

## Replay and retained data

```bash
python benchmarks/results/cuda-image-storage-20261028/reduce.py
```

`summary.json` retains every campaign's call count, failed timing ratios, maximum
ratio and paired/independent numerical errors, with the unchanged work and
acceptance gates. Default replay reads the original full `receipts.json.xz`
directly from existing Git objects through the repository's stock authenticated
reader. It verifies the manifest's original commit/path, byte count, SHA256 and
Git blob identity before decompressing. No network, generated cache, alternate
storage or new recovery framework is used.

The original receipt contains all 2,075 campaign observations and independent
references, source overlays and recipes, protocols, artifact identities,
instruction census, complete package-load witnesses and auxiliary
build/cache/test/failure/pilot records. Source recipes now live in that original
receipt rather than being duplicated in the current checkout. No failed timing
sample or original byte is removed from history.

Replay requires the pinned original Git object. It belongs to the original PR
history and is not an ancestor of the squash-merged mainline, so even
`fetch-depth: 0` does not supply it. The core-b CI job explicitly fetches the
manifest's immutable revision and authenticates its bytes before running the
offline replay tests. Other checkouts must likewise explicitly fetch/retain
that revision; missing history continues to fail closed. To materialize the exact 209,852 original bytes separately:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/cuda-image-storage-20261028/receipts.json.xz \
  --manifest benchmarks/results/cuda-image-storage-20261028/original-receipt.snapshot.json
```

The output is under ignored `.artifacts/retention-restore/`. The immutable source
revision and original identities are recorded only in the stock snapshot manifest.

Per-process energy/force
columns use a lossless little-endian IEEE-754 XOR encoding; `decode_samples()`
restores every bit. A one-off comparison checks restoration against the original
JSON observations. Paths are execution provenance and need adaptation elsewhere.

Instruction inventories retain canonical `(section bytes, SHA256, multiplicity)`
multiset digests/counts plus binary identities and collector recipes. Replay
checks these collection attestations; it does not decode absent binaries. Full
raw inventories and successful compiler warning streams remain ignored local
artifacts, with hashes and action/diagnostic summaries retained here. No cubins
or shared libraries are committed. The compact current summary and offline replay fit the existing
80-MiB aggregate evidence policy without changing historical evidence or budgets.
