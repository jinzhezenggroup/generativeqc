# H100 PBE0 Direct48: first cold pilot failed, finite stage spent

Refs #2731. This records the historical source-matched initial pilot, **not a
current-master numerical defect, an RTX 5090 quota failure, or a completed PBE0
benchmark**. The measured native source is
`cbd4d8074de35105717c8aac7cd1f86babbe90ec`; core DSO SHA-256 is
`5645dacffe534cff08bd5cc042eefc9ab1f5335cced0f16220387eefd61a38df`.
Publication may use a newer base without changing that scientific cohort.

| Arm / phase | Scientific requests | Complete E/F | Outcome |
| --- | ---: | ---: | --- |
| Standard / cold | 1 | 0 | FAILED: missing packaged stationary CUDA artifact for `pbe0_rks_spd` |
| Fast | 0 attempted | 0 | NOT_RUN: stopped after Standard failure |
| GPU4PySCF | 0 attempted | 0 | NOT_RUN: stopped after Standard failure |

Energy, forces and endpoint timing are **UNAVAILABLE**, not zero. No partial E/F
frame returned either. Internal computation may have occurred before the force
failure; SCF/Fock work, precision/fallback, peak memory and billing are unavailable.
No numerical-error, precision, OOM, speedup or formal-statistical conclusion is made.
The finite stage is **SPENT**: the two unattempted arms are not an unused balance.
DF, water96, formal repetitions, the full144-call plan and parent closure remain
incomplete. This PR does not authorize another attempt, SDK build or artifact copy.

The unique Job was `h100-pbe0-direct48-cold-20261024-e961-v1`, created as raw STRING
`1792947534000`, FAILED at raw STRING `1792967890000` (2026-10-26 06:38:10 UTC+8).
One release `ad95a9f8a5e3206b4957f58c6187c45e3c0b468f4d3acfd94b28fb18b3b6babc`
was published and consumed. Source/environment readbacks matched before/after:
exact environment identity `9a839b321b4fef6499530df9bc19ca3dbdd024a0f7c714509ddcf886187dcc44`,
48,416 regular files, eight links,19 external dependencies; HF source inventory
`9cd7117da928ff96eaf6524a10c588567934b6b7e90cd2b3c78328558e7c211b`.
H100SM90/CUDA0/PCI3B identity and cgroup matched before/after. This is prerequisite
identity evidence, not successful PBE0 E/F qualification. ABI scalar was not
explicitly recorded and remains unavailable.

Root177.031s, worker command44.019s, supervisor before export99.938s and engineering
charge279.938s have different scopes and must not be added. There is no complete
endpoint timer. Actual guard/argv/birth/nonce/terminal cleanup passed, with no
remaining owned children and no retry. Six retained API polls were RUNNING;
the separate final CLI original supplies FAILED. Platform logs retain all four
entries without truncation and32 events. Its captured project label contains
replacement characters; the reader uses the exact ASCII Job/status/time fields.

## Exact missing companion and frozen-source diagnosis

The loader directory is the resolved core library's parent:

`/inspire/ssd/project/chemicalreaction/czxs25220150/h100-paper-environment-20261024/build/hf-sm90-portable-attempt1`

Both exact required pathnames were absent in Root's subsequent bounded read-only
`lstat`, with parent inode/mtime unchanged:

- `libgenerativeqc_stationary_pbe0_rks_spd.so`
- `generativeqc_stationary_pbe0_rks_spd.json`

The force exception itself means library **or** manifest unavailable; the separate
two-path original strengthens it to both absent at that later observation. The
capsule retains both observations without rewriting the earlier uncertainty.

At frozen `cbd4`, `python/generativeqc/batch.py:798-826` selects a catalog-covered
all-electron packaged force, uses `compiler=None`, and passes the core parent as
`aot_directory`. `_stationary_cuda.py:2549-2584` and3527-3563 select the packaged
loader. `generativeqc_compiler/method/stationary_cuda.py:1051-1140,1410-1548` derives
`pbe0_rks_spd` from exact profile/spin/plan and requires that DSO/JSON pair.
Coverage is intentionally different from artifact availability: no silent JIT
fallback, filename rewrite or cache repair follows a missing declared package.

The eventual artifact digest is constructed only after manifest validation from
the identity and binary SHA. It binds source/contract/plan/weights, partition3,
SPD domain, SM/code objects and manifest integrity. It is unavailable here because
the loader failed before reading the manifest. An old native-cache object, a PBE
companion, a different source or sm120 binary is not a qualified replacement.

The frozen CMake graph explicitly separates core/native bindings from Python
stationary shared products. The producer targets are
`generativeqc_stationary_pbe0_rks_spd` and
`generativeqc_stationary_pbe0_rks_spd_manifest` (an independent ALL target).
`tools/generate_stationary_force_aot.py` emits the SPD wrapper; the manifest writer
binds actual primitive/source/binary/plan/target/precision provenance. Both DSO and
JSON install to `CMAKE_INSTALL_LIBDIR`. The archived HF SDK's actual successful
command requested only `--target generativeqc`; it did not request the independent
Python companion product. Core/native PBE0 source generation does not substitute
for that DSO/JSON pair. This establishes a historical deployment-closure gap, not
an option-OFF claim or a current production-PR regression.

Root has separately requested an H100 packaging/native-branch disposition from
the production owners in [the #2731 comment](https://github.com/jinzhezenggroup/generativeqc/issues/2731#issuecomment-6408960658).
No changes to #2789/#2795 or their scientific evidence are included here.

## Lossless storage and offline replay

`originals.json` restores **76 terminal originals (861,717 raw bytes) plus eight
exact supplements**. Every delivered original retains its exact bytes and recorded
Linux mode. Only `scripts/expected-original-input-manifest-v1.json` is
**OMITTED_IDENTITY_ONLY**: SHA-256
`2e7518ffbe13a36e9dcb00004230a69a6e92340cbb7ba45805385010d3a73a38`,
9,451,229 bytes, mode0644. The original 77-file terminal index is preserved verbatim
as historical evidence; it does not make the omitted original restorable here.

The omitted census describes the closed `source/cbs-qualified`,
`source/hf-dft-cbd4` and `env` trees, 11 enumerated fixed inputs, 48,416 regular
files /5,931,857,284 bytes and eight symlinks. Its separate external descriptor
binds19 dependencies. The capsule records those domains, counts and identities.
The frozen historical preflight really consumed the full manifest; its exact
before/after PASS receipts and their equal environment identity remain delivered.
This compact publication checks those recorded receipts and deployed input
identities. It **does not rerun whole-environment validation**, reconstruct the
48,416-row census or qualify a new environment. The prior full85 publication remains
in Git history and frozen local backups; this reader does not restore the census.

All other84 originals remain byte-identical. Other unique bytes share one bounded
pool; five published fixed Git inputs are restored by revision/path/mode/blob/size
and SHA. The measured source remains `cbd4...`; selected driver/diagnosis files use
byte-identical blobs in the published ancestor `7522994...` only as recovery
anchors, not whole-tree or new scientific-source equivalence. `artifacts.json`
describes the remaining members. The two census fragments and their now-unused
large-file storage descriptor are absent; no storage limit or policy is raised.

The521-file /28,134,391-byte Task inventory includes binary cache data retained
remotely. Those bytes are **inventory-only here**, not77 additional raw originals,
not reconstructed binaries and not evidence of a cache hit or qualified repair.
The original compressed transport envelope is also not reproduced; its identifier
remains in the audit. This publication restores the collected originals themselves.

From a full local checkout containing the fixed published commits:

```powershell
python -B benchmarks/results/h100-pbe0-direct-df-20261024/reader.py --repo .
python -B benchmarks/results/h100-pbe0-direct-df-20261024/reader.py --repo . --restore .artifacts/pbe0-failed-pilot-restored
python -B -m unittest discover -s benchmarks/results/h100-pbe0-direct-df-20261024 -p test_reader.py -v
```

Replay is standard-library only and never imports restored workers, native/SDK,
CUDA, PySCF or an oracle. It checks byte recovery, release/plan/consumption,
request/return accounting, input modes, source/device receipts, actual guard
cleanup, API raw/parsed bindings, final status, exact two-path stat and summary.
Restore requires a fresh directory and never executes its contents. POSIX chmod
is applied only on POSIX; Windows restores retain the source modes as witnesses.
History is never fetched implicitly. A shallow checkout may need an explicit
history preparation step; ordinary small reader tests do not require historical
objects, network access, a solver or the full raw publication.
