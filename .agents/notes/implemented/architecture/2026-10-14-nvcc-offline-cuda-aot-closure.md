# Decision: explicit offline NVCC 12.9 object closure

Status: implemented; delivery CI and independent review pending
Date: 2026-10-14

## Problem and decision

The CPU collector cannot qualify CUDA objects. Extend the canonical closure
owner with a separate manual CUDA entrance and caller snapshot/result types;
reuse recipe hashing, CacheClosure, finite compiler processes and durable store.
Preserve CPU entrypoints and identities. No production option/cache/schema,
runtime NVCC, packaging selector or scientific formula changes.

Bound NVCC to x86_64 Linux ELF 12.9, native system G++, whole-program objects,
sm_80 and narrow ordered flags. Observe dependencies/preprocessing and diagnostic
jobs; never execute dryrun output. Bind profile, tool selections, header bytes,
source/cwd/environment and complete asserted immutable transitive input coverage.
Command salt transports otherwise opaque closure identity to compiler caches.

## Rejected alternatives and invariants

Depfiles/ldd/guessed companion lists do not prove closure completeness. No new
cache owner or retroactive qualification of old objects. Unknown jobs/config,
unsupported path spellings and mutation/timeouts fail closed. Reusable pre/post
keys and a fresh source-matched object are required; caller owns the immutable
namespace snapshot, including unseen/absent headers and restored-between-probes
mutation. Ordinary runtime and installed bundle work remain separate owners.

## Evidence

Initial base 8f72a935a6cefc45473806df287066c8795bc039; fast-forwarded after
no-overlap audit to eae30bcdba6209238008344f458894fe91ab3ff4. b877 is the sole
source writer; PR ownership transfers before any subsequent PR mutation.
Initial local system Python lacked pytest; isolated ignored uv environment used.
Initial default pytest temp directory failed permission checks; explicit ignored
basetemp used. Initial CUDA fixture normalized Windows slash spelling and failed
source membership; fixture corrected without weakening production path checks.
Final Windows controls/legacy regression: 107 passed, 33 platform skips.
Cache/store/boundary checks: 62 passed. Compiler structure: 539 modules, zero
dependency errors. The Windows public-docstring scanner reports baseline
backslash path/contract classification errors, with no closure-module finding;
Linux required CI remains authoritative for that whole-repository gate.

Dedicated finite CPU Notebook i1123-nvcc129-1014: 2 CPU/8 GiB, official
pytorch:25.06-py3:25.06, NVCC 12.9.86, system G++ 13.3.0, verified sccache 0.16.0.
Requested priority 10, observed HIGH 6; project live priority is HIGH and CLI
caps against live project priority_name, but CLI read surfaces do not independently
expose that numeric cap. Preserve this limitation, not a claim of independently
proved maximal priority. Actual start 2026-10-14 08:09:48 Asia/Shanghai; configured
120-minute platform timer, status uptime+remaining=7200 seconds, derived deadline
10:09:48. No renewal, GPU allocation or other actor resource use.

Linux source-v7 LF archive 88f27660890bf45339423e7818d3099395cc9b3fa19730f17dc18143301c0dab:
49 passed, zero skips, including real NVCC pre/object/post key equality, actual
ELF/CUDA object, independent raw MD versus M dependencies, canonical store
publish/delete-original/read/hash, salt invalidation, ambiguous-header decoys,
symlink selection drift and finite FIFO timeout. Linux applicable legacy controls:
121 passed, 7 skipped, 12 real Clang cases deselected (Clang not installed).
Do not turn these counts into new Clang, GPU science, endpoint or installed gates.

Stock diagnostic fields and relative device job names were observed and admitted
explicitly after the initial unsupported model. NVCC emits `closure :`, unlike
the preserved GCC parser; normalize only the fixed NVCC target. `-M -Xcompiler=-H`
sees both host/device user branches and guards original path spellings, but only
a subset of NVCC's implicit runtime headers. Complete namespace coverage remains
the producer assertion. Non-blocking regular-file descriptors avoid FIFO swaps
blocking hash/macro/magic reads outside the shared deadline.

Retained failures include missing local pytest, default temp permissions, Windows
fixture slash normalization, initial Ubuntu PEP 668 install failure, initial MT
target INCOMPLETE, missing launcher on sanitized PATH, independent MD target
parsing, noncanonical compiler path in the test handoff, and an incorrect test
assumption that all internal host stages were cacheable. None is rewritten as PASS.
The SSH helper initially printed a credentialized transport URL to stderr;
subsequent calls filter URL/token lines and no such URL is included here or in
the source/evidence archive.

Actual sccache driver/device replay hits are retained; salt changes miss NVCC,
cudafe and cicc, while ptxas may hit identical PTX. One internal host job remains
non-cacheable even for the ordinary source. Do not claim a whole-object compiler
cache hit or force invalidation of semantically unchanged ptxas work. The ordinary
opaque config changes key/miss behavior but leaves identical object bytes;
the actual assembler payload changes both key and object bytes:
2bbc67b37baf7fe1951ae625c952fe9a1ef093107af767b9b1cda323b336cd74 ->
15ad9010784bb3731cd9603a4237a3f088fc2b049c27f641270c4a6a44af5570.
Its source SHA-256 is 14e0fb47c9471d58e5f43486fc2ac3222f82c6a092df481b4bc4095be10336e8;
full object receipt SHA-256 e2121620c5f3bb5bfb0f1152742f522e0c2fa15ca7a54875876306959fbdf058.

Ignored local archive `.cache/nvcc1123-evidence-v7.tar.gz` and remote directory
`/inspire/ssd/project/chemicalreaction/czxs25220150/i1123-nvcc129-20261014` retain
snapshots, raw observations, first failures, receipts, counters and actual stored
objects. Archive SHA-256 a88790c9bf1eabf844b88181118ce022c38d4604abc3eb826580cf339bb8a18e
was verified after download. The ordinary replay receipt subsequently gains full
source/recipe/closure metadata in source-v8 and is rerun as a bounded delta;
collector bytes remain identical to source-v7. Required CI, formal exact-head
independent APPROVED and protected merge/delivered blobs remain delivery gates.

Metadata delta: source-v8 LF archive
5ac5acd370122dea914ac76ae30e6f7039fc532180ce530222ef3fcf5a963a25;
real cacheable replay rerun 1 passed, 48 deselected. Full source/recipe/closure
receipt SHA-256 848bd63a77d415f80354292420c403110a005641fa20ff5c80d043864e077ae9,
ordinary source SHA-256 c6bf08a2695b5c3d07d8b783c6e533a57a3ba671e50e22293003151e1d782459.
Expanded archive `.cache/nvcc1123-evidence-v8.tar.gz`, SHA-256
319a79a28d4d7adeb6b2e98935099822860b8eaab97cff28c2296ed921dac9f4,
retains v7 and delta records separately. Collector LF bytes SHA-256
a585735a3dd60dd8b11e281e6f5dacd6e401084d63e51cb60d9d255d32db7814
are unchanged across both qualification snapshots.

## Review repair: alternate host sysroot

The initial PR head 35f6ddfd361b80798abf80d2dd83b7a1b128d0a4 admitted
GNU host diagnostic jobs with `--sysroot=/opt/sdk` or `--sysroot /opt/sdk`,
contrary to the bounded namespace contract. Failure injection reproduced two
incorrect reusable closures; the existing `-isysroot` control already rejected
the alternate namespace. Reject both GNU spellings in the diagnostic job model,
with three executable failure-injection cases. CPU entrypoints remain unchanged.
The PR was returned to Draft, outside the queue with auto-merge disabled; the old
exact-head approval does not qualify this repair or clear the later finding.

Local repair controls/store: 58 passed, seven explicit real-NVCC skips. Ruff
passed and compiler ownership checked 539 modules with zero errors. Fresh LF
source-v9 archive SHA-256
618195a61c369660318aa36369ef3a0c9723bba57f937f9dae035d42d8241ba7;
collector SHA-256
e7e5b3c2f893e3d164dc4b7045b05bc07d43a91e50ac8d792558fe652193681b.
The original dedicated CPU environment completed all 52 real tests with zero
skips, including fresh object/store receipts, pre/object/post identity, actual
driver/device cache replay and salt misses, opaque payload and path/FIFO controls.
The non-cacheable internal host stage and unchanged-PTX reuse limitations remain.
Downloaded ignored `.cache/nvcc1123-evidence-v9.tar.gz` SHA-256
611a6b65a9bb3c2b51bb9f505b64f92d0f69c06cb4517f481beb6c2bc9ed6f7e
matches the remote archive. Earlier v7/v8 evidence and failures remain retained;
new receipts qualify only source-v9, not relabeled historical objects. New-head
CI, independent delta approval and protected delivery remain required.

## References

- Refs #1123; CPU GCC #2218 and Clang #2444.
- `docs/developer/libxc_cuda_aot_closure.md`.
- NVIDIA CUDA Compiler Driver archive 12.9.1.
