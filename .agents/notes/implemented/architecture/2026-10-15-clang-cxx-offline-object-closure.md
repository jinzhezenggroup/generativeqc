# Decision: explicit native Clang C++20 offline object closure

Status: implemented
Date: 2026-10-15

## Problem and decision

The existing offline collector supports Clang C and G++ C++20 but rejects
Clang C++20. Extend the existing collector, manifest, CacheClosure and immutable
store contract through explicit compiler family and language selection. No new
cache, compiler-process, algebra or runtime admission owner is introduced.

Canonical clang++ often resolves to the same ELF file as clang. Preserve C++
driver mode explicitly after canonicalization; do not infer language/recipe
identity from physical executable equality or GCC-compatible predefined macros.
Require a single integrated C++20 cc1 object job, matching native target and
resource directory, disabled default configuration, full ABI macro output and
actual GCC installation/include-search selection. Repeat these observations
under the existing finite deadline and immutable caller snapshot contract.

## Rejected alternatives and invariants

- Removing only the language guard leaves the cc1 job fixed to C.
- Blindly reusing GCC diagnostic decoding assumes a printer protocol. The
  observed Clang C++ printer is LLVM `sys::printArg`: all job arguments are
  quoted and quote/backslash/dollar are escaped once. A narrow separate decoder
  follows that protocol; Clang C's existing decoder/recipe remains unchanged.
  Independently compare actual compiler depfiles and reject ambiguous
  backslash/control paths instead of normalizing identities.
- Neither the selected GCC directory nor ldd/header discovery proves snapshot
  completeness. The producer must assert unchanged executable, loaded library,
  resource, C++ standard-library/ABI/search namespace and opaque inputs.
- C/GCC/G++/NVCC recipes and default rejection remain unchanged. The new explicit
  Clang C++ identity does not establish link closure, production packaging,
  installed/runtime admission, scientific/CUDA or endpoint performance gates.

## Verification contract

`tests/python/test_bulk_aot_clangxx_closure.py` covers explicit language/standard,
ABI/search/selection drift, incomplete manifests, unsupported flags/PCH, native
standard-library/template C++20 object compilation, matching pre/post closures,
immutable publication, deletion and retrieval of the original object, independent
Decimal execution, compiler depfile/path controls and opaque-input/cache controls.
Measured Linux results are recorded only after execution on a fixed LF snapshot.
The current API contract is in `docs/developer/libxc_cpu_aot_closure.md`.

Refs #1123. This bounded object slice leaves the parent acceptance scope open.

## Investigation evidence

The private Ubuntu 22.04 image initially lacked canonical Clang. Ubuntu Clang
14 and 15 reject `--no-default-config` on actual preprocessing even though
`--version` prints a version successfully. Keep this negative evidence; version
output alone does not verify option compatibility. LLVM 18.1.8 packages were
retrieved from the official apt repository, verified against SHA-256 fields in
the signed apt index, and installed only in the task-owned CPU Notebook.

At LF tree `05b2f8c010fd9e17cbbb1e40a9239145aa0517dc`, real Linux tests recorded
171 passed / 2 failed / zero skips. Object-store/Decimal execution passed; the
failures were a test socket pathname exceeding the Unix socket limit and POSIX
shlex preserving Clang's dollar escape. A short test-owned socket isolates the
ordinary cache counters while retaining the persistent disk cache. Raw plans
confirm dollar and real backslash-dollar spellings; malformed grammar and
backslash identity controls supplement the actual depfile comparisons.

The decoder is grounded in LLVM 18
[`Command::Print`](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/clang/lib/Driver/Job.cpp)
and [`sys::printArg`](https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/lib/Support/Program.cpp).
These investigation results are not acceptance of the later repaired snapshot.

## Source-matched Linux qualification

The task-owned CPU Notebook `i1123-clangxx-1015` runs Ubuntu 22.04/x86_64,
Python 3.11.16, canonical Clang 18.1.8 and verified sccache 0.16.0. Clang selects
the GCC 12 standard-library installation; the separate canonical G++ control is
Ubuntu GCC 11.4. These are distinct observed identities, not the image's advertised
additional compiler version. Requested priority 10 was capped to 6/HIGH for
4 CPU / 16 GiB in CPU资源空间 / chemicalreaction. The Notebook remains retained
under the user's current resource authorization; platform reclamation still applies.

All 11,054 regular source files were checked before and after execution against
the complete size/SHA-256 manifest. The snapshot uses a raw Git LF starting
archive of tree `eed7a9cd389714e8fde2d7f5370c9b7d08339110`, built on base commit
`540ec82011232606374747727cd30fd7e1f93970`, with a raw Git tree delta, not
Windows working-copy bytes. Qualified tree:
`2e670e2269917e02479b74fa4dfdd0960d2233a5`; base archive SHA-256:
`251632abf15e2b0c68c620105b828671f643821c93eedf6c2a8ab4efeb5e1f18`;
delta archive SHA-256:
`bb24ad5552ab920128b5a6c73b8fa37f2c6316f2ecc682ac83cc38bd2170c223`.
Qualified collector/test Git blobs are respectively
`7d17ba3788e8e93f15f7e6d019a547e57eabe771` and
`3d566bb2b60f42d6f9de5ab9b078a9b3773a2951`; their raw SHA-256 values are
`b22028e7590905fb23668e30bb5f757e52b39160189cbe6b6ff371828604b823` and
`a76fab752f91d5c9c6438e0b92c2c77cc70c7f73eed7ec715935903bdb978e66`.
Later edits to this note record results; they do not alter qualified executable
inputs. No source identity is inferred from a working-copy checksum.

- Related closure/cache/store suite: **186 passed, zero skips**, at the repaired
  collector snapshot. Same-path/persistent-cache repeat: **44 passed, zero skips**.
  A subsequent test-only correction explicitly preserves C++ driver mode and
  disabled default configuration during the separate execution link; its final
  qualified test blob above reran **44 passed, zero skips** at the same path/cache.
- One-off baseline comparison against the base commit: **3 passed**, identical
  complete recipe, closure payload and key for GCC C, Clang C and GCC C++.
  This is migration evidence, not a permanent source-fingerprint test.
- Windows related suite: **142 passed / 44 skipped**; separate unchanged NVCC
  host-contract controls: **62 passed / 7 skipped**. Skips remain unavailable
  evidence for their native domains. Compiler structure: 540 modules, zero errors;
  changed Python files pass Ruff and formatting checks.
- Standard-library/concept/template C++20 source is collected, compiled through
  the verified cache launcher with independent `-MD` dependency equality,
  recollected with the same key, published, deleted at its original object path,
  retrieved, linked and loaded. Seven Decimal rows pass, including negative and
  fractional inputs. Actual source, compiler, stored-object and loaded-library
  identities are independently retained in `qualification.json`.
- Stored object SHA-256:
  `ba1423b2ad1bb3364cdd2cf2e8e422618a4b173416d1152426a8d5ec58e03918`.
  Actual final loaded library SHA-256:
  `3b3c2512c9134436eeab2cf67d46d479273a1cb7836bf1464317277ef6d422c4`.
- The ordinary compiler commands have test-owned sccache counter attribution:
  misses/hits progress **1/0 -> 1/1 -> 2/1** for first compile, identical replay
  and refreshed salt after a real manifest-covered opaque-input change. Changed
  assembler-only `.incbin` inputs also change the actual object bytes. No cache
  clearing, sloppiness or shared-server stop is used.
- Two actual negative compiler controls, omitting the salt or reusing its old
  value, return stale object bytes after the opaque input changes. Both independent
  payload oracles fail, as required; these invalid invocations are not qualified.
  The separate negative harness reports **2 passed** because it requires those
  failures. Its initial missing-fixture error is preserved separately.

Raw snapshots/manifests, commands, JUnit, first failures, cache statistics,
Decimal rows, actual objects/libraries and negative controls remain under
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/issue1123-clangxx-20261015`.
The ignored local archive `.cache/clangxx-evidence-final.tar.gz` has SHA-256
`bb9c3e96d6e7deeb13808fae5881f601be78bac44e1d9efeb6e69bdb66b3adc2`.
Remote and client execution exits are both zero for accepted runs. Earlier
transport CR errors, Clang 14/15 incompatibility, and the 150-pass/1-fail/19-error
and 171-pass/2-fail runs are not relabeled as acceptance. The evidence establishes
only this offline toolchain object slice, not link closure, installed bundles,
runtime/scientific/CUDA endpoints or completion of parent #1123.

## Review correction: transient ABI state and independent C++ entry

At `fe801e870c5e6fc463a5658bba013f68a640ca0f`, four real Clang controls define
`_GLIBCXX_USE_CXX11_ABI=0`, include `<string>` directly or through a wrapper,
then undefine or restore the macro. Actual `nm -C` object symbols have the old
ABI while the final `-dM` table hides the zero. All four rejection assertions
fail on that head; the old collector incorrectly admits the translation unit.
The preceding accepted-domain measurements do not cover this counterexample.

Clang C++ collection now observes active `-E -dD` macro-definition history,
rejects unsupported definitions even when later restored, and binds/rechecks
its digest alongside the final macro table. An inactive-branch control retains
the default ABI and independently checks the new-ABI symbol. This remains a
conservative native object domain, not alternate-ABI support.

The C++ test producer now resolves `/usr/bin/clang++` and its resource/loader
manifest independently. Physical equality with the C entry is not required by
the production contract and must not be a fixture prerequisite. The explicit
C/C++ comparison uses each canonical entry and a shared union snapshot.
These changes require fresh source-matched Linux qualification, required CI and
independent approval at the new exact head; earlier results/approval cannot
substitute for this delta.

The history check admits only the native object-like ABI definition `1`.
An additional four real controls use a function-like ABI macro: its bare use in
libstdc++ `#if` expressions selects zero, even with a body of `1`. These also
produce old-ABI symbols and bypass a history regex that assumes whitespace after
the macro name. The final guard checks the macro-name boundary, rejecting both
object-like zero and function-like definitions. The failed intermediate four
controls and an earlier harness parameter-substitution mistake remain separate
from the final acceptance records.

### Source-matched review-delta qualification

LF tree `64983cd9f20c785dbedc37317bf48fe08b778a38`, delta archive SHA-256
`36560f66eda0111926cf4a6ad25efacc6cd80dae720c968e5e2726e936c145cf`,
uses the same starting archive above. All 11,054 source files again match the
complete manifest before/after the full suite and subsequent controls.
Qualified collector/test blobs are
`13f9a0bf9ab3a79cb147ff9b32504207badbd1ac` and
`f280aaf72609a03371248cce9ca853f3989d341f` (raw SHA-256 respectively
`b0be6480be500e7c87f826a0df47ce202adc563b913f68ede1e6d1ff31be01ef` and
`e8248136404a78b2ba3141a620b61d8443e9bccfd14d0f6c5ae7342b46f9d167`).
Only later result-recording edits to this note differ from that snapshot.

- Full related closure/cache/store/boundary/publication suite: **224 passed,
  zero skips**. Same-path/persistent-cache C++ repeat: **54 passed, zero skips**.
  Eight real transient ABI negatives reject; the inactive-branch native-ABI
  control compiles with the independently observed new-ABI symbol.
- With canonical C entry at Clang 15 and C++ at Clang 18: **53 passed, zero
  skips, one explicitly deselected C/C++ cross-comparison**. This proves the
  C++ fixture no longer depends on physical equality or C driver capability.
  The C entry is restored to Clang 18 and independently checked afterward.
- One-off preserved GCC C/Clang C/G++ recipe/payload/key comparison: **3 passed**.
  Actual stale-byte omitted/old-salt negative controls: **2 passed** (each requires
  the independent payload oracle failure). Test-owned cache counts retain
  misses/hits **1/0 -> 1/1 -> 2/1**. Stored object, actual C++-linked library and
  all seven Decimal rows retain the identities above under the new recipe key.
- Windows related checks: **171 passed / 53 skipped**; fresh NVCC host-contract
  controls: **62 passed / 7 skipped**. Native skips are not promoted into evidence.
  Compiler structure and changed-file Ruff/format/diff checks pass.

The source-matched review journal is the task directory's `review-final/`.
Downloaded ignored `.cache/pr2525-review-evidence.tar.gz` SHA-256:
`9586ed60463902d47b944a3cdf3cfdf656e8c5a5ea7b7d12bbc4a9f65e3464a4`.
It includes the old evidence, red controls and intermediate snapshots; current
raw stored-object/library bytes are rehashed after download without extracting
or following links. Accepted remote and client execution exits are zero.
This toolchain qualification remains separate from latest-head CI, independent
formal approval, normal protected merge and parent-issue completion.
