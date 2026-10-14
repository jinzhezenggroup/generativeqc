# Decision: reuse verified same-build CPU derivative source text in the audit

Status: implemented
Date: 2026-10-14

## Problem

Compiler-heavy job [115726215592](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38525941059/job/115726215592)
at synthetic merge `819253eda13eee9d274102a4cd0fea738e398c44` spent
302.19 seconds in the repository derivative AOT audit. The same mandatory job
had already generated and compiled the full 46-unit CPU stationary inventory.
The audit regenerated that identical source inventory in another Python process.

On master `5bfda9d873c546bd5396088deeb0e9ab9cfb05ac`, source-only attribution
of the unchanged audit took 47.45 seconds locally: 45.16 seconds for 46 CPU
translation units containing 362 primitive requests and 23,068,887 bytes,
2.25 seconds for the 34 SR/LR programs, and about 0.009 seconds for the CUDA
header/package inventory that this particular test asserts. These are local
attribution results, not a prediction of hosted whole-shard savings.

## Decision

The normal build generator still unconditionally calls the original full CPU
emitter and writes all 46 source units. After those writes it publishes a
content-hashed receipt beside the outputs. The receipt is build-local metadata,
outside the generated scientific source text and package identities.
The helper's code is included in the existing native source-identity manifest,
and the normal codegen runner records its import in the build depfile. Generated
receipt bytes are not source inputs; changing helper code remains build-bound.
The producer compares and writes explicit UTF-8 bytes, retaining canonical LF
source on every platform. Existing CRLF outputs are normalized once rather than
being accepted by universal-newline text comparison and then failing the receipt.

The audit optionally reads that same-build source directory. It accepts a
handoff only when the current compiler Python/JSON input inventory, producer,
receipt helper, Python interpreter version, domain, shard count and byte limits
match. The entire compiler module/data inventory is hashed, including nested
files and additions/deletions, rather than maintaining a fragile list of leaf
imports. The inspected primitive emitter's transitive generation dependencies
are compiler-owned; emitted native header names are strings, not file reads.
Any future external generation input must be included before it can safely
participate in this handoff.

Every source unit is read into a bounded immutable string snapshot and checked
for expected shard name/order, bytes and SHA256. Source identities are checked
again before publishing and after reading. Missing or stale receipts retain the
original full-generation fallback. Present corrupt, partial, reordered,
truncated, symlinked or out-of-budget outputs fail closed. This is ordinary
trusted local-build integrity, not an authenticity signature against a malicious
writer that can replace both sources and receipt.

Only the existing repository audit test opts into the already supplied
`GENERATIVEQC_BUILD_DIR`. Standalone tests without that environment, old builds
without receipts, and changed-source builds still execute full generation.
The complete audit payload and all existing assertions remain unchanged. SR/LR
generation, CUDA metadata generation, numerical tests, compiler caches and CI
selection are unchanged.

## Rejected alternatives

- Replacing CPU generation with dummy data in the repository test would remove
  its standalone all-source smoke contract, even though CI also builds it.
- Trusting filenames, mtimes, sizes, a native library's mere presence, or an
  unverified build directory would admit stale source bytes.
- Dropping the stationary inventory's repeated source-identity call is a poor
  first target: source-only measurement found its six repeated emissions cost
  about 0.79 seconds locally, while initial primitives and shared wrapper
  lowering dominated. That test is unchanged.
- A persistent Actions cache or a new global source cache is unnecessary for
  a same-job build-to-test handoff and would widen the trust/storage surface.

## Evidence

- Original and updated producers emit all 46 CPU units byte-for-byte identically
  (23,068,887 bytes); no native compilation was needed for that comparison.
- The complete original-generation and verified-handoff audit dictionaries are
  equal, including statistics, provenance, identities and CUDA inventory.
- A verified 46-unit source read took 0.073 seconds locally. The complete cached
  audit took 2.46 seconds; a fresh-process invocation of the exact repository
  test passed in 2.64 seconds, with a 2.35-second call phase. This is one local
  source-only comparison on Python 3.13.5, not hosted CI or endpoint evidence.
- Focused suites: 81 passed, one native-library export test skipped because no
  native library was supplied, and the heavy repository test separately passed
  against the real generated handoff. No full native/GPU build was run.
- Negative controls cover input rewrites preserving size/mtime, nested JSON
  changes, module additions/deletions, producer changes, absent/stale receipts,
  malformed metadata, output deletion/modification/truncation/reordering,
  symlinks, byte limits and interrupted lowering/output publication.
- The original repository test rejects injected wrong CUDA target, missing SR
  family, empty generated header and missing production provenance. Uncached or
  stale-handoff audit calls still surface injected CPU generation failures.

## Revisit when

If source generation gains non-compiler inputs, account for those explicitly or
retain generation instead of reusing. If CI stops building this inventory first,
the audit naturally returns to its original cost through the missing-receipt
fallback. Hosted same-source runs must establish the actual shard-level benefit.
