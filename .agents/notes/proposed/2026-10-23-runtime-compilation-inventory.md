# Runtime compilation inventory: Windows source-byte follow-up

Status: proposed
Date: 2026-10-23

## Problem

While inspecting the bounded documentation slice for #2680 at
`bbd41a42764e8d22d15c1348f70a633961d4dcf5`, the existing fresh-process
semantic-source-cache test failed on Windows/Python 3.13.5. This is separate
from CUDA execution, compiler-cache availability and numerical acceptance.
No native compilation or GPU work was required to reproduce it.

`common/semantic_source_cache.py::cached_sources` records each generated
source's UTF-8 LF byte length. Its publication helper
`common/source_cache.py::cache_source` opens the file in text mode without
an explicit newline policy. On Windows that expands LF to CRLF. The manifest
hash is computed from the actual file and matches, but the recorded length
does not. A later fresh process rejects the entry in `load` with
`ValueError: source hash mismatch` (the message also covers length mismatch).

## Evidence

The unchanged executable contract is
`tests/python/test_stationary_semantic_cache.py::test_fresh_process_real_pbe0_sources_skip_ir_and_ad`.
It uses real PBE0 source emission but replaces binary object/link compilation.
The miss completes; the fresh-process hit fails. Setting `PYTHONUTF8=1`
removes locale-decoding errors but does not alter newline translation.

| Source | Manifest bytes | Actual bytes | CRLF pairs | Manifest/raw SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Primitive unit | 117211 | 119745 | 2534 | `a57d2306cfec2967f089ee2db68e85afda6f3ac72d06671704993950ab7f126a` |
| Stationary wrapper | 18267 | 18756 | 489 | `500066a10f44a8ca72abfd60ed7c5d38b512c6b739c825fc389bd7c3c875375b` |

Both actual-minus-recorded lengths exactly equal the number of CRLF pairs.
Normalizing those files to LF gives different hashes, so normalizing only
reads would not preserve their recorded binary witnesses.

## Proposed follow-up and boundaries

A separately owned change should establish deterministic source-byte
publication and an executable cross-platform cache round-trip contract,
including explicit handling of pre-existing translated entries. Reassess
downstream source/object identities before changing publication. Do not
silently rewrite accepted artifacts or weaken hash/length checks.

The inventory slice does not implement this proposal. It keeps the Windows
failure visible and does not turn mocked cache tests into a platform JIT
support claim. Existing packaged AOT routing is a different boundary.

## Rejected alternatives

- Calling the test a GPU/toolkit failure: no compiler or device runs here.
- Suppressing byte-length verification or swallowing corruption: that would
  weaken cache integrity rather than correct publication.
- Editing production cache code in the inventory PR: outside its authorized
  documentation-only ownership and validation scope.

## References

- [Current inventory](../../../docs/maintainer/runtime_compilation.md)
- [Issue #2680](https://github.com/jinzhezenggroup/generativeqc/issues/2680)
