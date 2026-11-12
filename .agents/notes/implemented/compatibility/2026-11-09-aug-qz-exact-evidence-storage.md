# Decision: retain augmented-QZ receipts in the existing exact FP64 container

Status: implemented
Date: 2026-11-09

## Problem and decision

PR #3057 adds complete independent augmented-QZ qualification, including a failed
root experiment. The original 36-file family occupied 1,118,571 bytes. Use the
existing exact-fp64-evidence v3 reader, rather than deleting receipts, rounding
numbers, introducing another codec, or recompressing unrelated campaigns.
The user approved a bounded 576 KiB aggregate allowance increase, from 89,391,104
to 89,980,928 bytes. The 1 MiB per-file and 2 MiB incoming review gates remain.

## Preservation and rejected alternative

The three authenticated streams occupy 549,209 bytes. One-off migration checks
against commit 7f404e3fa30355688642dcda08697d7dc6168a33 reconstruct every original
decompressed UTF-8 file byte for byte. The container retains 6,090 exact FP64
values, basis decimals, implementation/source/build provenance, and failed runs.
Do not parse/re-serialize native progress JSONL: its 49,399 C++-formatted records
include numerical spellings Python JSON would change. They remain raw text.
The original README and summarizer are also recoverable inside the container.

## Replay and qualification boundaries

The readable campaign README documents the authenticated restore/replay command.
Restoration refuses a nonempty destination; replay uses the unchanged historical
scientific gates and refuses optimized Python that would disable assertions.
Tests cover byte-exact restoration, both compressed-stream integrity failures and
rejection of a perturbed endpoint energy. Hash-pinned storage review authenticates
the large stream but does not itself certify numerical or performance claims.

Integration with master preserves its newer Libxc campaign, manifests and checker.
Official combined-tree catalog regeneration remains stale with zero current
qualified products/contexts. Historical CUDA fast-compile correctness receipts
are not a fresh combined-source campaign or release-performance/full-peak claim.
