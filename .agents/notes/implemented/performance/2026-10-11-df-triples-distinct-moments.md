# Decision: reuse equal occupied W seeds in FP64 DF triples energies

Status: implemented
Date: 2026-10-11

## Problem

The triangular occupied domain visits `i >= j >= k` but materializes six W
seeds even when occupied indices repeat. These seeds are functions of physical
`(I,J,K)` tuples, not distinct permutation labels. Repeated tuples therefore
repeat both original GEMMs without adding information. The complete ethane230
endpoint still spends about five seconds in triples after #2221 physical replay.

## Decision

The occupied-triples generator derives a first-identical source map from the
authoritative six permutations. The native energy owner skips only noncanonical
W builds and selects a separate remapped energy kernel for repeated-index tiles
when actual W storage, compute and accumulation are all FP64. All-distinct tiles
keep the original kernel; non-FP64 materialization remains unchanged. Existing
C++ types/functions, original energy kernel and response/Fock sources are
byte-identical, with additive helper/kernel symbols only.

Every original projected contribution, virtual permutation, occupied tile,
denominator degeneracy and finite check remains. The six-cube arena and panel
admission are intentionally unchanged. This is work reduction, not a memory
capacity claim. One-panel and generated-provider budget fallbacks remain.
The policy is audited as `cc-execution:df-triples-distinct-moments`, not an
invented SolverOptions flag. Response/forces still use full independent W and
cotangent materialization.

## Rejected alternatives

- Dropping repeated/all-equal occupied tiles changes the nonfinite failure
  contract even when their energy cancels algebraically.
- Collapsing six projected energy terms risks changing the independent scalar
  algebra, multiplicities and accumulation order. Only W source loads alias.
- Aliasing cotangents because primal moments agree is not justified by primal
  equality. This qualification does not cover derivatives.
- Packing fewer W cubes would change arena/resource admission and source
  addresses unnecessarily. Retain the bounded original layout.
- Extending to actual FP32 execution requires separate precision gates. A
  rejected FP32 request that admits strict FP64 follows the admitted FP64
  schedule, not the requested label.

## Invariants

The first canonical tuple must remain directly materialized. Slots skipped by
the producer may contain arbitrary/uninitialized values and must never be read.
The energy kernel still evaluates all six contributions. All-equal overflow must
fail before publishing a result. No CPU/PySCF/oracle enters production.
Exact W evaluation count is `o^3`, versus `o(o+1)(o+2)` previously; the number of
occupied energy tiles and virtual epilogue points is unchanged.

## Evidence

`benchmarks/results/df-triples-distinct-moments-20261011/` retains one separate
candidate-then-control complete endpoint pair. Do not pool it with #2221.
Control/candidate wall times are 49.596829078 / 48.494610759 seconds: 2.22%
shorter, 1.02273x. Triples is 5.155789883 / 4.135521826 seconds, 1.24671x.
Both arms have 19 observations/evaluations; CCSD and physical replay counters
are unchanged. Total/triples energies are unchanged in this pair and pass the
original independent 1e-8 / 1e-10 Eh and physical R1/R2 1e-10 gates.

For `o=9,v=221,Q=488`, W evaluations fall 990 -> 729, total triples GEMMs
2132 -> 1610, and contraction summands 3258407583236 -> 2610452107406.
The 152 integral panel builds and complete epilogue work remain. CC device
numeric capacity is unchanged at 8583749632 bytes; complete endpoint host/device
numeric capacity remains 8945677650 bytes. Peak process RSS adds 1212416 bytes
in this pair; this is not a universal memory bound.

Five independent CPU cases run actual emitted kernel bodies with poisoned
unwritten slots. Six GPU fixtures include a >256-point tail, original oracle,
exact work, one-panel, deterministic repeat, generated-provider fallback,
one-byte-short refusal and all-equal overflow with unpublished sentinels.
Memcheck and initcheck each report zero errors. The first collector reused a
sanitizer summary filename: retain its final initcheck record and both original
logs; a naming-only repair reran just the two small memcheck actions to obtain
an independent receipt. No endpoint or passing matrix was repeated.

The measured library freezes #2221 CC and #2171 HF consumer closure. Only two
objects change, and borrowed link inputs are checksum-verified immutable.
It does not qualify newer #2215/#2217 HF or #2219 force/response behavior.
Parent #2221 merged as 5da28bdca; its squash tree equals the rebased parent tree.
Latest-master regeneration matches all three GPU-qualified artifacts byte for
byte; the native owner is semantically unchanged apart from the three recorded
leading-space insertions described below. Rebase/merging alone
does not require repeated GPU tests when those consumed identities are unchanged.
Broad/statistical performance and production/force stages remain `not-run`.

## Consequences and revisit conditions

The energy-only default saves repeated GEMMs while preserving resource/ABI
contracts. The map adds small host/kernel parameters but no numeric scratch.
Revisit mixed precision, compact storage or response only with independent
domain-specific gates and complete endpoint evidence. Requalify when consumed
owner/generated artifacts change, not merely when unrelated master commits land.

## Post-publication layout and master assessment

PR #2232's format bot changed only three native continuation-line indentations
in f70cf2f99. The exact measured source bytes and hashes remain immutable in the
publication. The offline receipt test now permits only leading indentation
differences when comparing today's native owner to those qualified bytes; it
rejects raw strings and escaped newlines before that comparison, and does not
normalize tokens, literals or line wrapping. This layout-only change needs no
GPU rebuild, endpoint or matrix repetition.

Master advanced to 2b68d10a9 (#2218, offline CPU XC AOT dependency closures).
It changes no DF triples owner, generator or CC/scalar/tensor equation consumer.
This is an impact assessment, not new CPU-XC numerical qualification. The prior
measured source/library and current-master HF/force limits remain unchanged.


## Exact layout binding clarification

The initial post-publication comparison accepted arbitrary leading-indentation
changes. Its replacement binds the measured owner and current owner by separate
SHA-256 values and records exactly the three one-space insertions. The test
replays those edits byte-for-byte and rejects any other indentation, token,
line-ending or trailing-line change. The current-assessment byte-identical flag
is false; the measured source/build hashes, reconstruction patch, original
numerical gates and all 75 raw receipts remain unchanged. Regenerated .hpp,
.cuh and .cu artifacts still match the GPU-qualified hashes. No GPU rerun or
latest-master HF/force qualification is inferred from this provenance repair.
