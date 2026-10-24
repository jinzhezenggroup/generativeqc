# Decision: separate incremental paper mapping from campaign admission

Status: implemented
Date: 2026-10-24

## Problem and decision

The paper freezes RTX 5090 results at 7522994, while later delivered independent
CBS checks cover additional energies and directions. Updating old checked
markers would conflate archive state with new evidence. An empty H100 report
cannot represent completed experiments.

Add a new CPU/offline family with fixed Git-object inventories, independent
raw-array error recomputation and proposed claim wording. Preserve the paper
and all historical raw families. Separate merged delivery, formal approval and
numerical coverage: #2721/#2719 merged with COMMENTED reviews, an explicit
process exception, not APPROVED.

## Invariants and rejected alternatives

- #2725 owns scientific protocol/schema and execution admission; this mapper
  only adapts its versioned interface. No second protocol or resource scheduler.
- Unknown campaign counts remain null. Zero imported receipts does not imply
  zero attempts on the coordinator's resources.
- Preserve exact paths/bytes/Git modes, failures and source/library identities.
  Never execute historical scripts or duplicate scientific computations.
- Directional checks do not imply full-vector, H100, current-master, performance
  or true-CBS qualification. Conventional CPU and DF cohorts remain separate.
- Do not change old unchecked markers, infer Lambda causality from total cost,
  use CPU oracle time as a speedup denominator or treat job success as science.
- Proposed wording stays separate until an authorized paper slice applies it.
  Refs #2739; #2724/#2739 retain remaining work.

## Evidence and revisit conditions

The reader verifies fixed LF blobs and recomputes 72 energy, 48 directional,
24 translation and 12 conventional CPU comparisons. Behavioral tests cover
changed bytes/modes, duplicate/unsafe paths, bounded/truncated/trailing XZ and
explicit gates under optimized Python. The first local replay exposed original
metadata paths overriding capsule keys; a RED regression preceded the bounded
parser repair. Raw inputs remain unchanged.

Add accepted H100 receipts through the owner's normal handoff, with fixed
#2725 contract and complete attempt accounting. A later authorized paper PR
may apply scope-limited wording without rewriting the original archive.
