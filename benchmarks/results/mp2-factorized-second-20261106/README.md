# Factorized MP2 second-stage qualification

Before recovery in a fresh clone, fetch GitHub's retained PR head with
`git fetch --no-tags https://github.com/jinzhezenggroup/generativeqc.git refs/pull/3016/head`.
This fetch also works after a normal squash merge or source-branch deletion.
The recovery manifest records the ancestry check and exact member hashes.

The complete measured snapshot is retained in Git commit
`e1922f36c0e449a8dbd4509a096f3d469c35087b` at this same directory.
Recover a member with `git show e1922f36c:benchmarks/results/mp2-factorized-second-20261106/qualification.json`.
This is the initial occupied-support compaction, not a general endpoint-speedup
claim. Final six-site decisions are in the
[Agent Note](../../../.agents/notes/implemented/performance/2026-11-06-mp2-six-site-dispositions.md).

Exact member hashes and storage rationale are in
`manifests/evidence_migrations/mp2-residency-qualified-snapshot-20261106.json`.
