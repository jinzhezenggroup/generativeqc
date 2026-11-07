# Scientific DF links and named replay boundaries

Before recovery in a fresh clone, fetch GitHub's retained PR head with
`git fetch --no-tags https://github.com/jinzhezenggroup/generativeqc.git refs/pull/3016/head`.
This fetch also works after a normal squash merge or source-branch deletion.
The recovery manifest records the ancestry check and exact member hashes.

Slurm n4 job 1511 qualifies actual H2/LiH/water DF metric host-transform links
and named prepared RHF/UHF replay boundaries with unchanged scientific gates.
All live raw replay fixtures remain in `tests/reference_data/cupti/n4-1511/`.
Global implicit/graph-wait coverage is not asserted; the retained F-HeH failure
is not a passing capture. See the
[Agent Note](../../../.agents/notes/implemented/architecture/2026-11-06-scientific-df-residency-lineage.md).

The full report, frozen-source patch, allocation/build identities and validation
snapshot remain at this directory in Git commit
`e1922f36c0e449a8dbd4509a096f3d469c35087b`. Recover with
`git show e1922f36c:benchmarks/results/residency-qualified-n4-20261106/qualification.json.xz`.
Hashes/recovery: `manifests/evidence_migrations/mp2-residency-qualified-snapshot-20261106.json`.
