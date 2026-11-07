# Historical n4 residency qualification

Before recovery in a fresh clone, fetch GitHub's retained PR head with
`git fetch --no-tags https://github.com/jinzhezenggroup/generativeqc.git refs/pull/3016/head`.
This fetch also works after a normal squash merge or source-branch deletion.
The recovery manifest records the ancestry check and exact member hashes.

The complete measured report/provenance is retained in Git commit
`e1922f36c0e449a8dbd4509a096f3d469c35087b` at this same directory; earlier full
run bundles also exist in `12d5142a3`. Recover a report with
`git show e1922f36c:benchmarks/results/residency-n4-20261106/qualification.json.xz`.
All live raw calibration, wrapper and four scientific DF compatibility replay
inputs remain in `tests/reference_data/cupti/`, not historical lookup.

This snapshot preserves the moved-RHF force failure and unknown scientific H2D
links. It is not a global residency PASS. See the
[later decision](../../../.agents/notes/implemented/architecture/2026-11-06-scientific-df-residency-lineage.md).
Exact hashes/recovery are in
`manifests/evidence_migrations/mp2-residency-qualified-snapshot-20261106.json`.
