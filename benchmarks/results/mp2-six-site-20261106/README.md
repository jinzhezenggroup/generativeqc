# Six-site MP2 qualification

Before recovery in a fresh clone, fetch GitHub's retained PR head with
`git fetch --no-tags https://github.com/jinzhezenggroup/generativeqc.git refs/pull/3016/head`.
This fetch also works after a normal squash merge or source-branch deletion.
The recovery manifest records the ancestry check and exact member hashes.

All six measured retention decisions, gates, timing scope and rejected
alternatives are in the
[Agent Note](../../../.agents/notes/implemented/performance/2026-11-06-mp2-six-site-dispositions.md).
The complete source-matched summary, raw work/storage/component arrays,
independent oracle inputs and all timed E+F samples remain in Git commit
`e1922f36c0e449a8dbd4509a096f3d469c35087b` at this same directory, for example
`git show e1922f36c:benchmarks/results/mp2-six-site-20261106/samples.json.xz`.

Current reproduction drivers remain in `benchmarks/experiments/mp2_materialization/`.
Full local build/run traces remain ignored. Member hashes and the unchanged
82 MiB budget rationale are in
`manifests/evidence_migrations/mp2-residency-qualified-snapshot-20261106.json`.
