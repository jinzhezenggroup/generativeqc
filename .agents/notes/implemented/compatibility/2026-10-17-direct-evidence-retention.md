# Decision: recover already-published Direct evidence without a duplicate archive

Status: implemented
Date: 2026-10-17

Concurrent benchmark additions exhausted the aggregate `benchmarks/results/`
budget while this branch was being qualified. Its already-public packet and
README are therefore removed from the live tree, not relocated into source,
notes, another budget bucket, a new archive-only commit, or a release asset.
No foreign evidence is removed and the budget is unchanged.

The existing immutable commit `f40055ecdb7763175f59fe8108f87805c3f3f193`
retains both files in Git history. The read-only verifier retrieves that existing
object locally and checks its digest before parsing. It never fetches, writes an
archive, or replaces observations. Shallow/source exports must explicitly supply
`--packet` or obtain that existing history. Tests cover explicit input, corruption,
and missing historical objects.

Both input routes authenticate the same compressed historical bytes before
decompression. An internally consistent change to recorded timings must not pass
through `--packet` merely because its medians still agree. Synthetic fixtures
inject their own expected digest; the production historical digest is unchanged.

The existing published pull-request ref provides a concrete manual recovery
route independent of keeping its feature branch or merge strategy. In a Git
checkout with the project remote named `origin`, fetch the existing PR history:

```bash
git fetch origin refs/pull/2587/head
git show f40055ecdb7763175f59fe8108f87805c3f3f193:benchmarks/results/direct-relative-force-20261017/measurements.json.xz > /tmp/direct-relative-measurements.json.xz
python -m tools.generativeqc_validation.direct_relative_evidence --packet /tmp/direct-relative-measurements.json.xz
```

A source export can use the resulting file without Git or network access in the
verifier. The expected compressed SHA256 is
`b8ec7d7031d50f9fa0a0cafbc3b39ab9931133439d2303bb377628f9fe21b7a9`.
A fresh, initially empty bare repository independently recovered this exact
object through `refs/pull/2587/head` with a 64-commit shallow filtered fetch;
explicit-file and historical validation both pass. This is manual online
historical recovery, not a claim that a normal master clone contains the packet
or that a shallow/source export works offline without retaining it separately.

This preserves observation provenance, not an offline force replay or a formal
performance promotion: the compact packet does not contain full force arrays.
Original ignored raw artifacts remain untouched. Later experiments must not use
history-only publication as a way to bypass the repository's evidence budget.
