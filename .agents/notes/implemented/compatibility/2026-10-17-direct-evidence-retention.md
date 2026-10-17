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

This preserves observation provenance, not an offline force replay or a formal
performance promotion: the compact packet does not contain full force arrays.
Original ignored raw artifacts remain untouched. Later experiments must not use
history-only publication as a way to bypass the repository's evidence budget.
