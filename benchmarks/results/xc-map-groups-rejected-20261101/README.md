# Rejected cooperative XC map-group discovery

The candidate did not establish a useful complete-endpoint gain on A100,
RTX 5090 or PRO6000 and is not promoted. See the
[decision and complete timing table](../../../.agents/notes/rejected/2026-11-01-xc-cooperative-map-groups.md).

`receipts.json.xz` retains original full endpoint records, all independent
qualifications, source/library/scheduler identities, both source variants as a
base plus patch, launch scripts, cache/build logs, native tests, sanitizers and
the original A100 loader failure. Each member has its own SHA-256. Numerical
oracles and audited environment reuse contracts remain in their existing
archives and are bound by SHA-256 rather than duplicated.

The 24 complete populations / 48 energy-and-force calls pass independent
qualification. Blackwell arms share an allocation; A100 arms do not. Each arm
has one cold and one warm call. Small changes (-0.62% to +0.40%) are pilot
observations, not established speedups or statistical equivalence.

Parent production identity:
`b6b63f9063949ed70191c54b4d0f0b7efb788957274eb680c98f4242bbcf74f3`.
Candidate production identity:
`02cf4359fcfc27622f0b959d078c324df6212fef6b7aeb27104d89bbbea4f675`.
The patch applies to PR #2910 production sources at `1e07da405`.

Use `PYTHONPATH=python:. python benchmarks/results/xc-map-groups-rejected-20261101/report.py`
to authenticate member/reference bytes and recompute every independent gate.
This reader intentionally does not rely on Python assertions for integrity.
