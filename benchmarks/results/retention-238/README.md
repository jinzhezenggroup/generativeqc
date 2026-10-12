# Evidence retention audit

`inventory-before.json` records baseline tracked counts/bytes by class.
`migration.json.gz` audits every removed run artifact, its original checksum and
Git archive identity, and extracted scientific measurements/diagnostics.
The [shared policy](../../../docs/maintainer/evidence_retention.md) documents reproduction,
publication and external artifact retention. Scientific fixtures and raw
comparison samples remain versioned; no production choice or gate changed.

This is repository-maintenance evidence, not a GPU performance result.

## Lossless checkout storage

The complete original migration audit remains in `migration.json.gz`, including
all 161 records, extracted measurements, failed diagnostics and original Git
identities. Compression preserves its exact UTF-8 bytes without parsing or
reserializing the JSON. `migration.storage.json` records both stored and decoded
sizes and SHA-256 identities. Read it locally with `gzip -dc migration.json.gz`
or Python's `gzip.decompress`; no historical Git objects, network or GPU are
needed. The aggregate evidence budget and scientific gates are unchanged.
