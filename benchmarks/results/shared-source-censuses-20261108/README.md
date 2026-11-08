# Shared source censuses and exact evidence recovery

Fifteen containers published in master
`6746722694f7e59e291d8dd792e60acb58e998f7` share repeated source path/hash
maps. Their existing `.json.xz` paths now contain authenticated member descriptors.
`storage.json` binds every original JSON and XZ size and SHA-256, its exact XZ
recipe, and the physical storage identities.

`source-maps.json.xz` retains all path strings, base hashes, changed/removed
entries and original map order. Two complete campaign containers retain the
remaining original UTF-8 fragments: October 23–24 schedule campaigns and the
October 15 r2SCAN-3c degree-seven/final-validation campaigns. Numerical tokens,
embedded strings, failures, source identities and observations are unchanged.
No original campaign is selected or filtered by its outcome.

The stdlib reader `benchmarks._source_census_evidence` verifies bounded storage,
map inventories and complete original byte hashes. Existing reproducers use
`read_json_bytes`; consumers that authenticate an original compressed transport
use `read_xz_bytes`, which actually recompresses and verifies that transport.
The original presets are 6, 9 and 9 with extreme mode. Recovery is entirely
offline and requires no Git history, GPU, compiler or artifact service.

All fifteen original transports were reproduced in the migration runtime.
The [XZ manual](https://www.tukaani.org/xz/man/xz.1.html) notes that output can vary
with encoder version/build even for identical settings; transport
recovery therefore fails closed on a byte/hash mismatch. Original JSON recovery
does not invoke an encoder and always checks the complete original byte hash.

From the repository root, recover one original transport into a new file:

```bash
python -m benchmarks._source_census_evidence \
  benchmarks/results/df-md-charge-splits-20261023/source-inputs.json.xz \
  --xz --output .artifacts/source-inputs-original.json.xz
```

Omit `--xz` to recover the exact original JSON bytes. The command refuses an
existing output. The existing per-family reproduction commands still recompute
their original source identities, numerical gates and timing populations.
This representation changes no scientific qualification or retention limit.
