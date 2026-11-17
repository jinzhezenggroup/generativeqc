
## Lossless source-patch storage

The original source patches are retained as XZ files. Before applying a patch,
restore its original filename with `xz -dc NAME.patch.xz > NAME.patch`.
The [storage inventory](../pr-merge-storage-20261028/manifest.json) records both
stored and original byte counts and SHA-256 values. Measurement identities,
all samples, and acceptance decisions are unchanged.

## Historical source-census recovery

Only the build/source checksum stream moves to the already published Git
ancestor recorded in `source-census-snapshot.json`. Independent inputs, all
observations, failures and numerical gates stay in this directory. Restore the
exact original census before running the unchanged historical GPU recipe:

```bash
python tools/restore_retained_evidence.py --all --manifest benchmarks/results/becke-native-phases-20261006/source-census-snapshot.json --output /tmp/becke-source-recovery
```

The output preserves the repository-relative path. Supply that original file
to the copied reproduction bundle outside the checkout; no runtime lookup or
scientific fallback is introduced. Recovery is local Git only, with original
size, SHA-256 and blob identity checks; it never fetches missing history.
