
## Lossless source-patch storage

The original source patches are retained as XZ files. Before applying a patch,
restore its original filename with `xz -dc NAME.patch.xz > NAME.patch`.
The [storage inventory](../pr-merge-storage-20261028/manifest.json) records both
stored and original byte counts and SHA-256 values. Measurement identities,
all samples, and acceptance decisions are unchanged.
