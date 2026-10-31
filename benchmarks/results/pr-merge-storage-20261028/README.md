# Lossless historical evidence storage

This storage-only migration compresses 34 existing files from Git revision
`a3ec1f8fd8116fc70cf945ab2e7dd97f1e0d42a6`. It saves 1,394,547 payload bytes
before inventory and documentation overhead. The aggregate evidence limit stays
83,886,080 bytes. No new scientific or performance qualification is claimed.

`manifest.json` binds each original path, byte count and SHA-256 to its compressed
companion. XZ source patches recover the original filenames before application;
the practical-auxiliary reader already accepts gzip JSON/JSONL companions.
Publication manifests bind the new stored bytes. Scientific source identities,
measurements, failed gates, references and acceptance decisions are unchanged.

Run the opt-in byte verification from the repository root:

```bash
python benchmarks/results/pr-merge-storage-20261028/verify.py
```

Restore an individual patch without executing it:

```bash
xz -dc benchmarks/results/direct-phase-moved-20261023/driver.patch.xz > /tmp/driver.patch
sha256sum /tmp/driver.patch
```

Compare that digest with `original_sha256` in the inventory before applying it
to the historical revision declared by its own publication. Use `gzip -dc` for
gzip members. Decoding restores exact bytes, including whitespace, numeric
spellings and failure messages; it does not relabel a historical run as current.
