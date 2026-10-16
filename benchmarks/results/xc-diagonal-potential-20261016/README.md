# Guarded diagonal XC contraction, 2026-10-16

Full qualification, tables, failed trials, caveats and reproduction instructions
are preserved byte-for-byte in `qualification.md.xz`.

```bash
python -c 'import lzma; print(lzma.open("benchmarks/results/xc-diagonal-potential-20261016/qualification.md.xz", "rt").read(), end="")'
python benchmarks/results/xc-diagonal-potential-20261016/summarize.py
```

Both commands are read-only and need no GPU. The archived README is 9,714 bytes,
SHA256 `84d39840db447dbbf4067f3652211d8c8e4a0c10146262dbf64588175c4a63d0`.

Force gates: 32 offline-replayable, 168 collection-time receipt only. Frozen
sources; the force overlay remains unmerged.
