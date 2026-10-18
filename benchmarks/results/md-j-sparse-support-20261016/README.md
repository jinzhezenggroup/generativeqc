# Structural-support MD-J transforms: retained qualification

The complete 2026-10-16 qualification narrative, tables, negative trials and
reproduction/safety guidance are preserved **byte-for-byte** in
`qualification.md.xz`. Read it without creating or overwriting files:

```bash
python -c 'import lzma; print(lzma.open("benchmarks/results/md-j-sparse-support-20261016/qualification.md.xz", "rt").read(), end="")'
```

This is only lossless compression of this task's historical documentation to
retain subsequent XC evidence within the unchanged 64-MiB repository cap.
No scientific record, force array, reference, timing sample, executable
reproducer or qualification gate changes. `SHA256SUMS` covers every current
member. `recovery.json.xz` separately identifies all 44 original tracked members at
`ef02ef322e809a905cca09e7d2b3323b9d1972ae`, including full original envelopes and
negative-trial receipts; the pooled working view is not that full archive.

Both provenance JSONs are lossless archives. Inspect either without writing:

```bash
python -c 'import lzma,sys; print(lzma.open(sys.argv[1], "rt").read(), end="")' \
  benchmarks/results/md-j-sparse-support-20261016/recovery.json.xz
```

Use `source-and-build.json.xz` in the same directory for source/build provenance.

Recompute all 96 native E/F gates against eight independent geometry references
and export maintained safe reproducer scripts, using **fresh** destinations:

```bash
python benchmarks/results/md-j-sparse-support-20261016/summarize.py \
  --output .artifacts/md-j-summary-new.json \
  --export-reproducers .artifacts/md-j-reproducers-new
```

Keep the exported `output_safety.py` beside the scripts. Every GPU command still
requires a finite `srun` allocation and unchanged scheduler device visibility.
The archived narrative explains site environment paths, source/build provenance,
resource admission, original cache behavior and cleanup obligations. Do not
substitute archived output-overwriting scripts for maintained `reproduce/` tools.

The sparse exact coefficient layout is default under existing admission and
bounded fallback, without changing screening, Direct J/K or the 128-MiB cap.
Its standalone qualification does not imply qualification of other PR stacks.
The separate `../xc-diagonal-potential-20261016/` packet records actual composed
measurements; independent historical timings must not be added or subtracted
to manufacture a composed endpoint.
