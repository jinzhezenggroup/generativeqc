# Recover #2191 evidence

All eight original files remain in existing merged commit
`f81640c8cd7531be60c1d9a4d2323e90934e1d1c`; the manifest pins every size/SHA-256.

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/df-cc-ladder-dressing-factorization-20261010/snapshot.manifest.json \
  --output .artifacts/df-cc-ladder-evidence-restored
```

Every Git blob verifies before writing. No sample is discarded, cap raised or
archive published. New evidence: `../df-cc-energy-q32-default-20261010/`.
