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

## Homepage update

Only **GenerativeQC Direct r²SCAN-3c/48/96** changes in the homepage figure.
Other sizes/methods, all GPU4PySCF observations and DF remain
[historical](../readme-refresh-20261015/README.md).

| Atoms (AOs) | Cold (s) | Warm (s) | Moved (s) | Moved-warm (s) |
| --- | ---: | ---: | ---: | ---: |
| 48 (544) | 61.975930 | 11.375935 | 44.667711 | 11.426388 |
| 96 (1088) | 254.163584 | 42.716903 | 155.577024 | 42.753795 |

Complete E+F includes preparation and returned host forces; both warm columns
are five-call medians, without trimming or iteration normalization. Both
12-call populations finish within the unchanged 1800-second limit. Cold SCF
iterations/Fock builds: 19/19 and 24/24; moved: 13/13 each; every warm: 1/1.
Canonical spherical def2-mTZVPP, strict FP64, moving Becke 48×16×32 per atom,
screening and convergence are unchanged. Slurm job 9054 uses assigned RTX 5090s.
Maximum energy/force errors: 1.524e-11/3.868e-11 (48), 2.320e-11/6.739e-11 (96)
in Eh and Eh/Bohr, below 1e-8/1e-7 gates. The 96-atom force gates replay offline;
48-atom force gates retain collection-time independent-validation receipts.

These are `qualified-stack/candidate/r2scan-3c-{48,96}` observations: sparse
MD-J (#2563), guarded XC (#2567) at
`5940546542d0cf77b60c8e59a2df32f8852f8d0f`, and six force-compiler files
from #2558 at `719214668aeb69146174a7039490fd95d5607533`.
**Not XC alone, current-master measurements or a historical-cohort speedup.**
The archived qualification retains full source/binary provenance and matched
comparisons. No GPU or competitor rerun is claimed by this presentation update.

```bash
PYTHONPATH=.:python python tools/render_readme_r2scan3c.py \
  --output .artifacts/readme-r2scan3c-replay
```

This audits both publications and identical protocols, then regenerates the
figure and full mixed-source `timings.md` without duplicating raw evidence.
The figure is published as `assets/readme-dft-performance.svg`.
