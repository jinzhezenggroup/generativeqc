# Guarded diagonal XC contraction, 2026-10-16

Full qualification, tables, failed trials, caveats and reproduction instructions
are preserved byte-for-byte in `qualification.md.xz`.

```bash
python -c 'import lzma; print(lzma.open("benchmarks/results/xc-diagonal-potential-20261016/qualification.md.xz", "rt").read(), end="")'
python benchmarks/results/xc-diagonal-potential-20261016/summarize.py
```

Both commands are read-only and need no GPU. The archived README is 9,714 bytes,
SHA256 `84d39840db447dbbf4067f3652211d8c8e4a0c10146262dbf64588175c4a63d0`.

Original force gates: 32 offline-replayable, 168 collection-time receipt only;
the new small-case supplement adds 24 fully replayable calls. Frozen sources;
the force overlay remains unmerged.

## Homepage update

Only **GenerativeQC Direct r²SCAN-3c/3/12/48/96** changes in the homepage figure.
Other methods, all GPU4PySCF observations and DF remain
[historical](../readme-refresh-20261015/README.md).

| Atoms (AOs) | Cold (s) | Warm (s) | Moved (s) | Moved-warm (s) |
| --- | ---: | ---: | ---: | ---: |
| 3 (34) | 1.610058 | 0.181972 | 0.501545 | 0.182197 |
| 12 (136) | 8.018521 | 1.442532 | 6.128277 | 1.447230 |
| 48 (544) | 61.975930 | 11.375935 | 44.667711 | 11.426388 |
| 96 (1088) | 254.163584 | 42.716903 | 155.577024 | 42.753795 |

Complete E+F includes preparation and returned host forces; both warm columns
are five-call medians, without trimming or iteration normalization. All four
12-call populations finish within the unchanged 1800-second limit. Cold SCF
iterations/Fock builds (3/12/48/96): 13/13, 18/19, 19/19, 24/24;
moved: 9/9, 13/13, 13/13, 13/13; every warm: 1/1.
Canonical spherical def2-mTZVPP, strict FP64, moving Becke 48×16×32 per atom,
screening and convergence are unchanged. Slurm jobs 9100 (3/12) and 9054 (48/96)
use assigned RTX 5090s; job 9100 has a finite 12-minute allocation.
Maximum energy/force errors: 1.251e-12/2.508e-11 (3), 2.729e-12/1.791e-11 (12),
1.524e-11/3.868e-11 (48), 2.320e-11/6.739e-11 (96)
in Eh and Eh/Bohr, below 1e-8/1e-7 gates. The 96-atom force gates replay offline;
48-atom force gates retain collection-time independent-validation receipts.

The 48/96 cases are `qualified-stack/candidate/r2scan-3c-{48,96}` observations;
the new 3/12 cases are under `readme_small`, using the identical loaded binary
and scientific input identity. The source composition is sparse
MD-J (#2563), guarded XC (#2567) at
`5940546542d0cf77b60c8e59a2df32f8852f8d0f`, and six force-compiler files
from #2558 at `719214668aeb69146174a7039490fd95d5607533`.
**Not XC alone, current-master measurements or a historical-cohort speedup.**
The archived qualification retains full source/binary provenance and matched
comparisons. Only native 3/12 is newly run; no competitor rerun is claimed.

```bash
PYTHONPATH=.:python python tools/render_readme_r2scan3c.py \
  --output .artifacts/readme-r2scan3c-replay
```

This audits both publications and identical protocols, then regenerates the
figure and full mixed-source `timings.md` without duplicating raw evidence.
The figure is published as `assets/readme-dft-performance.svg`.
