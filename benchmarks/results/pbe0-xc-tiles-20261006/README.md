# Larger local SCF XC tiles: complete PBE0 endpoints

Same-binary 256/512 SCF tile comparisons on n2's RTX PRO 6000 Blackwell,
finite Slurm job 2506, CUDA 12.9 and ccache. Generated/local-AO execution and
ordinary force tile 256 remain unchanged; this publication changes no default.

| Atoms | Replay | Tile 256 median (s) | Tile 512 median (s) | Improvement |
| ---: | --- | ---: | ---: | ---: |
| 96 | warm | 22.861523 | 22.384145 | 2.088% |
| 96 | moved-warm | 22.932187 | 22.433680 | 2.174% |
| 48 | warm | 8.522234 | 8.260260 | 3.074% |
| 48 | moved-warm | 8.505666 | 8.250513 | 3.000% |

Five interleaved A/B pairs per row pass the greater-than-2%/relative-MAD
descriptor. Every timed call executes one actual SCF iteration/Fock build,
uses warm state, and has no fallback. Each arm independently converges and
publicly freezes its own native snapshot; density bytes are not identical by
contract. Preparation, cold/moved setup, priming, full vectors and all actual
histories remain in the raw records, excluded from these replay medians.

Independent GPU4PySCF 1.8.1 references use the exact grid and basis. Maximum
energy/force errors across all setup, prime and timed results are 1.015e-10
hartree and 3.134e-11 hartree/bohr, within unchanged 1e-8/1e-7 gates.

## Semantic work and limits

96-atom SCF map tiles halve from 9,216 to 4,608, but local AO-square summands
increase from 75,674,112,000 to 88,940,603,392 at the original geometry.
Force work remains identical: 2,359,296 points, 9,216 phased batches,
10,758,389,760 primal/reverse pair visits each and 64,512 phase launches.
Geometry-discovery fields are retained receipts, not repeated warm work.

AO-map reservation telemetry is not endpoint peak memory. Both prepared arms
coexist; concurrent full device/host peak, uncached compilation, automatic
tile selection, UKS96, other devices and cold/moved speedup are not qualified.
Cold iteration counts differ and are never normalized into acceleration claims.
The generic memory/compilation promotion envelope remains not-run; numerical
acceptance and the four positive matched replay comparisons do not change
production selection. Tile 256 remains the default.

Environment metadata is captured before preparation. The existing helper's
static after-benchmark nvidia-smi label is not post-measurement clock/thermal
evidence; no such claim is made. Slurm accounting storage is disabled, so the
zero-exit script receipt and startup allocation are retained without inventing
an unavailable accounting result.

## Exact frozen source

Base `83109befe5b97a5542b30e169da74c81d04719e6` plus `source.patch` produces
source identity `b0e32abf342aa53877f2a350801a1dfdc1bc74d341df0cebe82a738d74429d2f`.
The shared baseline includes ordered cooperative normalization, so this is an
incremental SCF-tiling comparison, not an additive or combined normalization
speedup claim. Raw null Git fields from the archived source remain untouched.
The fresh core SHA is `20100959aa8c2643b1f341f9c140e4778d618e496dda8f216548c52676137931`.

Retain the publication and `benchmarks/pbe0_xc_tile_pairs.py` outside a restored
checkout; check out the measured base, apply `source.patch`, copy that benchmark
into `benchmarks/`, then check `source-files.sha256`. Set `CUDA_ROOT`, `PYTHON`,
`CMAKE` and `CCACHE`, then run `bash <publication-directory>/reproduce.sh` from
the restored root on n2. Retained references are independent old data; set
`FRESH_REFERENCE=1` for a separate fresh reference campaign.

```bash
PYTHONPATH=python:. python -m pytest -q tests/python/test_pbe0_xc_tile_pairs.py
```

Selected raw files and receipts are checksum-bound by `publication.json`.
The offline consumer recomputes whole-vector gates, actual work and medians;
edited forces, iteration counts, selected tiles or work counters are rejected.
Full build/command/Slurm logs stay ignored locally and on n2; no Release is used.
