# Native DFT XC autotune qualification

`xc-receipts.json.xz` contains all raw paired reports, independent PySCF/libxc
references, observed native allocation receipts, semantic work diagnostics and
the Slurm/runtime replay transcript. It is an XZ-compressed JSON document, not
an executable profile or a release artifact. No native binaries are published.

The source base is `cd2b4514b`; each report records the exact dirty native source
inventory used for qualification. Slurm job 1441 ran on n4, RTX 5090/SM120,
CUDA 12.9.86, normal Release with fast compilation disabled. The local build
uses GCC 12 and ccache. The independent remote reference is PySCF 2.14.0.

| Workload | Batch | Final tile | Last accepted paired edge, ms | Speedup / lower 95% |
| --- | ---: | ---: | --- | --- |
| H2, PBE-RKS/sto-3g | 1 | 512 | 256 → 512: 36.490 → 31.344 | 1.164 / 1.164 |
| Water, PBE-RKS/def2-svp | 1 | 2048 | 1024 → 2048: 66.352 → 63.560 | 1.044 / 1.022 |
| Water, PBE0-RKS/def2-svp | 1 | 1024 | 512 → 1024: 103.277 → 98.632 | 1.047 / 1.010 |
| Water dimer, PBE-RKS/sto-3g | 2 | 2048 | 1024 → 2048: 148.385 → 134.001 | 1.107 / 1.089 |

These are frozen-density warm **complete energy and analytic-force** replays,
excluding preparation, cold convergence and warmup. Every edge uses five
alternating-order pairs, not selected fastest repeats. Earlier successful
tiles are marked superseded in reports; they are not numerical failures.
The table does not combine separately measured edges into an aggregate paired
speedup or generalize these molecules to all DFT inputs.

All samples pass independent energy/force, strict baseline/candidate parity,
translation and native memory-admission gates. Six additional real-worker
checks cover LDA-RKS, PBE-RKS, PBE-UKS, PBE0-RKS, B3LYP-RKS and r2SCAN-RKS.
The H2 profile actually replays at 512; explicit tile 64 stays 64 and a 1 GiB
ResourceBudget retains native tile 256. Their energies and forces agree.

## Reproduction

Build the matching checkout with CUDA SM120 Release, fast compilation off and
verified compiler launchers. Install the autotune reference extras. Decode the
receipt with Python `lzma.open` and take each XYZ geometry (Bohr), basis, method,
representation and batch from `reports[*].workload`. Under a finite Slurm
allocation, run `python -m generativeqc.tune_cli` with `--grid-level 3`,
`--xc-tiles 512 1024 2048`, `--repeats 5`, `--max-candidates 3` and an isolated
`GENERATIVEQC_PROFILE_CACHE`. H2 searches only tile 512. Preserve the assigned
device visibility. The current workflow and exact solver controls are in
`docs/user/local_autotuning.md`; new machine/noise conditions can select a
different winner or no winner.
