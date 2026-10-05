# Parallel Becke logs on the composed order-five baseline

Complete exact-direct PBE0/def2-SVP E+F qualification of PR #1950, on the measured master + #1830/#1833/#1847/indexed-policy + order-four/five stack. The PR phase-generation files are byte-identical to the measured candidate; these are not unmodified shipping-master timings.

| Atoms | Regime | Control seconds | Pair-log seconds | GPU4PySCF seconds | Native SCF iterations |
| --- | --- | ---: | ---: | ---: | --- |
| 48 | cold | 121.183339 | 121.536077 | 50.660632 | [25] → [25] |
| 48 | warm | 11.832915 | 9.159640 | 5.895901 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 48 | moved | 52.419001 | 52.628613 | 46.630073 | [12] → [12] |
| 48 | moved-warm | 10.194947 | 9.140124 | 3.855905 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 96 | cold | 269.894929 | 277.035697 | 77.793351 | [29] → [30] |
| 96 | warm | 25.672614 | 25.182758 | 13.503802 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |
| 96 | moved | 164.997179 | 128.451524 | 84.405941 | [18] → [13] |
| 96 | moved-warm | 25.750174 | 25.199025 | 10.014773 | [1, 1, 1, 1, 1] → [1, 1, 1, 1, 1] |

The 96-atom warm and moved-warm endpoints improve **1.91% and 2.14%**, saving 0.490/0.551 s. Their grid-response medians improve 8.6580→8.1656 and 8.6575→8.1633 s. The complete candidate remains 1.86×/2.52× the matched warm/moved-warm reference. This is a useful scheduling improvement, not the isolated 12.3% applied to the entire endpoint and not ten-second parity.

The 48-atom warm control fluctuates outside the changed grid stage. Grid response improves only about 0.067–0.068 s. Do not attribute the 2.673 s warm or 1.055 s moved-warm endpoint median gaps entirely to this schedule. All original samples and unmodified work are retained. Cold/moved slightly regress at 48 atoms; 96-atom cold/moved have different native trajectories (29/30 and 18/13). These differences are not assigned to the force-only treatment or normalized by iterations.

## Scientific and device qualification

All **288** same-geometry native/reference pairings pass unchanged 1e-8 Eh / 1e-7 Eh per Bohr gates. Maximum absolute errors are 1.06411e-10 Eh / 2.72972e-11 Eh per Bohr. Every cold/warm/moved/moved-warm sample, actual SCF history, Fock count, physical residual and force vector remains in the compressed raw JSON.

The reference rebuilds exact Direct J/K from full current density and includes moving-grid response. Basis, grid, convergence, screening and FP64 are unchanged. There is no DF/COSX substitution. Pair production remains 1,330,642,944 / 10,758,389,760 for the 48/96 grids; ordered atom traversals and four-word pair storage are unchanged. The treatment moves logarithms to parallel pair workers and rematerializes cheap ratio partials from the canonical graph.

Final source passes 79 Becke host tests and 17 resource tests, emitted CUDA tests (21 passed, 28 iteration-specialization skips), 24 shared-owner tests, and memcheck/racecheck/initcheck/synccheck with zero errors or hazards. The shared-owner probe is generated explicitly with pbe0_rks, matching its eight-row source contract. Native and all six stationary AOT modules build with verified CXX/CUDA ccache launchers. Tests and endpoint share frozen source and binary receipts in Slurm5928; the scheduler-assigned CUDA_VISIBLE_DEVICES is preserved.

This accepts numerical equivalence and reports the complete observations. A strict interleaved provider/default-promotion assay with isolated compilation/allocator peaks was not run. The compound stack is not promoted by these measurements, and #1895 remains open.

## Reproduction

- Control `8f90f0ac33ee6bd8a755fb85d71291e1f66c5884`; candidate `e92cf75b0a9e1a8fa3a6c1ea2f8bfffc809fddee`. The code in PR #1950 is `ef91a6617988a2a9b55dd743767371c6000c9920`.
- Apply the two source patches in reconstruction.json order to its retained base. Patches include the complete composed source and can be checked against the recorded scientific source identities.
- Retained drivers build native/AOT and probes, then run the finite Slurm device gates and 48/96 protocol. 48 runs control then candidate; 96 reverses the arms. Each arm/size owns a fresh stationary cache. Build/launch scripts retain historical checkout paths.
- Run `PYTHONPATH=python:. python benchmarks/results/pbe0-order5-pair-logs-20261005/verify.py` to validate hashes, complete geometry/phase/repeat inventory, all pairings, work, actual histories and timing summaries without a GPU.
- Full ignored binaries and logs remain in `/data/jzzeng/qc-1895-order5-pair-logs-20261005/.artifacts/` under endpoint-5928/, gpu-5928/, probes/ and build/. No release, release asset or external archive is used.
