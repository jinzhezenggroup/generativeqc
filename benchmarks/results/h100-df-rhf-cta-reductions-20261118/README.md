# H100: full-CTA fitted RHF energy, convergence and DIIS reductions

HF/def2-SVP with the README default auxiliary basis, DF mode, water-96 (768 AOs),
fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-opt-1118-qa`. Cold = constructor + preparation + SCF + analytic forces +
host export; warm = three frozen-own-density calls. Base is master `a6d5a3c00`
(native identity `c5f05cc9…`). The candidate binary (`d93ee912…`) was built from
master plus this change and the pre-format #3115 D4 change, which HF never
executes. The committed tree differs from it in those D4 files, in clang-format
whitespace, in declaring `kDfMatrixReductionThreads` in the DF kernel header rather
than `scf/cuda/scf_constants.hpp` (SCF dependency direction), and in omitting
`__launch_bounds__(512)` on the two DF reduction kernels (source-anchor stability);
none of these changes the arithmetic. `results.json.xz` retains every arm with all samples,
forces and qualification verdicts, plus both Nsight Systems kernel summaries.

## Attribution (Nsight Systems 2025.1.3, one cold + three warm calls)

| Kernel | Instances | Base median | Candidate median |
| --- | ---: | ---: | ---: |
| `update_diis_kernel` (cooperative DF owner) | 14 | 50.0 ms | 3.0 ms |
| `update_device_convergence_kernel` | 14 | 17.2 ms | 0.46 ms |
| `compute_device_energy_kernel` | 18 | 7.3 ms | 0.29 ms |

Total GPU kernel time for the profiled process: 5.22 → 4.22 s.

## Endpoint ABBA, water-96 DF-HF

| Arm | Build | Cold (s) | Warm (s) |
| --- | --- | ---: | ---: |
| A0 | base | 4.173 | 0.954, 0.951, 0.959 |
| B1 | candidate | 3.967 | 0.968, 0.923, 0.943 |
| B2 | candidate | 3.252 | 0.915, 0.914, 0.924 |
| A3 | base | 4.115 | 0.997, 0.968, 0.970 |

B1 is the slowest candidate process observed. The same source built with native
tests enabled measured 3.255 / 3.263 s cold and 0.910–0.923 s warm in two further
processes, and the untimed prewarms measured 3.38 s (candidate) and 4.27 s (base).

Every arm keeps 11 iterations / 12 Fock builds cold and 1 / 2 warm. Energies agree
to 1e-12 Eh, and errors against the independent dense reference are unchanged:
2.1048e-5 Eh/atom (the existing DF-HF energy approximation failure, retained as in
every earlier cohort), force max 1.9654e-5 and RMS 8.174e-6 Eh/Bohr, which agree to
within 3e-13. The GPU4PySCF 2026-11-17 reference for this case is 0.758 s warm
([readme-h100-20261117](../readme-h100-20261117/README.md)); this change alone
does not close that gap.
