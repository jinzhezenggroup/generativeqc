# H100: resident DF-RHF final fixed-point probe

HF/def2-SVP with the README default auxiliary basis, DF mode, water-96 (768 AOs),
fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-opt-1118-qa`. Cold = constructor + preparation + SCF + analytic forces +
host export; warm = three frozen-own-density calls.

Both builds contain #3115 and #3116 (pre-format, identity `d93ee912…`); the
candidate adds this change (`5947a9e1…`). The committed tree is master plus this
change only and differs from the candidate's source in those two PRs and in
clang-format whitespace. `results.json.xz` retains every arm with all samples,
forces and qualification verdicts, the host component traces of the baseline
and candidate, and the H100 native test log.

## Attribution (host component trace, three warm calls)

| Region | master `a6d5a3c00` | Candidate |
| --- | ---: | ---: |
| `final_state_fixed_point` | 79.3–84.1 ms | 12.9–13.3 ms |
| `final_state_weighted_density` | 11.2–13.8 ms | 4.0–6.6 ms |
| `final_state_fock_build` | 46.6–51.2 ms | 46.6–46.8 ms |
| `final_state_validation` | 14.5–20.9 ms | 17.1–21.0 ms |

The last two rows are unchanged work and show only run-to-run variation.
The master fixed-point probe downloaded the device physical Fock, re-uploaded it
for the eigensolve, downloaded the frame, validated it twice with fresh uploads
and downloaded the projector. The resident probe solves the staged F on the
device (8.4 ms `device_eigensolve`) and returns one diagnostic packet.

## Endpoint ABBA, water-96 DF-HF

| Arm | Build | Cold (s) | Warm (s) |
| --- | --- | ---: | ---: |
| A0 | #3115 + #3116 | 3.247 | 0.900, 0.915, 0.925 |
| B1 | + this change | 3.154 | 0.850, 0.844, 0.848 |
| B2 | + this change | 3.159 | 0.838, 0.838, 0.848 |
| A3 | #3115 + #3116 | 3.219 | 0.914, 0.908, 0.918 |

A traced candidate process measured 3.182 s cold and 0.843–0.852 s warm. Every arm
keeps 11 iterations / 12 Fock builds cold and 1 / 2 warm. Errors against the
independent dense reference are unchanged: 2.1048e-5 Eh/atom (the existing DF-HF
energy-approximation failure, retained) and force max 1.9654e-5 Eh/Bohr.

## H100 native tests

`generativeqc_df_final_validation_tests` (device final-state operations, now
including the resident probe and its rejection/promotion path),
`final_state`, `uhf_final_state`, `density_fitting`, `df_eigensystem`,
`df_capture_recovery`, `df_final_snapshot` and `df_spin_admission` pass.
`generativeqc_ks_final_state_tests` fails in its CUDA-only
`cuda_split_hybrid_identity` check ("split-hybrid final-state admission disagrees
with its physical J/K model"); that check calls `validate_final_state` without
backend operations, which this change does not modify. CUDA DIIS and density-seed
tests skip on this allocation.
