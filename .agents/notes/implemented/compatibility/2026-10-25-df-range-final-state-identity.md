# Decision: authenticate the supported DF-primary/exact-LR KS pair

Status: implemented
Date: 2026-10-25

## Problem and decision

PR #2287's retained native KS runner failed final-state validation. The ordinary
CUDA owner already accepts a matched density-fitted primary J/K plus a separately
prepared exact long-range exchange correction, but the final-state validator
required exact primary J/K. Permit the same matched primary pair in the validator.
Retain all spin, backend, operator, domain, screening, generation, physical-state,
canonicality, stationarity and energy gates. Fitted LR and mismatched primary J/K
remain rejected. This changes identity admission, not arithmetic or thresholds.

## Evidence and reproduction

The original failure log has SHA256
`24d0492425b385812edcbc85c78353ce3308ccbbb0ad64e3959b751f479074c4`.
Rebuilding the labeled runner from historical source
`7d8f4ba22883f9c3accfec70a3e8640510ffbe2c` against its retained library
(`ffa95a627075182dadb763218f746056b15139fcc7093f37bce750392816f45a`)
reproduced the rejection at `run_density_fitted_range_exchange_case(true)`.
All 68 transitive local headers were verified against that source. The three
preceding isolated chunk controls passed. This diagnoses that historical failure;
it does not establish that every retained endpoint failure shares this cause.

After integrating master at `39bbedf4a`, analytic nonorthogonal final-state tests
cover both spins, exact/DF primaries, zero/nonzero primary exchange, and invalid
provider/energy mutations. The valid fixture fails against the unmodified
validator and passes with this repair. Native CUDA controls compare energies and
exchange components with the independent CPU endpoint at `1e-10`, and retain the
actual primary and correction identities in the validated snapshot.

On node1 (`ssh n1`), CUDA 12.9.1, sm120, cached Release build:

```sh
build/generativeqc_ks_final_state_tests
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 build/generativeqc_ks_cuda_tests --df-range-exchange
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 compute-sanitizer --tool memcheck --error-exitcode=1 \
  build/generativeqc_ks_cuda_tests --df-range-exchange
```

All passed; memcheck reported zero errors. Slurm visibility was preserved.
Raw repair logs are under the ignored `.artifacts/repair-20261025/` and remote
`/home/jzzeng/tmp/qc-pr2287-repair-20261025/evidence/`.

## Current precision fixture and remaining boundaries

Master `af6a2ecf7a` selects strict generated J for pure-J proposals. The old
precision-census fixture incorrectly required mixed J in every AUTO solve.
It now checks the prepared provider's actual J capability and the local/dense AO
arithmetic separately, with independent CPU energy/density gates. PBE0 retains
mandatory mixed J and strict-refinement coverage. `--precision-census` passes
both spins and default/local/dense layouts on the same GPU.
The three focused host precision/local-AO modules pass 146 tests; changed-file
pre-commit, structural, retention and ownership checks pass.

The full current runner subsequently fails
`compare_wb97mv_nonlocal_chunk_history()` with "bounded WB97M-V nonlocal
SolverRegion changed the converged trajectory". Do not remove that assertion or
claim full-suite clearance. Historical large-system nonconvergence, unsupported
force domains, and negative/unknown performance evidence also remain unchanged.
No default promotion or profitability claim follows from this repair.
