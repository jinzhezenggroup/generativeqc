# Guarded default and CUDA DF RHF qualification

This source-bound campaign qualifies restricted CPU/CUDA DF correctness and
measures **CUDA DF RHF**, not DF KS performance. The user-requested default is a
policy decision, not evidence that all eligible endpoints clear the original
strictly greater-than-5% timing gate.

## Measured endpoints

Slurm job 7914 runs 18 fresh processes on its assigned RTX 5090, rotating
Hcore/Hückel order across three repetitions. The timer includes prepared-batch
construction, SCF, synchronized analytic forces and native atomic preparation.
Imports, Calculator construction, process startup, density export and independent
GPU4PySCF reference work are outside the timer. No foreign process is observed on
the assigned device before or after each sample; scheduler visibility is retained.

| Atoms | Hcore median (s) | Hückel median (s) | Wall reduction | Target iterations |
| --- | ---: | ---: | ---: | --- |
| 3 | 0.372555 | 0.379060 | -1.75% | 15 → 14 |
| 48 | 2.242193 | 2.242727 | -0.02% | 21 → 18 |
| 96 | 14.500008 | 13.387533 | 7.67% | 25 → 18 |

The 96-atom paired reductions are 7.85%, 7.58% and 7.71%; 3/48 atoms do not
cross 5%. Hückel preparation medians are 0.001652/0.110424/0.744293 seconds.
Each seed uses seven atomic iterations/Fock builds, one target attempt and no
discarded target work. Target SCF iterations are retained; the public CUDA DF
target-Fock field is unavailable (`null`) and is not inferred from iterations.

Both orbital and auxiliary sets are spherical def2-SVP. The auxiliary set is an
explicit benchmark input within the existing CUDA DF angular domain, not a
production substitution. Universal-JKFIT/Weigend oxygen g shells remain rejected.
The 48/96 geometries belong to `benchmarks.readme_hf_scaling.scaling_cases`; they
are distinct from the historical exact PBE0 campaign fixtures.

All 18 endpoints independently pass GPU4PySCF DF energy, force, final-density and
physical F(D) commutator gates. Maximum errors are 4.68e-11 Hartree, 6.07e-12
Hartree/Bohr, 7.10e-11 and 1.85e-12 respectively; gates are 1e-8, 1e-7, 1e-8
and 1e-8. These are independent numerical gates, not only agreement with Hcore.

## Validation and source identity

`validation.json` retains the actual final command receipt: Slurm job 7915,
210 passing Python tests (CPU and CUDA coverage), no failures/skips, and four
successful native executables. The DF tests cover RHF/PBE/PBE0 energy, analytic
force, final density, physical residual, warm precedence, preparation caps and
bounded retries on both backends. Local host checks add 411 tests and 22 subtests;
two source-owner DF retry regressions also pass. Compiler-cache statistics retain
actual before/after receipts, not an inferred cache-hit rate.

`summary.json` factors common provenance out of all 18 samples but retains exact
scalar/timing/work/isolation fields and SHA-bound original JSON identities.
The source digest hashes the sorted relative source paths/content under
`src`, `include`, `python` and `cmake` using the measured root harness algorithm.
It is `5b468396b152a0eb56a8239075f6d05f491734da68e281b08c5c6bb5adee416e`;
the release library is
`e15c5505ae954603f90b66825e8d68f904c41b9d16fc4b2fff4336e228d427c3`.
The current harness digest is retained and checked as well. Full JSON/JUnit/build
logs stay in ignored local task storage and on n1; no Release, tag or external
archive is created. Slurm accounting storage is disabled, so validation relies on
actual command/JUnit receipts rather than invented final accounting states.

Reproduce each sample with `python -m benchmarks.huckel_df_endpoint --atoms 96
--arm huckel --repeat 0 --output <ignored-run-directory>/huckel-96-0.json` inside
an explicitly finite `srun` allocation, preserving its `CUDA_VISIBLE_DEVICES`.
Use the qualified CUDA library, GPU4PySCF environment, one BLAS/OpenMP thread,
both arms, all three sizes and three separate processes per arm. The harness
rejects foreign GPU processes and source/library changes.

See the [default decision](../../../.agents/notes/implemented/numerics/2026-10-12-default-huckel-df.md)
and the [current user contract](../../../docs/user/initial_guesses.md).
