# Bounded cold incremental proposals, 2026-10-23

Measured base: `afa0a1690be3c0c46774b0f3c2044b8f9a37fd9e`. Incremental mode remains opt-in.

The measured V4 precursor used a PBE/PBE0 guard; the final head removes that
extra whitelist and inherits the existing provider/Standard/final-proof guards.
Three rotated-order V4 full E/F rounds on RTX5090, PBE0/def2-SVP, spherical AOs,
stock grid and Direct FP64 give these cold medians (seconds):

| Atoms | Master | V3 | Candidate | Candidate/master |
| --- | ---: | ---: | ---: | ---: |
| 12 | 3.16902 | 3.16159 | 3.08973 | -2.50% |
| 48 | 13.12015 | 12.81732 | 12.69048 | -3.27% |
| 96 | 30.40414 | 29.83235 | 29.03728 | -4.50% |

Cold includes construction, preparation and E/F, excluding Python/CuPy startup.
Fock counts stay 10/9/9; deltas increase from three to six. Warm stays one full
Fock. Independent-process 96-atom warm medians are 8.26583/8.36141 seconds
(master/candidate, +1.16%). Same-binary adjacent controls give 8.79706/8.55941
seconds (off/on), retaining all long tails; this does not establish a warm
speedup or exactly zero latency impact on arbitrary inputs.

All 160 recorded E/F endpoints pass independent full-density GPU4PySCF references
at E/atom <2.5e-6 Eh, force max <2.5e-5 and RMS <7.5e-6 Eh/Bohr. Observed maxima
are 6.02e-12, 2.04e-6 and 6.35e-7. Coverage includes PBE/PBE0 water clusters,
ammonia, methane, displacements, warm/reset, strict tolerances and interval
controls. Eighty-one host and two CUDA regression tests pass. These are sampled
default-grid solver checks, not quadrature-convergence or general speed claims.

[qualification.json](qualification.json) retains full-precision timing samples,
work counts, per-record worst observable errors, software/binary identities and
the exact measurement protocol. Recompute medians from each `timings` population;
summing the original validation counts gives 160, and each stored worst error
must satisfy the corresponding gate. Raw force arrays, detailed logs and source
snapshots remain in the original workspace evidence directory
`../evidence-delta-d-opt-20261023/`; they are not bundled into this compact record.
The CUDA regression test independently recreates the scientific reference.

Reproduce scientific coverage under a finite Slurm allocation with
`GENERATIVEQC_DFT_CUDA_TEST=1 python -m pytest -q tests/python/test_ks_incremental_convergence_cuda.py`.
For timing, use `benchmarks.readme_hf_scaling.scaling_cases()` water-12/48/96 inputs
and `benchmarks.readme_defaults.native_calculator("pbe0", "standard", "direct", input_dir)`.
In each fresh process synchronize before timing construction, prepare and the
first `execute(strict=True, properties=("energy", "forces"))`; then freeze warm
updates and time two further synchronized calls. Select the recorded libraries
and incremental flag in the recorded rotated order. The reference uses the
same scientific inputs, tighter tolerances and grid-response forces; suppress
both RKS density/potential history via `configure_reference_full_fock`.

Measured V4 CUDA KS SHA256:
`ab6e1e94a3423b22af0d010630e03f34f800f77da89f2510d449c69339e3c086`.
Measured V4 library SHA256:
`4789e2e8722c4151f82946bb40e7d29ae53add737363d9c3d5968836c1120b95`.

See the [decision and boundaries](../../../.agents/notes/implemented/performance/2026-10-23-bounded-incremental-cold-proposals.md)
and [current user contract](../../../docs/user/ks_options.md#experimental-incremental-direct-jk-and-resource-budgets).

The final head adds 40 independently checked endpoints: B3LYP/r2SCAN/M06-2X
12-atom cold/warm/moved/reset, B3LYP/r2SCAN strict controls and 48-atom resident
validation, plus a PBE0/96 bridge (master/V4/head cold: 30.18497/28.69142/28.83080 s).
The original and new cohorts stay separate
in `head_generalization`; the bridge is one sample, not another three-round claim.
All 200 recorded endpoints pass. Final-head tests: 81 controller/resource,
130 promotion-inventory and 5 independent CUDA tests passed.
Final CUDA KS SHA256: `7218190a885f66e287a4e8ee65052e3fbca05426a42c51229412cb30cab8b9ad`.
Final library SHA256: `16174b3618d8124e2b18e872c9b0a4908e7c7f9f3453508da2f04d78602e395f`.

## Coarser provisional screening follow-up

The next candidate explicitly requests 1e-9 screening for at most eight eligible
delta actions per solve, with existing producer caps retained. Prepared physical
screening remains immutable. Full/warm/force actions and the two closing physical
energies retain the original cutoff; unsupported producers fall back. Admission
adds no functional whitelist. The global incremental selector remains opt-in;
`GENERATIVEQC_KS_INCREMENTAL_COARSE_SCREENING=0` selects the previous policy.

Three new rotated-order rounds (same scientific settings and complete endpoint
protocol above) give these cold medians in seconds:

| Atoms | Master | Previous V5 | New | New/master |
| --- | ---: | ---: | ---: | ---: |
| 12 | 3.29684 | 3.33437 | 3.13103 | -5.03% |
| 48 | 13.13335 | 12.51437 | 11.73880 | -10.62% |
| 96 | 30.37949 | 29.01222 | 27.19843 | -10.47% |

Warm medians (master/new) are 0.39101/0.38869, 2.68802/2.69306 and
8.39806/8.33651 s; all perform one full Fock. Cold Focks remain 10/9/9 with six
deltas. The new 96-atom result is 6.25% faster than V5 in the same rounds.
Timing is sampled, not a universal latency guarantee. Separate intrusive traces
show J/K 8.81 to 6.86 s, including K 6.82 to 4.84 s; they are excluded from rankings.

[coarse-screening.json](coarse-screening.json) retains the new full-precision
populations, all 134 independently qualified E/F endpoints including pilots and
diagnostics, source/binary identities and worst errors. Maximum errors are
6.03e-12 Eh/atom, 2.04e-6 maximum and 6.35e-7 RMS Eh/Bohr. Seven independent
CUDA tests and 222 host/controller/inventory tests pass. The sampled functional
and lifecycle coverage is described in the
[decision note](../../../.agents/notes/implemented/performance/2026-10-23-provisional-delta-screening.md).
Raw source, forces, failures, reference receipts and runners remain in
`../evidence-delta-d-expand-20261023/`. The measured native base remains
`afa0a1690be3c0c46774b0f3c2044b8f9a37fd9e`; the PR integration base is `bbd41a427`.
The measured final library SHA256 is
`ad3f854f5a22351d470a59258a1f64e4e5cdc96fa60302fdd1dd3eb4bb6d4add`.
