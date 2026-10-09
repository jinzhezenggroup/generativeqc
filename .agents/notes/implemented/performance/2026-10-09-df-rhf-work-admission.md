# Decision: replace arbitrary DF-guess AO bounds with explicit work admission

Status: implemented
Date: 2026-10-09

## Problem

The [initial guarded default](2026-10-09-df-rhf-preconvergence-default.md)
used 200–400 spherical AOs as a conservative rollout interval. Its 230-AO
complete-force and 322-AO energy evidence did not establish crossover points
at either endpoint, or universal profitability inside that interval. Calling
the interval fully qualified overstated the measured coverage. It could reject
useful larger systems and smaller expensive Direct references for no algorithmic
or resource reason.

## Decision

Keep the existing compatibility limits (neutral singlet H/C, spherical orbital
representation, through-f orbital shells, no ECP) and independent bundled JK
basis. These match the supported/tested implementation, not a general theorem
about density fitting. Keep a genuine exact FP64 Direct reference, the original
physical convergence audits, and all existing bounded fallbacks.

Remove both numeric AO limits. Reuse `reference_quartet_direct` from the exact
reference source policy to skip cache-eligible references, conservatively even
when that optional cache might later be refused. Require the dense work proxy
`N_cartesian^2 / (32 * N_JK_auxiliary) >= 1`. This compares the unsymmetrized
full four-center volume with 32 three-center sweep volumes; the existing maximum
DF cycle count supplies the amortization horizon. It is deliberately a transparent
engineering heuristic, **not a calibrated timing predictor, actual integral/Fock
census, universal crossover, or proof of benefit for every admitted input**.
Contraction lengths, screening/symmetry and device/kernel throughput are not
modeled. Adding those to a future calibrated policy requires new endpoint evidence.

Resolve JK shape without allocating the provisional auxiliary. Use the shared
`df_preparation_storage` estimator on actual orbital primitive/shell metadata,
Cartesian/public dimensions and raw JK shell counts before preparing it. Reject
when that preparation peak exceeds the existing capped allowance. The native
DF tile/SCF planner remains the authoritative complete reservation owner and
retains its explicit resource refusal. The 32-cycle/512-MiB limits, minimum
256-MiB available allowance, density validation and cold retries are unchanged.

Expose source/work/resource admission diagnostics in the benchmark, including
standalone `auto-direct hf`, without exporting a fitted physical reference.
Standalone HF measurements are not substitutes for complete E+F evidence.

## Invariants

- No orbital/Fock/DIIS state from DF enters the accepted reference or response.
- Do not reinstate unexplained AO windows or call a proxy a physical work census.
- Preserve source/planner ownership, numeric capacity, explicit seed precedence,
  one preparation per endpoint and real cold fallback.
- Qualification names the actual tested molecules/bases/endpoints; an admitted
  family does not imply every member was benchmarked or every stationary root unique.
- Keep the original and revised binaries/cohorts separate and retain provenance.

## Evidence

Host policy tests cover cache refusal, unamortized work, actual metadata-budget
refusal, supported synthetic shapes on both sides of the former AO endpoints,
unchanged final controls, density validation and cold fallback. Numerical and
complete-endpoint boundary qualification is retained separately below.

### Frozen qualification before the upstream retry repair

While these experiments were running, PR #2162 merged as master
`d02fb5e0c1d5d6c6f6f22ac7ee620d1f6f3c359c`, including a correction that disables
the ordinary DF solver's additional host SCF retry for a preliminary guess.
The following cohort uses `e7f730e872b89d61686318410b4d0650ed52b761` plus the
retained work-admission patch, **before that correction**. Complete timers and
numerical comparisons remain valid for these binaries, but reported DF cycles
describe only the final attempt; internal DF retries cannot be excluded or
their cycles reconstructed. This cohort is not fresh qualification of the
repaired implementation. Latest-master results are retained separately.

The probe/library SHA-256 identities are
`125eabcafc4f041c270da406fafd329573bbd11bffaa387f3a5087693d0846b4` /
`18e486f8c850ee461482f4a9603e348863abc5cd8bbda145ae82f314dd4d0e95`.
All native samples use finite Slurm allocations on node2 RTX PRO 6000 Blackwell,
CUDA 12.9.1, GCC 11.4.0, Release/portable_cuda, architecture 120 and FP64. Matched
CC DIIS 8, packed DIIS and Lambda audit cadence 30 are benchmark controls, not
changes to the public CC defaults. Raw records remain ignored; compact reviewed
records preserve source reconstruction, hashes, allocation, every observation
and full-precision all-pair numerical errors.

- **144-AO ethane / cc-pVTZ**, Slurm 2782: five complete E+F pairs, every pair
  faster. Median RHF including DF changes from `23.251690740` to `16.273158865` s
  (30.013% lower); complete E+F from `105.180449679` to `98.037097872` s (6.792%
  lower). Actual Direct physical Focks decrease **18 → 12**. The proposed shape
  is 160 Cartesian / 484 JK functions, ratio `1.6528925619834711`, and preparation
  estimate 3,970,056 bytes under the 512-MiB cap. All 25 comparisons pass:
  maximum total-energy error `5.400124791776761e-13` and force error
  `1.1331380278534198e-10`. Independent conventional RHF gates pass. Independent
  same-Hamiltonian correlation energy error is `5.4427573559223674e-12`;
  directional-force errors at `1e-4` / `3e-5` bohr are
  `3.2116293444128807e-9` / `9.23985789963444e-9`, below the unchanged `3e-7` gate.
- **34-AO methane and 58-AO ethane / cc-pVDZ**, Slurm 2784: both are work-skipped,
  execute no DF cycles, and keep **13 / 18** Direct physical Focks respectively.
  Reference densities, orbital energies and endpoint energies match exactly;
  maximum force differences are `1.2656542480726785e-14` /
  `4.707345624410664e-14`. Independent conventional RHF gates pass. Proposed JK
  dimensions describe metadata only; no JK integral work is implied. Timing
  variation on these unchanged paths is **not a DF speedup claim**.
- **230-AO ethane**, separate Slurm 2785 pair: RHF `131.410457119 → 87.548135856` s,
  complete E+F `525.174438163 → 479.766835648` s, and **20 → 13** Direct physical
  Focks. Same-Hamiltonian energy/force and fresh conventional RHF gates pass;
  maximum paired force error is `3.2722147214059305e-10`. This allocation also
  passes 94 selected host/native tests. Its standalone 144-AO HF pair is retained
  but is not used as complete E+F performance evidence.

The unchanged exact-reference gates are `1e-10` reference energy, `1e-9` total
energy/density/orbital energies and `3e-9` paired forces. No recorded final guess
refusal or seeded-Direct fallback occurs in the admitted cases. DF physical-Fock
counts remain unknown/null; this statement does not exclude historical internal
DF retries. The small-case skips and the useful 144-AO measurements demonstrate
why 200 was not an experimentally established lower crossover.

## Revisit when

A device-calibrated cost model can account for actual primitive shell-class
work, fitted metric/preparation cost and resident/streamed schedules. Extend
compatibility only with independent exact-reference and final-force validation;
do not convert the accepted reference to DF-HF under this initialization policy.
