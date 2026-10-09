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
retains its explicit resource refusal. Qualification additionally showed that
streamed preparation can cost more than the saved exact Focks. Admission now
queries that same source-backed dense/packed planner with actual combined
orbital/auxiliary/dummy metadata and lazy DIIS charges, and requires full retained
three-center storage before generating factors. This is a resource/layout guard,
not a reintroduced AO upper bound. The 32-cycle/512-MiB limits, minimum
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

### Repaired-master work-only policy, before resident admission

The fresh build starts from merged master
`d02fb5e0c1d5d6c6f6f22ac7ee620d1f6f3c359c` plus the current admission change.
Its probe/library SHA-256 identities are
`8fe6c6a50468d4eb642173dfa944bd11767ff6ed074d04f49531a73b07c76c74` /
`bd433b0a2e3e0ecd68912fa9b54dde74df50b17dce477c8eb946cb7df7264773`.
Toolchain and matched native descriptor controls remain as above. The retained
sources exactly match immutable commit `7cc745041`; all 12 scientific/benchmark
hashes are verified against its Git blobs. To reconstruct the pre-repair cohort,
apply the retained reverse repair patch to that commit. The original longer
admission patch remains recoverable in the same commit's Git history. The current
resident-only change has its own forward patch against `7cc745041`. These deltas
are not standalone patches against master. Deterministic gzip stores complete reviewed
records within the repository's aggregate evidence budget, without changing any
of the seven original #2162 receipt members or publishing external archives.

Slurm 2786 supplies one fresh-process complete E+F pair per case, not another
five-repeat cohort and not pooled with the historical measurements:

| Case | Direct RHF (s) | Auto RHF incl. DF (s) | Direct E+F (s) | Auto E+F (s) | Direct physical Focks |
| --- | ---: | ---: | ---: | ---: | --- |
| 144-AO ethane | 23.761879381 | 16.356134614 | 106.234459272 | 99.032049898 | 18 → 12 |
| 230-AO ethane | 130.387988312 | 91.270814356 | 524.481676108 | 488.356870697 | 20 → 13 |

Complete E+F decreases **6.7797% / 6.8877%**, including the now genuinely bounded
single-attempt DF work. Each reports 13 DF SCF cycles with complete preliminary
cycle counters, no refusal or seeded-Direct fallback, and unknown/null DF physical
Fock counts. Paired state/full-force, fresh independent conventional RHF, independent
same-Hamiltonian correlation energy and both directional-force steps pass the
unchanged gates. Maximum independent correlation energy error is
`5.4427573559223674e-12`; maximum two-step directional-force error is
`9.239848795805639e-9`. Physical translation and response residual audits pass.

The same finite GPU allocation passes **96 selected host/native tests** in
138.73 s, including the inherited exhausted-guess/single-attempt regression and
ordinary-DF retry-preservation tests. The corresponding host-only run passes
84 tests with 12 optional CUDA/PySCF skips. Staged hooks and warnings-as-errors
Sphinx builds pass; the admission change does not modify generic compiler
`performance`/`production` promotion states or public CC defaults.

### Rejected streamed initialization

Slurm 2787 tests 414-AO butane (470 Cartesian / 876 JK functions) with the
repaired work-only policy. The 512-MiB plan streams factors. Preliminary DF takes
`621.562225737` s / 16 SCF cycles; exact Direct work falls from **23 → 18** physical
Focks, but inclusive RHF rises **1412.964970839 → 1727.131883383 s**. This is a
**performance rejection**, not evidence supporting a larger default domain.
The dense-volume heuristic alone misses this recomputation cost.

Paired reference energy/density/orbital differences are
`1.7053025658242404e-13` / `6.354180792644826e-10` / `1.2740364319085984e-11`,
but independent RHF density/orbital maxima are `1.995793696973891e-9` /
`2.137697463489019e-9`, exceeding the unchanged `1e-9` gates. The cold control
also contributes to these independent failures. Do not loosen the gates or call
this case fully numerically qualified. The separate 414-AO correlation oracle
hits its finite 2:30 Slurm limit; an attempted finite extension is denied and
no independent correlation result is fabricated. Larger complete-force and
independent correlation qualification remain open work, not prerequisites that
have silently been declared successful.

The current policy uses the shared executable resident-storage decision to skip
this streamed proposal before any DF integral/SCF work. It does not raise the
preliminary memory cap or alter exact-reference convergence controls to rescue
this case. The qualifying 144/230-AO full E+F observations remain distinct from
this rejection and from subsequent resident-guard validation.

The separate pre-repair complete **energy-only** pair (Slurm 2783, historical
probe/library identities above) reaches the same rejection: **4739.969042576 →
5070.842185943 s**, about 7% slower despite **23 → 18** exact physical Focks.
Its inclusive RHF is `1409.678828603 → 1723.071871936 s`, including
`619.865197047 s` of DF work. Paired total-energy error is
`2.5579538487363607e-13`; independent RHF gates still fail as above and no
independent correlation or complete-force qualification is claimed. This is
the complete-endpoint evidence against promoting streamed initialization, not
an isolated-kernel inference. Historical DF cycles still have the documented
retry-reporting limitation.

### Final resident-only policy qualification

Slurm **2788** on October 10, 2026 supplies new, separate complete E+F pairs
after adding the resident-layout admission. Probe/library identities are
`1f7fd6f62b1aaab88e0973e2bf99d19bf35255a3b2ba54a68c55afae0e503c04` /
`26c3e4b90c1ea5168de38cd4b97a59baa168f9f2a3d5b0f248eb08f9c34df015`.
The retained forward patch against `7cc745041` reproduces all 12 measured
source hashes; original #2162 receipt members remain byte-identical.

| Case | Direct RHF (s) | Auto RHF incl. DF (s) | Direct E+F (s) | Auto E+F (s) | Direct physical Focks |
| --- | ---: | ---: | ---: | ---: | --- |
| 144-AO ethane | 23.468501282 | 16.450619956 | 106.274643528 | 99.403362322 | 18 → 12 |
| 230-AO ethane | 130.483474699 | 87.504512813 | 522.489890368 | 477.593147083 | 20 → 13 |

Both complete endpoints improve (about **6.5% / 8.6%**). Each executes 13
single-attempt DF SCF cycles without refusal or seeded-Direct fallback. All
unchanged paired state/full-force, independent RHF, independent correlation
energy, two-step directional-force, translation and response gates pass.
Maximum paired force error is `3.2712099695686447e-10`; maximum independent
energy error is `5.4427573559223674e-12`, and maximum directional-force error
is `9.239845340236474e-9` under the unchanged `3e-7` gate.

The actual native helper's 414-AO admission query now returns **BudgetSkipped**
before any DF work: **zero DF cycles/seconds**, detached density empty, and
`0.000172461 s` complete query time. Its selected streamed value plan has
532,834,339 workspace bytes under 536,870,912 available bytes, but cannot retain
the full tensor. A fitting memory limit alone would have admitted that slow
schedule; the executable storage decision rejects it. This cheap query is an
admission regression, **not another complete 414-AO endpoint benchmark or an
independent scientific qualification of that case**.

The same finite GPU allocation passes **96 selected tests** in 138.84 s;
host-only validation and staged hooks/Sphinx are retained separately. Reviewed
full-precision records reside in `work-admission-resident.json.gz` in the
existing benchmark publication, separately from historical and rejected
work-only cohorts. No universal benefit claim or fixed AO interval is restored.

## Revisit when

A device-calibrated cost model can account for actual primitive shell-class
work, fitted metric/preparation cost and resident/streamed schedules. Extend
compatibility only with independent exact-reference and final-force validation;
do not convert the accepted reference to DF-HF under this initialization policy.
