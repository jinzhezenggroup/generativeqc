# Decision: use isolated resident owners for the 96-atom tuning experiment

Status: implemented
Date: 2026-11-03

## Problem

The first 96-atom water-cluster PBE0/def2-SVP pilot takes more than two minutes
to prepare, converge, warm and measure one endpoint. Repeating cold convergence
for every sample makes a multi-candidate, five-pair experiment disproportionately
expensive. The existing small-molecule receipts do not establish large-system
performance. The native DF admission inventory also has a separate, explicit
small-topology restriction that this experiment must not bypass.

## Decision

Add a benchmark-only resident-session controller. Each isolated child process
loads one native library and one immutable execution policy, converges its own
density, freezes warm-start updates and performs an untimed warmup. The controller
then alternates complete host-returned energy/analytic-force calls between two
owners, reversing side order on successive pairs. Owners are never executing GPU
work concurrently. A final comparison against the original default uses two
fresh owners, not the search's retained state.

The experiment searches existing XC tiles and J/K execution policies, not grid,
functional, precision, screening or convergence approximations. CPU PySCF/libxc
remains an independent untimed oracle. A standalone campaign script uses the
allocated CPU cores for the larger independent solve; the production reference
helper and the existing tuning CLI's one-thread behavior remain unchanged.

The experiment does not install a local profile, emit an execution-tuner recipe,
or change ordinary Calculator defaults. Its resident-session evidence is labelled
separately from the existing CLI's fresh-process-per-replay receipts.

## Invariants

- Every accepted comparison retains at least five alternating-order pairs.
- Both sides pass the native allocation-admission gate and independent E/F gate.
- Grid points, actual Fock/J/K census, source/device/library and numerical controls
  remain fixed. J/K searches additionally keep the resolved XC schedule fixed.
- A timing winner cannot override a failed numerical, work, identity or resource
  gate. The selected composition must pass a fresh final holdout against defaults.
- The two resident contexts consume memory simultaneously; their scoped allocation
  ledgers are not a claim about total driver/context/library memory or its reduction.
- No CPU reference result supplies a production density, integral or timed endpoint.
- DF's <=16 orbital-AO / <=128 auxiliary-AO admission limit remains authoritative.

## Large-reference convergence boundary

The initial direct-DIIS and default-Newton CPU runs stagnated at orbital-gradient
norms around 2.6e-10 to 4.6e-10. Reducing the Newton augmented-Hessian dependence
cutoff alone was insufficient. Retain these failed/interrupted trials rather than
treating their near-identical energies as a converged oracle.

The campaign resumes from its own independent CPU checkpoint, uses complete Fock
rebuilds (`direct_scf=False`), tighter integral screening (`1e-16`) and a Newton
augmented-Hessian dependence cutoff of `1e-28`. No native density or integral enters
this reference. The final solver phase is configured with a `1e-9` gradient
stopping tolerance, but the benchmark independently recomputes and **requires an
actual full-Fock orbital-gradient norm <=1e-10** before allocating resident owners.
Solver convergence is also mandatory. The less restrictive internal stopping
setting is not substituted for this stricter measured acceptance gate; energy
convergence stays `1e-12`, and all native E/F, parity and work gates stay unchanged.

Reference force evaluation includes molecular-grid response. Reference solver
controls and actual residuals must be retained with the receipts and must not be
silently described as the CLI's unmodified small-molecule oracle protocol.

## Source identity boundary

The initial search ran with a temporary CPU-reference threading change in
`python/generativeqc/_dft_autotune.py`. That helper belongs to the canonical
native compatibility inventory even though it did not execute in the timed GPU
endpoint. The resulting checkout identity did not match the native library.
The public CLI correctly refused replay; the search controller had checked the
library hash and native identity, but had not checked the full checkout inventory.

Restore the production helper instead of narrowing that inventory, overriding a
hash, or treating the unused helper as exempt. The maintained controller now
checks the complete checkout identity before loading native code and checks the
worker's native identity. The original 70 paired endpoint samples remain labelled
exploration, with their actual source and library identities and executed sources
retained. A separate, explicit-configuration qualification uses two new owners,
five new pairs and the restored, fully source-matched checkout. Only that latter
run supports the qualified performance claim. No profile is installed by either
protocol.

A first source-matched holdout (job 1464) passed its endpoint gates but omitted
the report-level device identity on the explicit-configuration path. Its separate
postcheck consequently failed before replay or regressions. Retain those ten
samples as an incomplete postcheck, not as the final result. Record the holdout
owner's identity regardless of whether a search ran, exercise that behavior in
the explicit-configuration test, and repeat the complete qualification in job
1465. That run passes its holdout and independent replay. Its DF diagnostic
assertion subsequently stops on the earlier HF inventory-v2 <=16-AO topology
guard, not the anticipated DF-specific auxiliary limit. Retain that failure and
verify the actual frontier in a separate scheduled postcheck (job 1466), alongside
the source-matched GPU regressions. Do not change either admission guard or repeat
the successful DFT timing merely to make the auxiliary diagnostic look clean.

## Rejected alternatives

Do not reuse small-system winners as 96-atom evidence. Do not mutate execution
environment variables around a shared or concurrent Calculator, mix separately
timed search edges into an aggregate speedup, lower scientific work, or remove
the DF inventory restriction to make a large experiment appear supported.

## Evidence and revisit conditions

`tests/python/test_dft96_autotune.py` exercises alternating order, retention of
all samples, independent-oracle rejection, both resource owners, unchanged
semantic work and the XC-versus-J/K scheduling boundary. It also checks that
source mismatches fail before native loading and explicit requalification skips
the search. Job 1465's source-matched five-pair holdout measures 11.789769 to
10.497078 seconds (10.96% less elapsed time) with XC tile 1024 and default J/K.
Its independent fresh native replay passes; job 1466 passes ten opt-in GPU
regressions. The CPU suites pass 91 tests, with ten GPU opt-ins skipped there.
Real-device receipts are retained with the benchmark campaign; warm results do
not establish cold-SCF speedups or a universal molecule-size crossover.

Revisit a reusable production tuning protocol only with explicit lifecycle and
combined-memory qualification. Supporting large DF additionally requires its own
complete native allocation inventory, the earlier HF topology/eigensolver
admission domain, and independent scientific validation; a
successful direct-DFT experiment is not evidence for fitted HF or fitted KS.
