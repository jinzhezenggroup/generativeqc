# Decision: VV10 confirmation must satisfy the active stationarity policy

Status: implemented; native water96 complete, independent water96 comparison unqualified
Date: 2026-11-05

## Problem

The default WB97M-V target now uses a physical global MO-gradient norm, but
the nonlocal energy-confirmation hold still checked only proposal-density RMS
and physical AO residual RMS/maximum. Those local/normalized bounds do not imply
the MO norm, especially as the system grows or the overlap becomes poorly conditioned.
A held density could therefore never reach the candidate gate. Similarly,
holding a density after its canonical proof was rejected could repeat the same
rejection without applying the still-owned DIIS proposal.

The prior water96 investigation and its analytic physical counterexample are
retained in
`../../proposed/2026-11-05-water96-direct-stationary-confirmation.md`.
That investigation did not establish all stages of the untraced 540-second
timeout, and this repair does not retroactively turn it into a measured result.

## Decision

Share the exact existing physical MO stationarity predicate between candidate
convergence and nonlocal confirmation. A qualified orbital-policy target may
hold its density only after that predicate and the original confirmation AO
and density bounds pass. A rejected final proof forbids the hold, letting the
owned proposal advance through the existing scratch-invalidation path.

Do not require energy convergence, full energy history, or a second physical
target observation just to select a hold: the next unchanged-density physical
evaluation exists to establish those conditions. Seed/frozen operators remain
excluded by the prepared owner. The helper does not authorize publication;
candidate history, target readiness and final canonical/force proofs still do.

Explicit strict/non-orbital callers retain their original AO confirmation
thresholds and need not supply an evaluated MO norm. Neither public tolerances,
the physical MO norm, nor grid/model/backend admission is relaxed. No GPU-name
dispatch, CPU/PySCF production work, Direct-to-DF substitution, new storage, or
extra device-to-host transfer is introduced.

## Observability

The optional existing progress journal records
`nonlocal_confirmation_orbital_blocks` when the old AO/density hold would have
passed but the active MO gate forbids it. It records
`nonlocal_stationary_density_holds` only in the actual publication/update
branch retaining D. These are observations of executed decisions, not new
acceptance gates or clean performance measurements.

## Evidence

The executable host regression constructs S=I, an idempotent eight-AO closed
shell D, an independent physical F and its eigensolver density proposal using
NumPy. The current production policy helpers must reject the old false hold:
AO residual RMS 6.36396e-6 and maximum 9e-6 pass 1e-5, while MO norm 3.6e-5
fails sqrt(1e-9). The same probe covers allowed energy confirmation without
history, the exact MO threshold, NaN/negative/infinite norms, rejected canonical
proofs, and strict-policy isolation.

Focused host tests pass (2); adjacent local host tests pass (78), with 22
nonallocated CUDA cases skipped rather than executed outside Slurm. Exact
new-runtime host qualification in Slurm 15941 passes all 114 selected tests.
Both independent tight same-grid small default/strict E/F/warm/moved tests
pass in 35.54 s, including actual default MO-path admission and strict isolation.

### Bounded water96 result

The single allocated 5090 run completes both native Direct E/F endpoints:

| Native phase | Complete seconds | Public SCF iterations | Total Fock builds |
| --- | ---: | ---: | ---: |
| Cold | 214.366170 | 18 | 19 |
| Own-density warm | 18.774825 | 1 | 1 |

The cold timer includes constructor, preparation, SCF, analytic forces and host
export. The native process rejects PySCF/GPU4PySCF imports and does not borrow
the reference density. These timings include progress instrumentation and
therefore are not a clean statistical performance population or a general 5090
non-regression result. The earlier progress comment's 19 trace completions must
not be confused with the final public cold count: 18 SCF iterations plus a
post-SCF build give 19 cold Focks; warm has one iteration/build.

The actual large input observes two old-AO-hold/MO-candidate mismatches, then
one allowed density hold, six physical MO checks and two accepted canonical
probes across cold/warm, with no rejected probe or unfinished native region.
This establishes the actual water96 trigger and successful solver escape, not
the exact stages or total attribution of the older untraced 540-second timeout.
Cold semantic work is 19 J/38 K, 13 provisional plus 5 target SCF Focks and
4 VV10 sweeps. Warm is 1 J/2 K, 1 target Fock and 0 VV10 sweeps.

The following stock GPU4PySCF Direct reference reaches its 360-second wall cap
without a complete cold sample. The overall shell/Slurm task therefore exits
124 **in the reference phase**, after the native arm returned successfully.
The raw reference JSON still says `running`, has zero samples and is explicitly
incomplete; it is not a valid 360-second endpoint time or a reference numerical
failure. No 96-atom cross-engine E/F acceptance or speed ratio is claimed, and
the cap was not extended or the population repeated. The separate small tight
independent scientific gates above remain valid.

For future work, the completed diagnostic attributes cold iteration XC
100.122151 s (84.114902 s of VV10 pair work **inside** XC), JK 28.732487 s,
optional LR preparation 20.675874 s and stationary forces 43.809581 s. Warm
stationary forces account for 15.696044 s. Nested regions must not be summed
as independent elapsed time. Optimize those measured operators rather than
reintroducing a hold that suppresses necessary density updates.

Native source identity:
`25555f38303c47ae970413639b16d208775763a22dd91c34f2b429379ae645e2`.
Core ELF SHA256:
`53a73ff59f0210c53d6f21afd641cb8c004c3994247e69f06474f478a076e9ee`.
The existing normal O3 SM80+SM120 build is incrementally rebuilt using verified
sccache 0.16.0 and fresh-mtime changed-file overlays. The actual changed
`cuda_ks.cpp` compiler invocation is retained; this is not identity-only relinking.
The build helper's final receipt extraction initially exited 127 because n1
lacked rg, after compilation/linking completed. A grep fallback recovered that
receipt without rebuilding; the original exit and recovery explanation remain.

The existing checked-source Libxc catalog is regenerated with its authoritative
checker; only the current input identities change, capability summaries are
unchanged and historical science remains stale. Its 63 host regressions pass.
This metadata refresh does not change the measured native source identity.

Final CPU-only audit SHA256:
`c9c429f94bc82df9a1c8473a49911908cb1e5253a5b3208359fb6339000d07f8`.
The machine-readable audit explicitly reports the native completion, reference
incompleteness, driver exit 124 and false overall large scientific qualification.

## Consequences and acceptance

Previously suppressed density updates may require more changing-density VV10
pair evaluations; that is necessary work toward the original physical target,
not a reason to hold a nonstationary state. Complete endpoint evidence and
independent E/F remain required before any performance/default claim. Small
system science and an isolated policy test cannot qualify water96 or A100
performance. Historical fitted-LR/DF endpoint times are not full-Direct baselines.

## References

- Issues #2976 and #2987; existing PR #2992.
- `tests/python/test_scf_convergence_defaults.py` and
  `tests/python/test_wb97mv_default_convergence_cuda.py`.
- Ignored local `.artifacts/issue2987/hold-fix-*` receipts and remote
  `/data/jzzeng/issue2987-defaults-20261105/hold-fix-qualification-15941/`.
