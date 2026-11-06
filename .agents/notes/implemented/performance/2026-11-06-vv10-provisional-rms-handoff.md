# Decision: stop oversolving temporary VV10 operators, not the physical target

Status: implemented; clean paired default qualification passed
Date: 2026-11-06

## Problem

WB97M-V water96 full Direct takes 18 native molecular Focks versus the retained
GPU4PySCF reference's 10 reported cycles / 11 effective-potential builds. Simply
listing native stages is not a causal attribution of the difference. The user
specifically requested isolating provisional oversolving and discarded solver
history, following the unsuccessful guess/DIIS-metric controls in PR #3019.

The incumbent sequence is nine semilocal/RSH initializer Focks, one full target,
four frozen-VV10 Focks, and four final physical observations. Both handoffs clear
CDIIS history and start a fresh target-energy epoch. Provisional energies and
stationarity cannot publish a physical target or analytic forces.

## Mechanism cohort

Actual A100 LSF225855, one fixed five-arm cohort on one source/ELF. Each completed
arm includes one cold constructor+prepare+SCF+analytic-forces+host-export endpoint
and one frozen-seed warm endpoint. A private journal records already-returned
host scalars without DF_TRACE, extra GPU transfers, or a different MD-J route.
These host-journal timings are mechanism evidence, not clean final timings.

| Arm | Cold/s | Molecular Focks | Provisional / target | VV10 sweeps |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 111.947746 | 18 | 13 / 5 | 4 |
| Provisional RMS handoff | 108.286345 | 16 | 11 / 5 | 4 |
| Bounded proposal-history continuation | 119.207417 | 23 | 19 / 4 | 3 |
| Both | 107.985260 | 16 | 12 / 4 | 4 |

All eight completed A100 samples pass all 16 repeat-paired comparisons against
the retained independent static reference with the original gates: energy
1e-5 Eh/atom, maximum force 1e-4 and RMS force 3e-5 Eh/Bohr. This is sampled
default-grid parity, not an independently grid-converged production ranking.

### What oversolving actually costs here

The initializer's actual AO maximum at steps 7/8/9 is approximately
1.835e-4 / 1.418e-5 / 2.718e-6, while its step-7 spin RMS is already 4.954e-6.
Thus the old AO-maximum 1e-5 proposal gate really does wait until step 9; the
previous RMS-only logs could not prove those numerical values.

Using the same 1e-5 tolerance on provisional spin RMS hands off at 7 instead of 9.
The first frozen solve also hands off sooner, but its work is redistributed
across more anchors: the new complete scope sequence is
`seed*7, target, frozen*2, target, frozen, target, frozen, target*2`.
It still executes four frozen Focks, five physical targets and four pair sweeps.
The net saving is **two initializer Focks**, not all of the eight printed-step
difference. J applications fall 18→16 and full/LR K applications 36→32.

Final physical convergence is not an arbitrary delay. Baseline MO gradients
at steps 15/16/17 are about 7.825e-5 / 8.319e-5 / 5.177e-6, against sqrt(1e-9).
Step 16 passes energy but not MO stationarity; step 17 passes MO stationarity
but its energy change is about 1.214e-9. The unchanged-density 18th evaluation
supplies energy confirmation. Neither the MO gate nor that energy epoch may
be replaced by provisional RMS.

### Why simply retaining history loses

The trial deliberately permits cross-operator entries only as approximate density
proposals, never as physical Focks, target energies or convergence evidence, with
a six-update expiration. Nevertheless, the first frozen inner solve grows 4→10
Focks. Total work grows 18→23 despite saving one expensive pair sweep. The
combined arm is only 0.3s ahead of RMS alone in one cold sample; it does not
justify the additional approximate-history mechanism.

This rejects **unrebased proposal-history continuation in this cohort**, not
all history transfer. Historical densities were not retained and old Fock/error
pairs were not rebased to the new operator. A future exact/linearized secant
transfer needs explicit density/storage ownership and an independently validated
mapping. Do not cite this experiment as proof that every reset is optimal.

### Continuous control has a resource-policy confound

The existing outer-off selector also disables automatic resident-LR source
admission: `prepare_phase_source` requires `nonlocal_outer_active` for that
automatic path. The control is therefore an **integrated outer-off/resource
ablation**, not a source-matched pure operator-sequence experiment.

It reaches only eight completed physical observations before the 180s A100
process cap and seven before the 280s RTX5090 cap in Slurm16517. Neither has
a complete cold E+F endpoint or accuracy qualification. Do not invent a
converged iteration count, attribute the whole slowdown to VV10 pair sweeps,
or claim to have fully explained GPU4PySCF's 10-cycle trajectory. The final
latest-master cohort additionally tests outer-off with existing explicit
`GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=1`, restoring bounded LR source admission
under the same 16GiB default ceiling. This does not by itself prove identical
realized resource bytes. Its outcome is retained separately from these timeouts.

The RTX5090 incumbent completes at 193.716044s cold and 14.316116s warm median
(three frozen-seed observations), 18 Focks and four pair sweeps. The later
small-water oracle test in that job is not reached after the control timeout.
The A100 driver retains every later arm after its continuous-arm timeout;
its overall exit1 therefore does not mean four successful arms failed science.

## Production decision and invariants

Use provisional spin RMS by default only for the existing qualified FP64
Standard full-Direct nonlocal owner, while retaining AO maximum for strict,
fitted, mixed, unsupported-model/device-chunk and other unqualified consumers.
An explicit `GENERATIVEQC_VV10_RMS_HANDOFF=0` restores the incumbent proposal
gate; unsupported owners cannot opt themselves into qualification with 1.
Warm starts do not initialize these provisional stages.

Keep all physical energy/MO-gradient and canonical publication checks, fresh
energy epochs, CDIIS clears, proposal/total budgets and final strict fallback.
No precision/grid/screening changes, Direct→DF substitution, CPU/reference
production dependency, GPU-model/size hardcoding or additional resident storage.
Remove experimental history/journal selectors from the final production image.

## Clean default qualification

The production image removes experimental journals and history selectors. With
all force-pair controls unset, each arm measures one cold endpoint and three
frozen-seed warm endpoints. The sole rollback is the RMS handoff selector.

| GPU / job | Baseline cold/s | Default cold/s | Baseline warm median/s | Default warm median/s |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF225862 | 111.882707 | 108.185959 | 15.021796 | 15.034534 |
| RTX5090 / Slurm16584 | 193.680082 | 189.379635 | 14.299161 | 14.340794 |

Cold improves 3.30% / 2.22%, with 18→16 Focks and four pair sweeps on both
devices. Warm retains one Fock and zero pair sweeps; median increases of 0.085%
/ 0.291% are noise-scale observations, not proof of literally zero regression.
Every matched call stays within the declared 1.05 guard. All 32 independent
repeat-pair E/F checks pass unchanged gates. Actual Slurm16584 also passes both
original/moved small-water independent GPU4PySCF tests (default/explicit strict,
cold/warm E+F). Production-image host policy tests: 23 passed.

Earlier host preflights were not scientific failures: the local run had 5 pass
and 18 missing-native-library errors; the first n1 run had 20 pass and three
missing-CCACHE errors. Supplying the actual native image and verified compiler
cache repaired the setup, giving 23 passes. Preserve those initial logs rather
than replacing them with the passing run.

Do not claim the separate warm 1.576s target or the entire 18-vs10 gap is solved
by this cold two-Fock saving. Further history work must rebase across operators,
not simply preserve stale Fock/error pairs.

## Evidence storage policy review

The existing 82MiB aggregate checkout allowance has less than 10KiB headroom.
The active offline replay needs full observable arrays, all negative controls,
and source-matched qualification receipts, not hashes or a favorable summary
alone. Its already-XZ-compressed evidence is small but exceeds that remainder.
This PR explicitly requests a bounded 256KiB allowance increase to 82.25MiB;
it does not waive per-file/review limits, discard prior unfavorable evidence,
move active oracle inputs to network/history-only storage, or publish an
external archive/Release. Review this policy change alongside the retained
payload, rather than silently increasing it for an unbounded benchmark dump.

## Evidence and references

- Issue #2976: plan comment 6676386404, interim decision 6676959159, and clean
  qualification / corrected-control progress comment 6677438637.
- `benchmarks/results/wb97mv-stage-handoff-20261106/` retains the experiment,
  partial controls, final qualification, failures and source/ELF receipts.
- Independent static observables:
  `benchmarks/results/direct-range-warm-anchor-20261105/publication.json.xz`,
  SHA256 `6659c9962ca035cc6c0b5411cf54c6f1f64409a922eda2d00aff29726b65880f`.
- Mechanism source `4ab3b32086eb87caf0041593befb995f46c9c851710cdccf7af338d18c846987`,
  ELF `9702f376c4942a0d1b8e3590416edd50a5effb0faa7aee449eccb88757c118e8`.
- Clean production source `f0b90fe38e919340b96a7bcc28da6a52930b4459de5cf2cb9e90ef3d63565ba4`,
  ELF `8f227dda0d7b29e2a873a845a49d9813ee83ed0845ef2ca6d1f78ad8010005fe`.
