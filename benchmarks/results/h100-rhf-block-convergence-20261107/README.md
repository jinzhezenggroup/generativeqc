# Default-off RHF block-convergence contribution

Three fresh native process pairs support retaining the opt-in 256-thread RHF
matrix walk. Median paired cold reduction is **2.63859281%**, with median saved
wall time **0.259096243 s**. This compares 32 and256 threads on the same measured
core/GPU. It is a component result, with the default still off.

| Pair | Execution order | Cold off (s) | Cold on (s) | Off/on | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Off→On | 10.0877508200 | 9.5558210750 | 1.05566552 | 5.27302621% |
| 2 | On→Off | 9.7951560050 | 9.5367017230 | 1.02710101 | 2.63859281% |
| 3 | Off→On | 9.8661326719 | 9.6070364290 | 1.02696942 | 2.62611756% |

All three cold directions improve. All nine matched warm on/off ratios pass
<=1.05; their maximum is 0.9906338856. Each fresh process performs one cold and
three same-owner, frozen engine-local cold-density warm complete analytic E/F
calls. Cold includes constructor, prepare, Hückel initialization, SCF, forces and
host export. Each arm starts with its own copy of the same authenticated prior
artifact-cache snapshot; tracing and observers are disabled.

Both arms retain automatic cold proposals (cold-screening selector unset),
DIIS 0, XY 1, weighted-Z 1, original requested FP64 and incremental off. Only
`GENERATIVEQC_HF_BLOCK_CONVERGENCE=0/1` differs. Every cold has 15 molecular
iterations/Focks: 8 FP64 proposal and 7 strict-target iterations; each initializer
separately has 7 preliminary iterations/Focks inside cold. Every warm has one
iteration/Fock. These are observations, not a required equality or an assumption
that RMS reduction ordering cannot change a trajectory.

## Numerical and source scope

All 24 complete 96x3 force arrays pass the original independent dense HF96 oracle
limits:1e-5 Eh/atom, 1e-4 maximum force and 3e-5 RMS force. Worst observed errors are
1.37372e-13, 3.63727e-9 and 1.69098e-9 respectively. All 12 full off/on comparisons
pass the original strict 1e-9/1e-7/3e-8 limits; maximum strict force difference is
4.88077e-13. Exact values and all per-call times/work remain in `summary.json`.
The original CPU-spot/reference-reuse qualifications and oracle bytes are retained;
no new reference/oracle calculation was performed for this screen. The24 native
complete E/F calls are new measurements.

Measured source is **513288b77bb6a654cffa33c9aa3aa6b39d134c16**, tree
9338a49061fc5a20a7474c438cf4a147731b415b, native input identity
0138841399c776a2616f2af0d5be63076950ee70e16463f708a2392389c70811.
Core SHA256: e3ef895c3a332103160cdbebd5fc5a13165788b6987f4bfe5422b42f6ad88ade.
Device: GPU-fe8a7e35-47a3-b7bf-8493-d4fecd7c37be, H100 80GB HBM3/SM90,
MIG disabled, driver 595.58.03, CUDA runtime 12090. The evidence commit/publication
tree is separate from measured513; it introduces no new device measurement.

The actual linked selected fixture covers32/256 matrix tails through 768 AOs,
nonfinite/coarse/inactive states, complete retention/copy, current-main energy
refinement and bidirectional native cached-policy refusal. Selected racecheck
and synccheck have strict zero terminal summaries. An old integral/resource
suite and memcheck were not repeated or relabelled as new qualification.

## Boundaries and retained failure

The initial plan used an unrecognized diagnostic-purpose string. Its original
`23ff20e7…` failed receipt and streams remain `FAILED_RETAINED`: refusal occurred
before native load/E/F, with no on stage. A reviewed external adapter changes
only that purpose to the endpoint's existing
`FOCUSED_DIAGNOSTIC_NOT_ALL30_ACCEPTANCE`; the f916 endpoint, core, science,
settings and gates are unchanged. The successful first pair and both added
counterbalanced pairs have fresh output/control/cache paths. No failed scientific
sample was removed or retried.

The first pair saved 0.531929745 s, larger than the prior separate intrusive86d
profile's 0.262619628 s convergence-component time. The profile used a different
source/device/timing regime. It does not justify attributing that first wall-time
change to this kernel. The two added pairs retain the smaller observed effects;
all three are reported rather than selecting the best result.

There is **no new GPU4PySCF timing comparison**, mechanism/kernel-speedup proof,
default promotion or original fixed30 acceptance. Each final qualifying round
still requires contemporaneous baseline/candidate/GPU4PySCF owners on the same
GPU for every original case, including the matched-native warm guard. Three
qualifying rounds must retain the original G4/candidate median>1.05 and
every-pair>1 conditions. These three native/native pairs do not satisfy that
three-arm population or campaign.

## Retention and offline replay

`evidence.json.xz` losslessly retains 139 original payload files/1,251,651 bytes,
including all 69 successful endpoint files, complete E/F arrays/plans/qualifications,
all 48 stage streams, original failure/control, compact source/build/image/review
bindings and selected fixture/sanitizer logs. It also includes the exact previously
retained oracle file, with its originating capsule hash in `cohort.json`. Native
binaries, objects, runtime caches, generated shards and large build logs remain
in their bound owner; hashes do not make omitted bytes recoverable here.

Run with standard Python 3.10 or newer:

```sh
python benchmarks/results/h100-rhf-block-convergence-20261107/replay.py
```

The reader authenticates every retained member and recomputes 24 dense E/F gates,
12 full strict comparisons, all nine warm guards, all six cold times, paired
medians, source/core/device/control/phase-work bindings and the original failure.
It loads no native library and runs no CUDA, SCF, reference engine or oracle.

## Post-measurement integration boundary

The PR also integrates master `7704dd93dec18db0cf506d78ce1c76f0d6d40fe1`
(#3039's shared-provider migration). Its combined native input identity is
`2c113229e644417e47abaf2d30a75a579a2f8096fe08e04a3ca66831cb4d80cc`.
The measurements above still belong to source `513288b77` and native identity
`0138841399`; they are not measurements of the combined source. Host integration
checks and offline replay preserve those receipts without a new GPU run or any
default/qualification promotion. Regenerated Libxc snapshots change checked
source identities only and retain stale status with zero qualified products.
