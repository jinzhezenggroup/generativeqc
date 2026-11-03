# 96-atom native hybrid-DFT schedule qualification

## Result and scope

Follow-up: [1024-point CUDA defaults](../default-cuda-xc-1024-20261103/README.md)
apply the measured tile to ordinary semilocal-XC calls without a profile. The
original campaign, source identity, gates and retained samples remain historical
evidence rather than a new-source performance claim.

Separate complete fresh-process cold-start measurements are retained in
`../dft96-cold-autotune-20261103/README.md`; they do not change the warm timing
boundary or historical classification of the observations in this report.

For the repository's `tools/dft_mp_v1/inputs/water32.json` water cluster
(96 atoms, 768 spherical def2-SVP AOs), PBE0-RKS selects XC tile **1024** and
keeps the incumbent J/K policy. A fresh five-pair holdout measures the complete
warm energy plus analytic-force endpoint at **11.789769 -> 10.497078 seconds**:
**1.123148x**, or **10.96% less elapsed time**. The paired bootstrap one-sided
95% lower speedup bound is **1.113716x**.

These are fixed-density warm replays, not kernel timings, cold-SCF speedups or
tuning-cost-amortization claims. Preparation, initial convergence and warmup
are excluded. No speedup is generalized to arbitrary 96-atom molecules, other
bases, other quadratures, UKS or fitted calculations.

Slurm job **1465** runs on n4 / RTX 5090 / SM120. It reuses the normally built
Release library from source `7a638f55c`; the native and runtime source files are
unchanged in merged master `46ad7c208`. No generated kernels are compiled for
this qualification. The complete checkout and native input identities both are:
`413127726534f605780873d80f46ed23e717969d07423c3e067a6293cbd32496`.
Native library SHA-256:
`54ae88e0820c08d869fd47ed7ccff744a6bd9e67e8982d62662769d77cd58328`.
Executed controller SHA-256:
`6e2fc9da7946261824e111f8cd13203c2c361fabff9f2b3b276d4d07a494357e`.

## Exploratory candidate search

Slurm job **1461** searched the following candidates. Its checkout had a
temporary threading change in the untimed CPU reference helper, making its
full canonical source identity
`b05f49f2bdad2a36516d1063f7055d8727b3b310386d255c5e7630101f97b137`,
different from the native library identity above. These rows are **exploration,
not fully source-matched qualification**. The original executed controller and
helper are retained unchanged; the historical identity can be reconstructed
from those bytes. The maintained controller now fails closed on a full-checkout
mismatch, and the production helper is restored to the native build's source.
Only a new source-matched, explicit-configuration holdout supports the headline.
The exploratory controller's SHA-256 is
`8276cbb32c3804f239c2827cc46431a3234b791ae2d4b0fafb2ef0ead182c3ab`;
it is distinct from the qualified controller identified above.

Every row retains five alternating-order pairs. XC candidates compare against
the original native tile 256. J/K candidates compare against the selected tile
1024 with incumbent J/K, so these rows must not be multiplied into an aggregate
speedup. The independent final holdout above measures the final configuration
directly against the original defaults.

| Change | Baseline -> candidate, seconds | Speedup / lower 95% | Decision |
| --- | --- | --- | --- |
| XC tile 512 | 11.780861 -> 10.758018 | 1.095077 / 1.089515 | Exploratory gain, superseded by 1024 |
| XC tile 1024 | 11.774048 -> 10.504712 | 1.120835 / 1.117297 | Search winner |
| XC tile 2048 | 11.768322 -> 13.298370 | 0.884945 / 0.882234 | Reject: 13.00% slower |
| Disable MD J, request Rys J | 10.514874 -> 15.045816 | 0.698857 / 0.695444 | Reject: 43.09% slower |
| K task schedule `fill` | 10.528292 -> 10.795785 | 0.975222 / 0.968206 | Reject: 2.54% slower |
| K task schedule `primitive` | 10.529405 -> 10.792763 | 0.975599 / 0.971889 | Reject: 2.50% slower |

All six candidates pass independent numerical, both native resource-owner,
native-identity and semantic-work gates. The initial controller did not check
the full checkout inventory, so those gates do not erase the source mismatch.
The slower rows fail the performance gate as well. In particular, the Rys J
policy that won small water cases must not be transferred to this large system
as a presumed improvement. No independently source-matched J- or K-specific
speedup is established; the final configuration keeps both policies at default.

## Scientific, work and memory controls

The functional and exact-exchange coefficient, Treutler level-3 quadrature,
FP64, spherical basis, neutral singlet state, batch size one and geometry stay
fixed. Native controls are `max_iterations=100`, DIIS history 8, energy tolerance
`1e-12`, density tolerance `1e-10` and screening tolerance `1e-14`.

Each timed call returns complete host energies and forces and reports **1,078,336
grid points, one actual Fock build, one Coulomb application and one exchange
application**. J/K trials additionally preserve the resolved XC schedule.
The selected configuration's fresh holdout has maximum independent energy error
`4.74e-14` Hartree/atom, maximum force error `6.02e-11` Hartree/Bohr, and force RMS
error `1.92e-11` Hartree/Bohr. The existing E/F and much tighter schedule-parity,
translation, same-iteration and significant 1.02x improvement gates are unchanged.

The selected tile increases the observed owned numeric device peak from
623,529,284 to 861,759,980 bytes. Both sides pass their native capacity admission
with no rejected allocations. This is a latency win, not a memory-reduction
claim; scoped native ledgers do not include all driver/context/library-internal
memory. The two idle/resident owners consume memory simultaneously, but never
execute GPU work concurrently.

## Independent reference and protocol

CPU PySCF/libxc 2.14.0 supplies the independent energy and analytic forces with
molecular-grid response. Slurm CPU job **1460** uses 16 allocated cores, full
Fock rebuilds, integral screening `1e-16`, Newton augmented-Hessian dependence
cutoff `1e-28` and an independent CPU checkpoint with a strictly verified
molecular model. No native density or integral supplies this oracle.

The final CPU solver phase is configured with energy convergence `1e-12` and
gradient stopping tolerance `1e-9`. The benchmark does **not** trust that looser
configured gradient setting: it independently recomputes and requires an actual
full-Fock orbital-gradient norm **<=1e-10**, measured **2.034778735e-11**, as well
as solver convergence. Earlier direct-DIIS and default-Newton attempts that
stagnated or timed out are retained separately and are not accepted references.

`benchmarks.dft96_autotune` amortizes initial convergence within each isolated
resident child. Each child freezes warm-density updates, warms once, then runs
only controller-requested complete E+F calls. Side order reverses on successive
pairs. The final holdout starts two fresh children. This is explicitly a
**resident-session** protocol, distinct from the tuning CLI's fresh process per
timed replay. It does not install a local profile, emit a production execution
recipe or change ordinary Calculator defaults. The user's profile cache remains
unchanged. For this exact workload, the explicit selected option is
`KsOptions(grid=GridSpec.treutler(3), tile_points=1024)`, with no J/K environment
overrides.

The independent fresh native replay in job **1465** passes numerical, resource,
work and schedule-parity gates. Job **1466** passes all **10** opt-in real-GPU
regressions (71 deselected, 122.54 seconds). The four targeted CPU test suites
pass **91** tests, with the 10 opt-in GPU cases skipped in that CPU invocation.

## DF boundary and reproduction

This is the historical boundary at the source identity recorded above. The
later explicitly budgeted RHF DF96 admission/qualification is documented
separately in `../df96-local-autotune-20261103/README.md`; it does not relabel
these earlier refusals or this campaign's direct-DFT timing as a DF result.

The positive result is direct PBE0-RKS, **not DF96**. The native DF admission
inventory still requires <=16 orbital and <=128 auxiliary AOs. This experiment
does not remove that restriction or claim large fitted-RHF / composed KS-DF
resource qualification. A separate native DF96 bounded-resource admission probe
actually stops at the **earlier HF inventory v2 ordinary <=16-AO topology gate**;
it does not reach the DF-specific auxiliary-AO limit. The separate public-CLI
attempt exhausts its explicit 60-second budget in the independent CPU reference
worker, before measuring any candidate. It publishes no qualified recipe. That
timeout is not relabelled as the native topology refusal. Neither probe is a
qualified DF96 timing or performance result.

`receipts.json.xz` retains the complete geometry and controls, 70 exploratory
endpoint samples, a further 10 source-matched endpoint samples from an incomplete
postcheck run, and 10 final qualified endpoint samples (five pairs). The earlier
source-matched job **1464** passed its timing/scientific gates but stopped before
replay and regressions because its report omitted a top-level device identity.
That reporting defect is fixed and protected by a behavioral test; job 1464 is
not relabelled as the final qualification. Job **1465** then passes both its
source-matched holdout and independent native replay. Its subsequent DF probe
stops a diagnostic assertion because the HF guard fires before the anticipated
DF-specific guard; this is not a DFT timing or numerical failure. Follow-up
Slurm job **1466** verifies the actual HF refusal and runs real-GPU regressions
separately, without rewriting or rerunning the job-1465 timing samples. Full
independent reference E/F/residuals, allocation/work
observations, fresh replay and regression transcripts, earlier unsuccessful
attempts, executed scripts and their hashes. CPU checkpoints remain ignored
local artifacts; their checksums and complete molecular input models are
retained without publishing native binaries or checkpoint binaries.

To repeat GPU measurement, decode `reports["plan.json"]`, bind its library path
and hash to a normally compiled matching source, and write the archived
`reports["reference.json"]` to the plan's reference path. The oracle can also be
recomputed independently from the retained molecular model and executed solver
script, using an independently generated CPU-only starting checkpoint. Run:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:20:00 \
  python -m benchmarks.dft96_autotune plan.json --output new-run --repeats 5 \
  --configuration '{"tile_points":1024,"environment":{}}'
```

Preserve Slurm device visibility and set the source checkout's Python/native
library environment normally. New runs must retain their own identities and
samples; do not pool them with this campaign or claim a universal optimum from
this bounded six-candidate search. Omit `--configuration` to repeat that search,
with a correspondingly larger finite Slurm time limit, instead of only running
the fresh qualification above.
