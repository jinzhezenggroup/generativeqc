# Decision: bind DF shell precision to the verified final density

Status: implemented bounded numerical candidate; useful acceleration remains open
Date: 2026-10-14

## Ownership and admission

RHF finalization passes an immutable allowance and synchronous binding to its
strictly verified final host density/token. The DF response validates the exact
host span, immutable plan/item identity, nonzero current solve epoch, determinant
generations, standard restricted full-range fitted model and strict precision.
This authorization is separate from optional SCF-factor/projection leases.
Neither the allowance alone nor an equal-sized external density authorizes it.
The binding remains alive through the response's successful or exceptional drain.

The first integration incorrectly also required an occupied-factor lease. The
default 96-AO cold endpoint then silently stayed strict after final correction,
while a frozen replay did narrow. Slurm 2990 exposed this; Slurm 2993 checks
all cold/frozen/moved phases after separating verified-density authorization
from cache eligibility. Do not repair this by forcing every precision mode onto
an occupied schedule or by adding another eigensolve just to authorize narrowing.

Admission retains 24–96 orbital AOs, at most 512 auxiliary AOs, 2–12 atoms,
through-d orbital/through-f auxiliary shells, full-rank metric and the default
`1e-10` metric cutoff. Convergence remains energy <=1e-10, density <=1e-8 and
screening <=1e-12; reference exports remain strict. The independent floors are
1e-6 Eh energy, 1e-5 maximum force component and 3e-6 RMS force, not a per-call
mathematical certificate. Explicit observable targets retain their actual audit.

Only the existing optimized shell schedule narrows. Generic metric/three-center
derivatives and all response-weight/conditioning work stay strict. The bounded
three-counter allocation adds 24 host/device/readback bytes to the response's
existing arena accounting. Resource failure drains the rejected arena and retries
without the precision request and consumed optional projection lease. No output
or precision evidence is published from that failed attempt.

The synchronized finite complete force publishes private model ID 7 with actual
task attempts, prepared primitive products and strict task retries. The parser
reports `rhf-df-shell-budget-fp32-v1`, not direct-HF weighted orders or XC points.
Energy-only calls and failed peers cannot inherit the record. The broader SCF
operator census remains incomplete; no Fock totals are invented for this owner.

## Independent evidence and negative fixtures

Slurm 2994, main library
`48eef8f7ec0910cb99b369ae478e75fba3a51a69d0831159a75338d96a3a1891`, passes
14 new complete force tests. PySCF independently supplies the fitted metric,
SCF and analytic force including auxiliary/Pulay response. Tests cover full-rank
Cartesian/spherical cases, bucket/group/packet execution, strict return, tight
budgets, energy-only calls and public default 96-AO cold/frozen/moved phases.
There is no CPU derivative oracle in production. The later failed-peer test is
additional and needs its own scheduled evidence.

The Cartesian cc-pVDZ-JKFIT water fixture has retained rank 130 of 131, condition
number about 6.5e7. Its missing mixed record was correct full-rank rejection,
not a license to weaken that gate. Failed jobs 2986/2988/2992 are retained.
The fixture now checks strict fallback on the **same** inputs; added Cartesian
def2-SVP auxiliary cases independently cover full-rank FP32 force execution.
No numerical tolerance was loosened. Molecular underflow guards legitimately
cause task retries; requiring zero retries was not the precision contract.

The existing strict regression's BLAS charge-dot assertion failed first in 2981.
Job 2984 reproduces it with both the pre-shell-work library
`3d854b3b05ceb17d2c330a4b231d752d65e18eec01a121c35be893fb7e0fe792` and the new
candidate. The old source predates integrated master, so this establishes a
pre-existing failure, not exact-master parity. Energy/force gates passed before
the generic-route counter assertion. Explicit `RESPONSE_SPACE=dense` and
`COULOMB_RESPONSE=panels` exercise that fixture's intended algebra; all 24 compact
cases pass in 2994. Do not relabel this as a pass of the unchanged default suite.

## Retained losing endpoint and causal profile

Job 2995 runs the actual README CLI with independent reference/FP64/Fast owners,
one cold, five frozen, one moved and five moved-frozen calls per mode/size on
n2/PRO6000. All 72 calls pass their numerical gates. At 96 AOs the maximum force
error is 4.80e-7 Eh/bohr; every Fast call uses one solve with matching strict/Fast
SCF iterations. Each reports 194,203 task attempts, 483,977 prepared primitive
products and 13,654 task retries. Fock counts remain unavailable, not zero.

| AOs | Cold ratio | Frozen ratio | Moved ratio | Moved-frozen ratio |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.9143 | 0.1663 | 0.4400 | 0.1674 |
| 96 | 0.9912 | 0.9503 | 0.9869 | 0.9481 |

Ratios are FP64/Fast. The 24-AO generic schedule has **no admitted FP32 work**
and pays two strict audit solves; it is not accelerated support. The 96-AO loss
cannot be hidden by reporting float task counts or a different strict schedule.

Intrusive Nsight job 2996 isolates three frozen calls. Strict/Fast shell kernels
take 85.732/86.118 ms, generic derivatives 10.572/10.487 ms, other kernels
68.452/68.411 ms. Class 223 improves 10.879 -> 6.003 ms, but low-angular 112 and
113 lose 3.737 -> 5.846 and 3.332 -> 5.394 ms. Guards, local summation and strict
geometry preparation can erase narrowing gains; this is not general acceleration.

## Rejected deferred-vote experiment

An experimental typed shell latched each representability/cache rejection per lane and
votes once at the final pre-scatter transaction boundary rather than twice per
primitive. Float generated graphs have fixed indices/bounded loops; nonfinite
arithmetic cannot change memory access. Every unsafe input remains latched;
all nonfinite output/compensation channels still trigger whole-task strict
replay. Strict scheduling and arithmetic remain unchanged.

This changes **attempted work**, not the error gate: a rejected two-by-two-by-two
task prepares eight products before replay rather than stopping after one/two.
Counters and tests report that actual work. Slurm 2997 passes all 48 native
independent tests and four concurrent/retry cases under each of memcheck,
racecheck and synccheck, with zero errors/hazards/warnings. Job 3002 also passed
15 HF-DF, 12 direct-HF, six independent finite-difference DF and 24 explicitly
dense compact-shell checks, without skips in those selected populations.

Clean README job 3003 nevertheless rejected the change: all 72 calls passed
accuracy gates, but the 96-AO cold/frozen/moved/moved-frozen ratios became
0.9867/0.8063/0.9371/0.8068. Prepared products increased from 483,977 to 619,571
with unchanged 194,203 attempts and 13,654 retries. Fewer votes did not compensate
for extra rejected work. **The experiment is reverted**; the current kernel again
rejects early. Preserve the experimental binary and overlay only as negative
evidence, not the current implementation or performance claim.

Job 3001 separately reproduced the existing default final-state iteration
assertions (`[2] != [1]`) on both the older coherent pre-shell source/library
and the new integration. This is pre-existing evidence, not exact-master parity.
The independent finite-difference subset above passes; the unchanged default
final-state suite is not described as passing.

## Cost-selected follow-up

Jobs 3004/3005 built a provisional selector retaining FP64 for orbital bra pairs
containing only s/p shells, and selecting FP32 only when either orbital shell
has angular momentum at least two. This follows the per-class losses above,
without changing the optimized schedule or the physical/error gates. It is
partial precision, not all-FP32.

Native dispatch value 2 identifies that selector. The Python reader reports
`df_shell_sp_bra_contractions` as strict only for model ID 7 with dispatch 2;
historical dispatch-1 binaries really attempted all classes and must not acquire
the new label merely by loading newer Python. Native probes retain coverage of
every typed class independently of production cost selection.

Job 3006 passed all 57 selected endpoint/regression checks with main library
`2288d259168fab460329e2155f204c9f4c3fc58e0b99edf1518ed896519dc199`:
15 new HF-DF, 12 direct-HF, six independent DF finite-difference and 24 explicit
dense/panels compact-shell cases. None of those selected cases skipped; the
previously reproduced unrelated default-suite assertions remain separate.
The host parser/policy/admission/runner subset passed 225 checks and touched-file
hooks passed after formatting. Historical dispatch-1 provenance remains tested.

Job 3007 ran the actual README CLI, again with 72 independently gated calls:

| AOs | Cold ratio | Frozen ratio | Moved ratio | Moved-frozen ratio |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.9007 | 0.1661 | 0.4360 | 0.1660 |
| 96 | 1.0035 | 1.0063 | 1.0027 | 1.0051 |

The 96-AO maximum force error is 6.78e-8 Eh/bohr. Each call reports 29,709
mixed task attempts, 52,053 prepared products and 99 strict task replays, one
complete solve and matching strict/Fast SCF iterations. Fock totals remain
unavailable. Retaining losing s/p classes in FP64 removes the measured regression,
but a roughly 0.3–0.6 percent ratio is **near-neutral, not useful demonstrated
acceleration**. The 24-AO generic-only route still runs two audit endpoints,
has no mixed work and is substantially slower. Do not call this all-size support.
Further narrowing must target the remaining actual derivative/geometry work;
positive task counts alone are not the goal.

The subsequent provenance-cache change removes repeated host model resolution,
not additional FP64 arithmetic. Jobs 3009/3010 requalify and measure that controller
with this same binary: the 96-AO frozen ratios become about 1.048 while cold
remains neutral and small generic-only HF still loses. See
`../performance/2026-10-14-fast-model-provenance-cache.md` for complete timings,
unchanged precision counts, skipped 3c oracle prerequisite and evidence limits.

## Retention and remaining goal

Raw scripts, failed/successful XML, timings, trace CSV/JSON and source snapshots
are under ignored `.artifacts/readme-budget-fast-20261013/`. Full Nsight reports
and the pre-experiment binary remain on n2 under the matching data directory.
No release or external evidence host was created. Larger README populations,
generic small-HF narrowing, broader SCF/integral precision and full-DF CCSD(T)/CBS
remain open; this continuation does not complete the user's all-method goal.

See `2026-10-14-df-shell-typed-runtime.md` for the shared typed equations and
transactional native kernel, and `../../proposed/2026-10-14-readme-budget-fast.md`.
