# Decision: type the efficient DF shell owner, with transactional strict replay

Status: implemented private candidate; complete HF-DF admission remains open
Date: 2026-10-14

## Problem

The raw FP32 moment-polynomial candidate is not an efficient replacement for
the production shell-shared or Rys consumers. Direct HF previously lost more
schedule efficiency than narrowing recovered when routed to a generic owner.
The same mistake must not be repeated for fitted response. The source of the
scientific equations, shell schedules, folded public weights and atomic sink
must remain shared with the optimized strict endpoint.

## Decision

`emit_df_shell_derivatives_cuda` and `emit_df_rys_shell_cuda` now accept
`scalar_type="float"`. Their optional namespaces have float axis caches,
moment/recurrence DAG temporaries, weighted component algebra, primitive local
sums and subgroup reductions. Low-angular component and high-angular
cooperative Rys schedules are unchanged. Geometry, Gaussian decay, Boys/Rys
preparation, density/AO weight folding and the global gradient remain double.
The two independent center channels widen before auxiliary-center recovery.
This is not an all-FP32 integral implementation.

Default strict emitter output was compared once against master `c2c670e2` and
remained byte-identical. This is migration evidence, not a required source-hash
test. Caller-supplied include filenames are protected from arithmetic/namespace
substitution; the default float Rys include names its float shell dependency.
Class-local combined headers retain strict and float owners together. Editing
one Rys class still invalidates only that class's mathematical compilation unit.

The native panel and packet kernels select the arithmetic as a compile-time
boolean. The existing internal host launchers expose a private, default-false request
and optional caller-owned three-counter sink. No environment override admits
the arithmetic, and no complete public HF-DF consumer selects it yet.

Every mixed subgroup completes into local channels before any global scatter.
Input casts reject overflow and nonzero-to-zero narrowing, including signed
response weights. Cache/local/reduced nonfinite results reject the whole task.
All lanes agree through their existing subgroup mask; a rejected task replays
all primitives in strict arithmetic before the sole scatter. A union reuses
the existing strict-sized shared cache during replay: reserving separate float
and strict arrays would exceed high-angular schedule capacity. Normal strict
instantiations contain neither these casts/checks nor precision-census writes.
An explicit lane-zero array-member assignment starts the selected union member's
lifetime before the first geometry rendezvous, also on float-to-double replay;
passing a differently typed pointer alone is not the lifetime contract.

Component/primitive contributions use compensated FP32 local summation. This
is still float arithmetic, not a double accumulator disguised by float inputs.
Strict instantiations retain their original accumulation order. The extra six
compensation channels address signed-contraction cancellation without changing
the polynomial/Rys equations or weakening application admission floors.

## Evidence and its limits

- Host compiler/scientific regression suite: 443 passed, 77 skipped. This
  includes all s/p/d/f polynomial classes and available Rys classes, signed
  contractions, coincident/asymmetric geometry, Cartesian/spherical projection,
  diffuse/tight scaling and class-local generation isolation.
- Slurm n2/PRO6000 job 2972: eight real-device arithmetic checks and two
  invalid-type checks passed; the earlier job 2971 timed out compiling all 64
  classes in one CUDA probe. The probe now compiles only its executed device
  witnesses, while the host still covers the entire class family. The finite
  compilation timeout and numerical gates were not relaxed.
- The same job passed 40 native-runtime tests covering SSS, PPP, DPF and FFF,
  all three schedules, component/cooperative Rys where available, full,
  symmetric and packed weights, clipped auxiliary panels, and panel/packet
  launches. References are independently contracted libcint derivatives.
- Injected overflow and signed-weight underflow retry exactly once. A failure
  after a valid primitive contribution checks complete replay rather than
  double-scattering partial work. Zero-weight tasks report no mixed attempt.
- Job 2976: 20 native tests each under memcheck, racecheck and synccheck; zero
  memory/synchronization errors and zero racecheck hazards/warnings.
- Job 2978 expanded simultaneous subgroup/tail coverage and found one FFF
  ordinary-summation failure: auxiliary-center x error `1.73462677e-5` after
  three repeated tasks, above the unchanged `1e-5 + 3e-6*abs(reference)` gate.
  Preserve that failed XML/log; original 40-test success was insufficient.
- Compensated FP32 local summation passes all 48 tests in job 2979, including
  the failed FFF case and pre-scatter strict replay. No class was dropped and
  neither the numerical gate nor the physical inputs were changed to pass.
  Four simultaneous-subgroup/retry cases also pass each of memcheck, racecheck
  and synccheck in that job, with zero errors/hazards/warnings. The complete
  package build is a separate check.

The counter entries mean attempted active tasks, primitive products prepared
by those attempts, and whole-task strict replays. A product rejected immediately
after strict geometry preparation does not prove a float DAG executed. Do not
call these counts FLOPs or fabricate a per-instruction precision census.

The native independent force contraction gate is absolute `1e-5` and relative
`3e-6` for these fixtures; strict uses `8e-11`. Fault-retry comparison is against
the same strict task with a `2e-13` relative gate. These tests do not establish
the complete molecular force allowance or endpoint speedup. Raw diffuse metric
negative evidence from the preceding note remains applicable; no gate was
loosened and no exponent whitelist was substituted for physical conditioning.

## Remaining admission work

Connect the immutable application allowance to the authenticated complete
HF-DF response, with bounded census allocation and publication only after its
successful synchronized finite endpoint. Generic metric/three-center tiles are
separate consumers. Preserve their strict fallback instead of claiming they
were narrowed by the shell flag. Qualify full cold/frozen/moved endpoint errors,
return-to-strict behavior and performance against the same optimized strict
baseline before publishing HF-DF Fast. The broader README/CBS goal remains open.

Ignored source snapshots, scripts, XML, failed timeout evidence and logs are
under `.artifacts/readme-budget-fast-20261013/`; node-local artifacts are under
the matching `/data/jzzeng/readme-budget-fast-20261013/` directory. No release
or external evidence host was created.

## References

- `../../proposed/2026-10-14-df-typed-polynomial-candidate.md`
- `../../proposed/2026-10-14-hf-budget-gradient.md`
- `../../proposed/2026-10-14-readme-budget-fast.md`

## Native HF-DF continuation

The private-only boundary above is superseded by
`2026-10-14-native-hf-df-budget-gradient.md`: verified-final-density transport,
bounded production census and independent complete HF-DF gates are integrated.
The first real README timing still loses, so this is not a speed promotion.
That note retains the cold-cache authorization failure, correct rank-reduced
fallback, baseline regression reproduction and the rejected one-final-vote
experiment. The experiment passed independent numerical/sanitizer checks but
lost clean endpoint timing through increased rejected work in Slurm 3003. It is
reverted: current kernels again reject early. The later cost-selected production
candidate leaves s/p-only orbital bra classes strict while private native probes
continue testing the full typed family. Jobs 3006/3007 supply that selector's
own complete qualification and timing: 57 selected checks and 72 gated README
calls pass, but 96-AO endpoint ratios are only 1.003–1.006. This removes the
earlier loss, not the remaining useful-acceleration gap. Neither older binary's
counters certify this work.
