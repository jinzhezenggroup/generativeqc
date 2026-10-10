# Decision: compensate exported RHF references and signed J/K response scatter

The generated-ABI routing limitation below is superseded by the
[2026-10-09 generated Fock sink follow-up](2026-10-09-generated-fock-compensation.md).
The original compensation and response decisions remain in force.

Status: implemented
Date: 2026-10-07 (UTC; node2 evidence directories use Asia/Shanghai dates)

## Problem

Issue #2019 retains pre-existing failures of the 230-AO / 488-auxiliary
DF-CCSD(T) force-repeatability gate (`atol=5e-10`, `rtol=0`). This work is
independent of the #1763 reduction/fusion changes and PR #2090. It does not
reclassify any historical failing sample as a pass.

The original cold force differences were 3.822e-9 and 4.468e-9. Same-primal
job 2486 failed at 5.520250923041203e-10; its serial repeat differed by
4.14979162144391e-10. Tightening RHF density convergence to 1e-12 did not
converge within 150 iterations and is not a fix. Job 2639 subsequently
failed a cold comparison at 2.9812392554617873e-9. These observations remain
in the issue and the
[historical reduction proposal](../../proposed/2026-10-06-indexed-gap-response-reduction.md).

Job 2492 already localized the dominant variability away from DF nuclear
weights: the DF gradient varied by 4.7874e-15 versus 3.1175e-10 in the
orbital/nuclear branch. Its returned orbital RHS followed the *second* weight
map; it was not evidence of an identical initial GMRES RHS.

## Localization

Baseline job 2641, based on master d27132b92, added diagnostic-only dumps of
already downloaded arrays, before GMRES and after each nuclear piece. Across
all six pairs of four fixed-seed replays:

| Stage | Maximum absolute difference |
| --- | ---: |
| Initial RHS | 7.421308449412811e-13 |
| Z solution | 4.103093597307081e-13 |
| Final overlap/Pulay weights | 3.162759290575856e-10 |
| One-electron gradient | 2.6733751068418766e-10 |
| Combined two-electron gradient | 1.4411805082659157e-10 |
| Orbital gradient | 2.406729344597792e-10 |

This observer-bearing sample passes the force gate but does not supersede
the failures. Signed J/K rounding is amplified into Pulay weights before
nuclear contraction. The one-electron variability exceeds the two-electron
variability; changing only nuclear-gradient atomics would miss this path.
The quadratic plus/minus magnitudes (7.40135 and 0.91065) also give no evidence
for catastrophic polarization cancellation as the sole cause.

## Decision and boundaries

Keep the scientific compiler's existing ERI permutation and RHF/UHF scatter
equations. Only the Direct emission accepts a runtime-owned output sink;
generated shell consumers retain their previous emitted source and interface.
The response's canonical restricted J/K route uses a compensated atomic sink:
the atomic returns the preceding sum, explicit rounded additions reconstruct
that addition's magnitude-ordered residual, and a second plane accumulates
the residual. Fold each correction once after all writers, on the owning
stream, before Cartesian-to-public projection and finite audit.

This reduces summation error without locks or a second integral contraction.
It does not guarantee bitwise reproducibility. Ordinary device J/K, UHF,
mixed-product/recurrence paths and ordinary energy/force SCF retain their contract.
Nonfinite input, intermediate overflow and nonfinite output remain errors.

Two caller-owned correction planes are explicitly charged to complete
admission. Their size is batch times canonical Cartesian dimension squared,
not public spherical dimension squared. Require exact size, current-device
storage and disjoint input/output/diagnostic/provider buffers before resetting
or enqueueing anything. No provider-side action allocation, host-density
work, success fence or four-index retention is introduced.

An admitted immutable resident source can replay the same compensated scatter.
Resident refusal retains compensated recurrence, and positive fixed masks
continue to use geometry-only screening. Independent exact audits explicitly
disable resident borrowing. If canonical storage itself is unavailable, retain
the existing bounded provider fallback rather than requiring an optional lease.

## Validation and evidence

- Causal observer intervention, job 2643: all six replay pairs have identical
  initial RHS, Z solution, hcore/Fock weights and overlap/Pulay weights. Maximum
  one-electron drift is 1.4210854715202004e-14, two-electron drift
  4.574118861455645e-13, and orbital drift 4.440892098500626e-13. All four
  stationarity maxima are 3.6001006590519303e-13. This is still observer-bearing
  localization, not four cold endpoints or a speedup claim.
- CPU equation/codegen/ownership checks: 29 passed; compiler structure audit
  checked 493 modules with zero dependency errors.
- Independent primitive test: 24 shuffled, differently launched dyadic
  cancellation multisets, whose closed-form exact sum is 6144. Infinity, NaN
  and intermediate overflow must stay nonfinite. Slurm memcheck: zero errors.
- Signed J/K probe uses independent dense PySCF/libcint contractions, fixed
  masks, spherical/Cartesian projection and through-f sources. It checks
  correction-size and input/output/error/census alias refusals before a valid
  enqueue; resident capacity refusal keeps the exact action.
- Clean job 2650: 71 real-GPU response/complete-force/same-primal/physical-replay
  tests pass, including independent energy-force finite differences and
  transactional refusal. Memcheck repeats 27 fixed-mask/resident tests with
  zero errors (25 unrelated tests deselected).
- Response-only clean same-primal job 2645 passes all six pairs, maximum
  7.647216193618078e-13; factor/Fock/coefficient source fingerprints match.
  However, cold jobs 2651/2652/2653/2654 fail *all six* force comparisons,
  maximum 3.757228306255911e-9. Serial repeat alone differs by
  1.7525385587191522e-9, with RHF iteration counts 24/25 despite final response
  stationarity below 6e-13. This is a retained rejection, not a resolved issue.
  The response-only change removes within-primal amplification but cannot
  remove upstream frame variability.
- Complete reference-energy, force-FD, residual, stationarity, factor and
  repeatability gates are unchanged. Primitive success alone does not resolve
  #2019; the full-candidate endpoint evidence appears below.

Ignored local evidence is under `.artifacts/force2019/`. The n2 roots are
`/data/jzzeng/force2019-20261007/` (frozen observer baseline),
`/data/jzzeng/force2019-compensated-20261008/` (causal intervention with observer),
and `/data/jzzeng/force2019-clean-20261008/` (unobserved response-only candidate). They retain
source patches/manifests, input/binary hashes, launcher versions/cache stats,
Slurm/device visibility, all-pair reports, complete wall times and work counts.
Every real-GPU command uses finite `srun` on node2 with `gpu:pro6000:1`.

## Exported-reference intervention

The full candidate applies the same residual recovery to the
restricted exported-reference Fock. Its one canonical plane is mandatory,
charged before CUDA/provider allocation, counted in retained device inventory,
held through graph replay, and freed on the bucket stream. Each Fock action
resets it and folds before projection/eigenframe construction. Small s/p
resident and fixed-output matrix references retain their existing path.

Generated page launch ABIs do not carry a compensated sink. Until they do,
exported bounded quartet references use the existing generic exact quartet
consumer across the complete domain, not a mixture of corrected and ordinary
scatters. Compiler-owned integral, spin and permutation algebra is reused;
there is no new SCF equation implementation or CPU/oracle path. This may cost
reference time; complete endpoint receipts, not an isolated scatter timing,
must establish the tradeoff. Ordinary HF energy/force selections are unchanged.
Reference energy/density tolerances and all force acceptance gates are unchanged
in this intervention.

## Full-candidate native endpoint qualification

The unobserved full candidate is retained at
`n2:/data/jzzeng/force2019-reference-20261008/`. Job 2656 completes one shared
primal and four response modes; cold jobs 2658/2659/2660/2661 independently
complete parallel/omitted/serial/serial-repeat endpoints. All six comparisons
in each group pass the unchanged `atol=5e-10, rtol=0` force gate:

| Pair | Same-primal maximum | Cold maximum |
| --- | ---: | ---: |
| serial / parallel | 4.600764214046649e-13 | 2.673417043297377e-13 |
| serial / omitted | 2.637889906509372e-13 | 3.490541189421492e-13 |
| serial / serial-repeat | 2.673417043297377e-13 | 2.355893258254582e-13 |
| parallel / omitted | 2.41140440948584e-13 | 4.2543746303636e-13 |
| parallel / serial-repeat | 5.591083152012288e-13 | 3.6060043839825084e-13 |
| omitted / serial-repeat | 3.1796787425264483e-13 | 4.591882429849647e-13 |

All energy differences are zero. Same-primal factor, Fock and coefficient
source fingerprints match. Every cold reference takes 19 iterations with
density RMS 1.835548764914266e-12; every final stationarity maximum is
5.761508849226979e-13. Lambda and true Z residuals are respectively
6.116413766248788e-13 and 1.3570438463417128e-13. Reference energy/density
tolerances remain 1e-12/1e-11. Each cold endpoint takes 20 CCSD iterations,
17 J/K actions (16 resident), 12 Z iterations and 13 Z operator actions.

Complete native cold endpoint times are 668.43/677.17/735.25/732.81 seconds
for parallel/omitted/serial/serial-repeat. Reference time is approximately
128–130 seconds. The complete shared-primal comparison takes 2143.54 seconds,
including its 235.59-second cold primal. These concurrent PRO 6000 samples
are qualification receipts, not a speedup claim. The generic compensated
reference route trades generated-page throughput for frame accuracy until
those page ABIs support the same sink.

The native endpoints return zero and finish their root progress events, but
the shared qualification shell script was extended while they were running.
After native completion, jobs 2658/2659/2660 execute a corrupted shell-file
offset (`sr/bin/time`, exit 127), and job 2656 encounters an unbound shell
variable (exit 1). These are retained **runner failures**, not successful
Slurm jobs. The original wrapper-status gate consequently rejects the run.
The separate `native-gates.py` report independently requires GNU time's
native exit zero, a completed root progress event matching the full endpoint
duration, valid complete JSON, the requested controls, and identical input,
source and binary manifests before evaluating every unchanged numerical gate.
It records all four runner failures rather than synthesizing completion
markers or replacing a numerical rejection with a rerun. Future qualification
runners must use immutable per-job script copies.

Final supporting validation:

- 59 CPU equation/ownership/reference-capacity/retirement/retry tests pass;
  the compiler structure audit checks 493 modules with zero dependency errors.
- GPU job 2662 reports 75 passes, including four independent d/f complete
  FCI force-oracle cases exercising the exported-reference intervention.
  Two Cartesian/spherical reference plan-reuse diagnostic assertions fail
  identically on the frozen original baseline in control job 2663; they are
  pre-existing and deliberately remain unfixed. The earlier job 2657 also
  retains its Conda/system GLIBC link errors; the system-C++ retry removes
  that toolchain problem, not the baseline assertion failures.
- Job 2663 memcheck repeats 27 fixed-mask/resident tests and four d/f
  complete-force reference tests, both with zero errors. The shuffled dyadic
  primitive also passes its registered CMake test and standalone memcheck.
- Qualified fingerprints match all 6235 tracked compiler/native/test/build
  source files checked locally. Ruff lint/format, clang-format and
  `git diff --check` pass. Compiler cache versions and before/after statistics,
  device visibility, source patches and all failures remain in ignored local
  evidence alongside `reference-gates.json` and the native-verification script.

The separate large-factor `3e-10` gates are unchanged and are not newly
qualified by this repeatability task. Historical failures and the failed
response-only intervention remain failures.

## Rejected alternatives and revisit conditions

- Looser force/response gates or tighter SCF convergence: neither addresses
  signed scatter accuracy; the latter already failed to converge.
- DF-only accumulation changes: contradict retained localization evidence.
- Polarization rescaling or derivative atomics alone: cannot remove the
  measured initial-RHS/Pulay amplification.
- Serial host/oracle summation: violates the production GPU/no-oracle boundary.
- Ordered per-output integral recomputation: potentially deterministic, but
  changes work/data movement substantially; revisit if compensated complete
  endpoints cannot satisfy the unchanged acceptance gates.

The atomic residual relies on the CUDA FP64 atomic returning the old value
and explicit rounded arithmetic, as specified by the CUDA 12.9.1 programming
guide. Revisit on a new accumulator backend, changed source projection,
changed precision policy, or a retained complete-endpoint numerical failure.
