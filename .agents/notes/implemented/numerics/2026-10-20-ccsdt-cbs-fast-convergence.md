# Decision: relaxed convergence and bounded FP32 warmup for CCSD(T)/CBS Fast

Status: implemented
Date: 2026-10-20
Base: `origin/master` at `eac2de2b33544b104bb318e53aa02aecbea5009f`

## Problem

The requested Fast mode lowers convergence requirements. DFT's admitted Fast
Fock route already does this: it changes energy/density tolerances from
`1e-10`/`1e-8` to `1e-6`/`1e-4`, then evaluates a fresh FP64 physical Fock against
the relaxed limits. FP64 final arithmetic does not imply reconvergence to the
original strict thresholds. Fast also defaults to FP32 auxiliary DF-CCSD warmup;
these two controls must be qualified together.

Changing only CBS's Python tolerances would fail: reference owners clamp them,
CCSD validates a strict residual ceiling, and Lambda/force consumers impose
independent stricter gates. Relaxing one solver without transporting the
allowance through its response consumers is incomplete.

## Decision

`CCSDTCBSCalculator(policy="fast")` uses explicit component-local convergence
allowances. Standard remains the default. The public budget type is the existing
`PrecisionPolicy`/`TargetErrorBudget`; for CBS it selects relaxed stopping limits
and bounded FP32 auxiliary DF-CCSD preconvergence. The current contract and numerical ceilings have one
owner in [the CBS guide](../../../../docs/user/ccsdt_cbs.md#fast-convergence).

Before allocating allowances, combine coefficients of identical method/basis
components. Divide each composite allowance by the resulting coefficient L1
sum, so the triangle inequality accounts for extrapolation amplification
without assuming cancellation. Arbitrary positive exponents and distinct
correction bases use the same term list as composition. The stopping mapping
is empirical; a component allowance is not a proved residual-to-observable
bound or a per-call certificate. No duplicate strict endpoint is hidden in Fast.

Reuse the existing size-tagged budget extension with a separate correlated
capability query. Native owners decode it before copying descriptor prefixes,
then retain a value-only `CorrelationConvergence`. Solver results retain it for
force consumers; DF groups pass it through the MP2 companion and independent
frame response. Zero preserves legacy strict behavior. Legacy RI-MP2 and
prepared correlated batches explicitly reject nonzero allowances; batch owners
cannot safely retain this extension through a shallow prefix copy.

Reference density is relaxed much less than DFT's because canonical post-HF
export still needs physically consistent orbitals. The canonical-frame,
denominator, gap and curvature safeguards are retained. Grouped DF Fast requests
the existing FP32 auxiliary graph at every shape, with switch residual
`max(default_switch, 100 * selected_cc_residual)` and at most 24 evaluations.
The FP64 epoch clears mixed DIIS history; only its physical checks can publish
convergence. Lambda, triples and force response remain FP64. CPU and conventional
CCSD retain FP64. Resource/provider refusal preserves the existing FP64 fallback.

The public performance descriptor is frozen, so a private six-counter query
copies precision admission and actual work from the native owner. Result metadata
labels mixed arithmetic only after actual FP32 execution; it fails if an admitted
DF Fast call lacks native telemetry. Batch and older-library Standard reads keep
unknown counters as `None`.

## Rejected alternatives

- Mixed warmup followed by original strict convergence does not implement the
  requested mode. Its historical timings are not evidence for this change.
- Removing global CC/Lambda guards would relax all callers and lose provenance.
- Reusing DFT's `1e-4` density ceiling would violate the canonical-reference
  contract without a broader scientific redesign.
- Repeating Standard on every call would obscure the intended speed/accuracy
  tradeoff. Independent oracles are qualification consumers only.

## Invariants

Preserve explicit tighter SCF/CC settings, bounded iterations and resources,
nonfinite rejection, independent physical residual replays, and complete
analytic forces (including Lambda, orbital, auxiliary-metric and center terms).
Retain the selected Hamiltonian and basis grouping on all routes. Numerical
failure must remain visible. Very small component budgets select Standard;
unsupported capability/transport must fail before calculation, not be ignored.

## Evidence

Source-matched receipts, input identities, raw complete-endpoint samples,
semantic work and independent oracle errors are retained under
[`benchmarks/results/cbs-convergence-fast-20261020/`](../../../../benchmarks/results/cbs-convergence-fast-20261020/).
These earlier receipts isolate the FP64 convergence-only ablation. They are
targeted qualification cases, not evidence for default FP32 warmup or a claim
over arbitrary molecules. Final mixed receipts and independent DF water checks
are retained in the [README campaign](../../../../benchmarks/results/readme-cbs-fast-20261020/README.md).
Water's small algebraic basis pair and the DF STO-3G/def2-SVP pair exercise
nonzero triples but are not systematic physical CBS sequences. The H2 DZ/TZ
case is a physical cardinal sequence. The DF oracle uses PySCF reconverged
energies and central finite differences, with both exact-RHF and RI-JK models.

Conventional CPU and CUDA Fast water pass `1e-5 Eh`, maximum force `1e-4` and
RMS force `3e-5 Eh/Bohr` gates, including the tighter allocated constituent
allowances. CPU CCSD iterations fall from 15 to 10; MP2 responses from 8/8 to
6/7. Standard water retains its independent strict-reference gates. CUDA
conventional H2, exact-reference DF and full DF also pass independent gates.
No changes to the integral/derivative work are inferred from iteration savings.

Earlier FP64-only n4 RTX 5090 complete E+F medians (three replay samples; every sample
recreates all basis-group owners):

| Case | Standard (s) | Fast (s) | Reduction |
| --- | ---: | ---: | ---: |
| Conventional water algebraic pair | 0.4086 | 0.3879 | 5.1% |
| Conventional H2 DZ/TZ | 3.3482 | 3.3241 | 0.7% |
| Water correlation-only DF, STO-3G/def2-SVP | 2.5216 | 2.2123 | 12.3% |
| Water full DF, STO-3G/def2-SVP | 0.9966 | 0.9631 | 3.4% |

These are small-domain observations with timing variability retained in the
receipts, not general speedup promises. In both DF models CCSD takes 16 versus
9 iterations, while the higher-basis MP2 response takes 13 versus 10. Fast DF
composite energy errors are about `5.3e-9 Eh`; maximum force errors are below
`1e-8 Eh/Bohr` against the independent finite-difference oracle. Conventional
Fast water's force error is about `1.75e-7 Eh/Bohr`, already dominated by the
pinned oracle's baseline difference. Smaller and saturated custom-budget
brackets also pass the conventional CUDA water observable gates.

Validation includes CBS composition, stronger caller controls, tight-budget
selection, detached metadata, missing native capabilities, malformed native
extensions, strict legacy ceilings and batch rejection. Existing CBS native
publication/recovery and precision-policy regressions pass. Real GPU work ran
under finite `srun` allocations on n4; CPU/CUDA builds used verified ccache.
Cache snapshots and local build/test logs are in `.artifacts/convergence-fast/`.

The final mixed build passes conventional CUDA water and exact/full-DF water
tests, including actual FP32 GEMM/evaluation counters. Its independent full-DF
water composite energy error is `5.87e-9 Eh` and maximum force error is
`1.25e-8 Eh/Bohr`; all allocated component gates pass. The final homepage series
remeasures both Standard and Fast with matching binaries on n4/n5 RTX 5090s.
Its seven vector comparisons are Standard controls; only C2 has independent
constituent/composite energies and two-step directional force checks. Single
cold observations do not establish statistical speedups.

## Reproduction

Run `tests/python/test_ccsdt_cbs.py`, `test_correlation_convergence.py` and
`test_ccsdt_cbs_native.py` with `GENERATIVEQC_LIBRARY` pointing at the matching
native build. CUDA tests require `GENERATIVEQC_CBS_CUDA_TEST=1` and Slurm.
`test_full_df_cbs_native.py -k cbs_complete` qualifies exact/full DF with
Standard/Fast; set `GENERATIVEQC_FULL_DF_CUDA_TEST=1` and provide PySCF.

`benchmarks/ccsdt_cbs_endpoint.py --policy {standard,fast} --case
{water-algebra,h2-dz-tz} --device cuda --repeats 3 --oracle --work --output ...`
measures complete conventional endpoints. The DF receipt embeds its exact
standalone qualification driver and all source/library identities. All samples
recreate same-basis owners; no converged reference survives between samples.

## Revisit when

Broader representative molecular benchmarks justify larger reference limits,
a residual-to-observable estimator becomes available, or a batch owner can
retain and report the complete correlated convergence policy. Additional mixed
consumers need their own joint qualification before composition with this mode.
