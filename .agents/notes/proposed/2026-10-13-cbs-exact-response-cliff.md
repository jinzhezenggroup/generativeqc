# Diagnosis: the C5 CBS exact-reference response residency cliff

Status: proposed
Date: 2026-10-13

## Problem

The README DF-CCSD(T)/CBS campaign at production revision
`05844f847b447e426523d4d9a559991ae8ac640b` exposed a sharp complete-E+F
increase between the fixed QM7 C4 and C5 cases. Correlation fitting does not
replace the conventional RHF reference or its analytic orbital/nuclear
response with RI-JK. A DF-HF initial guess is not a DF-HF final method.

The completed, uninstrumented node1 measurements are:

| Case | Complete E+F (s) | DZ group (s) | TZ group (s) | TZ response iterations / restarts |
| --- | ---: | ---: | ---: | ---: |
| C4 | 309.237849 | 65.257395 | 243.978961 | 17 / 0 |
| C5 | 2660.193940 | 178.854681 | 2481.337940 | 18 / 0 |

The TZ group owns DF-MP2 and its HF companion, not CCSD(T). It accounts for
93.3% of C5 wall time. These component groups do not overlap; the individual
companion component timers do overlap and must not be summed. Convergence
does not establish independent numerical qualification: consult the campaign
accuracy receipts separately.

## Evidence and mechanism

A separate node4 Slurm job (826, finite 01:02:00 allocation) replays C5 with
the same library and `GENERATIVEQC_DF_PROGRESS_TRACE` enabled. Its output is
isolated at `/data/jzzeng/readme-cbs-20261013/diagnostic-c5/`, mirrored under
the task worktree's ignored `.artifacts/cbs-readme/diagnostic-c5/`. No debugger
or profiler was attached to the uninstrumented primary jobs. The diagnostic
is not an additional primary benchmark sample.

The TZ `cuda_rhf_resident_values_prepare` trace explicitly reports:

```text
required_source_value_bytes = 15972157160
source_value_allowance_bytes = 8589934592
admission = canonical domain or value capacity refusal
source_values_built = 0
```

The replay reaches `exact_orbital_and_nuclear_response` after 91.284 s inside
the TZ native group, excluding its initial-guess wrapper. Its DF correlation
source construction took 8.180 s. Earlier TZ SCF iteration scopes took about
3.9 s each. This is not evidence for a 40-minute DF-MP2 energy contraction or
an exploding response iteration count.

The first two completed `gmres_arnoldi_action` scopes in the TZ exact-response
phase take **72.850 s** and **72.923 s**. Each response action applies exact
J/K to a density direction. These are instrumented action observations, not
additional endpoint samples or an additive ledger for the entire force call.

The exact reference and RHF frame response use optional canonical Cartesian
ERI residency. The default complete cache ceiling is 8 GiB. For these bases,
the unfiltered canonical inventory is `8 * pairs * (pairs + 1) / 2` bytes,
where `pairs = ncart * (ncart + 1) / 2`:

| Case | Public TZ AOs | Cartesian AOs | Canonical ERIs (GiB) |
| --- | ---: | ---: | ---: |
| C4 | 260 | 290 | 6.633 |
| C5 | 318 | 355 | 14.875 |
| C6 | 376 | 420 | 29.118 |
| C7 | 434 | 485 | 51.744 |

These are calculated source inventories, not measured memory peaks or
completed C6/C7 timings. C5's inventory is also confirmed by the trace.
The default all-bucket cache is refused as a whole when the cap is exceeded;
it does not automatically keep the most expensive subset. The bounded
fallback preserves the exact operator, but repeated actions regenerate ERIs.
The response defaults to zero screening, independently of screened SCF.

The grouped implementation shares a reference and DF source, but still creates
separate RHF frame-response owners for the HF companion and the correlated
force. Sharing the reference is not equivalent to sharing every J/K source,
derivative traversal, or prepared response owner.

## Proposed direction and invariants

Preserve the original benchmark observations. Diagnose and qualify changes
under a distinct source/library identity rather than silently replacing slow
points. Prioritize bounded expensive-bucket reuse and common exact-response
ownership, or introduce an explicitly different DF-HF/RI-JK method with
consistent analytic response and fitting derivatives.

Do not simply raise every memory budget: C7's complete cache alone exceeds a
32-GiB GPU. Do not drop physical residual audits, silently screen final
nuclear derivatives, or describe a DF-HF guess as full DF-HF E+F. Require
independent energy/force gates and complete-endpoint evidence for any change.

## References

- `benchmarks/results/readme-cbs-20261013/README.md`
- `src/scf/cuda/direct_jk_source_policy.hpp`: default source ceiling/mask.
- `src/scf/cuda/direct_jk.cpp`: complete-residency admission and fallback.
- `src/hf/rhf_frame_response.hpp`: separate response ceiling and screening.
- `src/hf/rhf_frame_response.cu`: exact response actions and final audits.
- `src/methods/df_ccsdt_force.cu`: grouped HF/correlated response ownership.
