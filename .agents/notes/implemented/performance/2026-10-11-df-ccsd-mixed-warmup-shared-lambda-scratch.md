# Decision: FP32 DF-CCSD warmup with strict polishing and shared Lambda scratch

Status: implemented, opt-in; complete-endpoint qualified for the recorded CUDA case
Date: 2026-10-11

## Problem

Reducing only (T) arithmetic does not address the dominant CCSD/Lambda work or
Lambda's simultaneous solver/replay/audit scratch reservations. The starting
master is `98263e06ca13aa88cee504f1f105f5cf3ef6d92a`.

## Decision

Retype the compiler-owned auxiliary primal graph to explicitly qualified FP32
storage and pedantic arithmetic for CCSD preconvergence. Cache only immutable
factor conversions within the native owner; update amplitude/tau conversions
for each evaluation. Widen each Q contribution before ordered FP64 accumulation.
At the residual threshold or finite warmup-action limit, discard mixed DIIS history and reevaluate
the warm-start state with the original FP64 graph. Only if the supplied guess
originally admitted restricted-pair coordinates, restore the internal doubles'
exact redundant-coordinate symmetry using the existing compiler-owned pair
map. Never project a supplied asymmetric guess or an FP64 iteration. An optional
packed history refused during warmup can reuse the larger full-history lease
with empty rows and its original packed stride/metric; actual allocation
accounting and the original FP64 rounding gate are unchanged.
Fit strict polishing at half the
warmup Q batch, including Q-independent slots and the existing qualified
occupied-pair workspace in the shared lease. Independently replay the accepted physical
amplitudes in FP64 before any energy or response publication.

Separately share Lambda's optional matrix scratch between the sequential
solver, parameter, expanded primal and independent audit stages. Their borrowed
outputs must be consumed/copied before the lease is reused. Persistent inputs,
cuts, cotangent seeds and optional invariant retention never alias this lease.
Resource fallbacks restore the original smaller layouts; numerical errors
still propagate.

## Rejected alternatives

- An FP32 (T)-only optimization leaves the larger solver costs untouched.
- FP32 final amplitudes, Lambda or force contractions require a different
  scientific acceptance contract; they are not introduced here.
- Carrying approximate DIIS history into strict polishing risks contaminating
  convergence and obscuring the precision boundary.
- Keeping separate FP64 batched polishing and FP32 warmup arenas defeats the
  storage benefit; reusing scratch with a smaller strict Q batch is bounded.

## Invariants

- Same RHF reference, DF Hamiltonian, equations and physical tolerances.
- No FP32-only convergence certificate and no reference-oracle production path.
- No TF32, low-precision denominator, precision-driven tolerance relaxation or
  replay/audit deletion.
- FP64 default callers retain their original contraction schedules.
- Sharing storage must not share cached scientific values across independent
  audit equations, and allocation fallback must not overlap pinned storage.

## Evidence

`test_df_cc_mixed_precision.py` protects equation structure, rounding bounds,
typed native emission, scratch slot geometry and strict final GPU certificates.
`test_df_triples_mixed_endpoint.py` also exercises the new CCSD-only candidate
against complete strict forces and an independent same-Hamiltonian energy.
Real-GPU tests and benchmarks require finite Slurm allocations on healthy nodes.
Complete endpoint timing and sampled device-occupancy peaks are recorded
separately from logical arena capacities. The scoped results below do not
establish a universal precision policy or a speedup for all supported sizes.

The first ethane cold pair rejected the initial compact schedule: endpoint time
increased from 303.359 to 353.916 seconds despite sampled device occupancy
falling from 31,529 to 27,547 MiB. CCSD switched after 12 approximate evaluations,
then used Q2 without the existing occupied-pair fold; it needed 48 evaluations
instead of 38. Q-independent arena slots made FP64 Q4 slightly larger than
half-width FP32 Q8. The revised plan includes this small fixed overhead, retains
strict pair admission, and switches at `1e-7` instead of `1e-5`. Packed DIIS may
still refuse approximate symmetry; its original FP64 bound is not relaxed.

The next energy-only diagnostic identified why admitting a strict pair workspace
was insufficient: all 20 strict evaluations still refused pair projection because
redundant-coordinate FP32 noise survived in the full-history state. Its energy
endpoint uses the existing input-residual mixing policy, so its 66.536-second
CCSD time is not a matched force-endpoint comparison. Restoring the compiler's
exact pair coordinates only at the precision boundary addresses this admission
failure without relaxing physical or storage-rounding tolerances, preserving
mixed history, or projecting an external asymmetric initial state.

### Final complete-endpoint qualification

Slurm job 7502 ran five paired cold ethane energy-and-force evaluations on one
RTX 5090 (`GPU-57db1e6a-5ff9-570d-67b0-6965fb3a77e6`), node1, driver 580.95.05.
The finite allocation used `main`, `gpu:5090:1`, four CPUs, and a one-hour limit;
device visibility was preserved. The case has 230 AO, 488 RI, nine occupied and
221 virtual orbitals, aug-cc-pVTZ / aug-cc-pVTZ-RI. Both modes use exact
conventional RHF plus correlation-only DF-CCSD(T), identical physical tolerances,
CCSD Q8 / DIIS8, Lambda Q32 / true-residual cadence30, and FP64 (T). Order
alternates AB/BA. Cold means independent processes with no supplied electronic
state, not recompilation of the executable or clearing the driver's code cache.

| Median | Frozen master | Candidate |
| --- | ---: | ---: |
| Complete native E+F, seconds | 305.987797 | 293.743219 |
| Complete process wall, seconds | 306.40 | 294.14 |
| CCSD including independent replay, seconds | 63.829337 | 51.812469 |
| Lambda, seconds | 78.945868 | 79.036453 |
| Sampled device occupancy peak, MiB | 31,529 | 27,547 |
| Lambda owned device bytes | 15,542,559,936 | 10,324,999,872 |

Endpoint native time decreases 4.002%; CCSD time decreases 18.827%; sampled
occupancy decreases 12.630% (3,982 MiB). Occupancy is sampled every 200 ms using
NVML and is not an exact allocator peak. Lambda eliminates a 5,217,560,008-byte
second scratch reservation without reducing its Q batch or scientific work.
There is no measured Lambda arithmetic speedup.

Semantic work is retained in all ten endpoint JSON records. CCSD evaluations
change from 38 to 36 (20 approximate, 16 strict); strict polishing uses Q4 and
all 16 strict pair admissions succeed. Packed DIIS is restored at its unchanged
rounding gate; its larger warmup-refusal history allocation remains charged.
FP64 GEMMs change from 54,862 to 48,520, with 23,180 additional FP32 GEMMs and
183,089,660 converted elements. Total contraction terms increase from
26,799,066,988,432 to 34,721,071,141,244, and packing traffic increases from
3,270,860,758,368 to 3,453,086,946,496 bytes. This is a measured precision/schedule
tradeoff, not a claim of reduced total arithmetic or data movement. Lambda
retains 21 iterations, 22 actions, Q32, core retention, expanded primal replay
and the independent matrix audit. No discarded JK attempt occurs.

Across the five pairs, maximum absolute energy difference is
`6.963318810448982e-13` hartree and maximum force-component difference is
`1.759259404821023e-11` hartree/bohr. Original FP64 CCSD replay residuals stay
below `1e-10`; independent Lambda, Z and stationarity gates stay below `1e-8`.
H2, water and LiH also pass independent same-Hamiltonian energy and two-step
directional finite-difference force gates. GPU qualification reports 51 passed,
nine expected CPU-parameter skips, and three unrelated (T)-mixed tests deselected.
Water complete E+F memcheck reports zero errors and zero leaked bytes in a
separate finite Slurm job. Final targeted CPU validation reports 62 passed and
76 CUDA/opt-in skips; additional precision/lifetime contracts report 55 passed.
Compiler structure checks cover 518 modules with zero errors; Ruff, formatting,
targeted type checking and `git diff --check` pass.

CUDA 12.9.86, GCC 11.4.0, sm120 Release and verified ccache 4.5.1 are used for
both builds. The frozen master library SHA256 is
`47f61e3a707297e0639ad70ad00b8f67c33a543eede338993f14e81df02c751e`;
candidate library SHA256 is
`9e05def80dbc2f80aac36168941b1cd0b6f6a4f00dac6f1bb0d74398adc43140`.
Reviewable source-matched evidence, controls, reproduction scripts, complete work
counts, sampled occupancy, acceptance summary, rejected measurements and source
patch are retained in
[`benchmarks/results/df-ccsd-mixed-warmup-20261011/`](../../../../benchmarks/results/df-ccsd-mixed-warmup-20261011/README.md).
Full transient logs, build products and rejected source archives remain in the
sibling `evidence-ccsdt-mixed-20261011/` audit directory. The retained patch binds
the dirty measured source, not a subsequently created publishing commit.

## Revisit when

Complete cold E+F qualification identifies whether the switch threshold,
strict polishing or conversion traffic limits the endpoint benefit.
Consider a mixed Lambda correction solver only with independent FP64 physical
residual/refinement gates and separately bounded persistent storage.
