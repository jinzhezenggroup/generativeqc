# Investigation: water96 Direct SCF and stationary-density confirmation

Status: proposed; phase diagnosis and host policy counterexample, not a fix
Date: 2026-11-05

## Problem

The #2992 qualification's 96-atom water cluster (768 AOs, WB97M-V/def2-SVP,
full Direct, RTX 5090) reached the 540-second cold energy-plus-force cap in
Slurm 15928. That untraced termination did not identify its last native stage.
It cannot establish a complete endpoint time, force result, or speedup.

## Verified phase evidence

One bounded diagnostic, Slurm 15933 on node1/main/5090, reused the identical
native image and unchanged endpoint inputs. Native progress tracing, Python
phase markers and one GDB stack sample were diagnostic instrumentation, not
clean performance measurements. The Python process stopped intentionally after
150 seconds using SIGALRM (exit 142); this was not a numerical/CUDA failure.

- Calculator construction ended at 1.260240 seconds from diagnostic start.
- Public preparation ran from 1.260272 to 3.672979 seconds, approximately 2.4 s.
- Public cold energy-plus-force execution began at 3.673161 seconds.
- Initialization inside execution prepared 14,985,773,728 bytes of optional
  resident FP64 LR source values in 20.717940 diagnostic seconds.
- Twenty-nine SCF iterations completed; all reported `converged=0`.
- Sixteen physical MO-gradient evaluations executed. Thus the newly admitted
  large-system stopping path was active, not silently falling back.
- No `ks_early_final_probe` region or early-final acceptance/rejection counter
  occurred. The observed tail failed before canonical publication proof, not
  because that proof was repeatedly rejecting candidates.
- No force region had begun. The unfinished thirtieth iterate was in Direct K.
- All recorded `cuda_error` values were zero.

Completed iteration regions totaled 125.207188 diagnostic seconds: JK 50.436724
and XC 73.314503. VV10 pair features accounted for 42.275563 seconds **within**
XC, not in addition to it. The two expensive pair evaluations were iterations
10 and 15 (21.126817 and 21.133236 seconds). Iterations 16-29 each spent about
1.1 milliseconds in this region, consistent with the exact VV10 key-cache
reuse path; these short regions do not by themselves prove unchanged AO D.

The approximately 35-second main-thread stack was a CUDA event wait through
`TraceRegion::finish()` in `CudaKsPlan::Impl::enqueue_legacy()`. Its high process
CPU usage therefore does not demonstrate CPU/PySCF electronic-structure work.
Neither the sample nor the first 150 seconds reconstructs the entire original
540-second run.

## Confirmed policy inconsistency; unconfirmed water96 trigger

The default physical target candidate requires energy change below 1e-9 and
the global MO-gradient norm below `sqrt(1e-9)`. The nonlocal confirmation branch
can instead retain D whenever the proposal-density RMS, physical AO residual
RMS and AO maximum pass 1e-5. It does not check the MO norm. These metrics are
not interchangeable, and the mismatch grows with matrix dimension and overlap
conditioning. The confirmation branch can therefore keep reevaluating a
nonqualifying state instead of applying the still-owned proposal.

A compiled host counterexample exercises the current production
`orbital_ks_converged` and `nonlocal_confirmation_tolerance` helpers. This is
an analytic eight-orbital policy probe, **not** CPU substitution for production
GPU computation and **not** measured water96 scalar data:

- S=X=I; D=diag(2,2,2,2,0,0,0,0), an idempotent closed-shell density.
- Physical F has occupied/virtual diagonal blocks -I/+I, with every element
  of its occupied-virtual block equal to 4.5e-6.
- Its exact two-level spectral projector gives proposal-density RMS
  3.1819805149528532e-6.
- Current AO residual RMS is 6.3639610306789281e-6 and maximum 9e-6: both
  pass confirmation, as does the proposal-density RMS.
- Its physical MO-gradient norm is 3.6e-5, exceeding the candidate threshold
  3.1622776601683795e-5. Even with zero energy change and complete physical
  target history, the current helpers report hold=true and candidate=false.

This proves a reachable mathematical inconsistency in the policies. It does
**not** prove which water96 candidate scalar failed: Slurm 15933 did not log
energy change, MO norm, or the hold decision. Lack of final probes plus cheap
repeated pair regions makes this the priority hypothesis, not a final attribution
of all 540 seconds. No runtime thresholds or solver logic were changed here.

## Decision and next acceptance gates

Do not repeat a broad benchmark matrix, weaken the physical MO norm, relabel
partial timings as complete E/F, or switch Direct to DF to conceal the issue.
The targeted repair should make a stationary-density hold imply the active
policy's stationarity gate (without requiring energy convergence prematurely),
or otherwise bound a rejected hold so the DIIS proposal can advance. Explicit
strict callers, other models, final canonicality and physical force proofs
must retain their contracts.

Before claiming the water96 cause or a completed repair, capture the failing
physical scalars/hold decision in a bounded run, exercise a coupled-policy
regression and perform independent E/F qualification. The present investigation
does not qualify water96 defaults or establish general RTX 5090/A100
non-regression. The old frozen WB97M-V capsule is a matched fitted-LR/DF
endpoint, not a like-for-like full-Direct baseline.

## Provenance and reproduction

- Executed native source identity:
  `be94892b51bb82b61b10f16d88a8fec0bde87e3ab0fba1c2385cf20e5d7cc44d`.
- Core ELF SHA256:
  `6bfa709dbc723b520194956e3b80794add5b525c5e4f74c25193fedfcde12781`.
- This is the retained pre-format 14e7245f5 image, not a newly rebuilt later PR
  head. Later line-wrapping and master-merge changes were not executed here.
- Raw remote evidence:
  `/data/jzzeng/issue2987-defaults-20261105/diagnose96-15933/`.
- Local receipts: `.artifacts/issue2987/receipts-repair/diagnose96-15933/`.
- Phase audit: `.artifacts/issue2987/water96-phase-audit.json`.
- Host counterexample: `.artifacts/issue2987/confirmation-counterexample.cpp`
  and `.artifacts/issue2987/confirmation-counterexample.json` (ignored local
  diagnostic artifacts; the complete analytic construction is described here).
- Host compilation uses verified sccache 0.16.0, C++20, `-Isrc -Iinclude`;
  compile the counterexample to an object, link, and run without a GPU.
- Issue #2976 is the external evidence/decision log; related #2987 and #2992.
- Prior qualification rationale:
  `2026-11-05-wb97mv-default-convergence-alignment.md` in this directory.

## Follow-up repair

The active-policy stationarity coupling and rejected-proof hold exclusion are
implemented in
`../implemented/numerics/2026-11-05-vv10-confirmation-stationarity.md`.
That note records the repair's separate source identity and qualification;
the original diagnosis and its uncertainty above remain historical evidence.
