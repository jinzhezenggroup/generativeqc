# Decision: promote owner-bound Direct RHF orbital-gradient convergence

Status: implemented; current-image qualification pending
Date: 2026-11-10

## Decision

After reviewing the explicit-mode qualification, the user authorizes making
orbital-gradient convergence the default and opening a PR. This supersedes the
opt-in-only decision in
[the initial alignment note](2026-11-10-hf-reference-convergence-alignment.md),
without relabeling that note's historical source or observations.

Absent or empty `GENERATIVEQC_CUDA_RHF_CONVERGENCE` now selects the physical
restricted MO-gradient norm for standard-control, all-electron FP64 Direct CUDA
RHF. Explicit `strict` retains the AO-maximum policy. Either tighter public
energy or density control restores strict admission even when the environment
explicitly requests `orbital-gradient`.

Default admission must not turn previously supported UHF, ECP, mixed-arithmetic,
application-budget or physical-reference requests into errors or silently relax
them. These owners keep strict convergence when the control is absent. An
explicit orbital request still rejects unsupported owners. DF HF and KS retain
their independent existing policies. Python resource/checkpoint and captured
native-plan identities continue to bind the selected policy.

## Accuracy and publication boundary

The complete endpoint and independent observable gates are unchanged. This is
not a new 1e-7 maximum-force guarantee: the prior HF48/96 experiment shows both
native orbital mode and default GPU4PySCF exceeding that stronger diagnostic.
Strict overrides and independent tighter audit solves remain available.

GPU4PySCF settings must remain completely unchanged. README figures for the
new default require a complete source-matched cohort, not selective replacement
of the two previously slower Direct HF points. Retain accuracy failures and
semantic work counts alongside endpoint timings. Qualification and evidence
identities are appended after testing the promoted default itself.
