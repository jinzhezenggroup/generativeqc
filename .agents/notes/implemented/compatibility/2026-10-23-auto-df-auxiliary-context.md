# Decision: resolve typed auxiliary AUTO admission against the prepared backend

Status: implemented
Date: 2026-10-23

## Problem

The current-master integration of PR #2526 exposed a mismatch before any SCF
execution: Python selected the values-only KS auxiliary handle for a CUDA
`density_fitting="auto"` request, but native role admission accepted only the
explicit CUDA enum. All 16 executable CUDA range-DF endpoint cases failed at
batch preparation in Slurm job 13058. The generic error incorrectly described
this as a correlation-only consumer mismatch.

## Decision and invariants

Both public preparation entry points pass the context's executed backend into
role admission. SCF and KS CUDA auxiliary handles accept explicit CUDA or AUTO
only on that CUDA backend. CPU AUTO, CPU-reference fitting, missing KS options,
wrong method families and use of auxiliary handles as orbitals remain refused.
Python's complete first-derivative SCF auxiliary selection likewise recognizes
AUTO on an explicit CUDA device. The correlation-only role policy is unchanged.

Do not repair this by changing endpoint tests to explicit CUDA: AUTO is a public
backend-selection contract. Do not admit AUTO without inspecting the context:
the auxiliary handle carries a provider-specific domain, not merely basis data.

## Integration diagnostics

The same GPU campaign exposed two separate qualification gaps. PBE with d
orbitals needs the packaged `generativeqc_stationary_pbe_rks_spd_manifest` target;
building only the ordinary PBE manifest does not qualify that endpoint.

The full-range shell consumer in current master already executes auxiliary g
through the generated raised-polynomial domain. The old g-partition test still
expected a separate generic-remainder panel counter. Its RHF numerical gates
passed before that stale assertion failed. The test now enables intrusive shell
diagnostics and requires actual `shell_*4_work_active_shell_tasks` work for the
complete shell route, while shell-sp and generic retain their independent force
checks. It also exercises AUTO for HF and KS with the unmodified default g fit;
the separate endpoint suite retains explicit CUDA coverage. No force tolerance,
grid, auxiliary column or metric mode is removed.

These diagnostic captures are correctness and ownership evidence, not timing
or default-promotion evidence. Original failures and subsequent exact-source
qualification records are retained in the PR's ignored local artifacts and
reported in its review description.

## Separate LR execution and raw-value ownership

After fixing admission, Slurm 13067 passed all 50 response/g-shell checks and
the four default/Direct selector cases, but all 12 explicit fitted LR cases
failed during execution. A scalar C API reproduction recovered the exception
hidden behind the batch status: the range correction called the Direct-only
adapter even when its prepared provider was fitted.

The correction adapter now routes an identical resolved LR fitted plan through
the existing shared DF executor. It checks the complete plan identity and keeps
the FP64 restriction. The existing stream/event synchronization remains intact;
Direct corrections keep their established path.

Current master also introduces optional prepared Hermite raw tiles. That owner
implements full-range integrals only, so it must decline an LR source before
allocation and retain the omega-aware generated fallback. Mixing full-range
three-center values with an LR metric would change the Hamiltonian. A dedicated
packed-single endpoint regression exercises automatic and explicit generic
mapping, checks actual primary Hermite work, and compares the complete energy
and force with independently fitted PySCF energies and finite differences.

The corrected CUDA campaign passes all 16 executable selector/practical/complete
LR cases and all 50 response/g-shell cases with unchanged numerical gates. The
one materialized-derivative case is an explicitly CPU-only diagnostic, not a
CUDA pass. Source manifests, binary hashes, compiler-cache receipts and Slurm
logs accompany the PR qualification. No performance promotion is inferred.
