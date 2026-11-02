# Decision: bounded full-rank fitted-HF execution recipes

Status: implemented
Date: 2026-11-03

## Problem

DF has several exact representations and native contraction schedules, but
changing the auxiliary basis, metric rank or arithmetic storage is not merely
a scheduling choice. DF value buffers and derivative response also have
different phase lifetimes. A resident microbenchmark cannot prove a faster
complete constrained-memory energy/force endpoint.

## Decision

Extend isolated native execution recipes with `--component df` for FP64
RHF. Fix the orbital/auxiliary bases, relative metric threshold and memory
hint before search. Retain native phase-aware resident/streamed admission,
fallbacks and the composed allocation ledger. Search only existing exact
packed/dense, exchange, shared-source and derivative mappings.

Use independent PySCF/libcint fitted-HF energies and analytic gradients including
auxiliary response. Rebuild libcint integrals from the exact immutable orbital
and auxiliary decimal tables. A first named STO-3G reference differed by about
1.55e-7 Hartree despite agreeing on the nominal basis name: reference-package
rounding is a different Gaussian input model. Do not loosen the 1e-9 gate to
hide that mismatch or import native integrals into the independent oracle.
Initially require a fully retained metric independently
above both native relative and reference near-dependence floors. Verify native
effective rank and observed Fock work at every sample. This avoids comparing
two different pseudoinverse Hamiltonians or silently changing force semantics.
Use strict absolute E/F floors (1e-9 / 1e-8 maximum / 3e-9 RMS), not DFT grid
discretization allowances. Independent work is tuning-only.

The initial consumer is the RHF small-native-eigensolver resource owner: at most
16 orbital and 128 auxiliary AOs. A first real water/def2-svp trial is correctly
refused at the ordinary HF inventory gate; switching off ResourceBudget would
not qualify a larger topology. Public KS/DF composition is also explicitly
unqualified. Preserve these refusals; do not disable the resource gate to
advertise a broader method domain. Accepted recipes retain
the same exact-workload/raw-holdout validation and process isolation as J/K.

The independent oracle also validates UHF inputs, but a real native UHF trial
does not publish a complete Fock census. Refuse UHF tuning explicitly until the
SCF/finalization owner counts every actual Fock, including retries and graph
replay. Do not infer work from logical iteration counts or drop the work gate.

## Rejected alternatives

- Tune basis/cutoff or packed-single storage: changes scientific approximation
  or precision rather than execution.
- Use conventional HF as the reference for fitted HF: different Hamiltonian.
- Omit auxiliary response in the reference gradient: invalid fitted derivative.
- Accept a truncated metric under reference default thresholds: different
  retained subspaces can look like schedule errors.
- Bypass KS/DF resource admission with only a sub-workspace hint: not a complete
  endpoint memory qualification.
- Force every workload resident: breaks bounded streamed fallbacks and lifetime
  assumptions, particularly in analytic forces.

## Evidence and invariants

Tests cover policy allowlisting, strict numerical floors, invariant metric rank
and Fock work, fixed auxiliary/capacity inputs, native resource receipts,
independent full-rank RHF/UHF references, and exact recipe replay. Opt-in Slurm
tests check all proposed schedules against independent E/F. Timing evidence
retains all samples, including refused/noisy candidates; no best-repeat claims.

## Revisit when

Independent rank-deficient metric/derivative semantics, complete native UHF
Fock observations and composed KS/DF resource requests are qualified.
Per-owner native execution descriptors can
later replace explicit subprocess recipes without process-global mutation.
