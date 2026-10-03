# Decision: integrate DF virtual actions into the native RCCSD iteration owner

Status: implemented
Date: 2026-10-03

## Problem

The generated one-Q DF actions omitted ovvv/vvvv storage, but no native solver
accumulated them. Reusing a conventional Problem with missing dense inputs would
otherwise allow conventional Lambda/triples/force consumers to read absent data.

## Decision

Keep a single CPU/CUDA RCCSD iteration and DIIS owner. An explicit naux and
Q-major B_ov/B_vv representation replaces exactly ovvv/vvvv; retained smaller
integrals must belong to the same fitted Hamiltonian. Fock and reference energy
remain supplied by the chosen reference contract, currently conventional RHF.

Generate retained shared iteration and independently expanded replay equations
from the existing inventory with external virtual corrections. Each evaluation
recomputes the complete Q sum for its current amplitudes, including DIIS trials
and convergence replay. Admission charges factors, accumulation, action scratch,
core scratch, histories, host retention and detached final amplitudes before
numeric execution. CUDA uploads factors once and retains them throughout solve.

The conventional admission function rejects DF unless an execution owner opts
in explicitly. Existing response/force/triples consumers retain that default.
No public DF Calculator method or force capability is registered by this change.

CUDA composition preserves one sticky arithmetic flag across the full auxiliary
loop and retained core; the standalone one-Q API continues to clear its flag.
An error in an early Q must not be hidden by a later successful action. Borrowed
outputs are accumulated on the same stream before their scratch is overwritten.

## Alternatives and limits

Duplicating DIIS/convergence policy in a second DF solver would create a second
numerical owner. Reconstructing ovvv/vvvv would undo the storage benefit. Both
were rejected. The present resident-factor route is an explicit internal
contract: a budget shortfall rejects, rather than falling back to dense tensors.

One-Q residual reassociation alone still repeats high-order T2 contractions Q
times. Auxiliary slice counts and operation counts include every trial/replay;
this implementation does not claim reduced formal CCSD scaling or a speedup.
A next structural optimization should aggregate smaller Hamiltonian-dependent
intermediates before their Q-independent T2 contractions and retain a bounded
fallback. Standard (T), native sources, Lambda and nuclear forces remain work.

## Evidence

The supplied-Hamiltonian native qualification compares CPU/CUDA DF solves with
complete dense native solves and a separately implemented determinant-space
oracle. It covers occupied-rich dimensions, multiple auxiliary slices, DIIS and
non-DIIS, complete Q counts, one-time factor uploads, exact/short budgets,
symmetric factors, early-Q arithmetic failures and default conventional refusal.
Five conventional CC/triples/Fock generated sources remain byte-identical.
The reproduction entry is `tests/python/test_df_cc_native_solver.py`; CUDA needs
`GENERATIVEQC_DF_CC_CUDA_TEST=1` inside a finite Slurm allocation.

The [retained qualification](../../../../benchmarks/results/df-cc-native-solver-20261003/publication.json)
contains twelve complete CPU/CUDA solver records with unrounded amplitudes,
dense comparisons, independent determinant or pinned molecular energies, and
complete Q work counts. Host checks pass 43 cases, real CUDA checks pass 25,
and two full unfiltered small-solve memchecks report zero errors in Slurm job
12120. CPU-only/CUDA CMake generation and explicit ccache commands are checked.

This is internal supplied-Hamiltonian solver qualification. Independent
hundreds-AO PySCF source states are validation targets, never production inputs.
A complete native DF CCSD(T) energy/force endpoint remains unqualified.
