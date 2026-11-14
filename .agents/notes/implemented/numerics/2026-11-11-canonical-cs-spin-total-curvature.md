# Decision: retain canonical Cs total curvature without broad ECP promotion

Status: implemented low-level qualification evidence, not public ECP admission
Date: 2026-11-11

## Problem and decision

Synthetic analytic-ECP UKS acceptance cannot certify the canonical def2-ECP46
potential, native-default grid, real-spherical basis or total B3LYP-D3(BJ)
curvature. Conversely, one canonical Cs dimer must not promote I, mixed Ni/ECP,
installed/public endpoints, scale or transition-state qualification.

Preserve separately completed CsH RKS and CsH+ doublet UKS cohorts using the
same native analytic library. Both compute every raw electronic 6x6 Hessian
column, two fresh electronic HVPs and exactly-once uncut two-body D3(BJ) total
curvature. Independent cold PySCF/Libxc Gaussian/VWN-RPA SCFs and moving-grid
gradients supply the electronic reference; independent simple-dftd3 gradients
supply dispersion. Neither oracle is used by production or cross-seeded from a
native density. No clipping, projection, post-hoc symmetrization or gate
relaxation supplies acceptance.

## Actual completed evidence

Both processes exit zero. The shared input is canonical real-spherical
def2-SVP / Cs def2-ECP46 with the native-default 33,516-point grid and 29 AOs.
RKS has five doubly occupied orbitals; charge-one UKS has populations `(5, 4)`.
These are deliberately separate declared electronic states, not a comparison
of spin layouts at an undeclared common electron count.

At independent gradient steps `0.003`, `0.001`, `0.0003` Bohr, the maximum
total raw matrix errors are:

| Case | 0.003 | 0.001 | 0.0003 |
| --- | --- | --- | --- |
| CsH RKS | 7.063509033511384e-7 | 7.3622138857532e-8 | 1.0704201497402366e-8 |
| CsH+ UKS | 5.295980505803932e-7 | 5.884503664899121e-8 | 5.2942326911331605e-9 |

All are below the unchanged `1e-5` matrix gate. Both fresh HVPs, raw
symmetry/translation, matrix-action and source reconstruction gates also pass.
The largest response residuals are `5.549225760578535e-12` (RKS) and
`1.237168811269154e-11` (UKS), below `2e-10`. D3-only matrix errors are at
most `1.507828555583231e-11`, below its separate `2e-9` gate.

UKS central-state energy/density/gradient maximum errors are
`2.2737367544323206e-13`, `1.4381074109337533e-10`, and
`1.1922685061449556e-12`; its independent orbital gradient norm is
`2.3874103451026033e-11`. Raw UKS symmetry error is
`1.9168694409543718e-13`. The electronic source inventory retains local and
nonlocal ECP, one-electron, Coulomb/exchange, XC AO/grid/weights, overlap/Pulay
and nuclear contributions before exactly-once D3 composition.

Complete wall times including independent references are 1465.9052 seconds
and 2214.6453 seconds respectively. The native matrix-plus-two-HVP scopes are
1405.9084 and 2136.5492 seconds. These include JIT and concurrent qualification;
they are not isolated performance claims. Receipts retain actual semantic
source/kernel/grid work, response solves/iterations, bounded workspaces and
outer-process peak RSS. Later H3 and iodine jobs overlap portions of the UKS
cohort; the recovery manifest supplements the driver's initial concurrency
description rather than rewriting its original receipt.

## Identity and recovery

The native library SHA256 is
`afb07c88c86bcfa2959cfa35bfcc138266c0d6fed8fe5abe16cac02b10ac0ec9`.
The driver SHA256 is
`13a3fbdd5ce1e9c67086e6f21adeaa97b8e79822bda86396652f121dcdcb24bb`.
It saves the raw electronic matrix before any later composition or validation.
Original failed RKS harness attempts remain failed and are retained by the
earlier stable-B88 recovery; they are not relabeled as total acceptance.

Raw matrices/actions, individual independent gradients, density/overlap/grid
arrays and source/library identities are in:

- `build/qualification/canonical-cs-rks-total-budgeted-20261111/`.
- `build/qualification/canonical-cs-uks-total-budgeted-20261111/`.

The separate verified recovery archive is
`build/qualification/uks-analytic-ecp-20261111/canonical-cs-spin-scoped-recovery.tar.gz`:
710 members, 54,999,517 bytes, SHA256
`dc7897496dbdc24f041d3e41ca36657d4bfa3ed26b066284eadb952a775fd837`.
Its manifest SHA256 is
`31415fe43f736b8dc6544daf473ae7cc0b580800444206979a2e055157ba3108`.
Every archive member and original source was independently rehashed. The
archive excludes live Ni, iodine and H3 calculations and the unpublished
saddle candidate. Its note snapshots predate this recovery link; they are not
restamped. The former stable-B88 archive remains unchanged.

## Boundaries and revisit conditions

This supersedes only the earlier absence of a completed canonical Cs UKS
matrix. I, mixed Ni/I/Cs molecular curvature, installed public ECP, 25–40-atom
work/resource qualification and the requested native TS/frequency endpoint
remain open. The independently frozen installed Ni process remains a separate
older source/binary cohort. No public capability, issue state, PR state or
release authority follows from these two dimer results.

## References

- Issues #2844–#2848; PR #3062.
- `2026-11-11-b88-spin-node-response-stability.md`.
- `../../proposed/2026-11-11-analytic-ecp-uks-spin-response-gate.md`.
