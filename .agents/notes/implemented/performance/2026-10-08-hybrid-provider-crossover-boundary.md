# Decision: freeze provider identity before hybrid crossover timing

Status: implemented
Date: 2026-10-08

## Problem

#2054 requests exact Direct J/K, DF-J plus exact K where supported, and full
DF-JK with occupied reuse. The current public KS selector exposes conventional
or fitted providers, while CUDA KS admission rejects fitted Coulomb with exact
exchange. A benchmark that silently routes the mixed arm would compare
different Hamiltonians under one name.

## Decision

The benchmark runner freezes the common PBE0 problem and records separate
approximation, auxiliary-basis and live provider identities. Exact Direct and
full DF-JK may enter ABBA complete E+force pilots. The mixed DF-J/exact-K arm
is `UNSUPPORTED` until a public production provider actually admits it.
Each supported result needs its own matched PySCF energy/full-force oracle;
missing, unconverged or mismatched evidence is `INCOMPLETE`. Clean timings and
intrusive trace profiles use separate processes. No production provider or
approximation selector changes with this benchmark.

## Invariants

- Do not equate Direct and DF numerical energies or forces just because they
  share a functional name; the auxiliary basis is part of DF's identity.
- Retain losing and failed attempts. A converged SCF energy with null forces
  is a failed complete E+force endpoint.
- Keep allocation/plan peaks distinct from measured memory peaks and keep
  unavailable work counters absent rather than zero.
- A small accepted pilot is not a 48/96-atom crossover, an AUTO selector, or
  closure of #2054/#1895.

## Evidence

The frozen installed H200 pilot at source `2952481f` and core library
`0f8fb7fc` passes independent three-atom and non-water holdout gates for the
two supported arms. On 48-atom water, two Direct cold solves converge but
force publication fails because prepared native integral derivatives are
unavailable within the ordinary stationary allowance; the public PBE0 force
router fixes that allowance at 512 MiB device / 256 MiB host for this route.
The full DF-JK arm completes 48-atom E+force and its occupied response profile.
The [pilot evidence](../../../../benchmarks/results/hybrid-provider-crossover-2054/README.md)
retains exact source, binary, AOT, grid, sample, oracle and failure identities.
No 96-atom run is admitted after the Direct48 failure.

## Rejected alternatives

Treating full DF-JK as DF-J/exact-K, inferring an executed provider from the
requested mode, raising the private force budget after seeing the failure,
or describing a provider allocation ledger as measured H200 peak memory would
erase the scientific and resource boundary found by this pilot.

## Consequences

This slice delivers a reusable negative-evidence benchmark and source-matched
independent oracle, while the large-system crossover question remains open.
The production owner must address the exact Direct48 derivative admission
boundary or provide an explicit supported resource contract before a matched
two-arm 48/96 endpoint claim can be made.

## Revisit when

The public KS provider admits a mixed J/K approximation with live proof and
matched analytic forces, or a reviewed source-matched campaign clears the
Direct48 complete endpoint, larger resource/work inventory and 96-atom gate.

## References

Issue #2054; `python/generativeqc/_ks_snapshot.py` provider proof;
`python/generativeqc/batch.py` stationary force allowance;
`src/dft/cuda_ks.cpp` mixed-provider admission.
