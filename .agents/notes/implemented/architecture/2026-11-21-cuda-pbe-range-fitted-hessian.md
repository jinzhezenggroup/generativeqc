# Decision: Differentiate the fitted range-PBE operator at its accepted omega

Status: implemented
Date: 2026-11-21

## Decision

Extend the [exact LR path](2026-11-21-cuda-pbe-range-hessian.md) to two fitted
radial sources. Bind the accepted LR `omega` to both metric and three-center
factorized derivative leaves. The radial parameter is fixed during nuclear
differentiation. Expanded leaves still require their declared signature; the
new parameterized leaves have distinct scientific identities.

Authenticate the fitted correction as its own accepted provider on the same
orbital system. Its rank may differ from the primary's; the relative cutoff
must match and each spectrum must be separated from that cutoff. The fitted
exchange-only response accepts the signed LR-minus-SR fraction. Full-range
J/K owners still reject negative exchange fractions.

The default stays exact LR for range-PBE so existing energy calculations do
not silently change. Explicit `df` and `auto` select the SCF Hamiltonian before
the first energy. The planner resolves the same model default as preparation
and charges the split fit and exact retry as alternative lifetimes. An
accepted SCF fit cannot be replaced by exact LR at derivative time.

## Evidence

- Slurm job 2292 (RTX 5090, SM120): public budgeted dense Hessian and HVP at
  `omega=0.33` for RKS with explicit `auto` and UKS with `df` and a negative LR
  correction. Both use s/p auxiliary bases and report zero exact quartet work.
  Independently reconverged Libcint/Libxc fitted-LR energies satisfy the
  `2e-6` Hartree/Bohr² Richardson HVP gate. Five WB97M-V resource regressions pass.
- Slurm job 2290: raw and weighted DF passes agree with displaced Libcint
  reference integrals for s/p at `0.3` and spherical f/g at `0.33`. A mismatch
  between live source `0.33` and derivative request `0.3` fails closed.
- Focused host identity tests admit matching fitted primary/LR and reject an
  exact primary or different LR cutoff. The unrelated split-hybrid assertion in
  the full host final-state suite also fails on the checksum-matched pre-change
  library; no split-hybrid behavior was changed here.

Logs and binary hashes: `build/hessian-gpu-sm120/pbe-ri-*`.

## Remaining scope

Scalar ECP admission is covered by the [follow-up decision](2026-11-21-cuda-pbe-range-ecp-hessian.md).
The [follow-up CLI decision](2026-11-21-cuda-pbe-range-cli-method-ir.md) covers
this MethodIR. Arbitrary correction graphs are still outside this domain.
The GPU qualification above is not cross-architecture evidence.
