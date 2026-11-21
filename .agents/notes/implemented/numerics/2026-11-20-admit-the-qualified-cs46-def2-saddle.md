# Decision: Admit the qualified Cs46 def2-ECP saddle protocol

Status: implemented
Date: 2026-11-20

## Decision

The public CLI and Python molecular saddle adapters accept a spherical scalar
Cs46 def2-ECP operator for direct CPU FP64 B3LYP-D3(BJ) RKS only when explicitly
using the [bounded cold proposal protocol](2026-11-20-bound-cold-rks-fock-proposals.md)
and accepted-frame warm starts. The number of frozen core electrons and the
basis filename do not identify the operator. Match the actual local and
projector terms at every ECP centre; harmless decimal spelling and radial row
ordering do not change identity. Ordinary derivative admission remains
independent of saddle admission. The Python adapter checks force eligibility
through the prepared native owner, not the conservative calculator summary,
and still never falls back to Python forces.

## Evidence and limits

Both installed candidates independently passed the same Cs46 first-order gate
with spherical def2-SVP, 2/2 accepted attempts, 108 independent displacements,
one unscaled `−571.052176 cm⁻¹` mode, and maximum raw Hessian error
`4.98e-7 Eh/Bohr²`. See `installed-ecp-saddle-candidate/cs46/` and
`installed-ecp-saddle-candidate/python-prepared-owner/full-cs46/` under
`/data/jzzeng/qualify-2844-2848-20261120` for artifact identities, work and timing.
These candidate results do not attest a subsequently rebuilt wheel. The
source-bound admission tests check the canonical potential and refuse a
coefficient change with unchanged provenance and core count before SCF.
UKS ECP, I28, and other ECP saddle protocols remain unqualified. Neither
the numerical gate nor the admission policy guarantees an arbitrary starting
geometry reaches the intended electronic root or a first-order saddle.
