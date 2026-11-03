# Decision: opt-in host electronic analysis of accepted native mean-field states

Status: implemented
Date: 2026-11-03

The host-only analysis decision is superseded by
[backend-following CUDA electronic properties](2026-11-03-cuda-electronic-properties.md).
Accepted-state capture and detached result ownership remain unchanged.

## Problem

The public single-point result exposed energies and forces, but not molecular
dipoles, atomic partial charges or a usable spatial electron density. Re-solving
with a reference package would violate the native production boundary and could
analyze a different determinant, fitting model or numerical grid.

## Decision

Electronic outputs are explicit single-point consumers of the same accepted
RHF/UHF or RKS/UKS state. HF moves its existing returned AO density into the
method result and copies it only for the additive density-capture endpoint.
KS uses the existing token-checked final-state exporter, including when the
public single-point endpoint delegates to its prepared-batch owner. No SCF is
replayed. Failed/unaccepted states do not publish electronic outputs.

The selected SCF backend is unchanged. Host FP64 density export, normalized
Gaussian position/overlap integrals, population contractions and spatial AO
sampling are explicitly part of the electronic-analysis contract. PySCF is
only an independent test oracle. Position integrals raise a polynomial power
in the shared overlap evaluator without re-normalizing the raised polynomial;
public Cartesian/spherical expansions and original contraction normalization
remain authoritative.

`charges` requests both Mulliken and symmetric Lowdin partitions. Dipoles
include nuclei and electrons with a documented coordinate-origin convention.
Density owns detached total and optional alpha-minus-beta matrices and frozen
geometry/shell inputs. Sampling prepares an independent native AO owner and
Cube output streams point blocks, rather than retaining a grid-sized tensor.

The existing method capability flags and default outputs are unchanged.
`supported_electronic_properties` describes the separate host consumers; prepared
batch output selectors still accept only their established energy/force flags.

## Boundaries and rejected alternatives

- Do not silently return an HF reference density for MP2/CC energies. Relaxed
  correlated densities require their own scientific definition and response gates.
- Do not treat the integer molecular charge as an atomic partial-charge scheme.
- Do not infer dipoles from point charges; include the full electronic position
  expectation, including on-site AO contributions.
- Do not use an integration grid to approximate analytic Gaussian dipoles.
- ECP and resource-budget analysis are not admitted yet. ECP needs an explicit
  valence/core and Cube-nuclear convention; host analysis needs its own capacity
  ledger before claiming a resource-bounded endpoint.
- Real-space sampling inherits the current native AO s/p/d/f domain. Dipoles
  and populations do not silently choose a reference SCF for other inputs.

## Evidence and acceptance gates

`tests/python/test_electronic_properties.py` checks direct and fitted RHF/UHF
against independent PySCF/libcint calculations using identical decimal basis
inputs, accepted KS state operators, normalized Cartesian/spherical spdf
position matrices, open-shell spin occupations, charge conservation, neutral
and ionic translation laws, density lifetime, streamed Cube ordering and invalid
inputs/nonconvergence. Full-SCF energy agreement is 2e-9 Hartree, density/dipole
agreement 3e-7 in atomic units, and fixed-state position agreement 3e-12.
GPU execution must be qualified through Slurm; CPU agreement alone does not
establish real-device qualification or a performance claim.

## Revisit when

Add native device-resident property contractions when an independently qualified
complete endpoint justifies them; add host capacity accounting before supporting
resource-budget analysis. Correlated, ECP, Hirshfeld/ESP and native CLI outputs
need explicit additional contracts rather than widening this one implicitly.
