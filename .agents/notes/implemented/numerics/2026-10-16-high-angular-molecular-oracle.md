# Decision: preserve normalization and physical density in high-angular oracles

Status: implemented test-oracle policy
Date: 2026-10-16

## Problem

Primitive leaf coverage does not establish molecular orbital relaxation with
high-angular orbitals. Extending the values-only molecular oracle to d/f exposed
two limitations of the test implementation, before any native numerical change:

- Libcint Cartesian d/f functions use a shell-radial normalization convention;
  the native packed basis normalizes each Cartesian component individually.
  Comparing their unscaled fitted ERIs was comparing different AO bases.
- Unconstrained density-root trial steps can produce indefinite density matrices.
  The direct-space VV10 value oracle correctly rejects negative density; loosening
  that check or accepting a failed density root would weaken independence.

## Decision

Derive Cartesian component scales independently from the diagonal of the libcint
overlap matrix. Apply those scales consistently to AO jets, overlap, core,
full/long-range four-center integrals and both orbital legs of three-center
integrals. Auxiliary scaling cancels in the full-rank `A V^-1 A` contraction;
the independently fitted four-index audit remains in place. Spherical bases keep
their existing convention.

For high-angular fixtures, independently orthogonalize the native starting
density in the displaced overlap metric and solve the occupied/virtual rotation
residual using values-only libcint, upstream Libxc and direct-space VV10.
Exponentials of real skew generators preserve the density rank, positive
semidefiniteness and electron count at every trial. The seed is not an oracle
answer: every displaced geometry is reconverged, and the original
lowest-occupation spectral fixed-point residual must still be below `2e-10`.
Legacy fixtures retain their original density-root solve.

## Acceptance boundaries

The origin energy, core, overlap, Fock and density audits remain unchanged.
Directional curvatures still use independently reconverged molecular energies
with the `0.004`/`0.002` Richardson stencil and `atol=8e-7, rtol=3e-5`.
Raw full matrices must retain symmetry (`atol=2e-7`) and HVP parity
(`atol=2e-8`); translation invariance retains `atol=2e-8`. No compiler/native
derivative or response equation enters this oracle.

The molecular fixtures now include direct RKS d/f and joint UKS/DF/ECP
orbital-f / auxiliary-g, with Cartesian and spherical representations. The
spherical f cases also request the raw full native matrix. The actual C++ CLI
and public native frontend gain the same simultaneous spherical f/g feature
combination. Optional quantitative receipts retain native outputs, both stencil
estimates, extrapolated HVP, identity inputs and error gates.

## Rejected alternatives

Do not clip negative densities, discard Cartesian cases, compare unnormalized
AO spaces, loosen convergence/curvature tolerances, or replace the independent
energy oracle with native/compiler derivatives. Those changes would conceal the
test-domain mismatch rather than qualify the migrated numerical endpoint.

## Molecular qualification evidence

Slurm 3233 completed all 30 maintained molecular cases and all four actual
C++ CLI/public-native-frontend cases, with no skips. This run used production
base `33c0fa9482a55943e942758a41fc9b89ae474cbf` plus the three test files
as subsequently committed in `f8ee2dd66299a5704626a662f6dbe05d7725e3ed`.
The measured library SHA256 was
`ada058063d00bad1a5178942fe70c0cf2458af3bd59a16a6c25297c01284d695`;
it is not the later complete-default/source-identity build.

The 29 optional directional oracle receipts have maximum absolute error
`3.362434830567196e-9` Eh/Bohr². The five newly added high-angular cases have
maximum error `1.3341827434509845e-9`. The thirtieth benchmark-H2 test separately
checks every raw Hessian entry against reconverged energies; it passes but does
not emit the optional directional summary. Do not confuse a receipt count with
the number of passing scientific cases.

The raw full spherical f matrices have maximum antisymmetric errors
`1.2927159342979166e-14` (RKS/direct) and `3.8816172498457036e-13`
(UKS/DF/ECP/f-g). Their matrix/HVP differences are respectively
`5.9674487573602164e-15` and `2.937754206566723e-14`. No posthoc
symmetrization or relaxed numerical tolerance was used. These are small 8/4/8
grid smoke fixtures, not chemically grid-converged production benchmarks.

The complete-default package receives its own clean-source endpoint
qualification; these earlier source/binary identities must remain separate.
