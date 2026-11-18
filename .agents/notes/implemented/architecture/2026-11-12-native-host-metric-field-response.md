# Decision: compiler-owned host metric, RHS and D/W response

Status: implemented
Date: 2026-11-12

## Problem

The fixed-geometry host CPKS action has no overlap motion or external nuclear
source. Reimplementing orbital/Pulay algebra in C++ would create a second
scientific authority. A diagonal-only weighted-density tangent would also lose
occupied-occupied energy motion and fail in degenerate occupied spaces.

## Decision

Lower the existing canonical KS TensorIR maps at build time through a checked
runtime-sized FP64 host backend. Every definition, including AD primal inputs,
retains finite validation. The backend materializes into one caller-borrowed
arena, reports exact workspace and scheduled terms, and publishes only after
all intermediates pass. No interpreter, runtime compiler, Python bridge or new
method formula participates in native execution.

`HostKsPerturbation` owns staging and composes the authenticated child response
with these metric/RHS/density/full-W maps. A new compiler-owned physical-residual
map reuses the canonical RHS expression and orbital-gap term. The consumer
rebuilds induced Fock on the full density tangent before accepting that residual.
The original GMRES tolerance remains the physical gate. Both outputs commit
together, never on an incomplete solve or a failed resource/lifetime check.

One contiguous numeric allocation holds the retained panels and largest map
arena. The resource envelope composes this with child response/action and
GMRES storage. It explicitly excludes borrowed provider/BLAS storage, metadata,
allocator overhead and process RSS. The separate work cap bounds only the
materializing maps; it does not imply bounded complete molecular work.

## Rejected alternatives

- Handwritten C++ orbital or weighted-density formulas duplicate compiler
  ownership and obscure occupied-degeneracy behavior.
- Dense Cartesian RHF integral AD does not supply general spherical hybrid/ECP
  nuclear sources and must not be relabeled to fill that missing contract.
- Finite-difference or PySCF production fallbacks do not become native analytic
  response. Displaced SCF remains an independent test oracle only.
- Linking these development components is not public Hessian/TS admission.

## Invariants and qualification

Existing map equation identities are unchanged; the physical-residual map is
additive. Full occupied-energy-matrix AD remains canonical. Independent
generalized displaced-SCF map tests include degenerate occupied and empty/full
spin populations. Molecular controls use cold independent Libxc/ERI SCF on
the exact exported grid, at two overlap/field displacement sizes. Native and
oracle execution do not cross-seed densities or orbitals. Particle-number
metric conservation, full D/W parity and unchanged physical residual gates are
required, as are failure/alias/budget/lifetime/reentrancy recovery controls.

Source-matched logs and JUnit properties live in the ignored qualification
directory `build/qualification/host-ks-perturbation-20261112/`. The preceding
fixed-geometry source cohort and its recovery remain immutable and do not
qualify these new files. No real GPU qualification is claimed.

The UKS displaced-field oracle deliberately accepts different alpha/beta
one-body fields. PySCF's UKS DFT energy helper delegates to a spin-independent
two-dimensional hcore contraction; it cannot consume that test field directly.
The test therefore supplies the matching spin-resolved one-body energy plus
the oracle's unchanged Coulomb/XC energy. Neither native production arithmetic
nor its SCF is replaced. Early test-harness failures (this energy shape and a
misplaced final assertion) remain retained separately from final acceptance.

## Consequences and revisit conditions

The optional static CPU target now exposes abstract overlap/field response,
not geometry-derived nuclear Fock sources or a full Hessian. Native moving
grid/integral/ECP source ownership, second derivatives, complete stationary
assembly, installed/current-source and representative endpoint qualification
remain required. Replace the materializing schedule only with independently
qualified semantic parity and explicit work/storage evidence.

## References

- [Fixed-geometry host response](2026-11-12-native-host-coupled-response.md)
- [Current contracts](../../../../docs/developer/dft_hessian.md)
