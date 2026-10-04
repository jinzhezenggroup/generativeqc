# Decision: close the remaining GFN2 CPU/CUDA scalar science split

Status: implemented
Date: 2026-10-04

## Problem

GFN2 already had compiler-owned CN/repulsion, H0 response, S/D/Q integrals,
Hamiltonian/VJPs, ES2/ES3/AES2, D4 scalar mathematics, spin scalar updates and
external point-charge force response. A final audit still found small scientific
conventions repeated in backend code: SCC electronic-energy updates,
plain/energy-weighted density scalar arithmetic, final Mulliken
charge/magnetization transforms, and terminal total-energy composition.

The conservative CUDA ownership ledger also still classified several runtime
files as handwritten science after earlier generated cutovers.

## Decision

Keep backend schedules backend-specific, while moving the repeated scalar
mathematics to compiler-owned programs:

- Shared `tensor.scf` density/weighted-density TensorIR remains the complete
  scientific identity for orbital density formation. GFN2 CPU/CUDA schedules
  bind that identity and use compiler-owned scalar lowerings for orbital energy
  weights, coefficient weighting and ordered updates.
- The #505 electronic runtime programs own the remaining Mulliken publication
  transforms.
- CUDA SCC electronic energy consumes the generated H0-density update and the
  shared SCC free-energy program.
- CPU and CUDA terminal energy consume one generated
  `E_SCC + E_repulsion + E_D4^ATM` program.
- CPU density remains a BLAS lowering while CUDA remains a tiled reduction;
  both consume the compiler-owned scalar weighting/update contracts.

Native code continues to own topology, ragged activity, reductions, BLAS/GPU
schedules, finite checks, stream/error handling and transactional publication.

## Rejected alternatives

- Moving occupation root solving, Broyden history, convergence policy or the
  eigensolver into TensorIR. These are solver/numerical runtime policy and are
  deliberately outside #505.
- Replacing CPU BLAS density formation with scalar loops merely to make source
  text identical. Scientific ownership is shared; execution schedules may
  differ.
- Routing GFN2 D4 through the generic complete fixed-charge scheduler. GFN2
  intentionally retains its SCC-resident cached traversal while sharing the D4
  scalar scientific owner.

## Invariants

- Restricted/unrestricted and finite-temperature occupation semantics do not
  change.
- Magnetization remains the pinned raw-alpha minus raw-beta convention,
  equivalent to N_beta-N_alpha for the negative Mulliken electronic
  contraction.
- No new full-size density copy or host staging is introduced.
- Backend reduction association remains independently qualified and must not be
  advertised as bitwise CPU/CUDA identity unless measured.
- Failure isolation and transactional publication remain native runtime policy.

## Evidence

Structural regression tests require production CPU/CUDA consumers to call the
generated helpers and forbid the retired local formula bodies. Existing GFN2
CPU/CUDA numerical and real-device CI remain the executable acceptance gate for
this source identity.

The generated density helpers reject nonfinite intermediate arithmetic before
the solver's final array checks. Every such data-failure exit must record
`EIGENSOLVER_FAILED` for that system: the batch caller intentionally continues
after a data failure and commits numerical staging only when the per-system
status is `SUCCESS`. Returning only the internal data-failure enum could reuse
a previous successful status and publish stale wavefunction/thermodynamic data.
The compiled regression in `test_gfn2_fused_electronic_codegen.py` runs the actual
solver and batch publication with an injected LAPACK/BLAS provider, first stages
a successful solve, then checks restricted and unrestricted arithmetic overflow
without publishing the previous values. This host regression does not qualify
an external linear-algebra provider or NVIDIA execution. CuMetal excludes the
native GFN2 CUDA target; the NVIDIA runtime/reference gate remains unrun locally.

## Consequences

The remaining GFN2 native files may still contain substantial
control/reduction code, but that is no longer evidence of duplicated method
equations. Ownership metadata should classify files by the mathematics they
actually own, not by their historical vendored origin.

## Revisit when

Revisit only if a backend requires a different physical density, population,
energy or spin convention. Pure scheduling/performance changes should preserve
the shared scientific programs.

## References

- #560
- #1813
- #1814
- #1815
- #505
- #1240
- #926
