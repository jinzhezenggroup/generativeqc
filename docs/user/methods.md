# Methods and long-term scope

VibeQC's long-term mission is to cover **all quantum-chemistry methods** in one
accelerator-native system. Public method discovery is rendered at Sphinx build
time in the [public method catalog](../public_methods.md). The catalog combines
stable native ABI registrations, compiler-discovered DFT selectors, public
composite selectors, and the automatic Libxc semilocal MethodIR inventory.

Stable native ABI IDs, providers, declared properties, batch capability, and
compatibility aliases remain owned by `manifests/public_methods.json`. DFT
scientific identity remains compiler-owned: representation never bypasses
backend, basis, grid, spin, derivative, or production-domain admission gates.

## Current method status

Run the Python frontend (`python -m vibeqc methods`) for the current public
discovery set, and use the generated
[public method catalog](../public_methods.md) for the documentation view across
native, compiler-discovered, composite, and automatic Libxc entry paths.

A discovered or listed method is not a blanket claim that every backend, basis,
grid, spin state, or requested property is qualified. The method contract and
execution-time admission checks remain authoritative for those combinations.
Compiler representation alone is likewise not a public execution guarantee:
unsupported lowerers fail closed. A Python-free native SDK install has a
separate `vibeqc methods` command that intentionally reports the C/C++ ABI/provider
registry only; it does not import the compiler catalog. Planned families and
development directions are listed in the
[implementation roadmap](../maintainer/roadmap.md), which does not promise
release dates or a fixed implementation order.

## CUDA global-hybrid forces

The Python `Calculator` exposes analytic forces for admitted all-electron
global-hybrid RKS/UKS compositions with direct J/K, FP64, an explicit `GridSpec`,
and device-fused XC. Force eligibility follows the actual MethodIR primitives;
the SCF owner still validates the semilocal composition. Request forces through
the ordinary single-point or prepared-batch interface:

```python
from vibeqc import Calculator, GridSpec, KsOptions

calc = Calculator(
    method="b3lyp-rks", device="cuda", precision="fp64", basis="sto-3g",
    ks_options=KsOptions(
        grid=GridSpec(radial_points=32, angular_polar=10, angular_azimuth=20),
        xc_schedule="device_fused",
    ),
)
result = calc.singlepoint(
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], properties=("energy", "forces")
)
```

The first call compiles a method-specific CUDA wrapper and requires a
discoverable NVCC toolkit (`CUDACXX` or `CUDA_PATH` can select it). Later calls
reuse the compiler cache. There is no CPU scientific fallback. Density-fitted,
range-separated, nonlocal, ECP and mixed-precision hybrid forces remain outside
this contract. See the [execution and resource limits](../developer/stationary_cuda_diagnostic.md)
and [independent force acceptance gate](../maintainer/hybrid_cuda_acceptance.md#public-global-hybrid-force-gate).

## Direct CUDA HF force state

Direct RHF/UHF force solves require the maximum physical AO commutator
`|F P S - S P F|` to meet `min(1e-8, density_tolerance)` in addition to the
energy and density-update criteria. The maximum includes both UHF spins;
nonfinite residuals cannot pass. Additional SCF updates remain within the
requested iteration limit and appear in the public iteration count.
An item's mixed-precision coarse stage keeps its previous stop; the exact FP64
refinement and exact-precision neighbors apply the physical force criterion.

Finalization projects the final physical-Fock orbitals to a determinant,
rebuilds its physical Fock, and rechecks that determinant's commutator before
publishing forces. Energy, forces and the returned warm state share this P/F(P).
The Pulay weight is `P F(P) P / 2` for RHF and `P_sigma F_sigma P_sigma` for
UHF. A failed final residual returns nonconvergence. Energy-only and detached
physical-reference execution retain their existing finalization contracts.

This force finalization adds one density projection, one physical Fock build,
four matrix products for residual validation and two for the Pulay weight.
Existing device scratch holds the products; a four-byte work counter is copied
at the existing completion fence. `VIBEQC_DF_PROGRESS_TRACE` records final
updates, physical Fock builds, residual checks and rejections separately from
iterative SCF updates. Complete endpoint costs include all this work. The
[decision record](../../.agents/notes/proposed/2026-09-17-consistent-direct-pulay-weight.md)
retains the independent diagnosis and qualification boundaries.

## Acceptance standard

A method becomes supported only when all of the following are true:

1. Its public behavior and mathematical conventions are documented.
2. Energies and relevant derivatives agree with an independent implementation
   over representative systems and basis sets.
3. CPU/GPU execution boundaries and unsupported cases fail explicitly; there
   is no silent fallback to an unvalidated path.
4. Reproducible benchmark artifacts support any performance claim.
5. Batched execution preserves per-system ordering, diagnostics, and failure
   isolation where the method permits batching.

## Expansion strategy

VibeQC expands method coverage behind the same registry-driven prepared
calculation interface. New public capabilities should reuse the existing basis,
integral, SCF, batching, resource-planning, and diagnostic contracts rather than
introducing method-specific API branches.

Near-term development focuses on completing scientific and backend coverage of
the method families already present in the registry: broader DFT execution and
analytic derivatives, density-fitted derivatives, correlated-method
qualification and derivatives, and wider basis/ECP/backend coverage. Compiler
and runtime work should remove duplicated handwritten execution paths while
preserving explicit numerical ownership and fail-closed unsupported cases.

A capability is promoted only after the [acceptance standard](#acceptance-standard)
is satisfied. Implementation details belong in the
[Developer Guide](../developer/index.md), performance qualification in the
[Maintainer Guide](../maintainer/index.md), and durable historical rationale in
`.agents/notes/`.
