# Decision: reuse the stationary native owner for CPU PBE/PBE0 UKS forces

Status: implemented
Date: 2026-10-12

## Problem

The complete CPU PBE/PBE0 force owner rejected unrestricted contexts even
though its final-state frame, XC point/atomic-grid derivative and density-fitted
integral response already supported separately occupied alpha/beta blocks.
The missing Direct binding was the compiler's two-spin integral adjoint layout,
not a missing physical gradient equation. Copying a UKS force implementation
would create another scientific authority and leave token/publication behavior
to diverge.

## Decision

Generate the two-spin PBE/PBE0 integral source weights from the existing
`StationaryGradientPlan` TensorIR AD alongside its unchanged restricted weights.
The CPU Direct bridge validates both D/W pairs and packs them into the
compiler-owned contiguous spin layout. It combines generated spin weights
before submitting each shell quartet once to the packaged CPU first-derivative
provider. Coulomb and exchange channels share the same primitive recurrence;
UKS does not duplicate that traversal. The four additional flattened matrices
enter both the preflight capacity and observed retained-capacity calculation.

The prepared owner authenticates spin count against its Fock specification and
the canonical exchange coefficient: PBE0 is -0.125 for RKS and -0.25 for UKS.
It then reuses the existing integral, XC/grid, nuclear, final-state token and
transactional force-publication owners. Both beta matrices are mandatory even
when the beta occupation is empty; absent spin storage must not masquerade as
an empty population.

## Rejected alternatives

- Handwritten spin Coulomb/exchange density equations: generated stationary AD
  already owns those coefficients and cross-spin terms.
- A force driver per spin or method: the existing shared source owner already
  accepts both spin blocks, so this would duplicate lifecycle and science.
- Molecular coordinate-by-AO-rank-four derivative tensors: shell-local weights
  preserve the existing bounded-storage and work-envelope policy.
- Expanding orbital grids to g during qualification: CPU integrals support g,
  but public orbital AO jets still stop at f. The actual DF endpoint intersection
  is orbital f and auxiliary g; qualification uses distinct primary/auxiliary
  spaces and preserves Cartesian versus real-spherical representations.

## Invariants

- Keep CPU, canonical full-range all-electron PBE/PBE0 and strict-FP64 admission.
  Direct additionally retains its packaged s/p/d source and work gates.
- Do not infer method identity from a PBE family label or from two spin blocks.
  Scaled hybrids, RSH, nonlocal, ECP and native correction-bearing graphs do not
  acquire this force capability.
- A frontend that separately decomposes an external correction may query a
  qualifying electronic native owner. That is not qualification of a complete
  native corrected model: correction orchestration remains a separate contract.
- No Python method/spin whitelist is added to batch dispatch. The prepared
  native per-item property query remains the force-routing authority.
- Failed/nonconverged states, revoked generations and insufficient source
  capacity never grant a stationary lease or partially publish force/source
  buffers. Invalid neighbors must not poison valid-item warm state.

## Evidence

`test_native_uks_forces.py` independently compares total moving-grid analytic
forces, final alpha/beta D and energy-weighted D with PySCF 2.14.0, including
an H2+ empty-beta determinant and a water cation with both populations. The
change of AO basis is measured from independently evaluated functions; tests
do not pretend spherical d/f/g and Cartesian spaces are interchangeable.
Angular fixtures cover Direct d and DF orbital f/auxiliary g in both
representations. Energy gates are 2–3e-9 Hartree; force gates are 2–4e-7
Hartree/Bohr, with final D/W gates of 3e-7 in the native AO basis.

Portable generated weights are compared with independent TensorIR execution
while generation forbids NumPy/runtime/provider imports. The retained native
Jet recurrence is an independent source oracle using unequal alpha/beta D/W;
it also checks actual shared primitive counts and exact/below-capacity gates.
Public-route effect tests block the Python scientific force driver. Cold/warm,
genuinely moved ragged batches, failed-neighbor recovery, nonconvergence and
stale/closed-state leases are exercised separately from science oracles.

Native C ABI, C++ SDK and CLI tests compare full UKS E+F publication with
freshly prepared/reconverged displaced energies for PBE/PBE0 Direct and DF.
`benchmarks/native_cpu_ks_force_endpoint.py` measures construction/preparation,
cold/warm/moved E+F and destruction, alternates paired execution order, checks
semantic SCF/Fock work and numerical equality, and retains a separate native
progress/capacity observation. Those named numeric observations are not a
complete RSS or simultaneous-live endpoint memory receipt.

## Consequences and revisit conditions

RKS keeps its original weights, buffer footprint and scientific traversal.
UKS pays four compact matrix buffers at the Direct binding boundary, not
another source recurrence. Revisit the packing only if a compiler-owned
disjoint-span weight ABI can remove it without copying the density equations
or hiding the extra live capacity. CUDA, wider native models, response/HVP,
complete resource/lease authority and remaining policy/resolution/correction
cutovers stay separate #2227 work; this slice does not complete R1–R6.

## References

- #2227 and #2302.
- `src/methods/dft_method.cpp`, `src/scf/stationary_integral_gradient.cpp`.
- `tools/generate_stationary_integral_weights.py`.
- `tests/python/test_native_uks_forces.py`.
- `tests/native/test_dft_integral_sources.cpp` and `test_cpp_batch.cpp`.
- Prior decision: `2026-10-12-native-cpu-direct-pbe-forces.md`.
