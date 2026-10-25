# Stable unrestricted electronic roots

`Calculator.stable_singlepoint` explicitly requires a qualified real internal
UHF/UKS stationary root before returning an energy or analytic force. Ordinary
`singlepoint` retains its existing convergence contract. Install
`generativeqc[stability]` for the bounded CPU linear algebra.

```python
from generativeqc import Calculator, SpinRootPolicy

calc = Calculator("uhf", basis="sto-3g", max_iterations=200,
                  energy_tolerance=1e-12, density_tolerance=1e-11,
                  initial_guess=None)
root = calc.stable_singlepoint(
    [("H", (0, 0, 0)), ("H", (0, 0, 4.0))],
    properties=("energy", "forces"), policy=SpinRootPolicy())
next_root = calc.stable_singlepoint(
    [("H", (0, 0, 0)), ("H", (0, 0, 4.02))],
    previous=root, properties=("energy", "forces"))
print(root.energy, root.spin_squared, next_root.root_relation)
```

The complete occupied-to-virtual real alpha/beta Hessian is twice the shared
spin response action. Every column and every eigenpair is checked with a fresh
action, including self-adjointness, residual and reference uncertainty. A stable
initial state is retained. A negative mode proposes both signs of each finite
rotation; each downhill proposal must reconverge through native SCF and pass an
independent complete stability and nonlinear Fock/energy check. An energy decrease
alone is insufficient. A near-singular or unresolved root fails qualification.
An empty rotation space still requires the nonlinear stationary-reference gates.
This is a local internal-stability certificate, without a global-minimum,
complex-orbital or spin-flip stability claim.

For each retained spin density `D`, physical stationarity also requires the
Frobenius norm of `C.T @ (F @ D @ S - S @ D @ F) @ C` to satisfy
`stationarity_tolerance`, where `F` is rebuilt from the actual density and
`C.T @ S @ C = I`. This metric-normalized norm is not divided by the number
of orbitals. Recanonicalizing `F` alone does not qualify a different retained
occupied projector.
The dimensionless Frobenius norm of
`C.T @ S @ D @ S @ C - diag(occupations)` must also satisfy the same tolerance;
this binds the density to the occupied columns even when `F` is degenerate.

(spin-root-backends)=
## Spin root backends

The endpoint is explicit native CPU, all-electron FP64, direct exact J/K. Native
SCF candidates use a screening tolerance no larger than 1e-12; certification
rebuilds unscreened J/K and the complete nonlinear SCF-domain XC potential.
Polarized LDA/PBE point-response models are admitted with their native grid,
tail and functional identity. Unsupported XC response domains, fitting, ECPs,
CUDA, composite dispersion and application precision budgets fail closed.
SciPy supplies CPU eigensolves and finite rotations; native SCF/J/K/XC and the
shared spin response own the scientific model. No PySCF or developer-tool oracle
runs in production.

`ump2` uses the same controller on its actual native UHF
reference. Restarts seed the existing native correlated batch, and the reference
returned by that batch is requalified before its correlated result is published.
Explicit `properties=("energy", "forces")` uses the same qualified reference for
[complete native CPU UMP2 analytic forces](methods.md).
`root.reference.energy` is the UHF reference energy; `root.energy` is the requested
method's total energy. UCCSD(T) is not admitted by this endpoint because it
does not expose the required CPU prepared-batch contract. This qualifies the electronic reference, not coupled-cluster
amplitude stability or the unrestricted reaction benchmark. Every attempted
correlated endpoint counts in the complete call's cost. Supported force properties
remain subject to the method's ordinary native capability and resource contract.

(spin-root-values)=
## Spin root values

Coordinates are Bohr, energies/curvatures Hartree, forces Hartree/Bohr, rotations
radians. The force is minus the nuclear gradient. `SpinRootPolicy` defaults bound
AO count (32), complete real dimension (128), response plus nonlinear model
actions (4096), signed rotations (six SCF attempts), and controller numeric memory
(256 MiB). Numeric memory covers references/response/rotation buffers; native
SCF/J-K/correlation arenas retain their existing resource contracts. These are work
and admission limits, not endpoint performance guarantees. Tolerances and policy
identity are retained with the receipt.

`SpinRootResult.reference` contains immutable two-spin AO projectors, canonical
orbitals, 1/0 occupations, physical Fock, overlap, orbital/reference energies,
residual and deterministic model/source identity. The receipt reports the initial
and final stability status, full eigenvalue spectrum, determinant `<S^2>`, alpha
and beta AO Mulliken populations, atom sums, root relation, native SCF attempts,
seed consumption, response/nonlinear work, projection receipts and synchronous
endpoint time including cleanup. Population values depend on the AO basis.

Geometry continuation projects the previous occupied spaces with native metric
projection and the cross-geometry AO overlap, then independently reconverges and
requalifies. Root identity uses the Hilbert-space occupied-projector distance,
which is invariant under occupied gauge and nonsingular AO basis changes. Global
alpha/beta inversion is equivalent only for equal spin counts. Equivalent
restart roots are collapsed only when their energies also agree. Arbitrary spatial
symmetries are not inferred: distinct projector classes remain distinct. Ordered
nuclei, basis/model, charge and multiplicity must agree across the sequence.

`tracking_tolerance` gates the maximum per-spin projector distance to the previous
root; its default is 0.25. It measures the actual geometry step as well as electronic
change, so large steps can require smaller geometry increments. A root switch
raises unless `allow_root_switch=True`, in which case the receipt says `switched`
and records the distance. This is a deterministic proximity policy, not a proof
that a distant state has the same chemical character.

(spin-root-errors)=
## Spin root errors

Bad policy/input types raise `TypeError`/`ValueError`. `SpinRootUnqualified` blocks
publication for failed admission, budgets, stationarity, instability, unresolved
curvature, model/lifetime mismatch or intended-root tracking. Existing native SCF,
capability and resource errors may propagate. Missing SciPy requests installation
of the stability extra. No failed call returns a partial energy or force result.
A finite SCF restart failure may try the next signed seed; scientific or ownership
qualification failures abort. A changed geometry/model always needs a new call.

(spin-root-ownership)=
## Spin root ownership

The synchronous call owns and closes all native batches, snapshots and Fock plans,
including failed and discarded attempts. Each restart has an independent native
owner so it cannot invalidate retained XC/state leases. Receipt numeric arrays
are backed by immutable bytes; diagnostic access returns a detached copy. The
nested ordinary `Result` retains its standard metadata ownership contract.
The retained numeric reservation includes the KS snapshot wire and its independent
grid copies, including closed models whose Python buffers remain live.
A previous receipt is detached evidence and a seed, never an executable provider
or a certificate for the next geometry. Do not mutate a Calculator concurrently
with this workflow.

## Independent qualification

`tests/python/test_spin_root_control.py` uses an analytic Hubbard dimer and failure
injection. `tests/python/test_spin_root_native.py` runs native UHF/LDA-UKS/PBE-UKS
stable and broken H2, open-shell Li, geometry tracking, equivalent spin roots,
post-HF references and publication failures. PySCF 2.14.0 independently supplies
libcint/Libxc, reconverged energies, determinant spin/populations, full response
spectra, multi-step finite orbital curvature and analytic moving-grid gradients.
Reconverged geometry differences verify published forces. The oracle is forbidden
during public production calls. Acceptance gates are explicit in the tests;
small fixed-grid equality is separate from production quadrature convergence.
