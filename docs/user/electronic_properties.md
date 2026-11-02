# Dipoles, atomic charges and electron density

`Calculator.singlepoint` can analyze the **same converged mean-field state**
used for its energy. Properties are opt-in and do not change the default output
or run a second SCF calculation.

```python
from generativeqc import Calculator

atoms = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]
calc = Calculator(method="rhf", basis="def2-svp", device="cuda")
result = calc.singlepoint(
    atoms, properties=("energy", "dipole", "charges", "density")
)

print(result.dipole)          # xyz, e*Bohr
print(result.dipole_debye)    # xyz, Debye
print(result.charges)         # Mulliken atomic partial charges, elementary charge
print(result.lowdin_charges)  # Lowdin atomic partial charges, elementary charge
print(result.density.matrix)  # spin-summed AO density, shape (nao, nao)
print(result.density.device)  # "cuda"; sampling follows this device
```

Add `"forces"` to the property tuple when analytic forces are also needed.
Unrequested electronic fields are `None`. `"energy"` remains mandatory.
These output selectors are currently Python single-point features, not native
CLI or prepared-batch output selectors. `calc.supported_electronic_properties`
lists them separately; the method registry and `capabilities` records retain
their native energy/force contract.

(python-electronic-values)=
## Scientific conventions

- Dipoles include nuclei and electrons:
  `sum_A Z_A R_A - Tr(D r)`. Their reference point is the coordinate origin.
  Translating all atoms by `a` changes a charged system's dipole by `Q*a`;
  neutral-system dipoles are translation invariant.
- `charges` means **atomic partial charges**, not the integer molecular charge
  passed to `singlepoint(charge=...)`. Mulliken populations use `diag(D S)`;
  Lowdin populations use `diag(S**0.5 D S**0.5)`, with the symmetric positive
  overlap square root. Both partitions sum to the molecular charge, up to
  numerical roundoff. A nonpositive overlap spectrum is rejected rather than
  silently truncating populations.
- Restricted densities include double occupation. Unrestricted `matrix` is
  alpha plus beta and `spin_matrix` is alpha minus beta. Restricted
  `spin_matrix` is `None`, and spin sampling returns zero. The AO conventions
  match the calculation's normalized Cartesian or real-spherical basis.
- These are HF/KS densities, not relaxed MP2/CC densities, Hirshfeld/ESP charges
  or a promise of numerical equivalence to every ORCA population scheme.

## Real-space sampling and Cube files

```python
rho = result.density.evaluate([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
spin_rho = result.density.evaluate([[0.0, 0.0, 1.0]], spin=True)

result.density.write_cube(
    "density.cube",
    origin=(-4.0, -5.0, -4.0),  # Bohr
    shape=(41, 51, 51),
    spacing=0.2,                # Bohr; scalar or xyz triple
)
```

Sampling returns electrons/Bohr³. Cube axes are axis-aligned and expressed in
Bohr, with the z index fastest; the header contains the molecular nuclei.
`spin=True` exports alpha-minus-beta spin density. Cube writing replaces an
existing path and streams blocks without allocating the entire grid or an
AO-by-AO-by-grid tensor. `block_size` controls the point block size; native AO
evaluation additionally enforces its own finite workspace budget. An I/O or
native failure can leave a partial file. Invalid grid parameters are rejected
before opening the file.

(python-electronic-errors)=
## Scope and failures

The current domain is all-electron RHF/UHF and RKS/UKS, including density-fitted
calculations, on CPU and CUDA. CUDA properties and real-space density sampling
support s/p/d/f bases. ECPs,
correlated methods, xTB and explicit resource-budget analysis are rejected
instead of substituting reference/oracle work or reporting an unrelated density.
A nonconverged solve never returns electronic properties.

Sampling rejects complex, nonfinite or incorrectly shaped points and
nonpositive/noninteger block sizes. Cube export rejects invalid integer counts,
nonfinite origins and nonpositive/nonfinite spacing before opening the path.
Native allocation/analysis errors and filesystem errors propagate; no failed
single-point call returns a partially valid result.

(python-electronic-backends)=
## Backend

SCF, electronic analysis and density sampling follow the selected backend:

- `device="cuda"`: native FP64 kernels evaluate analytic overlap/position
  integrals and dipole/Mulliken contractions; cuSOLVER and cuBLAS compute the
  symmetric overlap square root and Lowdin populations. Point sampling reuses
  generated CUDA AO evaluation and density contractions. There is **no CPU
  numerical fallback**, including when writing Cube values.
- `device="cpu"`: native FP64 Gaussian integrals/AOs and NumPy population and
  density contractions. Select CPU explicitly for host execution.

The current property endpoint detaches the accepted AO density to host and
uploads that snapshot for CUDA analysis/sampling. This is an explicit transfer,
not a second SCF or a resident zero-copy performance claim. Basis normalization,
input validation, point-grid construction, result download and Cube file I/O
are host work on both backends. PySCF is only an independent test oracle, never
a production dependency.

(python-electronic-ownership)=
## Lifetime

The returned density owns read-only AO matrix copies and immutable geometry and
basis data, not borrowed device pointers. It remains usable after the calculator
cache and native calculation are destroyed. Each sampling/Cube call prepares
its own native CPU AO or detached CUDA owner. CUDA retains the selected
`device_id` within the scheduler's visible devices, and calls restore the
caller's current CUDA device. Continue to run CUDA sampling under a valid GPU
allocation; detachment does not lease a GPU after the scheduler releases it.
Result arrays follow the usual
{ref}`result ownership contract <python-results-ownership>`.
