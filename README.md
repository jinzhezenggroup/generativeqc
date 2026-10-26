<!--
Keep this README concise and user-facing. Implementation details, capability
boundaries, and benchmark analysis belong in docs/ and benchmarks/results/.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/generativeqc-logo-dark.svg">
    <img src="assets/generativeqc-logo.svg" width="640" alt="GenerativeQC — AI-assisted creation and compiler-generated quantum chemistry">
  </picture>
</p>

<h1 align="center">GenerativeQC</h1>

<p align="center">
  <a href="https://codecov.io/gh/jinzhezenggroup/generativeqc"><img src="https://codecov.io/gh/jinzhezenggroup/generativeqc/graph/badge.svg" alt="Codecov"></a>
</p>

<p align="center">
  <strong>Quantum chemistry generated across the stack.</strong><br>
  GPU-native energies and analytic forces, with reusable batched execution.
</p>

GenerativeQC brings **generation from scientific intent to executable kernels**:
coding agents help develop human-reviewed source; declarative method definitions
and compiler IRs produce execution programs and specialized CPU/CUDA kernels.
The underlying quantum chemistry remains explicit and is checked against
independent scientific references—not replaced by a learned surrogate.

## Highlights

- **HF, DFT, and correlated methods.** RHF/UHF; RKS/UKS from LDA and GGA through
  meta-GGA, hybrid, range-separated, and nonlocal functionals (including PBE,
  PBE0, B3LYP, r2SCAN, WB97M-V, and r2SCAN-3c); qualified MP2, RCCSD, and
  RCCSD(T) paths.
- **Analytic derivatives.** Nuclear forces for qualified CPU/CUDA methods,
  with analytic Hessians and Hessian-vector products in selected domains.
- **GPU-native prepared execution.** Direct and density-fitted paths, reusable
  calculation plans, density warm starts, ragged batches, and per-system isolation.
- **Generated methods and kernels.** MethodIR and Libxc-backed discovery,
  compiler-generated execution/derivative graphs, and AOT-specialized CPU/CUDA
  kernels, with bounded runtime generation where supported.
- **More chemistry.** GFN2-xTB energies and forces; Cartesian and spherical
  Gaussian bases, custom basis input, and scalar Gaussian ECPs.
- **One native core, several interfaces.** Python, C, C++, and a Python-free
  native CLI, plus PyTorch and JAX integration.

## Install

Source builds require CMake 3.24+, a C++20 compiler, and Python 3.10+.
CUDA builds also require a supported CUDA toolkit.

```bash
python -m pip install .
```

For CPU-only installations:

```bash
GENERATIVEQC_ENABLE_CUDA=OFF python -m pip install .
```

For CUDA, select NVCC and an architecture matching your GPU (`120` is an
RTX 5090 example):

```bash
CUDACXX=/path/to/cuda/bin/nvcc \
GENERATIVEQC_CUDA_ARCHITECTURES=120 \
python -m pip install .
```

See the [installation guide](docs/user/installation.md) and
[CUDA/native build guide](docs/developer/build.md) for other architectures,
CMake presets, and SDK installation. The installed native CLI runs without
Python.

## Methods

Explore the compiled method registry and MethodIR compositions:

```bash
generativeqc methods
generativeqc methods --compositions --json
```

DFT selectors include `pbe0-rks` and `pbe0-uks`; the Python API also supports
`r2scan-3c`. Available properties depend on the method and execution context:
check `Calculator.capabilities` or see
[method discovery and support](docs/user/methods.md).

## Python quick start

Coordinates are in Bohr, energies in Hartree, and forces in Hartree/Bohr.
Use `device="cpu"` for CPU-only installations.

```python
from generativeqc import Calculator

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
result = calc.singlepoint([
    ("H", (0.0, 0.0, -0.7)),
    ("H", (0.0, 0.0, 0.7)),
])

print(result.energy)
print(result.forces)
```

Prepared batches reuse system topology and converged density state:

```python
systems = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("He", (0.0, 0.0, 0.0))],
]

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
with calc.prepare_batch(systems, warm_start=True) as batch:
    first = batch.execute(strict=True)
    second = batch.execute(strict=True)  # reuses compatible densities

print(first.energies)
```

## Benchmarks

**HF/DFT energy + analytic forces — RTX 5090**

With default settings, GenerativeQC outperforms GPU4PySCF in **64/64** measured warm cases.
For 96-atom Direct ωB97M-V, the measured warm speedup is **4.22×**.

![Direct cold and warm endpoint timings](assets/readme-direct-defaults.svg)

![Density-fitted cold and warm endpoint timings](assets/readme-df-defaults.svg)

Default XC/VV10 grids, native SCF policies and full grid-response forces;
all complete performance points are shown, including accuracy failures.
Cold includes setup; warm is the median of three frozen-density calls.
Independent energy/force accuracy results remain in the detailed tables.
See [measurements and conditions](benchmarks/results/readme-refresh-20261026/README.md)
for accuracy checks and detailed comparisons.

**DF-CCSD(T)/CBS energy + analytic forces — RTX 5090**

![GPU full-DF CCSD(T)/CBS complete endpoint timings](assets/readme-cbs-performance.svg)

MP2 correlation is extrapolated with cc-pVDZ/cc-pVTZ; the
DF-CCSD(T)−DF-MP2 correction uses cc-pVDZ. **DF-HF/RI-JK, analytic response and
gradients are fitted consistently**, with matching JKFIT auxiliaries and all
electrons. The curve covers C1–C7; each point is the median of two fresh complete
energy+force calls. Process/CUDA startup is retained separately. Basis snapshots
are outside timing.

The [measurements, work counts and independent validation](benchmarks/results/cbs-routine-standard-20261025/README.md)
retain every sample, source identity and failed candidate. Validation covers
13 physical-basis populations, full Cartesian forces and relative energies.

Other results are in the [benchmark archive](benchmarks/results/README.md).

## Documentation

Start with the [User Guide](docs/user/index.md) or
[Learn quantum chemistry](docs/learn/index.md).
Explore the [Reference](docs/reference/index.md),
[Developer Guide](docs/developer/index.md),
[Maintainer Guide](docs/maintainer/index.md), and
[Agent Guide](docs/agent/index.md).

## License

GenerativeQC is licensed under [GPL-3.0-or-later](LICENSE).
