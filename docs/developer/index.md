# Developer Guide

The Developer Guide covers the implementation: scientific methods, compiler IR,
integrals, derivatives and CPU/CUDA execution. Start with the build guide and
architecture, or go directly to a subsystem below.

## Getting oriented

1. [Build and CUDA configuration](build.md) — establish a development environment.
2. [Architecture](architecture.md) — locate public method and runtime ownership.
3. [Scientific compiler architecture](compiler_architecture.md) — generated programs and kernels.
4. [Electronic-structure boundaries](electronic_structure_boundaries.md) and [provider selection](provider_selection_boundaries.md) — method and backend responsibilities.

## Find a subsystem

- **[Compiler and IR](topics/compiler-ir.md)** — Scientific definitions, code generation, and lowering.
- **[Integrals and SCF/HF](topics/integrals-scf.md)** — Integral generation, Fock assembly, and SCF ownership.
- **[DFT and XC](topics/dft-xc.md)** — Functional evaluation, grids, and derivative terms.
- **[Density fitting](topics/density-fitting.md)** — DF sources, residency, replay, and response.
- **[Post-HF methods](topics/post-hf.md)** — MP2, RCCSD, triples, Lambda, and gradients.
- **[Derivatives and response](topics/derivatives-response.md)** — Stationary problems, response, and Hessians.
- **[Execution and backends](topics/execution-backends.md)** — CPU/CUDA plans, tensor operations, and precision.

For third-party-facing extension points, read [Extending GenerativeQC](extending/index.md). A stable external extension API is not yet guaranteed.

```{toctree}
:maxdepth: 1
:caption: Foundations
:hidden:

build
architecture
compiler_architecture
electronic_structure_boundaries
provider_selection_boundaries
```

```{toctree}
:maxdepth: 2
:caption: Subsystems
:hidden:

topics/compiler-ir
topics/integrals-scf
topics/dft-xc
topics/density-fitting
topics/post-hf
topics/derivatives-response
topics/execution-backends
```
