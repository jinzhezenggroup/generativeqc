# GenerativeQC documentation

Choose a guide based on what you want to do:

- [Learn quantum chemistry](learn/index.md) — charge, spin, basis sets, SCF, DFT and derivatives.
- [User Guide](user/index.md) — installation, methods and calculations.
- [Reference](reference/index.md) — APIs, units and supported capabilities.
- [Developer Guide](developer/index.md) — algorithms, code generation and backends.
- [Maintainer Guide](maintainer/index.md) — tests, benchmarks, builds and release procedures.
- [Agent Guide](agent/index.md) — repository workflow for coding agents.

## Build the documentation

From the repository root:

```console
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

The generated HTML is in `docs/_build/html/`. The [Python API](reference/api.md)
and [method table](public_methods.md) are generated from the code. Machine-readable
outputs retain their existing paths; see [generated files](maintainer/generated-files.md).

```{toctree}
:hidden:
:maxdepth: 3
:caption: Guides

learn/index
user/index
reference/index
developer/index
maintainer/index
agent/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Generated and root reference

public_methods
libxc_bulk_import
libxc_maple_coverage
cuda_ownership/README
```
