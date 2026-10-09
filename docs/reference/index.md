# Reference

Reference is for exact lookup rather than teaching.

- [Python API](api.md) — recursively generated from modules that explicitly declare `__all__`.
- [Python contract policy](python_contracts.md) — applicable contract facets and authoritative owners.
- [Python execution contracts](python_execution_contracts.md) — calculation, batch, result and Torch semantics.
- [Native C/C++ API](native_api.md) — ABI versioning, ownership, error handling and wrapper lifetimes.
- [Native symbol contracts](native_symbols.md) — source-owned C functions and C++ types and operations.
- [Public method table](../public_methods.md) — generated canonical method identities and declared capabilities.
- [Capability sources](capabilities.md)
- [Units](units.md)
- [COSX reference](cosx_reference.md)
- [D4 reference](d4_reference.md)
- [GFN1 parameters](gfn1_parameters.md)

Machine-generated `codegen_capabilities.json` and `cuda_ownership/` remain at the docs root because repository tooling currently consumes those paths.

```{toctree}
:hidden:
:maxdepth: 1

api
python_contracts
python_execution_contracts
native_api
native_symbols
../public_methods
capabilities
units
cosx_reference
d4_reference
gfn1_parameters
```
