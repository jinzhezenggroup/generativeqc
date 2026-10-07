# CUDA tensor and solver vendor boundaries

Run `python tools/check_vendor_boundaries.py` to check production native sources,
scientific compiler modules and `tools/generate*.py` entry points. The same check
runs in pre-commit. `--json` reports the current file/symbol reference inventory.

[`vendor_boundaries.json`](../../manifests/maintenance/vendor_boundaries.json)
classifies each file explicitly:

- `provider`: a retained primitive implementation with a semantic contract.
- `runtime_abi`: optional library ABI declarations and aliases.
- `infrastructure`: bounded non-scientific scheduling primitives.
- `diagnostic`: provenance or synthetic capability probes.
- `migration`: existing method-local coupling tracked by #1890.

The DF SCF adapter in `src/scf/cuda/df_scf_library.hpp` is retained as a reusable
matrix/eigensolver implementation. Its callers specify operands, strides,
alpha/beta and eigensystem work; its `DeviceSolver` owns numeric workspace and
retains the shared `solver::cuda::PreparedSymmetricEigenHandles` owner. That
move-only service uniquely owns solver/Params/Jacobi lifetime for GFN2, RHF/UHF,
ordinary KS and DF. Staged setup preserves the caller's stream order and Jacobi
settings. The enclosing device/stream owner settles work and explicitly resets
handles before releasing its stream or restoring the caller's device. The shared
service neither allocates numeric workspace nor synchronizes.

Retaining this adapter does not complete canonical binding integration
for its surrounding callers. Other SCF adapters with upstream implementation
selectors remain classified as migration work.

New files and symbols require an explicit classification and rationale. Existing
migration, diagnostic and infrastructure entries have exact reference counts;
adding another call or removing an old call requires a corresponding reviewed
inventory change. Remove retired entries when migrating a consumer. Do not
regenerate the manifest to accept a failing check without reviewing ownership.
Provider and ABI implementations may change the number of references to an
already classified symbol. No directory, including `src/tensor`, is exempt.

The scanner ignores native comments/string literals and Python docstrings. It
checks generator string fragments without executing Python or loading CUDA.
Function-address references, macro aliases and CUB/CUTLASS/CuTe namespace uses
are covered. Dynamically assembled API names, arbitrary external templates and
semantic changes at an existing callsite require review and runtime tests; this
lexical inventory does not prove the executed provider or replace scientific
acceptance. Implementation-selector API migration also remains separate work.
