# Geometry-only reset and the fail-closed capacity audit

## Motivation

PR #1557 adds a bounded nonlocal geometry accumulator and a reset entry that
uploads centers without unused D/W matrices. When integrated with protected
master 6609b6adfe3c1b7783c1c4612703100fcade65e6, the two capacity tests in
`tests/python/test_dft_mp_v1_tools.py` fail at `_source_limits()` with
`stationary CUDA initializer page contract changed`.

## Reviewed delta

The new FFI argtypes registration changes the initializer span. The new
`reset_geometry` method changes the complete `_CudaSources` span. The new
`stationary_geometry_reset` entry changes the complete native header hash.
All three are intentionally bound by strict fingerprints. Refresh precisely
those three constants in `tools/dft_mp_v1/qualify_capacity.py`; do not skip or
relax the comparisons, expand an allowlist, or refresh unrelated contracts.

Removing just the new registration and method reproduces both old Python
span hashes exactly. Removing just the new native entry reproduces the old
native-header hash exactly. The source allocation, owner layout, ordinary
reset, capacity definitions and admission predicates are unchanged. The WB97M-V
caller separately budgets two bounded source owners; this audit refresh does
not extend the frozen semilocal report to a new scientific route.

## Validation

The focused device-free reset suite executes the actual reset method and FFI
assignment with stubs only at the native boundary. It covers clearing page
state, repeat use, unchanged center/handle identity, native failure propagation
and exact pointer/tolerance argument types. A native-source check compares the
new reset body with the ordinary reset after removing exactly the two D/W
uploads and changing the error label. It therefore retains the ordinary
validation, draining and accounting operations without new allocation.

Local minimal-checkout result: 5 passed. The exact fetched `_CudaSources` span
and complete native header were used; the complete header's Git blob identity
was independently checked. The full capacity suite and report are left to
fresh repository CI; this is not a new CUDA numerical or performance run.

## Maintenance boundary

Future source-owner or native-header changes must be reviewed again before
updating their fingerprints. Keep all fail-closed source, budget and provenance
checks. A static capacity report remains preflight, not scientific qualification.

Agent: ChatGPT
Model: GPT-6 Astra Pro
