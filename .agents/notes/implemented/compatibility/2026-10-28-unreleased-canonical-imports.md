# Decision: remove unreleased compatibility entry points

Status: implemented
Date: 2026-10-28

## Problem

The project has not released an API, but package and ownership migrations kept
alternate imports, renamed callables and tests whose only purpose was to preserve
those old names. Several build/provenance inventories also still named forwarding
files rather than the implementation responsible for generated output.

## Decision

Follow the maintainer's explicit pre-release policy: migrate repository callers
and remove obsolete interfaces in the same change. No external migration period
is required before the first release.

- CC and MP2 equation consumers import `generativeqc_compiler.cc` and `.mp2`.
- Response/Hessian consumers import the installed runtime owners. The tools
  package's active API resolves its lazy exports directly to those owners.
- Integral production and tuning consumers import their semantic implementation
  modules. CUDA shell emission binds directly to the lowering subsystem.
- DFT owns native semilocal metadata. Range AOT generation consumes the
  method-neutral derivative registry, without the WB97M-V-era entry point.
- Remove renamed one-electron, matrix-function and nuclear-response aliases,
  the unused implicit CUDA target constant, the duplicate bulk-KS resolver, and
  the empty `cuda12` installation extra.
- Current native/header, numerical, resource and scientific identity contracts
  remain validation requirements. Removing obsolete Python entry points does not
  authorize removing a qualified execution route or a bounded resource fallback.

The canonical tuning CLI now runs as
`python -m generativeqc_compiler.integral.tuning.cli`. Its module executes `main`
itself; importing the module remains side-effect free.

## Provenance and tests

Update CMake dependencies and CUDA ownership anchors to actual implementation
paths. Regenerate the Libxc product records using their existing offline
generators; preserve pinned upstream inputs and generated scientific outputs.
Source inventories naturally change and existing binary/source cache checks
remain in force.

Retire tests that only asserted identity between an old name and its owner.
Retain independent numerical oracles, malformed-input checks, solver failure and
resource-lifetime coverage, no-tools runtime execution tests and current public
lazy-export checks. Test the canonical tuning command in a fresh interpreter.

Validation against the resulting checkout:

- Compiler/codegen, CC numerical, response algebra, lifecycle and import tests
  pass. The canonical CLI, CMake depfiles and native semilocal generation checks
  also pass after removal of the forwarding outputs.
- Build the current CPU library with CMake/Ninja, CUDA and optional stationary
  AOT disabled, using the verified `sccache 0.16.0` launcher. Generated Ninja
  compile commands contain that launcher. Run `test_implicit_response.py`,
  `test_hessian_directional.py`, `test_hessian_response.py` and
  `test_rks_hessian_production_owner.py` against it: 64 passed.
- `prek run --files ...`, Sphinx with warnings as errors, public Python contracts,
  compiler/native/CUDA ownership checks and offline source-registry verification
  pass. Libxc regeneration changes provenance records while scientific product
  bytes remain identical.
- Full Python collection uses `PYTHONPATH=python:.:tests/python` and pytest's
  `--import-mode=importlib` to resolve the repository's sibling test helpers in
  this development environment.

## Rejected alternatives

Keeping empty facades for hypothetical downstream callers would perpetuate the
same maintenance cost without a released compatibility obligation. Replacing
facades with copied implementations would split scientific ownership and is
also rejected. Removing cache or scientific validation merely because it uses
the word "compatibility" would discard correctness checks unrelated to old APIs.

## Revisit when

Define a versioned compatibility policy when preparing the first public release.
Do not resurrect old import paths based only on historical notes or benchmark
records; those records describe the source version at which they were produced.

## References

This supersedes the temporary shim-retention policy in
[the compiler package ownership note](../architecture/2026-09-15-compiler-package-ownership.md).
Current ownership is described in
[`docs/developer/compiler_architecture.md`](../../../../docs/developer/compiler_architecture.md).
