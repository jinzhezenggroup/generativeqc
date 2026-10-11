# Decision: native CLI owns non-compiling runtime workflows

Status: implemented
Date: 2026-10-11

## Problem

The native SDK could run only selected method families and left resource
planning and validated profile bundles to Python. Canonical correlated methods
also inherited the HF screening default, which their native owners reject.
Target-side Python basis/compiler machinery was not an appropriate runtime
dependency for an installed C++ executable.

## Decision

- Resolve every registered method through the native owner, with unscreened
  canonical correlated references and direct descriptor controls.
- Verify canonical local basis/provenance/decimal records in C++, expanding
  contractions and mapping scalar ECP data to the existing native constructor.
- Perform allocation dry runs in C++: metadata only, checked byte arithmetic,
  exact reserve/headroom rounding, real native layout/tile queries and explicit
  provider admission. ECP bounds consume the generated execution policy through
  a transactional scalar query, rather than duplicating quadrature constants.
- Validate/hash/export/import bundles natively with OpenSSL, nlohmann-json and
  libarchive. Keep the existing canonical Python hash and four-file ZIP contract;
  never relax numerical, source, device, toolchain or endpoint evidence gates.
- Keep immutable bundles and atomic locked index updates. CUDA kernel libraries
  can be selected on Linux by native re-execution, preserving Slurm visibility.
  CPU never selects local profiles. DFT schedule-only bundles are validated but
  must not bypass workload-specific KS admission by merely loading a binary.
- Installed Python console scripts prefer their packaged native executable.
  Source-only compiler interfaces retain compatibility. Autotuning that emits
  and compiles new source remains explicitly build-time Python work; the native
  binary never discovers or launches an interpreter.

## Compatibility and validation

The native resource JSON declares its own allocation-inventory scope and is
not a Python ProgramIR/ResourcePlan checkpoint. Unsupported domains must remain
unsupported instead of receiving an optimistic estimate. Opaque CUDA provider
allowances and detached cold retries remain charged in DF alternatives.

Host tests compare canonical floats/Unicode/SHA256 with Python, DF capacity
alternatives with the independent Python inventory, and correlated energies
with existing numerical reference values. Profile install/select/re-execution
tests use explicitly synthetic host-only identity stubs and private temporary
caches; they are not GPU numerical or promotion evidence. Real-device
qualification still requires an available supported Slurm allocation.

Automatic Libxc selectors and aliases are projected from the public bulk resolver
at build time, including its structural and blacklist gates. Runtime selection
uses single-program native CPU KS composition; it does not admit mixed bulk
graphs, invent CUDA lowerers, or consult Python qualification data on startup.
