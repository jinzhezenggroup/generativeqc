# Decision: native AOT generators import canonical compiler packages

Status: implemented
Date: 2026-10-01

## Problem

Thirteen native AOT entry points for shared HF/KS SCF, COSX/nonlocal DFT, RCCSD(T), and GFN2
installed synthetic `generativeqc_compiler.tensor`/`method` packages to avoid
NumPy-backed reference imports. The TensorIR and method package initializers now
support dependency-light compiler imports, so these per-family copies of package
setup were obsolete. In particular, the RCCSD generator unconditionally replaced
an already imported TensorIR package and its public reference APIs with stub
`execute` and `PackedLayout` implementations. Generation could succeed while
later imports in the same process observed the wrong module object/API.

## Decision

Remove those synthetic packages and use ordinary imports of the existing
canonical compiler implementation. Defer `PackedLayout` and `execute` imports
in the CC equation/reference modules to the four existing functions that
actually use those NumPy-backed services. Mathematical builders, AD, validation,
code emission, schedules, native execution, and independent references are
unchanged. The reference functions continue to use the canonical implementations.

This is a compiler-entry-point ownership slice of #926/#934, not a migration of
method equations or completion of either tracker. HF/DFT share the SCF entry
points; COSX is an additional DFT consumer; RCCSD includes the existing triples
and response generation; nine GFN2 generators consume the same canonical
TensorIR/method packages.

## Invariants and evidence

- Standalone generation must still run with `python -S` from an uninstalled
  checkout and a working directory outside the repository
- Importing a generator must preserve preloaded canonical module identity and
  public exports; importing the canonical package after generation must also
  work, including reference execution and packing
- Generation must not import NumPy, the public/native runtime, PySCF, Torch, or
  CuPy. The new subprocess tests reject those imports explicitly
- Against baseline `6ccecb1b41d604018d738f814a6cb44d2e191996`, all sixteen
  generated CPU/CUDA artifacts (3,364,332 bytes) are byte-identical. This includes
  SCF CPU/density CUDA, COSX/nonlocal pair, RCCSD CPU/CUDA including triples/response, GFN2 AES2
  CPU/CUDA, spin, SCC free energy, H0, ES2, ES3, pair, and electronic CPU/CUDA
- Those exact source comparisons also preserve the embedded mathematical hashes,
  function signatures, source-level diagnostics and failure checks. Existing CC
  reference tests retain independent numerical and invalid-input coverage
- `test_native_generator_imports.py` checks all fourteen generators both before and
  after canonical imports in isolated standard-library-only subprocesses, plus
  sequential cross-family imports with eager/lazy reference exports

Reproduce the artifact comparison by checking out the baseline and candidate in
separate directories and invoking each changed generator with `python -S` and
identical arguments. Ordinary generators use `--output`; AES2 uses
`--cpu-output`/`--cuda-output`; RCCSD uses `--cpu-header`/`--cuda-source`. Compare
complete emitted file bytes, not only embedded equation hashes.

## Consequences and boundaries

There is no native endpoint performance claim or required endpoint timing
campaign for this slice: generated scientific source and schedules are unchanged.
Canonical imports do load additional lightweight compiler modules; cold Python
import/source-generation cost can therefore change. No source-generation
speedup or nonregression bound is claimed. Repository source identities and
build-identity metadata correctly change with the edited source files; they are
not expected to match the old checkout. No artifact-cache identity is overridden.

The GFN2 pair generator retains only its geometry-package bootstrap; its
TensorIR bootstrap is also retired so it composes safely with the other
generators in the same process. The XC CPU
generator still has a broader bootstrap. These package initializers have eager
numerical dependencies, so removing those workarounds requires a separately
reviewed namespace/dependency change. Do not copy them into new generators.

## Rejected alternatives

A new shared helper that manufactures fake packages would preserve the module
identity defect and introduce another import framework. Globally lazifying
geometry/DFT/XC merely to remove the remaining workarounds is unnecessary scope
for this slice. Changing the IR, emitters or runtime to make the families look
more alike would require different semantic/performance qualification and is
not part of this decision.

## References

- https://github.com/jinzhezenggroup/generativeqc/issues/926
- https://github.com/jinzhezenggroup/generativeqc/issues/934
- `tests/python/test_native_generator_imports.py`
- `tests/python/test_rccsd_codegen_no_numpy.py`
- `tests/python/test_rccsd_generator_evidence.py`
- `tests/python/test_tensor_lazy_imports.py`
