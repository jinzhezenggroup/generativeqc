# Decision: preserve pkg-config OpenBLAS library operand order

Status: implemented
Date: 2026-10-10

## Problem

scipy-openblas32 0.3.34.0.0 emits an absolute shared-library path in
`scipy-openblas.pc`. CMake 3.24.3 and 4.4.4 FindPkgConfig classify that operand
as `INTERFACE_LINK_OPTIONS`, putting it before object files. ELF `--as-needed`
can discard the library before references are encountered; static archives have
the analogous extraction problem. The same imported target feeds capability
checks and the production `generativeqc` shared-library target.

This predates #2176 and is independent of its capability revalidation work.
With exact master `82d44319d13d109349c1122c3993c18ab95edae1`, the real provider
reports all three capabilities false through pkg-config, although global thread
control and LAPACKE symbols exist. The production CPU-linalg owner also fails
strict shared-library linking with unresolved `scipy_cblas_*` references.

## Decision

Normalize affected imported targets before either consumer sees them. Move
absolute library operands from `INTERFACE_LINK_OPTIONS` into
`INTERFACE_LINK_LIBRARIES`, preserving their original order relative to the
resolved `-l` operands. Keep include directories, compiler options and genuine
non-positional linker options on the original target. Ordinary `-l`-only
metadata and CMake-package discovery retain their existing behavior.

This is deliberately a bounded normalization, not a general linker parser.
Supported affected metadata contains recognized absolute library suffixes,
`-l`/`-L` operands, `-pthread`, and a single-argument `-Wl,-rpath,...` or
`-Wl,-rpath=...` option. Inputs must already be tokenized correctly by
FindPkgConfig. Positional or unknown forms, including group, whole-archive,
state-stack and as-needed toggles, do not get rearranged. Reject that pkg-config
candidate and try independent CMake metadata. Without a usable provider, the
existing explicit-provider error or automatic scalar fallback applies.

## Rejected alternatives and invariants

- Appending absolute paths after all resolved libraries loses mixed static
  archive ordering
- Copying them into both properties leaves the broken early link operand and
  can accidentally change archive extraction
- Adding `--no-as-needed`, accepting all probes, or changing only the probes
  conceals the broken production link contract
- Moving positional linker flags into/out of scope is unsafe; unknown forms
  must not produce a silently changed provider contract
- Actual compile/link checks still decide each capability. Missing APIs and
  deliberately wrong static-library orders remain unavailable
- No kernel, numerical tolerance, dispatch policy or performance claim changes

## Evidence

The integration suite exercises both native and SciPy symbol prefixes, shared
and static absolute libraries, ordinary named libraries, both mixed
absolute/named archive orders, real missing symbols, incorrect archive order,
provider compile options, runtime loading, reconfiguration, positional-flag
refusal, and independent CMake/scalar fallbacks.

The real scipy-openblas32 check builds the unchanged production
`src/tensor/cpu_linalg.cpp` as a shared library through the same PRIVATE provider
edge, then links and runs the existing `tests/native/test_cpu_linalg.cpp`.
Available capabilities become local=false, global=true, LAPACKE=true. The
existing test exercises explicit provider-parallel OpenBLAS BLAS/LAPACK calls;
automatic task-parallel selection correctly remains scalar without local thread
control. This
qualifies native provider linking and that existing test, not the full native
library, full scientific suite, GPU execution or an endpoint speedup.

The full top-level CMake configure also confirms the actual `generativeqc`
link command moves the real provider from before all 217 object operands to
after them. Full-library compilation was not needed for this link-order proof.

Tests pass with GCC 14.2 and verified ccache 4.14.1, using CMake 3.24.3/Unix
Makefiles and 4.4.4/Unix Makefiles/Ninja. The shared compiler cache is retained.

## Revisit when

Extend supported metadata only with order-sensitive negative tests and real
provider evidence. Remove the workaround only after the minimum supported
FindPkgConfig version correctly classifies these operands itself.
