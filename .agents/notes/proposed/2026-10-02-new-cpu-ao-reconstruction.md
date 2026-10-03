# Experiment: newly reconstructed CPU AO shape specialization

Status: proposed; focused numerical gates passed; endpoint qualification pending
Date: 2026-10-02

## Identity and scope

This is a NEW candidate against `9a5871dca1f371e5f6e9fcf9d6b076c83b894c14`.
It was written from the surviving rationale for the missing CPU AO experiment,
not recovered from its source bytes. It does not reproduce or inherit the old
candidate's source hashes, native qualification, endpoint reports, or timings.

New production `src/dft/ao_grid.cpp` SHA-256:
`0f039959e02fd420e2ba62f489f747824c607b0dd956782b99469be36c1b8279`.
The frozen comparator's source/header bytes are exact copies from the stated
baseline and are hashed by the regression test. They are reference fixtures,
not production fallback implementations and not the missing candidate.

## Problem and decision

The baseline already shares Gaussian radial factors and one-dimensional axis
jets. Nonzero axis polynomial derivatives still use runtime coefficient and
Horner loops. One C++ template now owns the unchanged recurrence for twelve
fixed shapes (`l=0..3`, `derivative=1..3`) and the original runtime fallback.
The derivative-zero direct Horner path executes before the bounded dispatch.
There are no basis-name, functional, system-size, grid-size, resource-budget,
public-option, or backend-selection heuristics.

No AO constructor, public header, packed layout, evaluator traversal, radial
factor, finite check, selection rule, or output index changes. No persistent
storage, preparation, host/device transfer, allocation, or new ownership is
introduced. The original public through-f/order-three limits remain in force.
Internal fallback tests stay within the original `l+derivative<=6` buffer domain;
this is not a new public promise for larger angular momentum or derivative order.

## Scientific invariants and risks

- Preserve coefficient initialization, increasing-k updates, derivative-step
  order, and descending Horner operations, including additions of zero
- Do not remove zero coefficients: finite `alpha` can overflow in `2*alpha`,
  and the subsequent `Inf*0` has observable NaN behavior
- Keep polynomial evaluation after the exact radial-zero guard. Far-field
  exponential underflow and zero contraction coefficients must not be turned
  into polynomial-overflow failures. Selected-out AOs must stay unevaluated
- Keep each jet's x/y/z product order and primitive/expansion summation order
- Template specialization is an optimization opportunity, not an assertion of
  bitwise behavior under every compiler/ISA/FP-contraction policy. Rerun the
  comparator under the actual flags of each promoted build

The tests compile real production construction and basis expansion tables;
old/new whole-AO comparison shares those tables and is not an independent
normalization or spherical-convention oracle. The independent long-double
Leibniz gate checks moderate finite arithmetic. Pinned libcint fixtures and
complete DFT force/response endpoints remain required separate gates.

## Evidence

The new test `test_cpu_ao_polynomial_specialization.py` checks every one of the
28 safe internal axis shapes, finite extremes, signed zero, subnormals, fixed
and runtime paths, deterministic random finite bit patterns, Cartesian and
real-spherical s/p/d/f jets, full/partial/sparse/empty selections, empty points,
output canaries, unchanged inputs, nonfinite failures, and exception messages.
Finite values, infinities and signed zeros compare bitwise; NaNs compare by
classification, without requiring their payloads. Runtime checks survive
`-DNDEBUG`.

Small standalone qualification on GCC 14.2.0 uses ccache 4.14.1 with one source
per cached compilation and no native library build:

- Production-like `-O2 -DNDEBUG`
- Current CMake Release arithmetic/codegen flags `-O3 -DNDEBUG -fPIC`
- Existing independent-oracle flags `-O3 -ffp-contract=off`

All 158 focused tests passed (133 new comparator/oracle tests plus 20 existing
independent AO-jet tests and five radial/axis-reuse guards). The radial
source-structure test now delimits the
zero-derivative branch by its own return instead of a later coefficient array,
so the guard remains meaningful after moving nonzero storage into a helper.

Matched standalone O2 object inspection (`-g0 -fstack-usage` additionally):
`.text` grows from 7026 to 9586 bytes (+2560); the nonzero helper grows from
266 to 2782 bytes; the evaluator hot symbol grows from 5388 to 5438 bytes.
Both helper stack reports are 24 bytes and both evaluator reports 832 bytes;
x86 red-zone storage is separate from the small explicit helper frame.
The helper remains out-of-line and higher-order/runtime paths retain some
bounded loops and stack traffic. This is compiled-code evidence only.

Current Release O3/PIC standalone inspection (`-fstack-usage` additionally):
`.text` grows from 11310 to 13077 bytes (+1767), with identical 986-byte cold
sections. The candidate has a 386-byte runtime helper and a 1766-byte fixed
shape dispatcher; the baseline's nonzero helper is 386 bytes. The evaluator hot
symbol is 9406 versus 9397 bytes. Its compiler-reported stack grows from 1184
to 1248 bytes (+64), a real cost that the unchanged O2 stack report misses.
These are standalone objects with matching arithmetic/codegen flags, not a
claim of exact full-library object identity or endpoint resource accounting.

No native endpoint timing or speedup is established by this note. Cache
statistics from this recovery workspace are shared with concurrent builds and
must not be attributed wholly to this experiment.

## Compiler and cross-method reuse

CPU AO evaluation serves the shared grid/jet layer beneath LDA/GGA/meta-GGA and
hybrid DFT consumers, including stationary derivatives and AO caches. It also
serves an explicitly selected LDA preliminary SCF guess. Conventional HF
integral/J/K evaluation does not consume this AO polynomial helper, so do not
claim a general HF kernel speedup. The retained value-only path is particularly
important for LDA and ordinary value-only callers.

CUDA currently lowers the compiler's `axis_expression` in
`python/generativeqc_compiler/dft/ao_cuda.py` through the existing scalar Graph
and CUDA emitter. Its factored symbolic lowering is not an operation-order
copy of this CPU recurrence. Replacing CPU evaluation with that generated DAG
would be a separate numerical/compiler-ownership migration, not a follow-on
mechanical cleanup. Preserve the independent CPU recurrence/reference until a
shared-lowering change has explicit extreme-value semantics and independent
CPU/GPU acceptance gates. Do not add another handwritten closed-form derivative
or method-specific version. Reuse the existing mathematical Graph/emitter if
such a migration is eventually justified, with native traversal/allocation
remaining native-owned.

## Promotion and revisit gates

Before promoting this new candidate, run pinned libcint/grid tests, AO cache
and resource/failure gates, derivative/response suites, native CTest, and
complete CPU DFT energy/gradient gates. The parent campaign owns these checks.
Then measure interleaved cold/warm/moved-geometry complete endpoints under exact
source/binary identities and matched scientific work. Preserve every sample,
including losses. Include value-only, higher-order through-f, cache-eligible
small GGA, larger streamed GGA, and constrained-memory controls. Use every
repeat's independent numerical errors and unchanged iteration/Fock/grid work.

Reject or narrow fixed dispatch if extra code/branch pressure or endpoint
regressions outweigh gains. Source and object inspection cannot decide that.
Persistent prepared coefficients were not selected: mutable packed state,
identity/lifetime, resource accounting, and unused CPU costs for GPU consumers
are unnecessary risks for this storage-free experiment.

## References

- `src/dft/ao_grid.cpp`, `src/dft/ao_grid.hpp`
- `tests/python/test_cpu_ao_polynomial_specialization.py`
- `tests/reference_data/cpu_ao_reconstruction/`
- `tests/python/test_cpu_ao_jet_numerics.py`
- `tests/python/test_grid_cpu.py`
- `docs/maintainer/performance_engineering.md`
- `python/generativeqc_compiler/dft/ao_cuda.py`
- `src/scf/preliminary_guess.cpp`
