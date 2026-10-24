# Decision: separate bounded polynomial point execution from built-in XC admission

Status: implemented
Date: 2026-10-20

## Problem

Issue #2330's v2 declaration has exactly two nonnegative FP64 density inputs,
including vacuum and zero minority spin. Built-in `XCProgram.validate_features`
has a separate three/seven-feature interior-v1 contract and rejects those
derivative boundaries. `NativeContractionProgram` also owns AO consumers that
are absent from a standalone scalar polynomial endpoint. Neither is an honest
wrapper for this slice.

## Decision

Add an immutable `UserExpressionProgram` recipe and an explicit native CPU
point/batch adapter. Reuse `output_set`, canonical Graph differentiation,
associative canonicalization, integer-power lowering, `ScalarCEmitter`, the
existing source publication helper, `CppCompilerAdapter` and `compile_runtime`.
Keep both declared inputs in the native ABI and validate them independently of
root reachability. There is no alternate parser, derivative algebra, feature
padding, catalog entry, `FunctionalSpec`, SCF callback or reference fallback.

The program retains the sealed expression and order, rather than a public mutable
Graph handle. Graph export reconstructs detached derivatives; audit exports and
program identity bind the declaration identity, ordered roots and graph. The v2
declaration's copy/replay/identity contract is unchanged. Native artifact identity
additionally binds emitted source and toolchain through the existing cache.
Compilation uses explicit contraction-off FP64 flags. Exact rational coefficients
remain exact through symbolic differentiation and round only at scalar emission;
this does not qualify generic Graph AD or large-rational ordering.

Calls copy validated input before entering native code. The native ABI checks
counts, pointers, all declared input values and output finiteness. It writes only
private scratch; the public adapter raises on failure and publishes detached
immutable bytes only after complete success. No caller output/donation API is
provided. Thus a late overflow cannot expose a partial result or modify an older
result. A raw private ABI caller must discard output on nonzero status.

## Bounds

Declaration budgets remain v2. Program graphs are limited to 8,192 nodes,
derivative exact coefficients to 512 bits, native source to 262,144 bytes and
batches to 65,536 points. Orders 0/1/2 have 1/3/6 output rows. A maximum order-two
call owns 1 MiB input, 3 MiB scratch and up to 3 MiB immutable publication bytes;
compiler/process and Python/graph metadata are separate. Budget failure is an
explicit refusal, not a silent change of backend or output contract.

## Rejected alternatives

- Padding rho with sigma/tau or loosening interior-v1 would change a built-in
  scientific contract without the corresponding method qualification.
- Reusing ContractionProgram would invent AO/observable ownership for an endpoint
  that consumes only two scalar features.
- Exposing mutable derivative graphs would let exported handles alter a program
  after its identity or compiled artifact was established.
- Calling a polynomial a named Libxc functional or treating compilation as
  physical calibration/SCF acceptance would grant unsupported entitlement.

## Evidence and remaining scope

### Pre-merge ownership correction

Review of the initial standalone adapter identified that its compiler-owned
class both compiled and executed the native ABI. Existing `xc/native.py` has
similar Python FFI precedent, but that precedent does not override the scoped
compiler/runtime ownership rule. This slice does not migrate that existing
adapter or claim to repair its broader ownership debt.

The corrected design keeps canonical graphs, scalar emission and artifact-only
compilation in the compiler. `UserExpressionProgram.compile` never imports the
public runtime, loads a DSO or allocates point buffers. The public-runtime owner
`generativeqc/_xc_user_expression.py` owns NumPy validation/snapshots, CDLL and
function lifetime, scratch, calls and immutable result publication. The public
`xc.compile_expression` factory connects this runtime to the compiler recipe.
The two factories and actual program/runtime classes are explicitly exported
and covered by the existing public contract manifest rather than bypassing it.

The fixed count/pointer/input-domain checks, batch loop and result-finiteness
boundary belong to the new `src/dft/xc_user_expression_point.hpp` template. It
contains no XC mathematics or AD; `ScalarCEmitter` supplies only the scalar
lambda's roots. The template is a declared wheel asset. Compilation stages a
checked copy beside the source under its stable basename and passes it through
`compile_runtime(headers=...)`, so checkout/installed resolution has equal
logical source/header identities without absolute asset paths. The native
schema deliberately advances to v2; prior standalone generated artifacts do
not qualify the changed runtime source. Missing installed assets, staged-header
tampering and stale artifact metadata fail closed. No shared cache, existing XC
domain, catalogue, CMake, Graph or old native adapter changes are required.

Moving only the Python class would leave the fixed execution template in the
compiler; keeping `program.compile` as a lazy reverse import of the public
runtime would violate the dependency boundary. Both alternatives were rejected.
Fresh source-matched point/batch execution is required after this correction;
the 121-pass receipt below describes the earlier baseline, not the new template.

The corrected source completed **125 passed / zero skipped** twice on the same
slice-owned Linux CPU. All 30 program controls remain active, including 14 native
cases. New executable controls prohibit public-runtime imports, CDLL loading and
native-output allocation during generation, prohibit DSO loading during compiler
artifact preparation, and compare actual checkout/declared-installed-asset
compilation and point execution. Installed resolution has no checkout fallback;
missing assets, modified staged headers and stale artifact metadata are refused.
All 11 measured source/test/header/config/doc SHA256 values matched the staged LF
blobs. Each run recorded 16 compiler and 16 link driver calls through sccache:
the first had 16 object misses, the replay 16 object hits, with zero compilation
failures. The separate corrected-source receipt bundle is
`24b71a9aeb0872b73aafac0c097ffaa530c5a5f8b180b600db6a582c62f35263`
in ignored `build/2627-ownership-native-receipts.tar`. The earlier receipts are
retained independently and do not transfer qualification to the changed source.
All four documentation helper suites also passed on the corrected source in a
separate docs environment: 50 public-contract, 15 Python API docs, 4 Sphinx link
and 23 native-contract tests, with zero skips. Full public Python/Ruff and native
contract checkers passed; compiler structure remained 570 modules / zero errors.
Preparing these docs tests first exposed a missing README in the minimal transfer;
that environment failure was retained and repaired using same-base repository
assets, not production edits. The docs dependency-download command timed out while
its original pip PID continued; the existing process was monitored to completion
without a duplicate install or changes to the scientific qualification environment.

`tests/python/test_xc_user_expression_program.py` exercises complete native CPU
point and batch calls against independent 70-digit Decimal energies and hand
gradients/Hessians, finite differences and spin/per-particle negative controls.
Controls include vacuum/minority spin, constant expressions with both inputs
validated in Python and the native ABI, near-one exact rational derivatives,
overflow without partial publication, immutable result lifetime, strict input
dtype/shape, graph/source/point budgets and artifact-cache replay.

The PR records exact-source commands and native execution results; generation or
skipped native tests alone are not native qualification. These polynomial tests
do not establish physical XC calibration, general AD admission, self-consistent
SCF, nuclear forces, response or CUDA. The rest of #2330 B and C--G remains open.

Source-matched qualification used base `70b51496eed2866b41ec8d8f623ab92102539563`
plus this slice's LF index patch, Python 3.11.16, GCC 11.4.0 and sccache 0.16.0 on
the slice-owned qz CPU instance (2 CPU / 8 GiB, effective priority 6 / HIGH).
`test_xc_user_expression_program.py`, `test_xc_user_expression.py` and
`test_extensions.py` completed **121 passed / zero skipped**. All six measured
Python source/test SHA256 values matched the staged LF blobs before commit.
The one-off harness recorded actual launcher-prefixed commands and statistics
before/after each private-server invocation: the initial run had 11 object cache
misses and 11 link/non-compilation calls, then the final run had 11 object hits
and 11 link calls, with no compilation failures. Initial regression checks had
four missing-asset failures because the minimal transfer omitted pinned
`upstream/libxc/7.0.0`; adding that same-base reference tree fixed the environment
without changing production or test sources. Earlier patch preparation also
failed when Git found a parent shared-directory repository; an isolated source
Git root prevented that implicit parent ownership. Both negative runs remain
in the retained receipt bundle, SHA256
`fa4bde7ce71a2e02f0d4fc6e35541e877792846cfe89862281d8c7955ff4ec03`.

The local Windows checks were 145 passed / 11 native prerequisite skips for the
focused expression/extension suites and 113 passed / 4 prerequisite skips for
existing expression/boundary/feature-mask regression controls. Compiler structure
checked 570 modules with zero dependency errors; Ruff lint/format and diff checks
passed. These are separate local and Linux endpoint receipts, not SCF or device
acceptance. Raw receipts remain in the task's ignored `build/2330-receipts/` and
its own remote `issue2330-native-20261020` directory; they are not release assets.
