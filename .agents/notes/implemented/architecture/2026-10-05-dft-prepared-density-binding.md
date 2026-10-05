# Decision: prepare native DFT density contraction portfolios

Status: implemented
Date: 2026-10-05

## Problem

Native KS projected an admitted precision schedule into a DFT-specific density
precision enum on every iteration. Generated launch glue separately chose the
scalar/tiled algorithm for every grid tile. Neither decision consumed the shared
lowering portfolio of #1889. Replacing this operation with plain GEMM would also
lose its explicit density symmetrization and mixed rounding semantics.

## Decision

The existing DFT scalar DAG owner defines the symmetric density/AO summand. The
compiler projects that operation into one canonical portfolio containing the
existing scalar and qualified tiled implementations at strict FP64 and qualified
FP32-compute/FP64-accumulation precision. Logical casts, fusion, reduction order,
method-owned strict refinement/audit, capture compatibility and zero extra
numerical workspace are explicit. Existing CUDA kernel arithmetic is unchanged.

CudaXcPlan narrows admission using the method's shared PrecisionDirective and
uses the common native selector during preparation. It binds fixed full/tail
launchers for admitted iterations and strict audit. Native KS submits the shared
PrecisionPhase and accounts for the selected binding's arithmetic. Response and
nonlocal feature export stay strict. Replay calls bound function pointers and
performs no portfolio search, heuristic, host allocation or provider preparation.

Local AO maps have an immutable launcher per tile, including empty tiles. Their
host storage is charged with map offsets in both retained and discovery-peak
bounds. New launch tables are built before publishing maps. An allocation or
admission failure cannot replace the previous dense binding with a partial table.
Rebinding is rejected once evaluation or capture has started.

## Evidence and limits

The existing incumbent is retained explicitly because complete target-specific
prepare/cast/pack/kernel/refinement/audit/fallback timing is unavailable. Unknown
costs are not zero, and no new tile or precision qualification is promoted. The
production tile remains 16; explicit compiler tile choices retain their existing
qualification requirement. Generated scalar/tiled code is one provider family.
This migration does not complete #1889's CC or competing-provider acceptance.

Canonical metadata identifies the AOT template. It is not a cross-device cache:
the live plan separately owns device, stream, runtime full/tail shapes, geometry,
and local-map lifetime. Toolchain/binary identity remains with the native build.
No arbitrary-shape or cross-module executable cache is introduced here.

Tests compare the scalar DAG to an independent NumPy contraction with deliberately
nonsymmetric densities and execute generated binding selection using a cached
host compiler. Existing tests cover precision work accounting and replay/error
publication. On n1, the native XC target was built with both CMake ccache launchers
verified in generated compiler commands. All three real-device suites passed via
Slurm main/gpu:5090:1: ordinary LDA/PBE/r2SCAN/WB97M-V E/V/state, tiled-tail/spin/
variational/capture, and local-AO independent CPU E/V/features/nonlocal/capture/
resource gates. Complete KS endpoint timing remains separate qualification.

## Rejected alternatives

Plain D*AO, reassociation of the explicit FP32 conversions, guessed timing,
per-iteration candidate selection, uncharged per-tile host tables, and promotion
of 8/32 tiles would each change a scientific, lifetime or evidence contract.

## Revisit when

Other providers implement the exact symmetric/mixed contract and complete
endpoint phase evidence allows profitable comparison against the incumbent.
The preserved native selector and canonical request should then be reused.

Related: #1886, #1889, #1899;
[shared selector rationale](2026-10-05-native-joint-selection.md).

## Integration with component precision admission

The landed local-AO composition intersects method admission with compiler-owned
XC execution capabilities. Preserve that intersection before preparing density
bindings: AUTO may retain mixed Direct J while a mapped density contraction stays
strict. Preparation validates the same capability instead of functional ordinals;
iteration accounting reads the bound density arithmetic. A host executable checks
dense/local capability, AUTO/FP64 admission and admitted/strict-audit phases together.
