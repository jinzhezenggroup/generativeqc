# Decision: retire duplicated GFN2 workspace arithmetic

Status: implemented
Date: 2026-10-07

## Problem

The embedded GFN2 runtime remains a production CPU/CUDA consumer. Removing its
folder before its capabilities migrate would remove GFN2. Its resource planners
independently implemented overflow-checked size arithmetic already owned by
`src/runtime/bounded_workspace.hpp` and consumed by other native methods.

## Decision and retirement ledger

Eight production owners now import the existing shared runtime checked-add and/or
checked-multiply operations. Their redundant implementations are deleted:

- CPU eigensolver, SCC driver, D4 planner, and common ragged Broyden mixer
- CUDA SCC eigensolver setup, input setup, topology setup, and topology staging

CUDA eigensolver setup also consumes the existing shared checked alignment helper.
This retires 16 duplicate helper implementations rather than moving the embedded
runtime into a differently named directory. The shared provider remains independent
of xTB. Scientific state, signed descriptor validation and publication remain in
their existing adapters.

## Invariants and deliberately retained contracts

- Overflow must not change the output argument, including aliased input/output.
- The shared generic alignment contract accepts non-power-of-two alignments;
  method-local power-of-two validators therefore remain local and unchanged.
- Input-setup `align_cursor` has its own output-publication convention and is not
  silently replaced in this ownership-only change.
- No arithmetic precision, mixing history, convergence gates, provider selection,
  device allocation sequence, kernel launch or synchronization changes.

## Evidence

Base: `303df5b31c97517f857ff5047befe2dccfeaab09`.

- New required cached native probe compiles the complete production Broyden mixer:
  ragged plan, exact/undersized arena binding, first-step damping, per-system
  iteration publication and failed-plan identity preservation.
- Overflow probes cover zero, ordinary values, size limits and aliased outputs.
- New source guards reject reintroduced duplicate size arithmetic in all eight
  migrated consumers. Shared-native dependency guards remain clean.
- Focused ownership/native/bridge/linkage probes: 27 passed, one unavailable CUDA
  toolkit skip. Complete native CTest suite: 75 passed.
- Built native CPU library; independent tblite energy/force oracle, retained
  runtime lifecycle and public GFN2 tests: 20 passed, 13 skipped (CUDA opt-in).
- Both kernel bodies in modified CUDA files are byte-identical to base. Removing
  only the migrated helper definitions/imports and normalizing the alignment
  function name leaves every remaining host token identical in all four files.
- Ten generated GFN2 headers match the prior qualification build byte-for-byte.
  Generator/compiler source is unchanged. This is generated-source preservation,
  not a device binary comparison or a new real-GPU performance qualification.

## Remaining migration, not completed by this slice

1. Generalized-eigen request/provider ownership and duplicated CPU/CUDA setup
   under #1240 and #1890, preserving exact overlap/capacity/failure gates
2. Density and weighted-density TensorIR/provider binding under #1814/#1879
3. Broyden semantic history operations and shared lowering under #1882, keeping
   method-owned numerical/restart policy and exact-order fallback
4. Pair-list/topology and persistent memory/handle orchestration under #933/#934
5. Move only genuinely method-specific GFN2 adapters into the method layer, switch
   consumers/build registrations, then delete the obsolete embedded runtime

Each slice requires actual consumer cutover and deletion after its checks. The
current embedded runtime cannot yet be removed wholesale. Source-byte preservation
should be used where possible to avoid an unnecessary GPU performance campaign;
a changed schedule or numerical lowering still needs its normal qualification.

## References

#1240, #1890, #1879, #1882, #933, #934
