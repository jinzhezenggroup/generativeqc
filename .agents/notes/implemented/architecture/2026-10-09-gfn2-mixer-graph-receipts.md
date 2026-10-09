# Decision: call-owned GFN2 mixer receipts inside the SCC device graph

Status: implemented; diagnostic qualified on H100 through 192 AO
Date: 2026-10-09

## Problem

#1882 needs evidence of modified-Broyden work and stage time inside the real
device-launched SCC graph before selecting another history provider. Whole-graph
profiler stalls do not identify mixer cost. #2109 moved the actual history folds
into compiler-generated CUDA fragments, so counting inferred work outside those
folds would miss nonfinite early exits.

## Decision

The native runtime owns a default-off, call-owned receipt arena bounded at
64 MiB. It resets the count before each synchronous inference and reads receipts
only after stream completion. Each active mixer CTA publishes its system,
pre-commit iteration/restart, vector dimension, live history, tentative slot,
actual coefficient/Gram/correction visits, stage-completion bits, status and
lane-zero elapsed cycles. The runtime records the SCC loop's submitted execution
mode, rather than inferring it from graph readiness.

The compiler's retained CUDA dot and correction emitters accept optional
identifier-validated visit counters. GFN2 binds them to the diagnostic kernel
specialization at the visited loop bodies. The default specialization compiles
the guards away and carries no history-count shared arrays. CPU generation and
the ordered arithmetic and failure order are unchanged.

## Rejected alternatives

- Inferring visits as dimension times history depth would misreport nonfinite
  dots that break before the end of a vector.
- Reintroducing handwritten CUDA history loops would create a second owner of
  the #2109 mathematics and lose generated source identity.
- Host polling or a separate mixer launch would measure a different execution
  path from the SCC graph.

## Invariants

- A receipt is an execution observation, not a scheduler input or a change to
  scientific publication.
- Overflow remains visible as attempted count greater than capacity.
- Inactive systems produce no receipt. Failed members do not commit history;
  their partial receipt can report completed earlier stages, but not the
  numerically failed stage.
- Cycle counts are per-CTA lane-zero intervals across existing barriers, not
  additive endpoint wall time.
- The 768-AO tblite discrepancy remains a failing scientific gate.

## Evidence and limits

With visit bindings omitted, all four generated CUDA fragments retain the
#2109 source identity
`f244c11c54c4124254f69177237a3a1dcccc09580f6a55c7e3f24b707092cf09`.
The instrumented generated identity is
`8592b95f849192ab17f614af64178651b92b25e5d1b4ca5d2e583378ff594784`.
The expanded instrumented CUDA translation unit is frozen separately in the
lowering test. Focused local generator, provenance and bootstrap structure
checks passed (151 passed, 3 device/tool skips). These checks do not establish
real-device numerical or timing acceptance.

Source-matched H100 qualification used master `d5a3173cc`, CUDA 12.9.86,
driver 570.124.06, `sm_90`, and ccache 4.5.1. The final library SHA-256 was
`738ffed59637b8ced302944c3a8028a4fc210c190fc8595b97bf0e9a6f9d492d`;
the bootstrap executable SHA-256 was
`a407a3d88122dd5d9a0620219f991fc936f7ef398183f581f991da11e0fd0f46`.
The ordered-history CUDA compile/device-link target passed, and the real-device
chronological-oracle fixture passed 10 stream/graph cases including ragged
batches, wrap, nonfinite visit counts, and failure isolation. The fixture's
standalone nvcc build explicitly targeted `sm_90`: the toolkit's default
`sm_52` PTX was not loadable by this H100 driver.

The private bootstrap ran default-off/diagnostic-on complete energy and
host-force endpoints for seven independent small tblite cases and water clusters
at 48 and 192 AO. Host and device input, graph replay/reset, first-call failures,
and recovery passed. Independent tblite energy/force acceptance was `5e-7`
absolute; A/B and replay agreement was `5e-14` absolute with exact iteration,
convergence, and status equality. The tighter A/B gate is above observed
default/default force variation (`4.16e-17` at 48 AO, `1.11e-16` at 192 AO)
and observed default/diagnostic variation (`5.55e-17` and `6.94e-17`).

Each 48-AO water endpoint recorded 17 active mixer invocations, 24,800
coefficient visits, 177,568 Gram visits, and 24,800 correction visits. Host
baseline/diagnostic/replay endpoint times were 35.301/36.336/29.831 ms;
device-input times were 34.973/35.368/30.882 ms. At 192 AO the corresponding
counts were 17, 99,200, 710,272, and 99,200. Host times were
148.266/107.886/86.770 ms; device-input times were
99.666/100.400/86.762 ms. These single-run timings are complete endpoint
observations, not a speedup claim. Per-CTA stage cycles are not additive wall
time. The independent 48/192-AO tblite receipt SHA-256s and test logs are in
`/inspire/ssd/project/chemicalreaction/czxs25220150/issue-1882-stage-20261008/`.

The 768-AO tblite discrepancy remains a failing scientific gate and was not
rerun in this diagnostic qualification. This change does not select or promote
a provider schedule and does not close #1882 or #560.

## Revisit when

An accepted provider schedule replaces the retained generated folds, or
measured receipt overhead is unacceptable for opt-in diagnostics.

## References

Issues #1882, #560; PR #2109; tests/python/test_ordered_history_lowering.py;
tests/native/test_ordered_history_consumers.cu;
tests/native/test_gfn2_cuda_bootstrap.cpp.
