# Decision: differentiate the shared D3(BJ) energy for native CPU curvature

Status: implemented, standalone low-level source only
Date: 2026-11-05

## Problem and ownership

Issue #2845 requires the complete two-body BJ second chain, including
coordination-number/C6 interpolation and mixed distance terms. Differentiating
only the final pair-distance formula would omit real nuclear response. A
correction-only slice can proceed independently of electronic endpoint promotion.

## Decision and invariants

`D3GeometryProgram.coordinate_hvp()` linearizes the existing generated coordinate
VJP with a fixed unit energy cotangent. GeometryIR/TensorIR therefore remains the
single mathematical owner of CN, normalized interpolation, BJ damping, and all
product-rule terms. Existing primal/first-derivative equations and pinned table
identities are unchanged. No hand-maintained second D3 algebra is introduced.

`generativeqc.d3_hessian.NativeD3SecondOrder` executes that generated HVP using
the existing compiled CPU TensorIR ABI. It snapshots geometry, binds parameter,
element, equation and coordinate identities, and assembles every full-matrix
unit column without symmetrization, translation projection or inferred entries.
No PySCF/simple-dftd3, finite differences, or electronic response solver enters
this implementation. Results identify the D3-only component and atom-major xyz
Hartree/Bohr² convention; they do not claim a total-method Calculator endpoint.

Only uncut two-body BJ, ATM off, is admitted for second order. Finite CN/pair
cutoffs and switch variants explicitly reject pending their own boundary
qualification; first-order support does not promote them automatically. Zero
damping and ATM remain excluded by the existing compiler specification checks.
Tests consume the existing canonical B3LYP parameter record, not a hardcoded fit.

## Bounded resources

A separate generation budget admits a conservative 32-panel CN/C6 table
reserve before graph construction. Native lowering separately checks the exact
generated execution byte/work requirements before compiler discovery. The
normal verified compiler-cache launcher and finite adapter deadline are reused.
The bound includes finite work admission, not just an output-memory cap.

Hessian output admission is separate from HVP support. Simultaneous runtime
admission includes native workspace, coordinate/response reserves and both the
assembly matrix and its immutable publication copy. A failed request publishes
no partial matrix; valid actions remain usable. Shared immutable tables, Python
metadata and compiler-process RSS are not represented as native workspace bytes.

## Independent evidence

The seven-case native suite uses a mixed Ni/C/O/H environment with six pairs and
nontrivial coordination response, as well as exact-zero isolated Ni curvature.
Independent simple-dftd3 1.4.0 has shared-library SHA256
`afaddf6a1e880b9a177ef4956f6a807ffe42969bd74b0bed398fec16b40367ae`, matching
the retained first-order oracle identity. All 12 raw unit columns are compared
to its independently evaluated analytic gradients at 3e-4, 1e-4 and 3e-5 Bohr.
Maximum errors are 3.99883e-9, 4.44310e-10 and 3.99604e-11 Eh/Bohr², below the
unchanged 2e-8 gate. Raw symmetry, translation, H@v, bilinear symmetry and finite
rotation covariance use a separately frozen 2e-11 gate.

The suite also covers geometry snapshot ownership, changed-geometry/parameter
identity, sensitivity to BJ parameters, unsupported ATM/cutoffs, zero direction,
invalid directions, generation/work/output budgets, an HVP-admitted but
Hessian-rejected simultaneous workspace, and subsequent recovery. A full raw
matrix uses 12 native HVP evaluations, zero electronic response solves and zero
displaced SCFs. Its generated plan reports 16326 scalar work units per action,
68720 native workspace/reserve bytes and 71024 simultaneous numeric bytes during
full-matrix publication. These are modelled numeric resources, not total RSS.

`d3-second-order-v6.{log,xml}` records seven passing tests in 49.42 seconds;
the entire process, including independent checks and additional compilations,
took 50.39 seconds and peaked at 441332 KiB. Launcher statistics and artifacts
are retained under `/tmp/gqc-2843-2849` and the data-disk JIT cache. These test
timings are not a composed electronic endpoint benchmark or a speedup claim.

The existing D3 compiler, ragged, native-reference and CPU production suites
passed all 76 selected cases in 9.11 seconds after their registered reference
sources were synchronized and all five source-file digests verified. The two
real-GPU execution cases were explicitly deselected, not counted as passes.
Compiler structure checked 584 modules with zero dependency errors; applicable
hooks passed. The initial missing-source-cache failure remains retained as
`d3-second-order-regression.{log,xml}`; successful replay is the `-v2` receipt.

## Retained failures and rejected alternatives

The initial 1e-3-Bohr unit-column reference difference failed the same 2e-8 gate
at 4.44319e-8. The four-step independent convergence diagnostic retains that
failure and the three smaller-step matrices in
`d3-second-order-reference-convergence.{log,npz}`; the error decreases as h².
No error tolerance was loosened. Earlier integer-input, scalar-cotangent shape
and generation-versus-execution budget failures also remain in the local logs.

Do not replace this source with a user-side finite-difference correction,
freeze C6/CN, silently include ATM, or promote total-method curvature from these
standalone checks. No external engine belongs in the production derivative path.

## Remaining gates and revisit conditions

Issue #2845 remains open. Generic molecular exactly-once composition,
electronic+D3 independent total Hessian/frequency checks, omission/duplication
negative controls, installed total-method endpoints, composed endpoint resource
and timing receipts, and any finite-cutoff second-order extension are separate
remaining gates. RKS/UKS, ECP, TS and frequency capabilities do not inherit
qualification from a geometry-only source.
