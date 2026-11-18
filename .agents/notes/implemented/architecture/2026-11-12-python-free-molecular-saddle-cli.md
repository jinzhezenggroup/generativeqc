# Decision: native molecular saddle CLI and explicit harmonic publication

Status: implemented
Date: 2026-11-12

## Problem

The native callback corrector, retained CPU derivative batch and orbital-root
bridge did not themselves provide a Python-free molecular task. Reusing the
minimum optimizer's automatic cheaper stages would change the requested
chemistry. The corrector's unweighted Cartesian curvatures also cannot be
published as mass-weighted harmonic frequencies.

## Decision and ownership

`generativeqc saddle` binds the shared native C corrector to one actual CPU
batch per energy/gradient/Hessian geometry. Prepared native force capability
is required before SCF. The token/geometry/method-authenticated native Hessian
bridge supplies the complete total, including native D3 exactly once. No
Python, JIT, reference SCF, displaced SCF or external optimizer is invoked.

Accepted progress observations alone replace the occupied-root reference.
Native cross-geometry AO overlap and each spin's occupied principal overlap
are separate from target-mode tracking. Empty spin populations are not
silently identified with the other spin. A rejected trial cannot reset the
reference. Canonical orbital-only export retains no AO/grid/Fock response
provider. Batches close after Hessian publication or on failure; the last
accepted geometry and occupied panels can outlive that batch.

The 100-attempt bound counts accepted and rejected trials. All five convergence
controls and fresh-final verification belong to the existing native corrector.
Derivative progress, optimizer observations, completed work and unknown
failed-stage work remain distinguishable. SIGINT/SIGTERM request cancellation
at the next synchronous owner boundary, never bypassing source cleanup.

Harmonic analysis has an explicit native supplied-Hessian owner. Ordered
positive physical/isotopic dalton masses are mandatory, not guessed from a
table. Mass-weighted geometry-derived translations/rotations are removed;
no small vibrational eigenvalue is deleted. Raw asymmetry is measured and gated
before roundoff averaging an analysis copy. The raw published Hessian remains
unchanged. Signed cm^-1 frequencies are unscaled; rank/index disagreement or
mode-publication failure preserves geometry but prevents task success.

## Resource and domain boundaries

Optimizer, derivative, root and result-panel bounds have separate ownership
scopes. They exclude primal/provider scratch, cross-overlap integral scratch,
JSON and RSS; none is a practical-work or process-memory claim. The method,
basis, spin, charge and native grid remain fixed. ECP/DF/CUDA molecular saddle
admission remains closed. Missing native capabilities refuse without fallback.

## Rejected alternatives

- Calling the Python molecular adapter from a nominally native CLI.
- Reusing the minimum-only `optimize` planner or implicit cheaper methods.
- Relabeling Cartesian curvatures as harmonic frequencies or deleting soft modes.
- Accepting an electronic-only Hessian for a corrected total method.
- Replacing independent heavy/UKS/complete-endpoint evidence with small controls.

## Evidence and remaining qualification

`tests/native/test_harmonic_modes.cpp` uses independent reduced-mass and linear
chain spectra, isotope scaling, signed soft modes, frame invariance, raw
symmetry/nonfinite gates and capacity refusal. `tests/native/test_cli_saddle.py`
covers actual CPU batches, refusal receipts, native UKS/corrected totals and
an opt-in complete NH3 endpoint with independent Python mass weighting.

New run evidence is kept separately under ignored
`build/qualification/host-native-ts-cli-20261112/`; predecessor receipts and
recovery archives are not overwritten. The earlier NH3 three-step independent
moving-grid oracle remains a separate input, not a newly executed oracle.
Small NH3/STO-3G and nonnegative H2 controls do not qualify the requested Ni
transition state, a positive genuinely open-shell UKS saddle, real Ni/I/Cs ECP
derivatives, or practical 25--40-atom complete electronic work. Those gates
remain mandatory. A one-electron H2+ candidate currently refuses in the linked
nuclear GGA response; it is not evidence of an accepted empty-beta Hessian.

The native source CLI and a newly repaired local wheel both reach the NH3
first-order saddle in three accepted attempts under the original five controls.
The source endpoint took 380.68 s and the installed endpoint 374.72 s, with
one OpenMP/OpenBLAS thread on a shared host; these are measured small-endpoint
times, not a speed comparison. The installed process runs with an empty
environment apart from explicit controls, no compiler/interpreter on PATH and
no loader path injection. Its native raw Hessian matches the prior accepted
raw matrix to 7.7e-14 at coordinates within 1.1e-13 Bohr; every column also
meets the unchanged 1e-5 matrix gate against all three archived independent
moving-grid stencils. Native mass weighting agrees with independent Python
analysis within 1.3e-12 cm^-1. No new displaced-SCF oracle was executed.

Current focused validation passes 196 Python tests, six native CTests, both
installed ELF fast-contract entries, actual CI Sphinx warnings-as-errors,
ownership/boundary/contract checks, and formatting. The opt-in NH3 unittest
is skipped in fast suites; the complete installed production endpoint is
executed and validated separately. Actual signal cancellation and a one-attempt
limit preserve accepted-state failure records. A too-short, buffered initial
cancellation harness and earlier failures remain in the evidence logs.

The initial default dependency install was stopped when it started resolving
unnecessary CUDA provider wheels. CPU-only isolation uses the repaired wheel
with `--no-deps` plus explicitly pinned minimal dependencies; it does not claim
that the default dependency resolver produced a CPU-only environment, or that
`pip check` is satisfied without optional provider packages. This local
manylinux_2_35 repair is not release/manylinux_2_28 portability qualification.

## References

- [Retained derivative-batch decision](2026-11-12-native-saddle-derivative-batch.md)
- [Current task contract](../../../../docs/user/geometry_optimization.md#native-molecular-saddle-command)

## Revisit when

New providers change root/source lifetimes, harmonic masses or projection rank,
or independent real-element/positive-UKS/practical-endpoint qualification
supports broader molecular admission. Never broaden it solely from a microkernel.

## Subsequent minimal UKS evidence

The [fresh installed SiH3/STO-3G UKS acceptance](../numerics/2026-11-12-installed-minimal-uks-saddle-acceptance.md)
records a new successful native ELF search and independent final matrix/mode
audit. It does not replace the original receipts or qualify Ni/ECP/scale.
