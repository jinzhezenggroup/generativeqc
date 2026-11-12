# Proposal: qualify complete augmented-QZ/5Z full-DF CUDA endpoints

Status: proposed; only the basis-operator inventory is implemented here
Date: 2026-11-09

The QZ implementation and its retained fallback decisions are superseded by
[the implemented closure note](../implemented/numerics/2026-11-09-full-df-orbital-g-auxiliary-h.md).
This proposal preserves the original diagnostic-only stage; its optional
heavy-atom 5Z extension remains proposed.

## Problem

Issue #3055 requests unmodified augmented orbital and JKFIT bases for full-DF
HF/MP2 and CCSD(T) energies and analytic nuclear forces. A first-overlap failure
hides the independent auxiliary, derivative and response boundaries. Increasing
a Python maximum would not supply the missing operators or qualify the endpoint.
The initial inspection uses master `76bb87c76d8ddd7da369684480796fe456ca7ef4`.

## Implemented inventory

`Calculator.basis_capability_report` and calculation preflight share one
operator-request traversal. The report preserves every supplied contraction,
both nuclear orders when forces are requested, and the actual selected auxiliary
consumer. Standalone `basis_capability` now exposes actual and supported angular
maxima and the selected consumer. These are mathematical admission diagnostics,
not evidence for amplitude, response, force, resource or endpoint closure.

The metadata regressions inspect the entire installed PySCF aug-cc-pV(T,Q,5)Z
and matching JKFIT records, including all primitive coefficient columns. They
are not numerical or independently reconverged molecular evidence. No new CUDA
energy/force qualification or performance claim is made by this change.

## Staged implementation and qualification

1. **Orbital g and full QZ auxiliary h.** Implement and independently qualify
   the missing S/T/V values and nuclear derivatives, spherical transforms,
   orbital three-center values/derivatives, metric values/derivatives and
   source/weighted nuclear sinks. Keep generic, SCF-DF and correlation-DF
   admission independent. Existing opt-in g components do not close the
   production route. Extend native packing/workspace and generated scheduling
   limits only together with the corresponding implementation and evidence.
2. **Full-DF RHF + MP2.** Independently reconverge the same complete basis,
   geometry, charge, multiplicity and all-electron root for CN- and a small
   heavy-atom molecule such as CH3F. Freeze explicit numerical acceptance gates
   for component energies, analytic forces and orbital response before
   promotion. Check at least two finite-difference displacements with separately
   converged displaced energies. Retain SCF/root diagnostics, fit rank and
   threshold, force sign/units and translational checks. Historical energy-only
   CPU numbers are not force references.
3. **Full-DF CCSD(T), separately.** Qualify CC amplitude, lambda/orbital response,
   triples and force contractions using the same complete auxiliary basis.
   Passing MP2, SCF or one-electron g checks is not CCSD(T) endpoint evidence.
   Retain HF, MP2 and CCSD(T) component energies/forces instead of only a final
   extrapolated difference. Require an independent same-Hamiltonian reference
   and two-step finite differences for this endpoint too.
4. **Optional Q/5 anchors.** Hydrogen aug-QZ requires orbital f/auxiliary g;
   C/N/O require orbital g/auxiliary h. At aug-5Z these become orbital
   g/auxiliary h for H and orbital h/auxiliary i for C/N/O. Implement and qualify
   those new domains independently; QZ closure must not promote 5Z admission.

At every stage preserve explicit missing-combination errors, complete basis
identity and bounded fallbacks. Record complete-endpoint wall time, peak device
memory, source/contraction semantic work and fallback selections for physically
relevant small molecules. Run all real-device validation and profiling through
finite-time Slurm allocations on permitted, available nodes. A missing allocation
is missing evidence, not permission to bypass scheduling or substitute CPU work.

## CBS observable and evidence contract

For cardinal numbers X < Y and fixed HF exponent alpha, let
`q = exp(-alpha * (Y-X))`. The HF CBS weights on `(X,Y)` are
`(-q/(1-q), 1/(1-q))`. MP2 *correlation* CBS weights are
`(-X^3/(Y^3-X^3), Y^3/(Y^3-X^3))`. Apply exactly the same constant
weights to the corresponding analytic forces. Add the small-basis correction
`CCSD(T) total - MP2 total`, and its force difference, only after both component
endpoints pass. Retain actual weights and absolute component values.

Compare D/T and T/Q while holding the correction basis fixed before changing
that correction from DZ to TZ. Q/5 anchors are a later, separately qualified
experiment. Preserve geometry, root, all-electron/frozen-core convention,
unmodified basis provenance and non-CP label definition. Different published
Hamiltonians, F12/CV composites or unverified CP conventions do not identify a
force error or prove monotonic improvement from augmentation. Ghost-basis forces
would require their own optional qualification.

## Rejected shortcuts

- Do not widen `LIMITS` or switch a correlation caller to the wider SCF consumer
  merely because a basis loads or an isolated generated component executes.
- Do not drop high-l auxiliaries, reuse an old binary as current evidence, use
  energy agreement as force qualification or make PySCF a production fallback.
- Do not mark #3055 fixed until the separate molecular energy/analytic-force
  closures and bounded resource evidence are retained.

## References

- Issue #3055; independent auxiliary-g UHF work #3032; opt-in orbital-g work #879.
- `docs/user/external_basis.md` for the current inspection API.
- `docs/maintainer/performance_engineering.md` for endpoint evidence requirements.
