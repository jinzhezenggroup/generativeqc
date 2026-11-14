# Decision: match GFN2 dispersion models before independent qualification

Status: implemented
Date: 2026-11-13

## Problem

Issue #560 retained 36 failed 768-AO observations from the source-frozen n2
campaign: native energy, native energy/forces, and xTBloom energy/forces at original
and moved geometries. Tightening SCC did not remove the discrepancy. The native
and comparator binaries agreed, but that agreement was not an independent oracle.

The reference producer constructed tblite 0.6.0 without `xtb_config`. That version
defaults to a **0.05-bohr dispersion cutoff switch**, whereas the native GFN2
parameter contract selects **sharp cutoffs**, with two-body/ATM radii 50/25 bohr.
This is a model mismatch, not an SCC convergence tolerance or compiler error.

At the original 128-water geometry, native fixed-charge D4 gives ATM energy
`0.023909817914186865` Eh; default tblite prints `0.023908544763071` Eh. Their
`1.2731511e-6` Eh difference accounts for the historical total-energy failure.
Explicit sharp-cutoff tblite reproduces the native total energy. Its atomic
charges are identical to the default-model calculation;
ATM is independent of the SCC charges. The force difference also disappears
under the original `5e-7` acceptance gate, without modifying production equations.

## Decision

- Construct the independent oracle with `xtb_config={"smooth_cutoff": 0.0}`.
- Record this model choice in every newly prepared oracle and endpoint report.
- Reject absent, smoothed, nonfinite, or otherwise unverified cutoff metadata in
  endpoint and lifecycle reducers. Regenerate older oracles; never relabel their
  frozen results by attaching configuration they did not use.
- Report the supplied experiment's 768-AO gate, including `NOT_EXERCISED`, instead
  of unconditionally copying a historical failure into every new report.
- Keep production parameters, SCC policy, generated equations, and acceptance
  tolerances unchanged.

## Evidence

The compact independent fixture `tests/data/gfn2_sharp_cutoff_tblite.json` places
three waters near the ATM boundary. Its sharp/smoothed force difference is
`2.5689631e-6` Eh/bohr, exceeding the public gate with only nine atoms. Both
controls retain full oracle energy/forces and the actual oracle-library SHA256.
The public CPU/CUDA golden test consumes the sharp control; an optional live
tblite test regenerates both controls. A behavioral producer test uses these
independent controls to reject an accidental return to tblite defaults.

The recovered campaign and matched independent references are retained under
`benchmarks/results/gfn2-sharp-cutoff-20261113/`. Reconciliation is explicitly
bound to the original production commit
`dc68eb0b2b496de18fd765bda577d836487b2883` and recorded binary identities, not a
claim that all later production changes are qualified. All 36 previously failing
768-AO observations pass the original gates with the matched oracle.

## Rejected alternatives

Increasing SCC precision cannot fix an additive, state-independent ATM model
difference. Changing the production model to match a library default, increasing
the tolerance, or excluding large molecules would hide the configuration error.
Regenerating historical reports in place would destroy the evidence explaining it.

## Invariants and scope

Independent references must match parameters, geometry, charge, temperature, and
cutoff policy, not merely the method name or library version. Native/peer parity
is not independent acceptance. A passing historical replay does not close all
of #560: current-source endpoint, lifecycle, and memory qualification remain
separate obligations. In particular, the current CPU 768-AO probe exhausted its
300-iteration SCC budget; it is a retained convergence failure, not a passing
energy/force observation.
Increasing history to 32 converged the original geometry in 131 iterations but
not the moved geometry within 300 iterations. Do not confuse that single
successful diagnostic with a completed original/moved CPU campaign.

### Current-source follow-up — November 14, 2026

The separate `benchmarks/results/gfn2-current-source-20261114/` campaign closes
those qualification obligations for production source
`95c3e547f96848e2fc3392d85fdfa1a68dff688b`, with full tracked-source manifests,
patches and actual CPU/CUDA library identities. All 120 current observations
pass the unchanged gates, including original/moved and five repeats per geometry
at 48/192/768 AOs. CUDA uses the original history 8 and maximum 300 iterations.
CPU explicitly uses history 64 with the same maximum 300 and tolerances,
converging the large original/moved geometries in 108/145 iterations.

Additional bounded controls explain why the prior CPU failures are not another
model mismatch: history 8 converges the original point in 702 iterations and
history 32 converges the moved point in 415, outside the original 300 budget.
Do not rewrite Broyden, increase the scientific gate, or silently retry controls
to conceal that trajectory. Explicit caller-selected history is part of the
receipt; it is not an automatic fallback or a new default.

Slurm 4463 runs the full current optimized SDK library, public CPU/CUDA oracle
and lifecycle checks, private CUDA host/device ingress and failed-transaction
recovery, and a separate zero-error/zero-leak Compute Sanitizer lifetime test.
Process RSS and device-wide memory snapshots are retained; exact device peaks,
workspace sizes and allocator counts remain explicitly unavailable. The native
moved CUDA endpoint is slower than its peer, so this acceptance makes no
speedup or per-stage cost claim. It does not admit ragged execution, true WARM
SCC, arbitrary-temperature/fractional occupations or unrestricted channels.

## Revisit when

A deliberate versioned change admits a smoothed GFN2 dispersion model. Such a
change needs its own independent energy/force qualification and must not inherit
the sharp-cutoff evidence.

## References

- Issue #560 and its retained n2 campaign, issue comment 5998880251.
- `docs/maintainer/gfn2_mixer_endpoint_qualification.md`.
- `src/xtb/native/src/model/gfn2/d4.cpp` and its sharp-cutoff parameter invariant.
