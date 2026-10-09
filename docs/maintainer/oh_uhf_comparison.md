# OH UHF determinant comparison policy

For the independently qualified neutral OH/STO-3G six-AO CPU fixture, validation
accepts an independently certified **spatial equivalence class** rather than
requiring one unique determinant orientation. The policy identifier is
`oh-sto3g-axial-spatial-equivalence-v1`.

The domain is O(0,0,0), H(0,0,1.834) Bohr and its moved bond 1.85234 Bohr,
spherical AO order O 1s, 2s, 2px, 2py, 2pz, H 1s, with alpha/beta populations
5/4 and the exact bundled STO-3G primitives. It is qualified by the retained
scalar and OpenBLAS CPU captures, not by a generic claim about every degenerate
UHF or UKS state. Other geometries, basis representations, spin states and
nondegenerate fixtures do not acquire a symmetry fallback from this policy.

## Independent decisions

`tools.oh_uhf_policy.evaluate_oh_density` recomputes the existing full-tensor
certificate from its `certify_oh` keyword inputs. It returns:

- `raw`: unchanged absolute 1e-7 full spin-density and metric-projector gates
- `symmetry`: the individual spatial-certificate metrics and gates
- `equivalence_decision`: `ACCEPTED_EQUIVALENT` only if all certificate gates pass,
  otherwise `REJECTED`
- `canonical_determinant_required`: false for this explicitly admitted domain

The raw outcome is independent: raw **FAIL** and `ACCEPTED_EQUIVALENT` can coexist.
There is no combined PASS that hides a raw failure. This density-comparison
policy is not an energy/force, convergence, stability or endpoint acceptance
policy; those requirements retain their own gates. A certificate does not prove
that the state is the intended ground-state root or is electronically stable.

The candidate transform must be one common axial SO(2) spatial rotation for
both spins. Acceptance requires nuclear-geometry and S/H/full-ERI invariance,
Fock covariance, Hermiticity, spin counts, metric idempotency, physical residuals
of both states and the transformed state, reconstructed/endpoint/reference
energy agreement, and transformed full-density/projector agreement. Fitting an
angle alone is insufficient. Unsupported and nonfinite inputs fail closed.
No arbitrary occupied/virtual alignment or independently chosen spin rotation
is admitted.

## Evidence and reproduction

Use [the fixture reproduction card](../../tests/data/oh_uhf_symmetry_1791.md)
for exact source/library/oracle identities, provider builds and capture commands.
Capture actual primary endpoint densities before batch closure; never replace
them with a later reference-export solve. Bind independent Hamiltonian tensors
to exact primitives, AO order, spin and geometry as the runner requires.

The original `tools/oh_uhf_symmetry.py`, capture runner and JSON fixture remain
byte-for-byte unchanged. Their hashes and historical policy text describe the
original diagnostic-only capture. The new policy layer evaluates those retained
tensors; it does not rewrite their timestamps, source attribution or decisions.
All four historical raw failures remain failures in their original capsule.

Run both `tests/python/test_oh_uhf_symmetry.py` and
`tests/python/test_oh_uhf_policy.py` to verify molecular and synthetic adversarial
rejection, raw/nondegenerate gates, evidence hashes and current policy. Replaying
these matrices is a CPU validation-policy check, not a new native endpoint run,
GPU qualification or performance claim.

See the [decision rationale](../../.agents/notes/implemented/numerics/2026-10-08-oh-uhf-equivalence-policy.md)
for the canonical-orientation alternative and conditions for expanding scope.
