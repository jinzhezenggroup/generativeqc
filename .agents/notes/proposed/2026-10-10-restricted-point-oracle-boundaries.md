# Proposal: resolve restricted-point oracle boundaries before performance promotion

Status: proposed; scientific qualification incomplete
Date: 2026-10-10

## Problem

After closing pure AD-width compaction because its actual CUDA instructions are
unchanged, a distinct PBE0 geometry prototype reuses exact spin equality. One
common density differential and three total-gradient differentials evaluate the
existing correlation algebra; exchange executes once, with its value added
twice in the original order. Exact density and Cartesian gradient equality is
checked per point; all unequal features use the original full-spin entry.
The compiler selects this entry only for explicitly unpolarized PBE geometry.
The native point/response interface and eight-direction layout stay unchanged.

This changes a computational specialization, not a functional formula. However,
new independent low-gradient/tail diagnostics expose incumbent numerical limits
that must not be hidden behind baseline/candidate equality.

## Evidence and present decision

Actual original/candidate host headers pass 40,832 point cases and 10,208 response
cases, all nine outputs and status bit-identical, including exact fallback and
broad symmetric companions. All existing 97 independently generated SCF-domain
cases pass their unchanged `5e-10*abs(reference)+1e-322` gate.

Additional 60-point restricted populations for each of PBE and PBE0 span densities
1e-280 through 1e100 and zero/small/large reduced gradients. The original rs/t2
formula is differentiated at both 450 and 550 decimal digits; rounded references
are identical. **Nine PBE and two PBE0 cases fail the strict relative gate in both
arms.** Every incumbent/candidate result in these populations is bit-identical.
The new independent qualification remains failed. No CUDA endpoint build, GPU
population or performance PR is authorized by this parity result.

The first failed qualification stops at a rho_s=1e-240 case. Its separately
retained v2 completes diagnostics for all inputs before reporting failure; no
case is deleted and no tolerance is changed.

Component analysis distinguishes two phenomena:

- Seven PBE cases with normal rho_s^(4/3) have almost-canceling X/C gradient
  coefficients. Their errors fit a 64-machine-epsilon scale of independently
  reconstructed component magnitudes. This is diagnostic conditioning evidence,
  not an accepted replacement for the strict point gate.
- Two cases at rho_s=1e-240 have nonzero subnormal gradients and subnormal
  rho_s^(4/3). The incumbent low-u exchange branch divides by a quantized product
  with relative error -1.1132817316953745e-5. The maximum exchange-gradient errors
  are 2.9948438398262264e-9 and 2.971729464763817e-8; correlation agrees to ordinary
  roundoff. These cases also fail PBE0, where X/C cancellation does not explain
  the error. The restricted prototype reproduces this incumbent limitation.

The component oracle is reconstructed from the separately high-precision
PBE/PBE0 combinations after FP64 rounding; it is suitable for diagnosing the
large component discrepancies, not a new authoritative component gate fixture.

## Invariants and next decision

Keep all original diagnostics, existing 97-point gates, independent formula
ownership, point admission and precision intact. Do not treat a raw relative
failure as passing merely because both arms fail, silently add an absolute
tolerance, or remove inconvenient subnormal/near-canceling cases.

Resolve the scientific acceptance boundary explicitly before GPU promotion.
Possible subsequent work includes independently scaled cancellation fixtures,
a rigorously bounded specialization admission with retained full-spin fallback,
or a separately scoped incumbent numerical fix. None is selected or implemented
as production behavior by this note. A performance task is not permission to
quietly fold unrelated numerical policy changes into its patch.

Only surviving actual device-work reduction followed by unchanged complete
endpoint scientific/work gates and formal timing acceptance warrants promotion.
No endpoint gain has been established for this prototype.

## Provenance and references

The measured qualified native snapshot remains #2185 plus default-off vector
scaffolding, SHA256
`cade369caa8eaa81d9a3f89f29e235a766e18129f94e16b576d069c72b7e15bb`.
Latest inspected master is b74815dba (#2196). #2195 changes fitted-only AO masks,
not the current direct PBE0 endpoint; #2196 changes CI cache lifecycle. Neither
justifies an old numerical/timing population or relabeling retained source.

Prototype header SHA256:
`c55e428014c51fca9c5173fd922426ad6b4be5d04fafc17be1db81f7fbd754fa`.
Actual host wrapper SHA256:
`1be9b84bd67ff2e2c688a3de6ab0e88f2f186aba29e3996ea9886d3369e499eb`.

- `.artifacts/xc-restricted-stage/current-state.md`
- `.artifacts/xc-restricted-stage/evidence/host/verified.json`
- `.artifacts/xc-restricted-stage/evidence/independent/xc-point-restricted-host-v2/`
- `.artifacts/xc-restricted-stage/diagnose-components.py`
- `.agents/notes/rejected/2026-10-10-compact-correlation-point-channels.md`
- `docs/developer/xc_scf_domain.md`

## Subsequent bounded PBE0 experiment

The original implementation and qualification in this note remain failed and
unchanged. A later PBE0-only producer-bound entry factors the subnormal exchange
ratio without changing the gate; all original PBE0 inputs and added boundary
cases now pass on CPU and GPU for that new entry. General PBE and production
dispatch are not qualified by that result. See
`../proposed/2026-10-10-pbe0-bound-subnormal-exchange.md` for the separate numerical
lowering, source identities, device evidence and remaining promotion conditions.
