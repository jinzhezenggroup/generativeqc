# Decision: revalidate a saved native determinant before physical component work

Status: implemented qualification procedure; no new public snapshot import or Hessian admission
Date: 2026-11-14

## Problem

The retained six-atom I–CH3–I native stationary frame contains canonical
coefficients and occupations, but neither orbital energies nor energy-weighted
density W. Earlier bounded ERI diagnoses explicitly used zero W. Those traces
cannot become physical frozen-integral evidence by relabeling them.

The earlier negative whole-endpoint stationary stage took about 546 seconds,
but its journal separates about 73 seconds of SCF and 465 seconds of stationary
integral/force work. Individual physical F[D] evaluations were about two seconds.
The aggregate stationary-stage duration is therefore not the cost of recovering
one actual physical F[D] at the saved determinant.

## Decision

For qualification-only saved-state components, reconstruct D from the saved
native C/occupations, then evaluate exactly one fresh native physical F[D] at
the original geometry, spherical basis/ECP, charge/spin, grid and method.
Reuse `PreparedFockPlan`, the native XC integrator and the production component
energy contraction. Match the total named program to its linked electronic
binding by normalized semilocal/exchange composition, not by assuming their
different method digests are interchangeable.

Diagonalize that actual non-DIIS physical Fock using the same native algebra as
production export, and require `validate_ks_final_state` to accept the resulting
canonical frame before requesting W. Mint a fresh diagnostic owner identity;
do not revive an old source token or claim that serialized labels authenticate
the lifetime of a new owner. The detached component's validator authenticates
its actual immutable in-memory D/W, basis and strategy snapshots.

Retain the original density/stationarity gates of 1e-11, energy gate of 1e-13,
screening of 1e-14 and zero final-state corrections. The physical component
energy excludes dispersion. Recover the requested D3(BJ) correction through its
canonical native parameters/runtime and check electronic+D3 against the saved
total energy; do not borrow an oracle correction or copy a numerical constant.

This is not a new production artifact-replay API. Production final-state export
already canonicalizes its retained actual physical Fock. A serialized frame
without F[D] instead needs the explicit fresh evaluation described here.

## Rejected alternatives

- Restarting the old molecular SCF or repeating stationary forces just to obtain
  W spends work unrelated to the missing native canonical energies.
- Using lagged SCF energies, half-trace KS energy algebra or independent-oracle
  orbital energies does not establish this native physical stationary frame.
- Zero W remains useful for a narrowly labeled primitive diagnosis, not for a
  physical frozen component or stationary Hessian.
- A failed saved-determinant gate is not permission to loosen tolerances, run
  hidden correction iterations or cross-seed from the independent state.

## Evidence

The retained cohort is
`build/qualification/host-saved-state-physical-20261114`. It uses source
`1bfcc8d018fbeaed432ab3737acc58b1a5d433d522cc939741459693abed8898`
and core SHA256
`6a2782d055cb03206631661875f843320a4bc42c3f9ac4f0c662c4a2bbd20b7b`.
Production sources are unchanged by this qualification.

- Native recovery: 6.654 seconds including preparation; one physical
  evaluation, zero SCF calls/corrections. Physical residual RMS 1.20e-14,
  maximum commutator 1.55e-13 and maximum density reconstruction error 7.52e-13.
  Reconstructed total energy equals the saved total at FP64 precision.
- Test-only comparison to the independently saved, matched physical state:
  D/W/energy-spectrum maximum errors 1.16e-11/3.10e-11/1.47e-11;
  minimum occupied-subspace overlap 0.999999999999997. No reference supplies
  native W or orbital energies and no independent SCF is restarted.
- Complete actual projected frozen-integral matrix at that nonzero native W:
  417.124 seconds, including 11.924 seconds for one-electron processing and
  405.199 seconds for ERI processing. These phases include maps/scatter;
  neither is a pure primitive timing. This is not an interleaved comparison
  or a complete physical-Hessian endpoint timing.
- The component returns all 343 fields `[L,gradient(18),raw-H(18,18)]` for
  `L=Tr(D(T+Vionic))+EJK(D)-Tr(WS)`. There are 136,320,861 actual ERI calls:
  5,723,832 scalar and 130,597,029 matrix calls; 639,412,191 local coordinates,
  3,596,397,327 ordered second partials, 9,548,100 one-electron calls and
  656,462,280 map terms. The combined first/frozen conservative ERI ceiling
  remains 882,874,977 within the original 1e9 cap. The component reserves
  188,096 numeric bytes; its measured process peak is 18,048 KiB. Recovery has
  a separate 361,984 KiB process peak, not included in the component reserve.
- Independent libcint first-integral-gradient differences at 2e-4, 1e-4 and
  5e-5 Bohr check every raw matrix element, with maximum discrepancies
  2.89e-8, 7.27e-9 and 1.82e-9. Value/gradient errors are 5.23e-12/6.51e-14;
  native raw asymmetry and translation residual are 3.55e-15/9.85e-15,
  without repair. Omitting W fails by 0.3143, far outside the frozen raw gate.
  All original gates are frozen before the native whole-component run.
- The independent reference retains the original ECP records and ionic charges,
  but isolates only the same S/T/Vionic/ERI component. Displacements/FD occur
  only in that test reference. There are 109 integral-gradient evaluations,
  zero SCF calls and no native evaluator calls in the reference.
- Initial validation incorrectly parsed the enabled host-region trace as
  progress-event records. Its failure/source are retained. Correcting only the
  trace parser validates the same native/reference outputs; no expensive run
  is repeated. Initial local-script lint failures and executed pre-annotation
  source texts are also retained.

## Consequences and invariants

This closes the missing authoritative native W for this saved frame and
qualifies its complete frozen **integral component**. It does not include
residual ECP, moving XC, nuclear repulsion, coupled response, total raw Hessian,
unscaled modes, Ni/Cs stationary fixtures, scale qualification or a native Ni
TS. Both public ECP saddle guards remain closed. Issues #2844–#2848 are not
declared complete.

Do not rerun this component unchanged, extrapolate an early primitive prefix
to its later owner classes or relabel the old zero-W evidence. Subsequent work
should consume the validated native frame for bounded actual response/composed
diagnosis while retaining the original scientific and endpoint resource gates.

## References

- `src/methods/dft_method.cpp`: CPU physical final-state export.
- `src/dft/ks_final_state.hpp` and `src/dft/ks_final_state.cpp`: shared gates/W.
- `src/dft/host_nuclear_integrals.hpp`: the isolated component contract.
- `build/qualification/host-saved-state-physical-20261114/acceptance.json`.
- `build/qualification/host-saved-state-physical-20261114/state-validation.json`.
- `build/qualification/host-saved-state-physical-20261114/matrix-validation.json`.

## Subsequent fixed-grid response and streamed-endpoint diagnosis

The same source/core and native frame subsequently exercise the actual
`HostGgaXcResponse` and prepared native J/K components, without SCF or a response
solve. Two occupied/virtual tangents are derived from the saved native canonical
frame; they are not captured GMRES iterates. Eight full-grid actions compare
streaming, exact point-partial preparation and authenticated warm replay, with
bitwise-equal native outputs. Independent Libxc/libcint checks at the identical
87,458-point grid give maximum XC/J/K/assembled errors
2.90e-15/7.62e-15/1.05e-14. Raw two-direction bilinear error is 2.64e-16;
the wrong-exchange negative control fails by 0.03486.

Warm fixed-grid XC median is 1.37855 seconds, of which the actual AO reader is
0.40355 seconds. The remaining 0.97531 seconds includes contractions, scalar
AD/coefficient work, accumulation and validation; it is not attributed to one
leaf. J/K median is 0.04073 seconds. These are source-component timings, not a
complete CPKS/nuclear endpoint comparison. Optional XC storage is 9,181,640
bytes inside the original 128 MiB cap; the streamed scratch is 785,672 bytes.

A separate actual streamed total-HVP request freshly validates the native
physical frame and enters the composed owner, independently moving the first
iodine ECP center along z. It reaches its finite 900-second cap with exit 124,
publishing neither a gradient/HVP panel nor acceptance. Twenty-six GMRES
operator/exact-residual actions return in 38.816 seconds, but returned actions
are not proof of a published coupled response or a successful HVP. The rest
of the composed call is not resolved by those timings. Attempted stack
sampling is denied by ptrace policy; no function-level attribution is claimed.
Do not call all remaining time pure ERI work or silently raise the cap.

The retained independent column has six converged, matched-root gradient states
at the original three step sizes, with spread below the original raw-HVP gate.
Their whole reference matrix remains failed: the finest second-order matrix's
raw asymmetry/translation are 4.17e-8/4.20e-8 against unchanged 2e-8 gates.
Accepting a separately audited column does not promote that matrix or modes.

Evidence lives in `build/qualification/host-saved-response-diagnosis-20261114`
and `build/qualification/host-saved-physical-hvp-20261114`, retained together
under `host-saved-response-and-streamed-negative-20261114`. A subsequent complete
matrix request must evaluate its own shared coordinate/frozen owners and all
18 responses under the original complete-request caps. It cannot inject the
retained frozen numeric matrix or infer complete-endpoint success from these
fixed-grid actions. No production source or public admission changes here.

## Subsequent composed matrix and recovered reference boundary

The distinct, previously sealed fourth-order reference must not be confused with
the failed original second-order matrices. The November 13 cohort archive
`a3d9afb5a9d209500fb9ffdbed2292224ca4dd6501eac6a878c15d3df1abf019`
already contains a genuine four-point gradient stencil and independent modes.
A read-only reauthentication verifies that archive's 344 members, all 108
saved-state input hashes, the exact original stencil, and the unchanged gates.
Raw asymmetry/translation are 1.08379e-8/1.92923e-8, and the reference's single
imaginary frequency is -256.18589345 cm^-1. No new stencil, SCF, extrapolation,
transpose repair, or reference/native cross-seeding runs. The failed original
second-order receipt remains failed. The reference matrix/modes are available
for a future **complete** native comparison; their absence is not a remaining
blocker for this particular I–CH3–I saved frame.

The first complete current-source native composed request freshly revalidates
F[D], prepares its own first/frozen sources, and requests all 18 independently
solved coordinate columns. No retained numeric component is injected. It
terminates with exit 124 at its unchanged 1800-second cap (whole wall 1800.03 s,
peak 440684 KiB), after returning 16 columns. No energy/gradient/raw-H panel is
published; native acceptance and the independent validator remain false.

This request supplies actual phase evidence, not an extrapolated prefix:
physical revalidation 6.63749 s; coordinate first sources 192.28832 s; frozen
matrix 472.96020 s, including 460.49837 s of traced integral ERI and 12.46016 s
of one-electron work. The 16 completed columns consume 1127.60941 s: 384
returned GMRES actions total 574.02025 s and the measured non-GMRES remainder
is 553.58916 s. That remainder includes multiple composed sources/assembly and
is not attributed to an individual leaf without another bounded diagnosis.

There is a separate scientific failure: coordinate 5 (carbon z) reports solver
status `converged` under the default relative stopping rule but physical
residual 2.79357417864e-10, above the original independent absolute 2e-10 gate.
Neither that solver label nor the 16 completed diagnostics qualifies the full
response, matrix, or modes. Future qualification must request tighter solving
and retain the original absolute physical acceptance check; raising the gate
or merely extending the whole-request timeout is not a fix. Preserve these
negative results and do not rerun this unchanged full request.

The terminal request, passing fourth-order reauthentication, explicit negative
matrix validation, and phase/response-gate audit live under
`build/qualification/host-saved-physical-matrix-20261114`. Production source and
the two public ECP saddle guards are unchanged. The complete issues remain open.

## Bounded carbon-z response and separate source attribution

A distinct component experiment investigates the coordinate-5 residual failure
and the previously unattributed per-column work, rather than repeating the
timed-out complete request. It freshly validates F[D] at the original strict
state limits, prepares its own first-source coordinate panel, and runs exactly
one actual `HostKsNuclearResponse` solve for carbon z. Relative stopping is
disabled and absolute GMRES tolerance is 1e-12; the independent physical residual
acceptance remains the original 2e-10. It returns 8.40601e-14 after 14 iterations
and 28 GMRES actions. This is a stricter diagnostic invocation, not a production
default change or proof that every coordinate passes the original absolute gate.

The complete bounded experiment takes 249.53057 s within its declared 600 s cap
and peaks at 437888 KiB. First-source preparation is 170.25870 s; the actual
coupled response is 52.03319 s, including 40.47935 s of returned GMRES actions
and an 11.55384 s non-GMRES source/assembly remainder. With the actual newly
solved D'/W', separately measured native sources take:

- integral density-response contraction: 0.00621 s;
- moving-grid XC column: 18.06809 s;
- residual ECP column: 2.46861 s;
- uncut native BJ column: 0.00507 s.

Only the measured owners are attributed. The remaining 11.55384 s inside the
coupled solve is still not assigned to its individual first-source/assembly
leaves. These are one-direction component measurements, not a complete-endpoint
speedup comparison or a measurement of every molecular coordinate.

The response numeric ceiling remains 128 MiB, aggregate ceiling 512 MiB, with
342467888 bytes conservatively reserved. No frozen integral matrix is evaluated
or injected, no total HVP/raw Hessian is assembled, and no native/reference SCF
is restarted. Both full D' and full occupied-energy-matrix W' are retained along
with source panels, exact probe/library/source bindings, phase logs, compiler
cache statistics, the initial compile failure, and passing component validation.

Read-only diagnostic comparisons use the six already audited independent
carbon-z states. D' errors at 0.003/0.001/0.0003 Bohr are
1.82118e-6/2.00239e-7/1.22786e-7; W' errors are
3.48988e-7/3.90448e-8/3.28839e-7. No new derivative-comparison acceptance gate is
invented, and the finest W' result is not falsely described as monotone
finite-difference convergence. These diagnostics do not qualify a complete
independent response/Hessian or modes.

Evidence is under
`build/qualification/host-saved-carbon-response-phases-20261114`, with the
previous complete-request negative and its exact source/state/reference
dependencies nested in the retained archive. Future optimization can now target
a measured moving-XC column or first-source remainder with bounded component
inputs, rather than blindly rerunning the 1800-second request or its 473-second
frozen component. All missing full-matrix, full-endpoint, Ni/Cs/mixed, scale and
native Ni-TS gates remain open; production source and public guards are unchanged.
