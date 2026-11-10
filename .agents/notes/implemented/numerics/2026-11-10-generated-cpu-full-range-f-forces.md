# Decision: reuse the shared Cartesian recurrence for native CPU f-shell forces

Status: implemented
Date: 2026-11-10

## Problem

The requested spherical def2-SVP Ni doublets include f polarization. A native
energy can converge while the prepared Direct force capability rejects those
shells and the bounded Python diagnostic rejects angular momentum above d.
Removing only the resource guard would not provide their derivatives. Expanding
the Cartesian component-specific s/p/d AOT table through f would multiply the
number of component programs without reusing the existing through-f recurrence.

## Decision

Package a full-range specialization of the compiler-owned Cartesian recurrence
already used for CPU/CUDA range exchange. Preserve the existing component AOT
programs for s/p/d quartets. Native shell contraction uses the new specialization
only when a quartet contains an f component; normalization, public spherical
expansion, signed weights, independent slot ordering and J/K channel sharing stay
in their existing owners. No production PySCF, finite differences, runtime JIT,
or fallback to the independent host recurrence is added.

The generated primitive wrapper validates finite centers, positive exponents and
the through-f domain before publishing anything. Three forward-AD center seeds
recover the fourth center by translation. Work receipts count all three
recurrence passes rather than recording one evaluation for that primitive.
Both explicit weight channels still share those passes.

The complete nuclear source skips a shell quartet whose four slots move with the
same physical atom. Its integral is translation-invariant, irrespective of shell
or AO labels. This shortcut is not valid for independent-slot derivatives or
orbital/basis response, so it is confined to the physical first-force contraction.
Resource admission, exact/one-byte-short refusal and final-state checks remain.

The existing bounded diagnostic remains s/p/d; its limits are not enlarged to
make a missing native provider appear supported. Unavailable compiled artifacts
and g+ shells retain explicit refusal. Failed SCF states are never consumed.

## Evidence and scope

- The actual generated primitive is compiled and compared with independent
  libcint analytic Cartesian derivatives for all four slots, f/f quartets,
  coincident/separated centers and tight/diffuse exponents. Independent multistep
  value differences additionally check nuclear signs. Invalid/numerically failed
  calls retain caller-owned sentinel outputs: 36 tests.
- Complete default-grid B3LYP/VWN-RPA RKS H4 and UKS H3 endpoints include an
  explicit f shell in Cartesian and real-spherical representations. Independent
  cold SCF must converge and pass orbital-gradient, density, energy and full
  moving-grid force gates. The test forbids the Python scientific force route:
  four tests pass without native/oracle density seeding.
- Native sparse spherical/Cartesian f-shell channel tests compare the compiled
  owner with the independent host recurrence, including high-order f/f cases.
  Existing source, capacity and MP2 fallback tests retain their numeric checks;
  the unsupported-artifact control moves from f to g.
- The slow asymmetric NiH3 neutral-doublet test passes against the newly
  installed wheel, requiring a separately converged atomic-guess oracle and
  complete moving-grid force error below `2e-8` Eh/Bohr. It explicitly forbids
  the Python diagnostic and retains the public UKS curvature refusal.

These are CPU force qualifications, not installed requested-workflow Ni
Hessian/HVP acceptance. The small one-primitive RKS H2 probe failed native SCF
before a usable energy/state was published, with or without automatic guesses;
no derivative oracle or success claim is inferred from it. H4 supplies the
independently converged restricted qualification fixture. No universal SCF
convergence claim is made for arbitrary through-f inputs.

## CI follow-up

The earlier diagnostic-grid extension left the native projector rejection test
at historical CPU limits 512/96. It now independently exercises the larger CPU
limits, retains both checked production grids, and refuses 1537/193 before
changing outputs. CUDA limits and ECP numerical gates remain unchanged.

The Libxc checker also requires a current source-bound catalog snapshot after
native/compiler changes. Refresh it with the official writer only after checking
that the complete retained summary is unchanged. Historical measurements remain
stale; refreshing input hashes is not a new Libxc endpoint qualification.

## Remaining work

Complete default-grid Ni UKS raw Hessian/HVP and displaced-state checks,
mixed Ni/I and Ni/Cs ECP accuracy and scalability, both-spin ECP curvature,
total-method D3 composition over those qualified domains, and the fixed-method
molecular TS API/CLI remain separate acceptance obligations (#2844–#2848).

## NiH diagnostic follow-up (2026-11-11)

The unchanged neutral-doublet NiH fixture at Ni `(0.01,-0.02,0.03)` and H
`(0.17,0.11,3.1)` Bohr now completes one cold native B3LYP/default-grid
energy-and-force request with the canonical spherical def2-SVP records. It
retains the earlier energy `-1508.6876441475483` Eh, 198 SCF iterations and
physical residual `8.6972588701e-14`. The Ni force is
`(0.00163816316033, 0.00133068371071, 0.0313601033966)` Eh/Bohr; the H force is
opposite within floating-point reduction error. This resolves its former missing
f-shell force route, not its independent electronic-state/curvature acceptance.

The diagnostic takes 280.18 seconds from basis/preparation through force
publication; process peak RSS is 378724 KiB. These are one energy/force-request
and 198-SCF-iteration diagnostic observations, not a speedup, work-boundedness
or complete requested-workflow performance qualification. The scientific receipt
is based on the working patch before formatting/final packaging, not a claim
about a future installed binary.

A further independent atomic-guess Newton attempt tightens the orbital substeps
(`ah_start_tol=1e-12`, `ah_conv_tol=1e-14`, `max_cycle_inner=50`, `kf_interval=1`,
180 macro cycles). It remains unconverged and reaches a different root, with
energy error `0.56527` Eh and density error `1.0286`. It is rejected; neither
native nor oracle densities seed the other implementation. The earlier failed
state-match evidence remains applicable, and no Hessian oracle is constructed
from this retry.

## Nondegenerate Ni pilot (2026-11-11)

Do not repeatedly use the nearly axial NiH or nearly symmetric original NiH3
as the only possible state fixture. The requested Ni-doublet domain also permits
an asymmetric neutral NiH3 at Ni `(0.01,-0.02,0.03)`, H `(1.8,0.1,0.03)`,
H `(-0.5,2.4,0.3)` and H `(-1.7,-0.8,1.1)` Bohr. With the unchanged pinned
real-spherical def2-SVP, B3LYP/VWN-RPA and native default grid, Direct/no DF,
this candidate independently matches from two cold starts. Production uses its
own initial guess; the independent oracle uses its own atomic guess. Neither
implementation receives the other's density or orbitals.

The installed native energy is `-1509.5720618893429` Eh after 123 iterations,
with physical residual `1.2572e-13`. Independent atomic-guess UKS reaches
orbital-gradient norm `8.2232e-11`, maximum density error `1.8451e-11`, and
energy error `4.3201e-12` Eh. The frozen pilot state gates are `2e-9`, `2e-7`
and `2e-8` Eh respectively. There are 46 spherical AOs and `(16,15)` occupied
spin orbitals; the independent frontier gaps are `0.13036`/`0.16465` Eh.
Frontier gaps alone are not a response-stability proof.

The pilot's native energy request takes 109.19 seconds and state matching
finishes at 127.08 seconds; these are not a complete force/Hessian/TS timing
or a speedup claim. The installed native library SHA256 is
`3cd6419275e937d75daa2c3304b3d87e9bae0681dc0686cd1ed9dd50c589857a` from the
working source patch on parent `10bd0290082c85838d627031ff4244056dca4608`.
This is one accepted electronic-state pilot, not UKS curvature promotion.
The failed original NiH/NiH3 fixtures remain visible. The slow force test now
retains this candidate and an explicit independent atomic-guess protocol.

The installed Ni force test also passes: one native cold SCF/force request,
one independent cold SCF and one independent moving-grid analytic gradient,
with the same frozen state gates and a `2e-8` Eh/Bohr force gate. Pytest reports
782.62 seconds; whole-process wall time is 783.82 seconds and peak RSS is
572344 KiB. These include reference/pytest work and are not native-only timing,
source-work qualification, or a speedup claim. The source-work counter semantics
are independently checked by the compiled sparse channel tests. This positive
first-force fixture does not erase the failed original state fixtures or grant
a public raw UKS Hessian/HVP capability.

## Final local validation

The full CPU CTest suite passes 141/141. Independent new primitive/endpoint
qualification totals 41 passing tests, including the slow installed Ni force
fixture. Broader installed native force, UKS low-level curvature, restricted
public Hessian/lifecycle, mixed-range force, ECP and D3 tests pass 173 distinct
cases with 36 configured skips. The first local commands omitted the repository
root needed by test-only `tools` imports; all 16 import failures pass after
correcting `PYTHONPATH`, while package and native-library imports still resolve
to the newly installed wheel. No production code is changed to fix that test
environment mistake.

The installed saddle C ABI and C99 header checks, new primitive ASan/UBSan/leak
probe, compiler/API/catalog checks, changed-file pre-commit and Sphinx
warnings-as-errors also pass. The compiler/catalog/API Python checks pass 112
tests and 25 subtests. Builds use verified sccache 0.16.0 with retained pre/post
statistics; unrelated shared-cache activity prevents attributing global hit
counts to this patch. These are CPU-only checks, not real-device GPU evidence.
