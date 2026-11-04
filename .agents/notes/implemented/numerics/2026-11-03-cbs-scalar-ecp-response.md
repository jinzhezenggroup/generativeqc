# Decision: share the scalar ECP Hamiltonian across correlated CBS endpoints

Status: implemented
Date: 2026-11-03

## Problem

Open-shell CBS already composes native unrestricted energies, Lambda/triples,
orbital response and conventional/fitted forces. ECP admission was still closed,
although the common reference and CPU weighted one-electron derivative providers
already include the checked scalar local/nonlocal residual. The generic CUDA
weighted one-electron response evaluated only ionic Coulomb attraction; its
stationary HF paired consumer already added ECP derivatives separately.

## Decision

Admit scalar ECP references in native MP2/UMP2 and restricted/unrestricted
CCSD/(T). Reuse the existing checked providers, physical reference identities,
generated correlation/response programs and ionic nuclear repulsion. Add the
missing residual to the generic CUDA potential cotangent using the existing
device ECP sink with derivative scale -1: the sink normally produces forces,
whereas the weighted response returns positive nuclear gradients. Pure overlap
or zero-potential cotangents need no ECP quadrature. Do not add another ECP
recurrence, stage integrals through a reference package, or retry CUDA on CPU.

Reserve a conservative ECP numeric workspace alongside reference capacity before
SCF. Derive the envelope from the generated refined grid and radial schedule;
include checked-grid matrix/coordinate panels, host/device grids, staging and
primitive/term metadata. Retain the existing radial allocation fallback and
two-grid numerical acceptance gate. This reserve is not a claim of exact RSS.

CBS permits ECP orbital snapshots only when every actual atom has the same
nuclear charge, removed core and decoded potential parameters in both MP2 bases
and the correction basis. Check before native preparation; unused elements and
JSON whitespace are irrelevant. Basis identity still includes the original ECP
parameters for artifact invalidation. Auxiliary fitting affects only the
electron interaction; it must not replace the ECP Hamiltonian. An ECP core is
not permission to freeze additional valence orbitals.

Restricted fitted response previously rejected ECPs in the RHF frame owner and
its optional recycle identity. Replace that admission boundary only alongside
an exact scalar-Hamiltonian identity: ionic populations and every ECP atom,
channel, radial power, exponent and coefficient are bound bitwise, in addition
to basis/geometry and physical frame matrices. Unrestricted exported references
use the same stronger source binding. Coulomb-only `BasisGeometryIdentity`
intentionally remains independent of ECP potential parameters; it must not be
misrepresented as an identity for the full one-electron operator. Reserve the
new payload before preparing a response accelerator, and clear stale images
on potential changes even if the caller supplies unchanged orbital matrices.

## Rejected alternatives

- Removing only admission gates would leave CUDA correlated forces missing the
  ECP residual response even though stationary HF forces were correct.
- Comparing only core counts would extrapolate between different potentials.
- Replacing ECP bases with all-electron snapshots or approximating forces by
  numerical differences would change the requested production model.
- Enabling standalone HF/DF or other ECP methods without their own admission and
  complete endpoint qualification is outside this change.

## Evidence and invariants

`tests/python/test_cbs_ecp_native.py` reconstructs independent PySCF/libcint
Hamiltonians from primitive AO/ECP inputs and reconverges displaced references.
The Na/H fixture uses the physical LANL2DZ Na potential with compact test-only
valence bases: it is not a systematic CBS sequence. Coverage includes neutral
doublets with nonzero triples, charged restricted singlets, empty-beta triplets,
HF/MP2 companions, standalone MP2 and complete CBS forces. Conventional and
fitted response must retain independent energy/force, translational-sum and
failure/recovery gates. `tests/python/test_ccsdt_cbs.py` independently protects
potential/core/all-electron mismatch rejection before execution without PySCF.
Native RHF identity/recycle tests also reject individual ECP field changes and
discard previously solved images. Energy/force acceptance is at most 4e-8
Hartree / 4e-7 Hartree/bohr, with translational sums bounded by 3e-9 Hartree/bohr.

Real-device qualification must opt into `GENERATIVEQC_CBS_ECP_CUDA_TEST=1` and run
through finite-time Slurm on an allowed node; node3 remains excluded. Production
uses no PySCF and no finite-difference force. No performance claim follows from
these small scientific endpoint fixtures.

The November 3 qualification composes dependency #2944 at
`34bb3a2c8c2e039470bbf9b77597f180e739b666`. The combined CPU CBS/ECP/public
correlation regression passes 224 cases (46 skips), with 47 additional host
reference/capacity/frame contracts and all four selected native CPU tests.
Slurm job 1477 on node4 (`main`, one RTX 5090, finite 30-minute allocation)
passes all 30 selected ECP/parent regressions: 16 ECP CUDA cases, 12 parent CUDA
cases and two CPU fitted cross-checks. The native RHF generated-preconditioner,
complete ECP identity and exact-budget executable also passes in that allocation.
The CUDA 12.9/sm_120 development build uses verified sccache 0.16.0,
`GENERATIVEQC_CUDA_FAST_COMPILE=ON` and `GENERATIVEQC_ENABLE_AOT_SHELLS=OFF`;
its shared-library SHA-256 is
`727f6921d1535aa1e6f4984d3f2704397e6fd059a0969ed488112c41fba3f1a5`.
These settings qualify correctness, not release performance or exact RSS.

## Revisit when

Adding spin-orbit ECPs, independently qualified new angular/element families,
frozen-core transport or a changed generated grid/staging contract requires new
scientific and resource qualification, not relaxation of these identity gates.

## References

- Builds on PR #2944 and the unrestricted analytic CUDA/DF response decision.
- Current contracts: `docs/user/ccsdt_cbs.md`, `docs/user/ecp.md`, and
  `docs/user/unrestricted_cc.md`.
