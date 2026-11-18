# Evidence: public canonical scalar-ECP curvature endpoints

Status: implemented
Date: 2026-11-12

## Scope and ownership

The [exact-linked admission decision](../architecture/2026-11-12-exact-linked-b3lyp-public-ecp-admission.md)
opens CPU direct FP64 B3LYP RKS/UKS Hessian/HVP interfaces for scalar ECPs,
optionally with the qualified uncut two-body D3(BJ), ATM off. It does not open
PBE0, changed-exchange graphs, DF, automatic Libxc, CUDA full-range or ECP
molecular saddle interfaces. No first-derivative capability changes here.

Actual public/installed qualifications are retained under
`build/qualification/host-public-ecp-20261112/`. They use the pinned canonical
external real-spherical def2-SVP records, correct ionic cores, manifest
charge/multiplicity and unchanged native v4 level-3 grid. SCF gates remain
`1e-13` Hartree and `1e-11` density, and response gates `rtol=1e-11, atol=1e-12`.

## Fresh source public control

The opt-in heavy harness now distinguishes retained-batch and fresh public
endpoints without relabelling either. The new `source-cs-uks` execution passed
with actual exit zero: complete public H and two public HVP calls, each with
its own native SCF, plus a cold independent central state and 36 fresh
displaced states across three complete gradient stencils. The full scientific
driver took 438.64 seconds. This is not a pure endpoint or memory measurement.

## Isolated installed Python endpoints

The separate prefix contains the repaired wheel and its explicitly selected
minimal CPU dependencies, but no PySCF, dftd3 or CuPy. Import guards reject a
reference-engine import. Isolated Python execution excludes repository Python
paths; nonexistent compiler/search paths cannot silently run a JIT fallback.

Each installed case executes a supplemental cold native central-state export,
then fresh complete public H and two fresh HVP calls. The public calls include
their own SCFs. Central densities/occupations and total energies match sealed
independent inputs. Raw matrices and HVPs pass the original symmetry,
translation, bilinear and three independent-gradient-stencil gates. Installed
qualification **reuses** sealed independent states/matrices; it does not claim
new independent SCFs. The accepted root audits retain actual displaced states
and per-spin occupied-subspace checks, not just convergence flags.

Accepted complete-call H/HVP0/HVP1 times in seconds:

| Case | Core | H | HVP0 | HVP1 |
| --- | --- | --- | --- | --- |
| CsH RKS | Cs46 | 160.17 | 29.63 | 29.61 |
| CsH+ UKS | Cs46 | 259.92 | 47.11 | 47.19 |
| IH RKS | I28 | 865.29 | 147.12 | 146.96 |
| nonaxial IH2 UKS | I28 | 2110.15 | 251.44 | 251.90 |

All four installed raw matrices match the corresponding immutable native
matrix bit-for-bit. Their largest raw errors at `0.003/0.001/0.0003` Bohr are,
respectively, Cs RKS `7.067e-7/7.320e-8/9.015e-9`, Cs UKS
`5.296e-7/5.886e-8/5.299e-9`, and I RKS
`1.393e-6/1.588e-7/3.023e-8`. I UKS errors are
`5.608e-6/6.240e-7/6.169e-8`, below the same unchanged `1e-5` gate. Its
independent root audit retains 54 actual displaced states; the fresh installed
central density/energy errors are `1.852e-10` and `1.080e-12`. The full installed
matrix's response residual is `6.154e-12`, and its first-plus-curvature nuclear
work is `95,817,582 + 95,817,582` visits under the explicit 300-million cap.
These timings are
not a speedup or practical-scale claim. The scientific Python drivers also
read and authenticate parent archives, so their process high-water RSS must
not be presented as electronic-endpoint-only memory.

The installed numerical cohort's source identity is
`11dc4348962edbbf38aecf17ebcecafedaec00984843d8ae5a64a510ebc9db20`,
with repaired core SHA256
`b903e6e7709566437be4c9dc12398baa521dfca47f0fdd7c9fdb443fc8c6340a`.
Initial installed driver attempts failed on an incorrect reference-array key
before any public H call. They remain failed; distinct `installed-v2` attempts
use the sealed total-energy field without changing any numerical gate.

## Installed standalone CLI

A subsequent repaired wheel exposes the already-produced native one-electron,
XC and ECP work counters in CLI JSON, alongside nuclear ERI, response and D3
work and selected finite work caps. Its source identity is
`9f648efbe1d578fe5c7a5ecae11ce6ede0175bf1872cdb70b59e01382ac47656`,
with repaired core SHA256
`f3f2927c1c897af1a3304097b6caf5507cbe01302009d6b75500be0f58d89b55`.
The source identity is separately checked against the current production tree.
This does not restamp the earlier wheel's results as executions of a newer binary.

Its actual native ELF passes CsH RKS full H and two HVP calls without any Python
path, against the independently root-audited three full reference matrices.
The complete calls take `159.81/29.58/29.55` seconds; their raw matrix also
matches the installed Python result exactly. The full matrix reports
`4,398,324 + 4,398,324` first/curvature nuclear primitive visits, 164 induced
Fock actions, 161640 ECP kernel blocks and 18 D3 pair visits. Nuclear counts
exclude the integrals inside induced Fock actions. These counters do not imply
complete CPU instruction counts.

Analytic Gaussian ECP work uses kernel blocks, not radial quadrature. Its
quadrature-only radial-layer/AO-point/contraction counters are correctly zero.
The first CLI audit incorrectly required them to be positive and remains a
failed audit; a distinct v2 audit checks nonnegativity and positive kernel work,
without rerunning or changing the successful endpoint outputs.

## Remaining boundaries

Each physical spin/charge composition has its own terminal installed receipt;
I UKS acceptance is not inferred from I RKS, source/retained-batch I UKS or
central-state agreement. Ni full matrices, independent matched roots, installed Ni, ECP TS
integration, complete 25--40-atom electronic work/time/memory and the requested
exact-level Ni saddle/frequencies remain separate requirements. No issue or
PR is declared complete by this evidence, and no GPU or publishing operation
is performed.

Legacy admission tests also migrate to the exact native exception. Their
original negative B3LYP/ECP and B3LYP-UKS assumptions remain captured in the
failed initial run. Current negative controls still refuse PBE0 UKS, DF,
unqualified angular/precision domains and ECPs on noncanonical graphs. Native
resource-failure/recovery tests check the native receipt and actual raw outputs,
not obsolete private-frame hash/shape diagnostic fields. Independent electronic
state/root coverage remains in the scientific controls rather than being
replaced with an implementation fingerprint or compatibility alias.
