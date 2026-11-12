# Decision: compose native D3(BJ) curvature only after qualified electronic curvature

Status: implemented
Date: 2026-11-09

## Problem and decision

The standalone GeometryIR/TensorIR D3 owner already differentiates all
two-body BJ distance, coordination-number/C6 and mixed terms. However,
`Calculator` rejected every dispersion-corrected Hessian/HVP. An electronic-only
matrix is not a B3LYP-D3(BJ) result, and correction availability cannot promote
unqualified UKS/ECP response or a native molecular TS task.

Intersect the existing CPU/direct/all-electron/full-range restricted electronic
qualification with the standalone uncut, two-body, ATM-off D3 compiler domain.
Execute one additive correction after closing the live electronic owners.
Retain electronic response diagnostics and add the correction identity/work
record and one source. Stream correction columns into a private electronic
matrix copy; only publish after every action succeeds. Never introduce another
CPKS solve, displaced production SCF, reference backend or method substitution.

Keep the first-order MethodIR serialization unchanged. Second-order executable
admission is narrower than the existing representable first-order correction,
and changing that serialization merely to advertise a new consumer would
unnecessarily invalidate otherwise identical energy/gradient artifacts.

## Invariants and alternatives

- D3 compiler validation authenticates the retained data identities and variant;
  no duplicated B3LYP parameters or second dispersion equation are introduced.
- D4, zero damping, ATM, finite cutoffs, DF, CUDA, UKS and scalar-ECP electronic
  domains cannot inherit this new public composition.
- Retained electronic arrays, correction workspace, the private total and
  immutable publication count against simultaneous correction admission.
  Corrected full-matrix output needs three matrices, not the uncorrected two.
- Discard primal SCF/dispersion result storage before correction execution.
- Failure is transactional; never recover by returning the electronic-only
  result, omitting a source, symmetrizing or synthesizing missing columns.
- Capability errors identify the physical MethodIR and its execution carrier
  separately. A PBE-named carrier does not imply that B3LYP changed to PBE.

## Independent-reference protocol

The numerical gates and exact water/STO-3G/spherical/grid fixture are in
`tests/data/d3_second_order/manifest.json`. Compare total raw columns with
independently reconverged PySCF moving-grid gradients plus simple-dftd3, using
the exact VWN-RPA component expression instead of a configurable B3LYP alias.
Standalone Ni/C/O/H correction tests retain their independent upstream oracle.

Preserve the initial three-point-reference failure: at `h=0.003 Bohr` its
full-column error was `4.806239251320399e-5`, exceeding the unchanged `1e-5`
physical gate. This highly nonstationary fixture has appreciable central-
difference truncation. Use a symmetric five-point fourth-order gradient stencil
at the same three steps; do not enlarge the physical acceptance tolerance or
change the native matrix. The original failed receipt remains part of evidence.

## Remaining boundaries

This delivery advances #2845 and improves #2846 diagnostics. It does **not**
close #2844 (public unrestricted spherical/Ni qualification), #2846 (mixed
Ni/I/Cs force and scale gates), #2847 (complete public scalar-ECP matrices), or
#2848 (native installed molecular TS API/CLI). The latter still has only an
internal mathematical saddle corrector. No 1,074-case or complete requested
transition-state workflow claim follows from this small all-electron fixture.

Revisit public domain admission only after its independent whole-endpoint
acceptance, preserving the existing numerical, state and resource guards.

## Retained evidence

Refreshing the native Libxc catalog updates only its checked source and native
input identities after the public Calculator change. The retained measurement
remains stale, its capability summary is byte-for-byte unchanged, and no native
energy/force product is promoted or newly qualified by this metadata refresh.

`benchmarks/results/d3-public-curvature-20261109/` retains the installed receipt,
compact failed-trial and test records, source identity and compiler-cache
observations. Original raw archives are ignored locally and recoverable from
Git commit `b1d20f9f6ade91f08f27c024177df3209aa10936`; no evidence-budget increase
or external publication is needed. The installed producer runs without
source-path imports or an external electronic
engine; its Python/fixture digests match the checkout. JIT-artifact replay and
fresh SCF are explicitly distinguished, and memory high-water marks are not
misrepresented as simultaneous process-tree totals.

## Low-level UKS and Cs-ECP total actions (2026-11-11)

The shared additive owner now has independent coupled-spin and scalar-ECP
qualification tests. These are new low-level composition receipts, not public
UKS/ECP admission or an extension of the earlier installed RKS qualification.
Production numerical code and the live Ni full-matrix source/binary identities
are unchanged by this test-only follow-up.

`tests/data/d3_second_order/uks_manifest.json` freezes the asymmetric H3 doublet,
occupations `(2,1)`, spherical STO-3G, explicit 8/4/8 grid, three displacement
steps and gates before execution. The native corrected SCF matches independent
PySCF energy, forces, both densities, electron counts and orbital stationarity.
The actual coupled electronic matrix/action is composed only after its owners
close. Tests reject omitted, reversed, duplicated and already-present correction
sources; raw symmetry, translation and H@v remain checked. D3 introduces no
additional electronic response solve and never changes the spin state.

For every raw column, a five-point stencil reconverges only the independent
UKS and adds independent simple-dftd3 gradients on the same moving grid. There
is no displaced production SCF even in oracle preparation. Maximum total raw
matrix errors at 0.003, 0.001 and 0.0003 Bohr are respectively
`1.602063420813593e-8`, `1.8194549603056664e-8` and
`5.6288377466018336e-8`, below the preselected 1e-5 gate. The independent HVP
gate remains 2e-6 and raw invariants 2e-8. Three tests pass in 147.06 seconds;
complete process wall time is 148.80 seconds with peak RSS 618,268 KiB.

`tests/data/d3_second_order/ecp_manifest.json` freezes the Cs-ECP RKS total
action gates. Both original CsH directions and every existing electronic
oracle threshold remain. After electronic owners close, the real additive
owner receives elemental `(55,1)`, not ECP-adjusted `(9,1)`. Independent D3
gradient differences augment the already-converged independent electronic
actions without repeating their SCFs or nuclear response solves. The original
normalized 1e-5 total-action gate is unchanged. Observed total errors at 0.003,
0.001 and 0.0003 Bohr are:

- Original direction: `7.914266203837047e-8`, `9.90284575199316e-9`,
  `4.009399084895942e-9`.
- Unscaled unit-Cs-x: `1.1228007537550424e-6`, `1.253671335030789e-7`,
  `1.2092561415266025e-8`.

Both tests pass in 1164.00 seconds. The complete two-fixture validation process
takes 1165.29 seconds with peak RSS 659,668 KiB. These warmed-artifact tests
include independent reference work and run alongside other CPU qualifications;
they are neither installed endpoint benchmarks nor paired speedup claims.
The three JUnit record-property warnings across both suites are reporting-format
warnings, not numerical failures. Raw logs, XML, process timing and compiler
cache statistics use `/tmp/gqc-uks-d3-total` and `/tmp/gqc-ecp-d3-total` prefixes.

Unrestricted ECP totals, mixed Ni/I/Cs forces/second order, a complete Cs raw
matrix, installed Ni curvature, fixed-method molecular TS and its final
frequencies remain outside these receipts. No issue is closed by these tests.

An ignored local recovery bundle at
`build/qualification/uks-ecp-d3-20261111/local-recovery.tar.gz` retains the exact
test inputs, raw logs/XML/timing/cache receipts and reference-quadrature
diagnostics. All 42 members are verified; its 473,995 bytes have SHA256
`c3d3251e83e7e8fe1d1ebd76b3e946272d409949d33700e6467c6a632b98a123`.
The input-manifest SHA256 is
`1b6d84cf5c9ea8a21eb8e1a60f3459fb1e18081e4ea71694479a8169e269c464`.
It records the actual 16b5 parent plus dirty test/note files before this
recovery-pointer paragraph, not a clean later-commit run. The native binary is
referenced by its verified identity and prior local Ni recovery archive, not
duplicated or uploaded to an external asset host.
