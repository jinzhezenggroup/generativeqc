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
