# Decision: Compose CPU PBE UKS with native two-body D3(BJ) curvature

Status: implemented
Date: 2026-11-23

## Decision

Resolve the exact named PBE-D3(BJ) UKS total MethodIR to the existing
spin-polarized PBE point binding and geometry-only D3 owner. Both the HVP and
scalar-column Hessian add the correction exactly once. The correction neither
changes the converged density nor requests another electronic response solve.
The SDK, CLI, and Python keep the same prepared owner and charge for the
correction workspace before SCF; there is no fallback to a bare electronic
result or a Python second-derivative implementation.

This admits uncut two-body D3(BJ), ATM off, only for the exact named
all-electron Direct FP64 CPU program in Cartesian and real-spherical AOs.
LDA remains bare, and D3(0), ATM, switched/cutoff D3, D4, DF, ECP, and UKS
recycled columns do not inherit admission. The owner already composes the
same geometry-only curvature for RKS; duplicating or spin-scaling that
operation would produce an incorrect total.

## Evidence

The independent test reconverges PySCF on the moving discrete quadrature and
adds s-dftd3 1.4.0 gradients. For the distorted UKS water cation, the maximum
raw total Hessian errors at a `1e-4` Bohr step were `1.70e-8` and `1.17e-7`
Hartree/Bohr² (Cartesian STO-3G and spherical def2-SVP), below the unchanged
`3e-7` gate. The spherical cation's error was `3.09e-7` at `3e-4` Bohr;
using the finer step resolved the finite-difference error without relaxing the
gate. The correction-only fresh HVP agreed within `2.1e-15` Hartree/Bohr²
against the independent D3 gradient difference for both representations.
The RKS controls, energy and gradients, raw symmetry, and HVP-to-matrix
agreement also pass. Installed SDK/CLI/Python parity, host-capacity boundaries,
moved/replayed geometry and all 152 CPU CTest cases pass.
An independent H₂⁺ check also recovered all six raw columns for LDA, PBE,
and PBE-D3(BJ) with an identically zero reference beta density. Its maximum
Hessian error was `3.75e-9` Hartree/Bohr² (`check_empty_spin.py`).

In five measured samples after one warm-up, the full spherical STO-3G
water-cation endpoint medians for bare/corrected UKS were `0.445/0.452 s`
for HVP and `2.488/2.491 s` for dense Hessian. Both did 16 SCF Fock builds,
26/222 induced Fock actions and 56,160/479,520 XC response point visits;
correction work was 9/81 D3 pair visits with no displaced SCF. The retained
reservation increased by 11,548 bytes in each case. Paired before/after
measurements on the existing D3 RKS path found unchanged work and values;
median HVP/dense times changed by `-0.28%/-0.33%`. This is CPU endpoint
evidence, not CUDA qualification. Reproduction commands, raw measurements,
and installed-binary checksums are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/native-pbe-d3-uks`.
