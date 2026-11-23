# Decision: Bind canonical CPU LDA UKS to the native stationary Hessian

Status: implemented
Date: 2026-11-23

Canonical Dirac/PW92 LDA now shares the prepared all-electron Direct FP64
stationary owner with PBE UKS. The fixed-grid action, nuclear RHS and frozen
curvature keep separate spin densities and use the density-only local point
derivative. Normalization of gradient slots does not introduce GGA response;
the local point supplies zero Cartesian-gradient coefficients. Coulomb sees
the summed spin density, with no Fock exchange.

UKS contexts without a separate composite qualification query the exact total
native binding instead of importing another list of admitted names into the
Python facade. Missing bindings still decline before SCF, and qualified
composite routes retain their previous contracts. Recycled groups, corrected
semilocal UKS, DF, ECPs and CUDA do not inherit this admission.

The independent reconverged PySCF reference checks energy, alpha/beta D/W,
energy components, forces and every raw Hessian column for distorted water
cation. Refining the same-grid central stencil from `1e-3` to `1e-4` Bohr
changed maximum errors from `2.14e-6` to `2.13e-8` for Cartesian STO-3G and
`2.56e-6` to `1.20e-7 Hartree/Bohr²` for spherical def2-SVP. The raw columns
also agree with the native HVP and the translation/symmetry gates.

Installed SDK, CLI and Python agree for spherical HeH. The native SDK covers
capacity refusal without publication, exact HVP replay, moved geometry, stale
leases and refused recycled groups, with no Python or compiler found on PATH.
CPU CTest passed 152 tests and the installed Python qualification passed 580
tests, with 18 skipped and 25 passing subtests.

Five post-warm-up paired complete endpoints preserve outputs and modeled work:
HVP/dense changes are `-0.14%/+0.06%` for LDA RKS and `+0.24%/+0.33%` for
PBE UKS. New LDA UKS HVP/dense times are `0.322/1.756 s` with 16 SCF Fock
builds, 24/216 induced Fock actions and 51,840/466,560 response point visits.
No native before timing exists for LDA UKS. Reproduction data and source/binary
receipts: `/data/jzzeng/gqc-2227-sdk-JsefEq0x/native-lda-uks`.
