# Decision: Admit real-spherical CPU LDA/PBE Hessian and HVP through the linked owner

Status: implemented
Date: 2026-11-23

## Decision

The exact bare LDA/PBE RKS point binding controls the XC closure and absence
of exchange; it does not depend on the orbital representation. Reuse the
same prepared-state token, AO packing, native integral projections and XC
maps for Cartesian and real-spherical AOs. The derivative directions are
always atom-major Cartesian coordinates. This supersedes the Cartesian-only
restriction in the [original linked semilocal decision](2026-11-22-native-cpu-semilocal-hessian.md).

Keep spherical CPU LDA/PBE exclusive to the linked owner. A package without
the exact binding, or a request for an unlinked block schedule or recycle
width, does not enter the Python science path for spherical AOs. The earlier
Cartesian fallback domain remains in place. The admission still requires
bare, all-electron, Direct FP64 RKS; this does not add fitted, corrected,
open-shell, or CUDA contexts.

The CLI dry run accepts the same explicit tensor grid controls as CPU
Hessian/HVP execution for linked registered methods, including LDA, which
does not need a generated CLI composition to own its grid. Keep the grid
restriction for ordinary energy dry runs and retain the linked method and
provider checks for derivative dry runs.

## Evidence

On base `cae2959e1`, independently reconverged PySCF moving-grid gradients
yield maximum raw Hessian differences of `2.347e-7` (LDA) and `2.524e-7`
(PBE) Hartree/Bohr² at a `3e-4` Bohr step for distorted water with spherical
def2-SVP, including oxygen d polarization. A `1e-3` Bohr step produces
larger errors, consistent with second-order finite differences. Raw symmetry,
translation nulls, dense/HVP consistency and four-direction recycled
agreement pass without displaced SCF work in the native endpoint. The tests
also forbid a Python science fallback.

The CLI budget test exercises packed real-spherical H₂ at its exact
composed host cap and refuses one byte less before the SCF trace. SDK, CLI
and Python cross-check dense, HVP and recycled results, with no Python or
toolchain in the native executables' search path. Qualification artifacts
are under `/data/jzzeng/gqc-2227-sdk-JsefEq0x/cpu-spherical-semilocal`.

Five interleaved process timings per variant on one EPYC 7K62 core included
SCF preparation, response, curvature and publication. Eight unchanged
Cartesian LDA/PBE and spherical PBE0/B3LYP-D3(BJ) controls matched energy,
derivatives and work counts exactly, with median deltas from −1.83% to +0.08%.
The newly admitted spherical water cases have no prior linked endpoint:

| Method | HVP / dense (s) | SCF iterations / Fock builds | HVP / dense response actions | HVP / dense curvature ERI visits |
| --- | ---: | ---: | ---: | ---: |
| LDA | 1.466 / 3.497 | 19 / 23 | 24 / 204 | 398,786 / 326,255 |
| PBE | 1.602 / 4.013 | 18 / 22 | 24 / 198 | 398,786 / 326,255 |

All four cases use 315,722 first-integral ERI primitive visits. These are
CPU-only receipts, not general size or CUDA qualification.
