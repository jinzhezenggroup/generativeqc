# Decision: pass the composite AO reservation to its native evaluator

Status: implemented (bounded allocation fix; Ni Hessian qualification incomplete)
Date: 2026-11-04

## Problem and decision

The first real-spherical external def2-SVP Ni/H trial passed SCF state and
basis matching, then failed before HVP execution: the third-order native AO
evaluation required 94,392,480 numeric bytes but inherited an unrelated
64 MiB default. The composite caller had admitted a 512 MiB grid budget and
already excluded an AO reserve from its scalar-grid allowance, but never
passed that reserve to `NativeAO.evaluate`.

Include native basis storage and point/publication buffers in that reserve
and lend it explicitly to the native evaluator. Keep it excluded from the
remaining grid budget. Do not raise the global AO default, silently allocate
without admission or borrow scalar-grid capacity twice. A contract regression
uses a conservatively enlarged synthetic basis capacity, verifies the same
native grid result when sufficiently funded and rejects the smaller budget
before native evaluation. It is not a large-allocation or Ni Hessian receipt.

## Scientific evidence and remaining boundaries

The offline Ni/H def2-SVP fixture is generated from source data whose five
files were compared byte-for-byte with the primary BSE repository at
`4adaf1372c7101620ca1a9f3130be9ae97fb8f30`. The generator verifies those input
hashes and the retained BSD-3-Clause license. Its payload checksum protects a
canonical external scientific input, not locally generated production source.
An independent PySCF comparison checks the actual spherical d/f AO values and
first/second spatial jets; the 51-dimensional spherical space is not replaced
by the 56-dimensional Cartesian space.

The NiH4(2-) singlet cold-start probe matched independent B3LYP/VWN-RPA energy
and overlap on the same explicit quadrature. One density trial passed a
2e-9 gate, but another looser-converged reference trial missed it; these
exploratory results are not a complete or universally matched state receipt.
Tighten convergence and retain all failed trials before accepting derivatives.

The trial also exposed a distinct boundary: public CPU Direct KS stationary forces still
use an s/p/d component-dispatch owner with explicit 16-AO and angular/work
caps. Ni def2-SVP includes an f shell and is rejected there. Do not merely lift
those caps: a shared bounded through-f first-gradient owner and independent
complete force qualification are required. The composite Hessian first-source
owner is different and must be qualified on its own complete endpoint.

Neither AO matching, SCF convergence, response construction nor an in-progress
HVP qualifies #2843. Preserve the public method gate, installed/default-grid,
independent raw-matrix, lifecycle and scale requirements. Cold compilation,
semantic work and full endpoint time remain part of the eventual receipt;
bounded numeric storage alone is not work-bounded or a performance result.

## References

- Issues #2224 and #2843–#2849.
- `tools/generate_global_hybrid_hessian_basis.py`.
- `tests/python/test_spherical_global_hybrid_basis.py`.
- `docs/developer/dft_hessian.md`.
