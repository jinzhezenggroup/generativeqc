# Decision: construct coupled nuclear response around one native grid/frame owner

Status: implemented (development consumer; endpoint qualification is separate)
Date: 2026-11-12

## Problem

Separate overlap/integral, residual ECP, moving-grid XC and abstract metric/field
CPKS components do not themselves supply an actual nuclear response. Merely
adding matrix callbacks could mix a nuclear source from one density/grid with
an induced-Fock reader from another, despite matching dimensions.

## Decision

`HostKsNuclearResponse` constructs the existing coupled response and metric
consumer internally from the exact verified state, normalized system, prepared
Fock and retained geometry. Its fixed-grid reader uses that geometry's actual
AO basis, points and weights. Borrowed-owner validators authenticate the epoch,
prescription and contents. A copied frozen density from the authenticated frame
feeds both integral and moving-XC sources; every source validation checks it
against the current frame. The residual ECP source borrows the same system.

The existing compiler-owned `method.ks_cpks` map inventory adds the per-spin
frozen-Fock source sum. Native code only binds its integral, local residual,
nonlocal residual and moving-XC panels. Ionic attraction is already in the
integral source, so no second Coulomb tail is added/subtracted. Geometry-only
nuclear repulsion and D3 do not alter electronic CPKS and remain separate.

Keep explicit source/map caps and a conservative composed owned-array bound:
all live child panels, reader staging, residual-ECP workspace, copied density,
assembly arena, transactional S/F/D/W and GMRES. Provider/borrowed state,
internal BLAS, stack, allocator overhead, RSS and total instructions/time are
not that bound. This does not remedy the ordered integral source's quartic
work amplification across nuclear directions; batching/source reuse still
needs complete endpoint qualification.

Both members of an S/F or D/W output pair publish only after final owner
validation, and D/W additionally require the fresh physical residual gate.
Failures, cap refusal and nonconvergence leave external outputs unchanged.
Avoid overlapping/reentrant calls; retain all borrowed owners until destruction.

## Rejected alternatives

- Accept unrelated nuclear and fixed-grid reader callbacks: matching matrix
  extents do not establish physical source identity.
- Contract ECP from ionic attraction alone: omits residual projectors.
- Add residual ECP to hcore twice or alter its Coulomb tail: changes chemistry.
- Populate nuclear RHS in Python or with displaced production SCF: fails the
  native ownership and analytic-source contract.
- Treat first response as a full Hessian: second source contractions and
  geometry-only terms are still required.

## Validation and boundary

`test_host_ks_nuclear_response.py` links current native sources with the exact
scalar inventory and an ordinary cold native SCF frame. The independent oracle
uses libcint/Libxc primal operators at displaced geometries and separately
reconverges KS D/W on the retained grid prescription. It does not seed the
production frame with reference-engine densities. Existing independent grid,
integral, ECP and XC component controls retain their own gates.

These compact controls are not representative atomic ECP or installed endpoint
qualification, second derivatives, a stationary Hessian, a positive saddle or
final vibrational frequencies. Their evidence must remain separate from the
previous accepted ECP source cohort and from future full-endpoint receipts.

## Revisit when

A complete native Hessian owner composes this response with all second sources,
or batched source traversal removes repeated first-integral work. Preserve exact
frame/grid ownership, bounded fallbacks and independent physical acceptance.
