# Decision: shared GFN2 moment-origin primal and automatic pullback

Status: implemented; complete GFN2 endpoint qualification pending
Date: 2026-11-07

For the subsequent ownership census and source-migration closure, see
`2026-11-08-derivative-retirement-closure.md`. Earlier qualification states
below describe their original experiments, not the final campaign.

## Problem and decision

Primitive S/D/Q values and ket-coordinate derivatives already use the compiler
Hermite Graph, but CPU/CUDA still duplicate the ket-to-bra origin translation
and its handwritten weighted pullback. This is representation mathematics, not
just matrix scatter. It must not be omitted from the derivative ownership audit.

Extend `integral.gfn2_sdq` with the physical dipole and traceless-quadrupole
translation primal, using the existing xx,xy,yy,xz,yz,zz ordering. Differentiate
its weighted objective through Graph with respect to overlap, local dipole and
displacement. Emit both value and additive VJP helpers in the existing host and
CUDA S/D/Q artifacts; migrate native consumers directly and delete the old
pullback helpers, without aliases. Native contraction retains the identity
quadrupole-adjoint passthrough, spherical transforms, shell traversal, screening,
finite checks, fixed reduction order and force signs. Do not move those resource
or representation-order contracts into a second scheduler.

## Evidence and limits

The generated host helper matches an independent Cartesian-matrix translation
and complex-step differentiation of that primal over dense off-diagonal weights,
nonzero additive seeds, zero displacement and displacement scales 1e-4/1/1e4.
The primitive s/p/d Cartesian quadrature and coordinate finite-difference gates
remain intact. Native probes use verified compiler-cache launchers. Device
helper tests and actual CUDA translation-unit compilation are separate gates;
full molecular energy/force oracle and no-regression endpoint comparisons remain
required before the repository-wide derivative-retirement request is complete.

The host/device origin-shift gate has since passed all eight parameterized
cases on node1 (main/gpu:5090), with the same matrix/complex-step oracle and
additive seeds. Actual `gfn2_integrals.cu` also compiled for sm120 against the
emitted S/D/Q and H0 headers. These are not complete GFN2 singlepoint timing;
GFN2 does not support prepared batches, which must not be substituted for its
public endpoint.

Provenance retains the pinned upstream hash and updates the actual adapted CUDA
file hash; no source-integrity check is weakened. This change is not authority
to publish a release, and does not retire periodic or other method-response math.
