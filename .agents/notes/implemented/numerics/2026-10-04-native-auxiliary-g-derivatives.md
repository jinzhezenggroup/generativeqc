# Decision: native auxiliary-g source and weighted nuclear derivatives

Status: implemented
Date: 2026-10-04

## Problem

The physical DF source used by the 230-AO ethane and 264-AO benzene CC work
contains auxiliary g shells. Its values already work, but the native source
rejected derivatives. The generic weighted derivative consumer also packed only
three Cartesian terms per public AO; spherical g can require six. Merely removing
the source rejection or increasing a local array would therefore be unsafe.

## Decision

Bind the explicit auxiliary-g derivative emitter from #1804 to the existing
runtime product traversal. The adapter copies response fields into the existing
policy ABI; it does not introduce a second recurrence. Lower angular classes
still call the original evaluator. The g evaluator admits the internally raised
orbital power needed by f/f/g derivatives and owns F0–F11 work arrays.

Raw A, fixed-transform A, and metric derivative tile APIs admit auxiliary g with
the existing coordinate/batch guards. The source's immutable angular flag still
restricts its value-math policy. Orbital g and auxiliary h remain outside this
source domain. Legacy bulk exporters and method force capabilities retain their
separate through-f admission.

The standalone full-weight and strided-tile bridges select an explicit expansion
stride: three for through-f calls, six when auxiliary g occurs. Both orbital and
auxiliary metadata use that same stride, including lower-angular shells sharing
a g-containing basis. A launch cannot infer the stride from a component's angular
degree. Host admission includes the larger metadata and checks retained capacity.

The weighted traversal contracts all center channels for each A or M element in
one visit. Weights address full unit-weight `[mu,nu,P]` and `[P,Q]` arrays; no
triangular doubling is implied. Shared atoms collect all their center channels,
including atoms carrying only auxiliary functions. Serial and cooperative
schedules share the generated polynomial definition. No coordinate-sized
integral derivative tensor is materialized by these production consumers.

## Rejected alternatives

- Globally increase the legacy SCF expansion stride: existing callers construct
  three-term metadata, so this would silently change unrelated launch contracts.
- Evaluate each nuclear coordinate separately to consume the final weights:
  bounded storage would still repeat integral work in proportion to atom count.
- Promote public CC forces from source derivatives alone: orbital/Z, overlap and
  complete-endpoint stationarity must still be composed and qualified.

## Evidence

The independent test tier uses normalized libcint Cartesian integrals/center
responses with independent public-basis transformations. Fixtures include
signed contractions, orbital f/auxiliary g, Cartesian/spherical/mixed bases,
permuted shell ordering, shared and auxiliary-only atoms, ragged tiles, and two
batch items with different geometry and metadata offsets. A dense fixed
transform checks derivative auxiliary indexing independently of the metric VJP.
The weighted tests also compare a nuclear direction with recomputed independent
values at two step sizes and reject insufficient budgets/nonfinite weights.

Acceptance gates are `atol=2e-9, rtol=2e-11` for native derivative arrays and
weighted contractions, `2e-10` for translational cancellation, and
`atol=3e-8, rtol=3e-7` for the independent nuclear finite differences. These
small-fixture gates do not qualify the separate strict large-factor gates.

Host multi-translation-unit policy linkage and all static hooks pass. Slurm job
12212 passed all 14 native tests in 99.23 seconds on the RTX 5090, including the
full-weight serial and cooperative schedules. The qualified library SHA256 is
`d0487331f52e8777d6d3d2c508d0729837a909c5dda31ad636a9d89d3d906eb2`.
Slurm job 12215 passed all 53 new/through-f regression cases under memcheck
(`--target-processes all`), including complete RHF/UHF force regressions, in
1022.63 seconds with zero reported errors. The numerical gates and full-weight
serial/cooperative cases are unchanged. Ignored artifacts live under
`.artifacts/native-g-response/`.

Two earlier attempts stopped at validation-harness admission, before invoking
the new kernels: the public Python CUDA orbital gate and then the general CUDA
system-construction gate each retain their through-f boundary. The validation
bridge can explicitly construct/normalize metadata under a CPU context and then
invoke the standalone weighted function under its separate CUDA context. That
option performs no CPU value, derivative or SCF evaluation. Do not widen public
method admission merely to construct a test's auxiliary metadata.

## Consequences and remaining work

The native primitives needed for a DF nuclear sink are available, but the
standalone bridge still consumes host spans. The physical source response should
feed a persistent sink on its existing stream, with combined live-owner admission
and publication only after successful final drain. Full DF-CCSD(T) forces remain
unfinished until that composition, orbital/Z response, Pulay response, and
independent complete-force gates pass. No complete endpoint speedup is claimed.
