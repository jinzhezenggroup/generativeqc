# Decision: discover the selected Python environment's OpenBLAS metadata

Status: implemented; GPU endpoint performance validation pending
Date: 2026-11-01

## Problem

A standalone native build can have the project's OpenBLAS build dependency in
its Python environment while CMake cannot find that environment as a library
prefix. The build then selects scalar CPU linear algebra. GPU execution still
uses CPU initialization; A100 sampled stacks identified Hückel eigen and dense
products as some of that preparation cost. Those samples diagnose work and do
not establish an endpoint speedup.

## Decision

After ordinary pkg-config and CMake discovery fail, query the selected Python
interpreter's distribution metadata for the installed LP64 package's CMake
configuration. Do not import `scipy_openblas32`, which loads a provider into the
process, and do not search unrelated Python environments or download a library.
Respect explicit scalar selection, disabled package discovery and target/host
separation during cross-compilation. Existing discovered providers retain
priority and all real symbol/link probes remain authoritative.

## Thread-control boundary

Discovery is not runtime admission. The benchmark environment's
`scipy-openblas32` 0.3.34.237.2 has global thread control but no local setter;
task-parallel automatic execution correctly remains scalar. The repository's
pinned 0.3.34.0.0 Linux x86_64 wheel, verified against the existing dependency
manifest, exports `openblas_set_num_threads_local` despite declaring a prefixed
spelling. The existing unprefixed-symbol probe recognizes that ABI. Do not
weaken task-parallel safety or silently substitute global thread switching.

The pinned wheel was extracted into an isolated experiment directory. Existing
benchmark environments were not upgraded or replaced. Any endpoint result must
retain the selected provider, binary/source identities and scientific gates;
provider discovery alone is not evidence that GPU4PySCF has been surpassed.

## Verification

The discovery regression uses synthetic distribution metadata whose package
raises on import. It exercises automatic selection, explicit provider priority,
scalar selection, package-disable and cross-compilation. A separate fixture
links and executes the production CPU linear-algebra contract against an actual
installed provider. Existing capability/cache tests retain negative transitions
and independent current links. Local pinned-provider execution: 29 passed;
28 pkg-config cases skipped because this host does not provide pkg-config.

## Revisit when

The Python dependency layout changes, native cross-builds need explicit target
provider metadata, or the existing thread-control admission policy changes with
independent evidence.
