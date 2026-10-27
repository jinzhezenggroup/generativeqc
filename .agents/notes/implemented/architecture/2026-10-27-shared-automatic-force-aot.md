# Decision: share automatic CUDA force consumers by execution ABI

Status: implemented
Date: 2026-10-27

## Problem

The per-functional native force package at commit
`7cfec2c98b8d47b7363b0b5c431ee8bee05f996b` emits the same geometry, AO pullback,
partition response and transport machinery for 394 automatic spin profiles.
Point mathematics differs by functional; most force machinery does not. Source
text deduplication alone does not remove repeated device kernels or compilation.

## Decision

Keep the existing native prepared owner, seven-source reduction and compiler
mathematics. Split automatic force execution into a point producer and a shared
consumer, with one bounded `DevicePointValue` tile in the existing arena and the
same borrowed stream. Group consumers by exact ingredient/spin/weight/reduction,
precision, partition and geometry contract. Registration-specific host bindings
retain method, plan, model and source identities and attach the immutable linked
point descriptor. Group those bindings into host translation units rather than
creating one CUDA or host compile task per functional.

`xc/automatic_cuda_package.py` reuses the existing automatic point emitter. It
erases only the emitter-owned function-name token for executable comparison.
Work floors, branches, spin layout, outputs, ingredient domain, point transport
and launch schedules remain in the identity. The canonical scientific identity
is retained in the registration, independently of the shared worker identity.
The unpolarized VWN 1–4 implementations provide a real reusable example. Their
polarized bodies differ. CHACHIYO and its modified variant have different work
floors and must not share a production point worker merely because an interior
formula can coincide.

A selected fused automatic profile remains an explicit build-time alternative.
The staged schedule is the compact default; complete cold/warm/moved SCF+E+F
qualification must stay within 5% median regression or select a measured fused
specialization. This is a package choice, never runtime compilation or scientific
fallback.

## Invariants

- The point producer reads only declared feature channels. Unused gradient/tau
  channels need not have been initialized by the grid owner.
- Producer errors are sticky and prevent reading stale point values. Launch
  errors are checked before the consumer can observe the intermediate.
- A staged XC value is not a prepared Becke seed. The existing cooperative
  template's `precomputed_point` path assumes phased partition response and
  skips inline Becke contraction. Ordinary staged tiles must still run point
  setup and either contract partition response or publish its phased seed.
  A complete H4/APBE force-vector requalification exposed a 0.0212945 Ha/bohr
  error when that flag was selected unconditionally; energies and densities
  were unchanged, so point-only and SCF checks could not catch the omission.
- Tile bytes and point launches/counts are explicit costs, including the minimum
  workspace alternative. No extra full-grid allocation or host transfer appears.
- RKS density-only forces use the existing unpolarized point convention; other
  forces retain polarized derivatives. Packaging does not inherit energy-only
  equal-spin optimizations as a new scientific force rule.
- Callback addresses and descriptor addresses both bind the executed module.
  Point workers and their validation wrappers have private ELF visibility, so
  another loaded core cannot interpose a worker behind an authenticated wrapper.
  Scope hidden visibility to emitted definitions after runtime includes. A
  target-wide hidden default also hides the inline resource ledger and its TLS
  owners, splitting allocation/release bookkeeping across native clients and
  the core. The native KS density-provider destruction gate exposed this.
  Private diagnostic v1 remains 7 identities/2 owners/40 work words; v2 adds
  shared consumer/point identities and costs with 11/4/43 extents.
- Dynamically initialized static bindings can occupy anonymous ELF BSS pages.
  Authenticate the `dladdr` module base against the file mapping inode. Checking
  the BSS descriptor page inode itself wrongly rejects a valid loaded library.
  Replacing the library file still invalidates the receipt.
- Source identities are not complete persistent binary-cache keys. Build
  dependency tracking and the compiler cache remain the native package owners.
  The existing offline object store retains its explicit closure domain and
  rejects incomplete/unsupported target or device-link closures.
- The existing capability identity includes the compiler's XC source inventory;
  point expression and method provenance therefore change across this packaging
  revision even when point mathematics is identical. Do not suppress that
  invalidation to obtain matching baseline hashes. A one-off comparison of all
  394 polarized/restricted emitted point bodies and physical contracts found no
  mathematical byte differences against the exact incumbent. Retain both sets
  of provenance and bind numerical comparisons to their actual library bytes.

## Rejected alternatives

A family-name alias would conflate different point mathematics or boundaries.
Removing only duplicate source text leaves per-registration device compilations.
A Python point/force driver would break the single native execution owner and
installed no-compiler/no-oracle contract. Unbounded resident point storage would
silently add a resource prerequisite. Discarding inode validation to accommodate
BSS would allow old mapped libraries to borrow replacement-file identities.

## Evidence

Executable sharing and genuine mathematical alias tests live in
`tests/python/test_shared_automatic_cuda_package.py`. Compiled receipt tests
cover legacy extents, BSS storage and replacement detection. Audit tests reject
unbound callback artifacts, missing executable identities and unaccounted point
work. `tools/census_native_automatic_aot.py` joins the active compilation database,
linked/installed bytes, CUDA entry symbols, cubin resources and measured ptxas
spills. It exports only `compiled-cuda` envelopes through the existing Lane-F
capability owner; object compilation cannot grant molecular science.

Real-device numerical, resource, lifecycle and endpoint measurements must be
retained against the exact per-functional incumbent before claiming #1123 done.
The full catalog build is separate from the narrower force qualification matrix.

## Revisit when

A new ingredient, derivative, precision, partition or geometry ABI is admitted;
a point producer layout changes; or measured launch/intermediate cost warrants
an explicitly packaged fused specialization. Requalification must compare
complete endpoints and actual artifact costs, not kernel timing alone.

## References

- https://github.com/jinzhezenggroup/generativeqc/issues/1123
- https://github.com/jinzhezenggroup/generativeqc/pull/2827
- [Current device contract](../../../../docs/developer/xc_native_cuda.md)
- [Offline CUDA closure](../../../../docs/developer/libxc_cuda_aot_closure.md)
