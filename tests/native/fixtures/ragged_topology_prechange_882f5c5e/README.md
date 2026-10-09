# Frozen ragged topology extraction oracle

The complete native common schema header and implementation are byte-for-byte
`git show` copies from `882f5c5e0bc060b5777d5060c9cb89714bb4429b`. That commit was
verified as live master on 2026-10-08 at 21:34:38 UTC. `manifest.json` records
original paths, Git blob IDs and SHA-256 checksums. Never refresh this oracle to
make a changed implementation pass.

The native preservation probe is compiled separately against these files with
`-Dgenerativeqc=frozen_generativeqc` and against the real production common
wrappers plus `src/runtime/ragged_topology.cpp`. The frozen build has only this
fixture's include path. The same API probe compares every diagnostic triple,
all native ABI member offsets and enum values, and complete scalar and pointer
identity for successful borrowed views and failed output clearing. It covers
ragged and empty systems, every topology scalar/count/pointer, invalid offsets,
maps, matrix extents, bucket permutations, explicit pair ordering, hashes,
plan tokens, and binding with the input and output as the same object. Legal overlaps through
int64 output subobjects also prove output clearing happens before host inspection.

CUDA/HIP structural cases point to distinct PROT_NONE pages. Every pair of the
14 topology arrays is tested for alias rejection and conflict ordering. These
cases crash if any structural validator dereferences device metadata. C/C++
allocation hooks require zero allocations around every production call. Normal
host linking and symbol/dependency inspection prohibit CUDA linkage or calls;
no device execution is implied. Native probes are Linux-only; source and provenance
assertions run on every platform. The Python test also guards exact preservation
of retained spin, geometry generation and physical pair-list policy bodies.

These files are test-only and must never enter a production source manifest.
