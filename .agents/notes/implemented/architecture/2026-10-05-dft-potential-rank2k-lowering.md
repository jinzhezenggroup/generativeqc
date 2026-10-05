# Decision: prepare native XC symmetric products through shared lowering

Status: implemented
Date: 2026-10-05

## Problem

#1886 needs real DFT consumers of the canonical request/candidate/binding
boundary. Grid density/orbital projections are covered by #1951; native density
preparation is independently covered by #1907. Repeating either migration or
optimizing unrelated AO/Becke kernels does not move this architecture forward.

## Decision

Express the existing compact potential panels as a TensorIR cross product plus
its transpose. A shared recognizer projects this fused operation into the
existing lowering contract, preserving scientific identity on CPU and CUDA.
The native owner retains one immutable bounded provider binding. The generated
incumbent keeps its exact packing, tiled/scalar choice, point-total fusion,
scatter, initialization and finite-error behavior.

The alternative shared provider executes pedantic FP64 rank-2k. Contiguous
jet/point axes flatten into its reduction; column-major lower storage maps to
the existing row-major authoritative upper triangle. Publication mirrors that
triangle and preserves sticky finite errors. Packing still derives its scalar,
gradient and kinetic coefficients from the original compact bilinear. No
functional formula is copied into TensorIR or the provider.

Optional rank-2k qualification is test-only until complete endpoint evidence
supports a production profile. Providing a budget does not invent performance
evidence. Resource admission and provider-unavailable cases retain the generated
incumbent without changing precision or backend. Unsupported local maps use the
generated callback even with a qualified dense binding.

## Invariants

- No vendor names or calls in the native scientific owner or its method API.
- The exact numeric XC arena remains unchanged. A 16 KiB host reservation
  covers the binding and bounded selection scratch. Optional provider storage
  is a separate 96 MiB allowance, not a second unused arena allocation.
- No handle/plan search, allocation, or host-data staging during execution.
- Capture records only device work. Physical replay publication owns counts;
  semantic summands are `spins * n * (n+1) * jets * points` per nonempty tile.
- Captured graphs and borrowed buffers must not outlive the prepared owner.
- Source-generated density and nonlocal Vxc are not claimed as migrated here.

## Rejected alternatives

Calling SYR2K directly from DFT would recreate the ownership problem. Expressing
only a GEMM would omit the symmetric publication/triangle contract. Promoting a
library from a kernel timing would ignore AO, density, point XC, packing,
publication and provider preparation. Reusing the ordinary grid binding would
violate its explicit capture rejection and omit replay accounting.

## Evidence

Canonical host tests compare against independent NumPy products and reject
asymmetric/scaled expressions. The native qualification target exercises actual
AO-to-density-to-point-XC-to-Vxc consumers, independent CPU E/Vxc references,
energy directional derivatives, spin/functional/tail domains, resource fallback,
changed-input graph replay, local maps, response and exact arena canaries.
Detailed measured outputs are retained under ignored `.artifacts/1890-vxc/`.

The initial native PBE endpoint comparison on Slurm node1 / RTX 5090 used two
synthetic Cartesian bases, 2304 points, 128-point tiles, and five warm samples
after one cold call. Timing includes AO, density, point XC, packing, Vxc,
explicit density H2D, and E/Vxc exports. These are fixed-density XC endpoints,
not complete SCF or forces.

| AO | Generated warm median | Rank-2k warm median | Six-call summands |
|---:|---:|---:|---:|
| 82 | 0.003152967 s | 0.003813108 s | 94,086,144 |
| 182 | 0.003617064 s | 0.004280671 s | 460,422,144 |

Each route submitted 108 spin/tile products over all six evaluations. Maximum
Vxc error against the independent CPU integrator was below 1.6e-14. Rank-2k was
18–21% slower on these endpoints, in addition to provider preparation (12.3 ms
for the first library context and 0.168 ms for the later one). This is retained
negative evidence: keep generated execution as the production incumbent.

The complete legacy native suite also exposed a stale test-message check:
precision rejection now says "layout", while the old test required "functional".
The test now checks the stable precision-rejection prefix; admission and the
last-good-result assertions remain unchanged.

## Revisit when

A representative complete XC/KS endpoint profile demonstrates a benefit and
qualifies simultaneous provider resources. That promotion belongs in a separate
reviewable change, with preparation and physical replay evidence retained.

The subsequent [384/768-AO crossover qualification](../performance/2026-10-05-large-xc-rank2k-crossover.md)
extends this evidence to larger dense domains; it does not invalidate the
small-domain loss or by itself promote a full SCF/force production default.
