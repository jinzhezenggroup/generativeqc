# Decision: one direct CUDA HF host/fleet capacity owner

Status: implemented
Date: 2026-10-17

## Problem

The topology-aware CUDA HF allocator already owned arena/staging capacities,
but Python and the native CLI independently expanded host metadata bounds and
composed retained fleets. Moving only the allocator leaf did not transfer this
remaining production capacity ownership.

## Decision

Share an allocation-free C++ scalar owner between the CLI and the private
`generativeqc_resource_hf_cuda_fleet_v1` adapter. Its compact inputs contain
topology plus capacities from the existing allocator v2 and ECP authorities.
The bridge validates every item and all aggregate signed-int64 peaks before a
second publication pass; input/output buffers must be disjoint.

Python retains metadata presentation and a missing-symbol compatibility path.
Discovered native refusal or inconsistent publication is authoritative, never
permission to retry the Python host polynomial. The CUDA allocator's admitted
domain, solver eligibility and scientific execution are unchanged. Density-
fitted HF retains its distinct existing owner.

## Rejected alternatives

- Tightening the retry to `max(arena + ECP)`: the old contract deliberately uses
  independent maxima, which can belong to different fleet items.
- Creating a CUDA provider in the host query: scalar composition must remain
  usable without device selection, allocation or an execution context.
- Applying CLI batch multiplicity after fleet composition: retained storage
  would be counted twice, while a serialized retry must not scale with batch.
- Calling equal capacity formulas complete memory authority: provider retention,
  physical allocations, leases and simultaneously-live RSS remain separate work.

## Invariants and evidence

Compiled CPU-only contracts exercise malformed later items, overlapping buffers,
address/capacity overflow, transactional append/publication, order independence,
and homogeneous/ragged multiplicity. Python comparisons retain an independent
compatibility bound and exercise actual compiled ownership with synthetic CUDA
leaves, missing symbols, authoritative refusal/corruption and exact/below byte
admission. Synthetic leaves are not real-GPU or numerical qualification.

Ready still requires exact-head CPU/CUDA builds, real allocator and endpoint
execution under finite Slurm allocations, independent science and fixed full-
endpoint non-regression gates. R3 and complete #2227 closure are not implied.

## Revisit when

A separately qualified allocator/provider/lifetime contract supersedes the
conservative capacities, or old-library compatibility is explicitly retired.

## References

- #2227, #933, #934
- `src/scf/hf_cuda_resources.hpp`
- `docs/maintainer/resource_planning.md`
