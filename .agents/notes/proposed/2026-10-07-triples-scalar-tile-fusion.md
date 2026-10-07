# Proposal: fuse the occupied triples primal and W/V seed traversals

Status: proposed, opt-in only; complete-endpoint qualification pending
Date: 2026-10-07
Related: #1763, merged #1999, #2042, #2045; deferred repeatability issue #2019

## Problem

The merged joint triples/Fock traversal reuses the six forward W cubes but
still executes separate energy, gathered W-seed, and six V-seed kernels for
each physical occupied triangle. Each W output invokes six individually
pruned scalar AD functions, recomputing compatible projections and repeatedly
reading the same virtual-coordinate source values.

## Proposed schedule

`occupied_triples_response.fused_tile_program()` composes the original primal
and its existing AD outputs. Each gathered W derivative rebinds inputs by its
inverse virtual-coordinate permutation. Ordinary TensorIR optimization/value
numbering shares equal producer nodes. There is no second CC equation set,
new amplitude iteration, precision change, or floating-point atomic reduction.

One grid-stride virtual-cube kernel publishes six W seeds, six V seeds, and
energy partials. The original block energy reduction tree, triangular energy
slot order, downstream packed reverse GEMMs, and complete Fock page domain
remain unchanged. Scientific derivative evaluations are not claimed to have
been omitted; only common scalar work is shared.

Keep six separate ordered denominator inputs. The algebraically symmetric
gap does not license using a differently ordered FP64 subtraction for a
permuted derivative. V boundary inputs similarly retain occupied and virtual
coordinate labels. W gathers preserve the existing virtual-label order.

## Admission and fallback

The joint owner admits six V cubes instead of one reusable cube: an additional
`5 * v^3 * sizeof(double)` before alignment. The W scratch remains six cubes.
If the optional allocation is not admitted, use the unchanged unfused schedule
before sacrificing existing page/panel residency. Exact minimal legacy-budget
refusal remains transactional. Full T3, full ovvv, and CPU mathematical
fallbacks remain prohibited.

The new flag defaults to false in both internal APIs. Gap-cotangent demand
also selects the retained unfused all-output schedule. Molecular force
composition still supplies mandatory full same-space Fock response. Standalone
primal, pullback, and full-Fock APIs remain qualification controls.

## Work and evidence contracts

The scalar region changes from eight launches to one per occupied triangle;
the downstream energy reductions, reverse GEMMs/audits, Fock kernels, Lambda,
orbital, and nuclear work are not credited as removed. Both paths write twelve
W/V seed values per virtual point; fusion increases simultaneously resident
seed storage rather than claiming to eliminate necessary derivative outputs.

Generated receipts count source scalar reads (including V products and ordered
denominator construction), seed/energy-partial writes, and FP64 scalar
arithmetic. They exclude finite checks, integer indexing, cache behavior,
compiler instruction transformations, and all parent phase work. They are not
measured memory traffic. The complete endpoint separately reports transfers,
admitted memory, phase timings, and scientific residual/stationarity checks.

The compiler graph has 376 prepared nodes instead of 1,743 summed nodes across
the separate scalar functions. Source-level modeled scalar reads per point
drop from 1,794 to 216, and arithmetic operations from 2,496 to 763. These
include ordered denominator and V product construction, not finite checks.
The compiler schedule identity binds the equation hash, coordinate and
denominator order, FP64 scalar emission, seed storage, and energy tree policy.
Node counts are not FLOP counts or a measured
speedup. CPU interpreted and compiled FP64 tests compare all thirteen outputs
to the original individual AD calls, including distinct ordered denominator
bindings. Native combined tests compare seven cotangents to the independent
host reverse composition, energy to original triples algebra, full Fock to its
standalone owner, transfer counts, page/panel fallback, and exact-budget refusal.

Local CPU qualification: 38 passed, 86 skipped (real-GPU tests require a finite
Slurm allocation). Compiler structure, CUDA ownership (333 files), vendor,
electronic-structure, provider-selection, and default-promotion inventories
were checked without errors. These CPU/structure results are not GPU gates.

Raw source/build/test/endpoint evidence stays ignored under
`.artifacts/1763-tile-fusion/` and on n2 under
`/data/jzzeng/triples-fusion1763-20261007/`. No Release or external backup is
authorized. Append actual job IDs, binary/source hashes, GPU/sanitizer gates,
28/56/230-AO matched cold endpoints, launch census, and measured device peak
before making any complete-endpoint performance claim or promoting a default.

## Rejected shortcuts and remaining gates

- Do not merge the six ordered denominators solely using mathematical symmetry.
- Do not count fewer scalar calls as fewer mathematical derivative outputs.
- Do not sacrifice a full occupied page just to fit optional fused seeds.
- Do not treat shared-source work from #2045 as newly removed by this schedule.
- Do not investigate or relabel #2019 as part of this scoped fusion work.
- Do not claim completion of general W/V/response fusion in #1763 from this slice.

Promotion requires independent energy/triples/full-force gates, unchanged
residual/stationarity criteria, finite-GPU sanitizer/resource qualification,
matched 28/56-AO and larger DF endpoint measurements, and complete launch/
materialization/transfer/peak receipts. Until then retain the unfused default.
