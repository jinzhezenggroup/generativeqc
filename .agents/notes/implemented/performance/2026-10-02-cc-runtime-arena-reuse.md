# Decision: reuse dead native coupled-cluster intermediates

Status: implemented
Date: 2026-10-02

## Problem

The runtime-shape native generator assigned a fresh arena region to every
non-input node, retaining all intermediate storage through return. This inflated
both CCSD energy and CCSD(T) force-response admission as occupied/virtual spaces
grew. Bounded admission alone did not make those allocations necessary.

## Decision

Compute last-reader intervals in the existing emitted topological order. Reuse
a slot only after its last reader has completed, and only for the same symbolic
product of runtime extents. Keep all program outputs live through return because
owners borrow their pointers. Inputs retain their external ownership. Every
operation writes its complete destination, including explicit initialization
for scatter/reduction nodes, so slot reuse needs no added clearing.

Use the same planner for CPU emission, CUDA emission and generated admission.
CUDA kernels and captured replays execute on one ordered stream; no host wait
or additional device synchronization is required between slot users. Production
graph tests require CPU/CUDA program and slot-plan identity because native CUDA
owners use the host-generated capacity functions.

## Invariants

- Never overwrite a node's inputs while that node executes.
- Never recycle output storage before the caller consumes the returned pointers.
- Reuse depends on symbolic shapes, not representative sizes: o, v, q and o+v
  remain independent runtime dimensions, even when some happen to be equal.
- Preserve equations, precision, reduction order, graph hashes, node/work counts,
  independent replay and failure publication behavior.
- Preserve checked products/additions and fail admission before execution.
- This change does not remove the <=12-AO analytic-force qualification boundary,
  dense MO/AO response weights, or their remaining scaling limits.

## Evidence

For the generated representative programs evaluated at o=20, v=80, q=16,
logical arena capacities in bytes are:

| Graph | Previous | Reused slots |
| --- | ---: | ---: |
| CCSD iteration | 2,656,236,872 | 1,754,560,048 |
| Independent expanded replay | 2,376,153,672 | 2,171,276,848 |
| Lambda transpose | 4,176,560,000 | 1,960,028,800 |
| Triples response | 2,606,745,640 | 1,019,500,680 |
| Hamiltonian weights | 31,656,889,600 | 14,855,129,600 |

These are generated capacity calculations, not executed large-system timings or
whole-process peaks. Arithmetic work is unchanged. In an executed 7-AO CPU
water force endpoint, complete numeric admission decreased only from 9,136,936
to 8,961,320 bytes because the final derivative stage remains dominant. Cold,
warm and changed-geometry times were approximately 2.5-3.0 s for both schedules;
no complete-endpoint speedup is established by that result.

The [retained CPU summary](../../../../benchmarks/results/ccsdt-arena-20261002/cpu-summary.json)
binds binary hashes, exact basis/geometry/settings, complete endpoint timings,
per-phase work counts and all-repeat oracle errors. It covers 7-AO forces and
14/28-AO energies, each cold, twice warm and at changed geometry. Baseline and
candidate energies/forces are bitwise equal; semantic work counts match.
Maximum errors against PySCF 2.14.0 are 8.7e-13 Eh for energy, 1.5e-14 Eh for
(T), and 1.3e-8 Eh/bohr for force. Match the packaged STO-3G coefficients in the
oracle, and supply explicitly corrected `ccsd_t_lambda` amplitudes to its
gradient; PySCF's default named basis and uncorrected Lambda are different inputs.

The 28-AO baseline warm endpoint takes about 40 s: independent expanded replay
alone takes 28.4 s, compared with 4.73 s for 47 iteration graphs, 6.47 s for MO
preparation and 0.25 s for (T). Arena reuse leaves that work unchanged. A separate
compiler census identifies degree-eight work in the deliberately unreassociated
expanded replay, versus degree six under existing contraction reassociation.
Any follow-up must preserve independent expanded equations and oracle gates;
removing replay or relaxing convergence is not an acceptable optimization.

`test_rccsd_arena_liveness.py` checks every native production graph's intervals,
output lifetimes, symbolic-shape admission and CPU/CUDA plan agreement. Existing
compiled Lambda, Hamiltonian and runtime triples-response tests compare executed
outputs with TensorIR. Public CPU CCSD(T) energy, analytic PySCF gradient and
energy finite-difference gates cover the complete path. The allocation-intercept
force test retains exact-budget success and one-byte-short refusal; its allocation
observation uses the current triples arena rather than a historical byte floor.

Real-device CUDA qualification is required before this change is ready to merge.
Compilation or agreement of generated plans alone does not qualify CUDA execution.

## Rejected alternatives and revisit conditions

Do not compare concrete representative tensor sizes: that would alias o/v
products unsafely at other shapes. Avoid a general runtime allocator or
recomputation schedule here; neither is needed to reclaim dead equal-size
buffers. More aggressive packing across symbolic sizes, graph reordering and
fused reductions may improve the remaining peak, but require separate work and
complete-endpoint evidence. The retained expanded replay reduces less than the
iteration graph because its original topological order keeps more terms live.
