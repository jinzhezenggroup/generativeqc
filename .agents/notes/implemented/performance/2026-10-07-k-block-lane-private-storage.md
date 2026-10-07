# Decision: preserve incumbent lane storage in K-block candidates

Status: implemented
Date: 2026-10-07

## Problem

The optional restricted raw-K block-contraction candidate introduced in #2065
replaced the incumbent capability set with just `streaming_fock` and
`k_block_fock`. This dropped `local_packed_streaming_fock` from the `psss` and
`psps` selections. Their streaming lowering consequently moved the entire task
and recurrence state into 32-entry shared arrays, not merely the new bounded K
accumulator. The qualified sm_120 binaries used 26,628 and 32,004 shared bytes
respectively, compared with 1,284 bytes for each incumbent streaming kernel.

This supersedes the blanket lane-local **shared** placement described in the
[original candidate note](../../proposed/2026-10-07-packed-k-shell-block-contraction.md).
The block-contraction algorithm itself does not require shared storage.

## Decision

Carry the incumbent's explicit `local_packed_streaming_fock` capability into
the value-only K-block selection, in addition to its two required capabilities.
The existing streaming emitter then keeps each lane's task, recurrence state,
and bounded exchange buffer private. The compact survivor queue remains shared.
Selections without that capability retain shared task/lane arrays. Custom
profiles control the opt-in; shell names are not hard-coded in the compiler.

Do not copy the entire capability set: mixed-Fock/precision policies are not
automatically valid for a specialized value-only producer. Do not globally move
all packed classes to private storage, since that would also change the qualified
incumbents and unsupported fallback resource policies.

## Invariants

- Preserve the incumbent Fock schedule and specialized integral IR.
- Keep the same bounded K eligibility, screening, symmetry, and atomic scatter.
- Keep UHF, Coulomb J, HF-weighted exchange, and unsupported-class fallbacks.
- Leave `GENERATIVEQC_DIRECT_K_FOCK_LOWERING` opt-in; default is still `incumbent`.
- Test the streaming emitter in isolation: generic paged/persistent kernels in
  the full shard intentionally still contain shared lane arrays.

## Evidence

The pre-fix regression run failed five assertions: production capability
inheritance, both private streaming classes, and both custom-profile opt-ins.
After the fix, the selected compiler/schedule/scatter/recurrence suite passed
1,266 tests; the two optional NVCC schedule compilation gates passed separately
with CUDA 12.9.1 and ccache. Ruff and the compiler dependency-boundary check pass.

`cuobjdump --dump-resource-usage` on the actual sm_120 restricted streaming
objects reports:

| Class | Original block registers | Fixed block registers | Original block shared bytes | Fixed block shared bytes |
| --- | ---: | ---: | ---: | ---: |
| `psss` | 151 | 118 | 26,628 | 1,284 |
| `psps` | 173 | 168 | 32,004 | 1,284 |

Both fixed kernels match their incumbent register/shared counts and report zero
stack and local bytes. The `ssss`, `ppss`, and `dsss` block allocations remain
23,300, 29,956, and 30,980 shared bytes respectively. These are compiled resource
counts, not measured achieved occupancy or bank-conflict evidence.

The real-device fixed-density gate uses an RTX 5090 on node1 through Slurm job
6379 (`main`, `gpu:5090:1`, finite `00:35:00` allocation), retaining its assigned
device visibility. Frozen independent PySCF MINAO densities and raw-K matrices
cover 48/96-atom water clusters, spherical def2-SVP, 384/768 AOs, and synthetic
density scales `1`, `1e-3`, `1e-6`, and `1e-14`. These scales are not actual SCF
delta-density trajectories. Maximum raw-K oracle error is `1.91e-11` against a
`1e-8` acceptance gate. Every per-class admitted-task census and prepared
resident-byte count matches between incumbent and block policies.

Same-binary ABBA has five clean wall samples per leg, ten per policy/case. The
wall endpoint includes density upload, transformations, K contraction, projection,
export, and synchronization; device-event/census samples are separate.

| Full-density case | Incumbent K wall (s) | Fixed block K wall (s) | Reduction | Admitted K tasks |
| --- | ---: | ---: | ---: | ---: |
| 48 atoms | 1.011104 | 1.009667 | 0.14% | 23,480,495 |
| 96 atoms | 1.901113 | 1.876451 | 1.30% | 81,907,624 |

This is a storage-policy correction, not a claim of a substantial endpoint win
or a matched old-block-versus-fixed-block speedup. The pre-fix resource counts
come from the earlier #2065 qualification, not this timing job.

Complete cold PBE0-RKS measurements use two independent solves per policy and
size, in ABBA policy order. The endpoint includes calculator construction,
preparation, and execution, excluding process/library loading and compilation.
Incremental Direct-JK is off. Every solve has 22 iterations and 22 Fock builds;
the public API does not export the full per-iteration trajectory, so this is not
a bit-identical-history claim. Maximum energy spread across policies/repeats is
`5.46e-12` Hartree against a `2e-9` gate.

| Case | Incumbent cold SCF (s) | Fixed block cold SCF (s) | Reduction |
| --- | ---: | ---: | ---: |
| 48 atoms | 71.987177 | 71.626375 | 0.50% |
| 96 atoms | 146.715713 | 146.358348 | 0.24% |

These small timing differences do not establish a meaningful performance win.

The native CUDA Fock-provider suite also passes in the same Slurm allocation
with the block selector, including independent J/K selection, direct through-f
values, s/p derivatives, and order-two derivative/CPU finite-difference gates.

The qualified scientific tree is master `0ad23e791` plus the compiler/test patch
with SHA256 `a174f3b7649ffeb0153aabbf6a17fea11b6a6d37ad20bd823edf35adba9beda9`.
Scientific source identity is
`5bc7638fae203889b8f1daf623338b68ce7ebb4da2a68bf2c3aa7225a977fd75`;
the tested library SHA256 is
`1259531987d6cfd90c684ba7e014848675bc9f4ecf9b00f9202276e0d20251d9`.
Only generated CUDA shards 1 and 2 differ from the original #2065 qualification;
all other generated CUDA shards retain their exact source bytes.

Raw inputs, oracle matrices, scripts/probe source, ABBA observations, source and
binary hashes, generated-resource receipts, and before/after ccache statistics
are retained locally under
`/home/jzzeng/codes/vibeqc-scf-solver/.artifacts/qualification/k-block-lane-private-20261007/evidence/`.
The remote job evidence is under
`/data/jzzeng/k-block-lane-private-20261007/evidence/` on node1.

## Revisit when

A separately qualified profile demonstrates a better storage policy with
complete-endpoint timing, unchanged admitted work, independent numerical gates,
and explicit register/shared/local resource evidence. Lower shared allocation
alone is not an achieved-occupancy measurement or a performance promotion.
