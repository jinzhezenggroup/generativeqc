# Proposal: bound the direct fallback compiler working set

Status: partially implemented; weighted callables shared, further native investigation pending
Date: 2026-11-16

## New authoritative evidence

The resource-only PR head `027c4b8b9` completed wheel run 43728794724 / job
132856447352 with a contained build failure, not a lost runner. The build/test
step ran 2026-11-15 15:15:10–16:22:39 UTC. At task 61/1122, compilation of
`src/scf/cuda/direct_bounded_fallback.cu` failed with:

```text
FAILED: [code=9] CMakeFiles/generativeqc_direct_native.dir/src/scf/cuda/direct_bounded_fallback.cu.o
nvcc error   : 'ptxas' died due to signal 9 (Kill signal)
```

The new always-run diagnostic survived and reported:

```text
OOMKilled=true
memory=12884901888
memory_swap=12884901888
```

The host runner remained alive. The resource/cleanup step succeeded at
16:22:42 UTC, and "Save completed compiler cache entries" succeeded at
16:22:45 UTC. This breaks the observed lost-runner/no-cache-save loop, but the
current source does not yet fit the cold-build envelope. The signal-9 error
and Docker state establish an OOM event for **this run**. They do not prove that
the historical November 14/15 lost-runner master jobs had the same cause.

The failed command keeps `-O3 -DNDEBUG -std=c++20 -rdc=true`,
`-Xptxas=--disable-optimizer-constants`, and all wheel targets:
`compute_80 + sm_80`, `sm_90`, `sm_120`. No NVCC split-compile/threads flag
appears in the failed command. The log does not identify which architecture's
ptxas process was killed. Ninja completion timestamps overlap other work and
cannot establish an isolated per-source duration or peak memory.

## Ownership and next investigation

`cmake/GenerativeQCCuda.cmake` places this file in the device-linked
`generativeqc_direct_native` archive alongside direct J/K, reference/ordered
forces, class-domain and other bounded implementations. Its one large
`bounded_direct_shell_quartet_kernel` template spans unrestricted/restricted,
screening, force/Fock, angular-order and radial-operator variants. The source
is only 68,315 bytes, so source-line count alone is not compiler working set.

Prior history already measured expensive legitimate misses for this owner:
`2026-10-14-cli-cache-snapshot-fallback.md` records roughly 1,269/1,765-second
direct J/K/fallback compilations after changed headers. Snapshot transport or
more retention cannot turn changed-input cold compilation into correct hits.

Next steps:

1. Inventory actual template instantiations and generated/shared numerical
   callees before choosing a partition. Read applicable integral instructions
   before changing their owners or generation inputs.
2. Investigate separating force/Fock or radial/angular instantiation groups,
   and eliminating duplicated retained device code using the existing bounded
   device-link owner. Do not introduce a giant unity translation unit.
3. Measure complete cold compilation of the affected owner and target, with
   semantic kernel/work inventory and genuine compiler peak-memory evidence.
   sccache client RSS is not worker RSS. Preserve cache correctness and reuse.
4. Qualify generated dependency closure, public launch bindings, device/host
   linking, independent scientific oracles and finite real-GPU runs under Slurm.

Do not simply delete high-angular/range/force fallbacks, drop wheel
architectures, change strict numerical flags, remove the cgroup to hide an
OOM, or claim the response-kernel pooling fixed this separate owner. The
UCCSD/RCCSD/DF pooling commits retain real reductions but execute later in the
build graph. Their complete wheel acceptance remains pending.

## Follow-up experiment: share large weighted arithmetic

The order-four/five generated weighted headers force-inline their complete
scalar arithmetic into spin, radial and output-layout adapters. The current
source experiment instead emits header-safe `__device__ inline __noinline__`
callables for 22 order-four and 24 order-five bodies, including FP64, FP32,
relative-response and pair-separable variants. Complete-class wrappers retain
the original 64-component partitions, global component indices, partition
addition order and all native admission/fallback policy. The low-order header
is byte-identical. A one-off comparison confirms that every other generated
byte in the two large headers is unchanged; this is investigation evidence,
not a permanent source-fingerprint CI gate.

The compiler's public scalar emitter accepts an explicit `cuda_noinline`
selection. Small/default CUDA helpers and CPU emission retain their previous
behavior. A new executable two-owner host shim preserves actual noinline/inline
linkage and checks both consumers' values and all four-center derivatives.
The focused independent weighted/native/range suite passes **568 tests**, and
the compiler structure check covers 604 modules with zero dependency errors.

Local comparison uses NVCC 12.9.86, verified sccache 0.16.0 and its existing
cache. The original complete fallback emits **153,750,985 PTX bytes** for
`sm_80`; its compiler is still running at this evidence checkpoint. The owned
foreground cache server is wrapped in GNU time so final resource usage includes
its reaped compiler children, rather than measuring only a lightweight cache
client. Before/after object, cache, wall and worker-memory acceptance remains
pending. Single-architecture data cannot stand in for the complete three-target
wheel envelope.

An additional RDC CUDA probe compiles and links both original and candidate
headers, exercising complete high-order strict/mixed/relative and separable
callables. This is **not** physical GPU evidence. The independent Libcint
device comparison was submitted as finite Slurm job 12282; it remained pending
because the only local `main` node reports `drained / maintenance`. The owned,
unstarted job was cancelled rather than bypassing Slurm or device visibility.
No GPU test, complete Direct endpoint timing or runtime nonregression is claimed.

Before treating this experiment as a wheel fix, qualify the complete fallback and
class-domain objects, their bounded device link and the wheel architecture
inventory. Real-device numerical/resource and complete-endpoint qualification
must still pass when an authorized scheduler allocation is available. A lower
caller register count in a transport probe does not establish a lower complete
call-tree resource requirement or an endpoint speedup.

## Retained outcome

The shared high-order callable experiment is retained for its measured compiler
working-set reduction, not a compile-time or runtime speedup. The complete local
fallback maximum RSS drops from 8.37 to 5.32 GiB, with all 108 kernel entries
retained. PTX shrinks only about 3%, so native density/component specializations
remain an important next investigation. See the superseding implemented record
`../implemented/performance/2026-11-16-weighted-cuda-callable-working-set.md` for
full object/cache evidence, instrumentation caveats and outstanding wheel/GPU gates.

## References

- Run: https://github.com/jinzhezenggroup/generativeqc/actions/runs/43728794724
- Job: https://github.com/jinzhezenggroup/generativeqc/actions/runs/43728794724/job/132856447352
- Implemented resource evidence: `../implemented/performance/2026-11-15-wheel-cold-build-resource-envelope.md`
- Implemented response work: `../implemented/performance/2026-11-16-native-response-cuda-kernel-sharing.md`
