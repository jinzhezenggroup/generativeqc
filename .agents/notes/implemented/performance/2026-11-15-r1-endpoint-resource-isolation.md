# Decision: verify endpoint resource placement, not just Slurm CPU requests

Status: implemented
Date: 2026-11-15

## Observation

The rebuilt R1 GPU population on Slurm job 17440 completed all 408 comparisons
but failed two BAAB warm endpoints: M06-2X UKS/DF at 1.0858688997266084 and
WB97M-V UKS/DF at 1.0891689425778566. Scientific errors stayed inside their
unchanged gates and semantic work matched. Preserve this population as FAIL;
neither the earlier source rebuild nor a later measurement changes its result.

An instrumented, non-acceptance diagnostic on job 17443 recorded wall and
thread CPU time, voluntary/involuntary context switches, GC events and Python
threads for every call. Multi-millisecond stalls occur inside the outer Python
`PreparedBatch.execute` wrapper, without a coincident GC event or extra Python
thread, with nearly unchanged thread CPU time and extra voluntary switches.
The original `native_execute_*` labels do not isolate the C ABI. Direct FFI and
driver/syscall probes are needed; neither a native cost nor a driver/OS cause
is established. Changing GC or removing samples would be unjustified.

The resource audit found a concrete qualification defect: eight CPUs were
requested, but the actual process affinity covered all 128 logical CPUs.
The node's Slurm TaskPlugin was unset. Job 17445 independently recorded the
actual CUDA UUID and PCI address, an empty assigned-device compute-context list,
the assigned CPUs and a controlled first-assigned-CPU binding. That diagnostic
still has occasional waits; binding is not claimed to eliminate them or to
retroactively validate the failed population.

## Decision

Bind every variant and block to the same first Slurm-assigned logical CPU,
before preflight, untimed profiling and timing. Preserve CUDA_VISIBLE_DEVICES;
the launcher verifies actual CUDA identity rather than choosing a faster GPU.
Record the allocation, inherited affinity for every timed block, controller
source and exact loaded binary. The production runtime and scientific inputs
are unchanged. This is explicit benchmark resource isolation, not a promise
that arbitrary uncontrolled host scheduling has the same latency.

Require a separately collected complete matrix to pass the original numerical,
semantic-work, sample-count and 1.05 per-case/phase/order performance gates.
Keep all previous failed and incomplete observations offline. A focused
100-sample diagnostic is not a replacement for the original 11-sample matrix,
and a resource-control change is not permission to retry unchanged populations
until one happens to pass.

## Revisit

Controlled job 17446 completed all 408 comparisons with no incomplete endpoints
and two failed ABBA warm gates: PBE UKS/DF ratio 1.0569072806471687 and
WB97M-V UKS/DF 1.0785614498702347. Scientific errors and semantic work pass.
Preserve the full FAIL result; CPU binding is not a fix.

Job 17450 separately measures the actual ctypes C ABI invocation as well as the
outer Python wrapper. Both variants have multi-millisecond FFI stalls without
matching thread CPU growth. Correlated strace intervals include NVIDIA driver
ioctl waits; Nsight intervals include long memcpy, free and synchronization
calls. These traces establish driver-call latency, not its ultimate cause, and
are not performance acceptance. FFI timing includes the ctypes/GIL boundary.

A subsequent host inventory found a still-running GPU sampler from August 26
and an orphan Nsight agent from October 7 on node1. An empty assigned-device
compute-context list does not exclude either process or shared-driver effects.
Do not terminate unrelated jobs or assert a causal attribution from this
inventory alone. A clean-node comparison must bind its own actual allocation,
GPU, binaries, Python environment and before/after source identities.

If controlled full endpoint timing still fails, investigate it as a real failed
gate. Do not increase samples/warmups, discard pauses, move cleanup outside the
endpoint or relax the threshold. A future scheduling explanation needs direct
evidence rather than an assumption based on equal kernel counts.

## Independent clean-node control

Node5 job 2041 fails before any scientific observation: the copied installed
ELFs require GLIBC_2.38, but this host provides glibc 2.35. Retain the original
loader error; it is neither a numerical failure nor a timing vote. A private
copy of the matching node1 glibc 2.39 loader/runtime launches only the Python
process with an explicit library path. It does not replace the system runtime
or modify the measured ELFs. Authenticate these runtime files before and after
the job, and retain the launcher source and actual Python/package versions.
The node5 Python 3.13.9 is the uv/Clang build, not node1's Anaconda/GCC build.

Node5 diagnostic job 2042 has no millisecond-scale wall-minus-thread gap in its
600 ordinary warm C ABI calls, unlike node1 job 17450. This intervention changes
the resource environment and Python build; it does not isolate a unique root
cause or prove that a sampler caused the earlier failures. It justifies a
separate, predeclared full qualification on this controlled allocation rather
than repeatedly retrying the unchanged node1 campaign. Preserve every scalar
row from all sixteen diagnostic streams, not just medians or selected pauses.

Job 2043 runs the unchanged complete endpoint protocol on an exclusive node5
allocation, bound to its first Slurm-assigned CPU with scheduler GPU visibility
preserved. Its full result, not the focused diagnostic or equal profile counts,
must determine acceptance. Do not modify its active controllers or admit any
partial timing result while it is running.

The completed job passes all 408 comparisons with no incomplete endpoints or
failures. Its maximum ratio is 1.0224901148355194 (BAAB warm PBE UKS/DF), with
identical semantic work and unchanged numerical gates. The untimed profile has
identical kernel, explicit allocation/peak/live-byte, static image, graph and
transfer counts for both variants. All earlier failed populations remain FAIL.
This accepts this complete resource-bound population, not a universal causal
diagnosis or a guarantee for arbitrary uncontrolled hosts.
