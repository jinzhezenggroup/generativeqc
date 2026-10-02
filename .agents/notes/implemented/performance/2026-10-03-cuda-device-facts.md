# Decision: query fresh CUDA resource facts without unrelated properties

Status: implemented
Date: 2026-10-03

First-process timing caveat: see the later
[lazy-library-load correction](2026-10-03-xtb-cold-library-timing.md). The
historical xTBloom first-process timings here omit an untimed native/provider
load; warm/changed timings and scientific gates retain their scope.

## Problem

After AES2 peer scheduling, small molecular cold calls still paid approximately
1.5–1.8 ms more than repeated calls from an otherwise similar xTBloom runtime.
Native context creation and compiler-profile admission each queried all of
`cudaDeviceProp`. An isolated Nsight Systems constructor trace attributed
32.73 ms across 21 calls to `cudaGetDeviceProperties`, with a 1.447 ms median.
The required architecture and resource attributes themselves are inexpensive.

## Decision

Native context initialization and the compiler-facing tuning probe share
`runtime/cuda_device_facts`. On NVIDIA/Linux, query the eleven required runtime
attributes and use the already loaded driver's stable metadata interfaces for
device name and total memory. Resolve `cuDeviceGet`, `cuDeviceGetName` and
`cuDeviceTotalMem_v2` lazily through an optional `RTLD_NOLOAD` reference. Retain
only that function/library lifetime; read every device/resource fact afresh.
No CUDA ordinal, device handle, topology or target descriptor is cached.

CuMetal and other platforms use the original full property query. NVIDIA also
retains that fallback if an attribute is unsupported or driver metadata is
unavailable. Other runtime query errors propagate without qualification. Failed
queries clear the candidate outputs, and successful publication is complete.
No NVIDIA header dependency or new required driver symbol is introduced into
the baseline library. CPU callers still never probe a GPU.

The xTB comparator now additionally reports `cold_total`, constructor plus
first-call time. The existing `cold` metric remains the first calculation
alone. Both phases are required before claiming cold superiority: the two
public APIs charge initialization to different phases. Cold-total comparison
fails closed when constructor timing is missing, negative or nonfinite.

## Alternatives and invariants

- A process-global device-ordinal cache would avoid the query but complicate
  visibility/context ownership and invalidation. Fresh selected queries avoid
  that assumption entirely.
- Omitting names, memory or resource limits would change profile identity.
  The complete-property oracle must match every reported field exactly.
- Linking new mandatory CUDA driver entry points would weaken CPU-only load
  and alternative-provider compatibility. Optional discovery preserves the
  existing fallback without introducing another provider into the process.
- Moving work into the constructor is not an endpoint improvement. Keep raw
  phase timing and the combined cold endpoint in performance evidence.

## Evidence

A Slurm microprobe on n2 RTX PRO 6000 compared the same facts for 100 queries:
full properties averaged 1.769 ms; selected fresh queries averaged 0.0029 ms.
This is diagnosis, not a complete endpoint performance claim.

The native oracle passed on separately allocated RTX 5090 and RTX PRO 6000
devices. It compares every resource field, name and total memory with the full
query, rejects invalid ordinals, clears failed outputs and preserves the current
device. Host tests inject changing device facts, repeated/changed ordinals,
unsupported attributes, missing drivers/symbols, driver failure, propagated
runtime errors, full-query failure and the CuMetal fallback.

n2 complete endpoint qualification passed 41 public tests and all 110 numerical
and SCC iteration gates using the AES2 PR binary as baseline. First-call times
in milliseconds changed as follows:

| Case | AES2 baseline | Selected device facts |
| --- | ---: | ---: |
| H2O | 14.681 | 12.911 |
| NH3 | 14.819 | 13.264 |
| CO | 16.666 | 15.182 |
| HF | 11.703 | 10.177 |
| SiH4 | 23.150 | 21.522 |

Construction-plus-first-call times improved by about 13–24% for these small
cases, but several remained slower than xTBloom by 0.5–0.9 ms. Process startup
and the 192-atom cold endpoint still need separate work; do not treat the
query-only speedup as universal end-to-end superiority.

Final source qualification also passed all 41 public tests and 110 numerical/
SCC gates on n1 RTX 5090. All ten repeated/changed medians remained faster than
xTBloom. First-call-only medians won in eight cases; construction-plus-first
call won in four. In particular, process initialization and the 192-atom cold
case remain slower. The native source/probe ABI, host fallback, local profile,
target policy and benchmark tests passed (67 tests, one separately executed GPU
opt-in); all pre-commit checks passed.

Pinned binaries: AES2 baseline
`213babf344a7bf90eaa194554dd9a3f98276e1c99f10e2954992f18b27cf9256`,
final qualified device-facts candidate
`8f311992409446d3bb8d47b4050b379aff44d3ee19bb20e915531b968b6ad5eb`.
The earlier n2 receipt predates removal of a redundant preprocessor guard and
formatting only; scientific/runtime query behavior is identical.

Ignored local evidence: `.artifacts/n2-construction-profile.log`,
`.artifacts/n2-construction-nsys.log`, `.artifacts/n2-device-facts.log`,
`.artifacts/n2/device-facts/`, `.artifacts/device-facts-qualification/`.
The scientific work, SCC settings, FP64 policy and all generated kernels are
unchanged from the AES2 baseline. No GPU profiler/benchmark bypassed Slurm.

## Revisit when

A provider exposes an equally cheap complete resource query, the CUDA runtime
adds direct name/memory metadata, or an unsupported metadata field is required
by a new compiler schedule. Preserve fresh visibility and the bounded complete
query fallback when extending this owner.
