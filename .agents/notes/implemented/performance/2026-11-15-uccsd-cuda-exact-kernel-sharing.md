# Decision: compile identical UCCSD CUDA workers once

Status: implemented; complete wheel and real-device qualification remain gates
Date: 2026-11-15

## Problem and history

The cold wheel investigation also found real generated-code duplication, not
just missing caches. The November 3 open-shell CBS force/CUDA change, #2944
(`2eda35ecb`), added unrestricted analytic response programs and their native
CUDA emitter. The current inventory has 46 programs, including 24 full Fock
moments, four triples cotangents, four triples-vector programs and six UMP2
energy/cotangent programs. These are required scientific endpoints, not optional
debug expansions to drop from the wheel.

At master `0597b33e1`, generation emits 4,700 CUDA kernel definitions and 4,700
launch sites. Many definitions repeat the same emitted arithmetic across
programs, distinguished only by their entry symbol and literal diagnostic node
number. Per-program kernel ownership makes compilation and packaged code grow
with graph duplication, not just distinct executable work.

This is an accumulated workload, not evidence of a new November 14/15 graph
explosion: the examined November 12/14/15 wheel logs already have 1,122 Ninja
tasks. The native stationary and automatic Libxc CUDA targets also remain heavy,
but already share authenticated consumers and point workers. Their object
completion windows are not isolated target timings. See the companion
`2026-11-15-wheel-cold-build-resource-envelope.md` for the lost-runner/cache loop.

## Decision and invariants

- Pool kernels within the UCCSD CUDA translation unit/namespace using their
  **complete emitted signature and body**. Abstract only the entry symbol and
  diagnostic node number. Exact source equality, not a mathematical family label
  or matching dimensions alone, establishes executable equivalence.
- Pass the original graph node number as a launch argument. Preserve positive
  nonfinite/index errors, negative singular-denominator errors, first-error
  publication and all original failure checks. The node number must not be
  captured from the first program to register a worker.
- Keep every launch, operand, launch configuration, reduction order, coefficient,
  precision, arena allocation/lifetime, graph dispatch and method registration.
  No runtime graph fusion, CPU fallback, graph-work reduction, new public ABI or
  scientific qualification is introduced.
- Keep unpooled emission as the default of the shared RCCSD emitter. Only the
  UCCSD package selects a pool. Externally owned/reused kernel schedules cannot
  also request local pool ownership.
- Keep wheel architectures, optimization/numerical flags and all acceptance
  gates unchanged. This is not permission to delete response programs or widen
  scientific admission.

## Evidence

One controlled local compilation pair used the same generated CPU/response
headers, NVCC 12.4.99, host toolchain and compiler flags. Source inputs were
generated from the PR parent and this change, respectively:

| Complete generated UCCSD CUDA object | Before | After |
| --- | ---: | ---: |
| Programs | 46 | 46 |
| Kernel definitions | 4,700 | 1,328 |
| Launch sites | 4,700 | 4,700 |
| Generated CUDA bytes | 4,961,454 | 2,992,096 |
| Generated CUDA lines | 84,717 | 36,835 |
| Object bytes | 20,764,096 | 10,618,824 |
| Cache-miss compiler wall time | 271.65 s | 161.44 s |

The unit's source shrinks 39.7%, object size 48.9%, and compilation wall time
40.6% in this single local pair. This measures the **complete generated object**,
not a selected kernel, but is still not a complete wheel build or a runtime
method-endpoint measurement. Do not extrapolate it into a whole-wheel speedup
or claim improved runtime energy/force throughput.

Both commands use the existing, verified `sccache 0.16.0` and default shared
cache, with statistics retained before/after. Two top-level requests produced
eight CUDA/PTX/device/CUBIN misses, zero hits and zero compilation failures; no
cache was cleared or bypassed. NVCC arguments:

```text
-std=c++20 -O3 --fmad=false --expt-relaxed-constexpr
-gencode arch=compute_89,code=sm_89
-gencode arch=compute_89,code=compute_89
-DGENERATIVEQC_HAS_CUDA=1 -I <checkout>/src -I <checkout>/include
-I <variant-generated-directory> -c <variant-generated-source> -o <variant-object>
```

These local architecture flags describe only the comparison toolchain, not a
change to the wheel's `80;90-real;120-real` inventory. `/usr/bin/time` surrounds
the sccache client; its RSS excludes daemon-owned compiler workers and is **not**
compiler peak-memory evidence.

A one-off migration comparison checked all 4,700 launch sites: each corresponding
kernel body/signature agrees exactly after parameterizing only the entry name
and diagnostic ID. Every launch configuration and operand agrees, with the
original node number appended. The remaining host orchestration agrees too.
Regenerated CPU and response headers are byte-identical. This is investigation
evidence, deliberately not a permanent source-fingerprint CI requirement.
The unpooled RCCSD CPU header (1,968,242 bytes) and CUDA source (3,017,317 bytes)
also agree byte-for-byte with the parent emitter in a one-off comparison.

Permanent executable tests check independently written matrix/scalar formulas,
bitwise unpooled/pooled parity, empty/asymmetric runtime extents, arena guards,
NaN/infinity/overflow, signed zero-denominator errors, invalid runtime maps,
caller-specific diagnostic IDs, and the scalar-reduction provider fallback.
Contract tests keep differing arithmetic, layouts, dtypes, batch signatures and
reduction policies separate. Host CUDA launch shims are not real-GPU evidence.
The complete generated CUDA object also compiles successfully with NVCC; the
full manylinux wheel and real-device endpoint gates remain necessary.

## Rejected alternatives and revisit conditions

Do not combine every generated CUDA file into a mega-translation-unit: larger
compiler working sets can worsen hosted-runner pressure. Do not drop force
graphs, fold physical spin domains merely because extents match, or relax
compiler numerical modes. Automatic Libxc parameter sharing is already enabled;
enabling it again is not an optimization.

Consider broader pooling only after qualifying each emitter's ownership and
diagnostic contracts. Consider splitting the large CPU response header as a
separate change: this patch does not remove its host compilation cost and does
not establish why the historical runner disconnected. Retain whole-wheel cold
and warm qualification before revisiting the conservative resource envelope.
