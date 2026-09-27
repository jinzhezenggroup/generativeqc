# CUDA KS small-native solver-region replay

Date: 2026-09-27

## Decision

Advance #370 with one deliberately narrow replay consumer rather than enabling
CUDA Graphs for every KS shape. Direct all-electron strict-FP64 RKS may request
`VIBEQC_CUDA_KS_REPLAY=1` on top of the existing two-step
`VIBEQC_CUDA_KS_CHUNK=2` qualification route. Replay is admitted only when
the prepared KS eigensolver is the capture-safe small-native family (currently
at most 16 AOs). Larger provider-backed `Xsyevd` execution remains ordinary.

The shared `SolverRegionCudaExecutor` and `CudaGraphRegion` own warmup,
capture, replay, invalidation and fallback. KS continues to own convergence,
DIIS/history, occupations, density/warm publication and failure semantics.

The replay callback is device-only: it enqueues the stable XC/Fock/SCF body
without publishing host generations or evaluation counters. After the shared
runtime has selected and successfully submitted exactly one warmup, captured
launch, cached replay or ordinary fallback, KS publishes exactly one logical XC
generation per physical step. Capture probing can therefore invoke the device
body without double-counting host diagnostics. Replay binding includes the
mutable warm-update policy so toggling that policy invalidates the captured
shape.

## Failure boundary

A submission error, asynchronous completion error or invalid device-control
iteration count invalidates the shared replay executable before the KS owner can
be reused. Scientific convergence failure remains method-owned and does not by
itself redefine the graph shape.

## Validation boundary

Host lifecycle coverage asserts the distinction between capture launch and
cached replay. Native CUDA coverage now requires the small-native eigensolver to
capture and replay, while the allocated 24-AO ordinary-solver test requests KS
replay and must remain outside capture.

No real-GPU endpoint or performance claim is made by this note. The configured
node3 GPU runner was unavailable during this implementation pass; #370 remains
open until allocated cold/warm/changed-geometry and ragged/failure-isolation
qualification is recorded.

Agent: ChatGPT
Model: GPT-5.6 Sol
