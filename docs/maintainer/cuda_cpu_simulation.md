# CPU-interpreted CUDA smoke test (experimental)

The workflow [`cuda-cpu-sim.yml`](../../.github/workflows/cuda-cpu-sim.yml)
checks a **small, real GenerativeQC CUDA primitive** on a standard CPU-only
GitHub Actions runner using [PantheonSim](https://github.com/pantheongpu/pantheonsim).
It is not a substitute for native CUDA correctness and performance qualification.

## What is exercised

- CUDA 12.9 NVCC compiles [`test_cuda_cpu_sim.cu`](../../tests/native/test_cuda_cpu_sim.cu)
  for virtual `compute_80`, linking the shared CUDA runtime.
- The test includes the **production**
  [`compensated_atomic.cuh`](../../src/runtime/compensated_atomic.cuh)
  implementation. Its CUDA scatter kernel launches with several block/warp
  configurations. This is not a reimplementation of the device arithmetic.
- Six small cancellation-heavy FP64 cases compare the combined sum and
  correction planes with an independent exact dyadic result. A separate
  nonfinite case checks propagation.
- The pinned simulator interprets PTX on the CPU against its H100 profile.
  The job fails on failed compilation, a missing shared CUDA runtime,
  simulation/runtime errors, or incorrect outputs. It does not silently skip
  a missing simulated device.

The workflow is intentionally separate from the required CUDA 12.9 **compile**
job and the physical-GPU scientific acceptance suite. It runs on pull
requests that change the probe, its workflow, or the tested production header;
it also supports manual dispatch once the workflow exists on the default branch.

## What a passing result does *not* establish

- No actual NVIDIA GPU, J/K contraction, SCF/DFT endpoint, or gradient was run.
- No device timing, occupancy, memory bandwidth, numerical performance,
  cross-architecture reproducibility, or real-device memory safety was measured.
- The simulator may not implement a later production PTX instruction or vendor
  API. Such cases must be reported as **unsupported**, not as a pass.
- A CPU-oracle comparison under an emulator complements, but never replaces,
  the existing real-device and complete-endpoint gates.

## Reproduce locally

With NVIDIA's CUDA 12.9 compiler, a Linux build of PantheonSim and no
physical GPU:

```sh
nvcc -std=c++20 -O2 -arch=compute_80 -cudart shared -I src \
  tests/native/test_cuda_cpu_sim.cu -o /tmp/generativeqc-cuda-cpu-sim
VGPU_SASS=0 vgpu run --gpu nvidia/h100 /tmp/generativeqc-cuda-cpu-sim
```

The `-cudart shared` requirement is essential: the simulator replaces the
CUDA runtime dynamically. `VGPU_SASS=0` forces the PTX interpretation path,
making this a PTX correctness probe, **not** an SASS or performance result.

## Next expansion gate

If this probe passes reliably, try an isolated, small-shape production J/K or
DF contraction kernel with a truly independent CPU numerical reference and
bounded memory/work. Review its CUDA dependencies and generated-code identity
before promoting it to a broader CI gate. A successful simulator-only run
cannot authorize a production-path change or claim a GPU speedup.
