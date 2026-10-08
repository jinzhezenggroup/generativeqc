# Decision: use CUDA 13.0 for the pinned simulator probe

Status: implemented
Date: 2026-10-08

## Problem

PantheonSim `0cd395b000d2bd499d22f364fb48e2b68435ca7e` guards its
`cublasEmulationStrategy_t` alias with `CUBLAS_VER_MAJOR >= 13` in
`nvidia/src/cublas_api.cpp`. CUDA 12.9 already declares the enum and its
getter/setter, so the shim's fallback `int` signatures conflict with the vendor
header. The experimental run failed during simulator installation; our CUDA
primitive compilation and execution were skipped, not passed.

## Decision and boundaries

Keep the simulator source pin and select CUDA 13.0 for this isolated experiment.
The correct enum aliases are enabled there, and this exact upstream commit has
successful CUDA 13.0 simulator build/test evidence. This avoids a downstream
simulator patch or downgrading below GenerativeQC's CUDA 12.9 minimum.
The required CUDA 12.9 compile gate and physical-GPU scientific gates are
unchanged. Only a fresh GenerativeQC workflow run can establish whether our
primitive compiles and executes with this pair; upstream success is not that
qualification. No silent skip, numerical relaxation, or hardware claim is added.

## Evidence and revisit condition

- Failed GenerativeQC installation: <https://github.com/jinzhezenggroup/generativeqc/actions/runs/37803038977/job/113400366806>
- Upstream enum guard: <https://github.com/pantheongpu/pantheonsim/blob/0cd395b000d2bd499d22f364fb48e2b68435ca7e/nvidia/src/cublas_api.cpp#L508-L516>
- Successful upstream CUDA 13.0 simulator build and tests: <https://github.com/pantheongpu/pantheonsim/actions/runs/37778264038/job/113314989978>

Revisit when a reviewed, pinned upstream revision supports CUDA 12.9 or the
experiment needs a different toolkit. Requalify the actual primitive after any
source/toolkit change; never infer GPU qualification from simulator success.
