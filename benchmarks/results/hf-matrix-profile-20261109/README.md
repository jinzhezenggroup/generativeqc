# Direct-HF matrix-profile diagnostic qualification

This is **routing and numerical qualification, not performance qualification**.
The CUDA image uses `GENERATIVEQC_CUDA_FAST_COMPILE=ON`; its endpoint times
cannot qualify release performance, a calibrated winner or a default promotion.
The records deliberately impose synthetic native/cuBLAS orderings, not measured
effective rates. No production calibration profile is distributed here.

## Provenance

- Execution source revision: `68795e7cf` (including parent PR #3052 updates).
- Native source identity:
  `a8118e137f1beb299cc4b882323ebea8258e8f09a29db75a47ac99434f05167f`.
- Executed library SHA-256:
  `cc3030c5080cdf6083be8fa31d7ffd23f06851ce67fefb7fd06e8080e9a4610e`.
- CUDA toolkit: 12.9.1; Release diagnostic image; native `sm_120` code.
- Node: `node5`, NVIDIA GeForce RTX 5090, driver 570.86.16.
- Slurm jobs: **2028** (three native fixtures), **2029** (matrix-profile memcheck).
  Both preserve Slurm-assigned `CUDA_VISIBLE_DEVICES=0` and use a ten-minute limit.
- `artifacts.sha256` identifies the library, executables and private OpenBLAS
  dependencies whose copied bytes were verified on the execution node.
  The logs also retain actual native/runtime/driver/cuBLAS ELF build IDs.
- Verified compiler launcher: ccache 4.5.1 for CXX and CUDA. The retained before
  and after counters belong to a shared cache, not exclusively to this build.

## Results and limits

`diagnostic.log.xz` retains successful resource-layout, incremental Direct-J/K
and matrix-profile fixtures. The matrix fixture checks 56 complete cached SCF
calls (84 item results): RHF/UHF, batch sizes one/two, energy-only/energy-forces,
no profile, native cold/warm, cuBLAS cold/warm/moved-warm, and stale-profile
fallback. It checks actual prepared bindings, capture reuse/invalidation,
iterations and physical Fock work, rather than only cost-model outputs.

Independent retained PySCF 2.14.0 HeH+/HeH STO-3G references gate energies at
`1e-8` Hartree and forces at `1e-7` Hartree/Bohr, including a 0.03-Bohr geometry
move. The exact system, reference values and gates are in
`tests/native/test_hf_matrix_profile_cuda.cpp` and
`tests/native/test_incremental_direct_jk_cuda.cpp`.

`diagnostic-memcheck.log.xz` repeats the new matrix fixture under CUDA Compute
Sanitizer with `--tool memcheck --leak-check full --error-exitcode 99`:
**zero errors and zero leaked bytes**.

The printed `owned_device_bytes` is not a complete device-memory peak.
There is no matched optimized baseline, angular-schedule promotion, measured
ERI profitability result, cross-device performance result or speedup claim.
Issue #597 remains open for these separate gates.

## Reproduction

Configure CUDA 12.9.1 with the relevant options in `build-options.txt` and a
verified CXX/CUDA compiler-cache launcher, then build:

```bash
cmake --build BUILD --target generativeqc \
  generativeqc_hf_resource_layout_tests \
  generativeqc_incremental_direct_jk_cuda_tests \
  generativeqc_hf_matrix_profile_cuda_tests
```

Execute the three fixtures through a finite-time Slurm allocation on a compatible
GPU. Repeat the new fixture under Compute Sanitizer in a separate allocation.
Do not modify device visibility or treat this diagnostic image as a release
benchmark. Recreated binaries have their own ELF identities; the test derives
compatible synthetic profiles from the actual loaded execution stack.

Verify the retained evidence files with `sha256sum -c sha256.txt` in this directory.
