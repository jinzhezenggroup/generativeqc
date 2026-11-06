# Same-source H100 and RTX 5090 CCSD(T)/CBS cohorts

Both GPUs pass the original 13-population and QM7 C1–C7 gates on unmodified source `7109b3d208d23352f266e2bf7440fdfcc71df974`, tree `33dff27444736e511f6679a37d64f46eaea42b09`. The common native input identity is `94d7b0c9b4275ea7300ca9d7d237103072c0992fdc2606eba5aa229d73129b75` (2092 inputs). This adds measurements and a CPU-only reader; production implementation is unchanged.

The two new curves use **the same source and scientific protocol**. They replace neither the historical H100 `6707989c` observations nor the old RTX 5090 `98531471` plus candidate5 observations. Those publications remain untouched. The post-`6707989c` closed-shell DF fallback change in #2982, alongside the integrated #3007/#3008 changes, is included in both new builds.

## Exact inputs and scientific scope

The frozen scientific inputs and independent reference records are `cbs-routine-standard-20261025` at `4370d6c4b1bf4033818f31a9fcc622d3045a33bf`, records SHA-256 `03b2722d016c13daaf6025a021de18e0ac333ebbb6efe94943c57748a57f5d3a`. Prepared basis bytes and identities are identical on both GPUs: 11 full-DF populations, two correlation-only DF populations with exact RHF, and QM7 C1–C7 at zero-based indices 0, 1, 8, 48, 183, 293, 5604. Full-DF QM7 uses all electrons, cc-pVDZ/cc-pVTZ extrapolation plus the cc-pVDZ correction and matching JKFIT. Diffuse hydroxide retains augmented orbital and JKFIT bases. No new independent-oracle solve was performed.

Unchanged gates are energy `1e-7 Eh`; maximum force `3e-6 Eh/Bohr`; composite RMS and two-step finite-difference stability `1e-6 Eh/Bohr`; constituent energy locality `0.01 Eh`; energy-only parity `1e-8 Eh`. Full independent Cartesian force coverage applies to the 13 small populations. QM7 retains same-source tight controls and the carried independent energy/directional checks. Original individual displaced C2 energies remain unavailable; C4/C7 directional references are not full-vector references.

Stretched N2 retains the same internally unstable RHF branch and independently reconverged local displacements. Agreement does not establish ground-state/root selection, dissociation accuracy, true CBS-limit accuracy, vibrational accuracy, force continuity or long-time energy conservation.

## Complete default endpoints

Every timed call constructs a fresh outer CBS calculator and includes inner preparation through returned host energy and analytic forces. Prepared basis tables, imports, process/CUDA startup and compilation are excluded from the default samples. Each process retains its first complete startup call separately. Default and tight calls remain distinct in raw records.

**Two calls provide descriptive medians and ranges only**, without confidence intervals or statistical performance promotion. The comparison applies **only to this configuration**, without a general hardware acceleration ratio. There is no matched GPU4PySCF CBS timing denominator. Host CPUs, Python environments and drivers differ as recorded below; these endpoint observations do not isolate an intrinsic GPU-only hardware effect.

### H100

| QM7 case | Atoms | Default median (s) | Two-call range (s) | CCSD iterations, first default | Final CC residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 5 | 1.138 | 1.133–1.144 | 12 | 3.991e-10 |
| C2 | 8 | 3.202 | 3.193–3.212 | 13 | 1.461e-09 |
| C3 | 11 | 8.892 | 8.864–8.921 | 13 | 3.758e-09 |
| C4 | 14 | 23.039 | 22.973–23.105 | 13 | 7.949e-09 |
| C5 | 17 | 55.344 | 55.324–55.365 | 13 | 1.074e-08 |
| C6 | 20 | 125.014 | 124.959–125.070 | 13 | 1.343e-08 |
| C7 | 23 | 267.271 | 267.205–267.337 | 13 | 1.517e-08 |

| Population | Energy error (Eh) | Force maximum (Eh/Bohr) | Force RMS (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| full-df--water | 7.082e-09 | 9.996e-09 | 5.021e-09 |
| full-df--water-stretched | 1.456e-08 | 2.266e-08 | 1.011e-08 |
| full-df--ammonia | 6.032e-09 | 1.151e-08 | 6.401e-09 |
| full-df--ammonium | 1.007e-10 | 1.729e-08 | 9.176e-09 |
| full-df--hydroxide | 1.198e-09 | 3.800e-09 | 1.866e-09 |
| full-df--formaldehyde | 6.131e-10 | 3.718e-08 | 1.483e-08 |
| full-df--nitrogen | 1.649e-09 | 1.394e-09 | 8.049e-10 |
| full-df--water-dimer | 1.140e-09 | 2.507e-08 | 9.004e-09 |
| full-df--hydroxide-diffuse | 2.846e-09 | 1.898e-08 | 8.928e-09 |
| full-df--hydrogen-stretched | 1.155e-10 | 4.852e-08 | 3.374e-08 |
| correlation-df--water-stretched | 1.454e-08 | 1.455e-08 | 8.227e-09 |
| correlation-df--ammonium | 1.004e-10 | 3.352e-08 | 1.662e-08 |
| full-df--nitrogen-stretched | 5.581e-09 | 1.190e-08 | 6.730e-09 |

Small-population errors are maxima over the two default complete calls.

### RTX 5090

| QM7 case | Atoms | Default median (s) | Two-call range (s) | CCSD iterations, first default | Final CC residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 5 | 3.211 | 3.182–3.240 | 12 | 3.992e-10 |
| C2 | 8 | 8.597 | 8.594–8.600 | 13 | 1.461e-09 |
| C3 | 11 | 24.579 | 24.567–24.591 | 13 | 3.758e-09 |
| C4 | 14 | 67.264 | 67.231–67.297 | 13 | 7.949e-09 |
| C5 | 17 | 172.382 | 172.368–172.396 | 13 | 1.074e-08 |
| C6 | 20 | 464.892 | 464.739–465.044 | 13 | 1.343e-08 |
| C7 | 23 | 1018.547 | 1016.686–1020.408 | 13 | 1.517e-08 |

| Population | Energy error (Eh) | Force maximum (Eh/Bohr) | Force RMS (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| full-df--water | 7.082e-09 | 9.995e-09 | 5.020e-09 |
| full-df--water-stretched | 1.456e-08 | 2.267e-08 | 1.011e-08 |
| full-df--ammonia | 6.034e-09 | 1.151e-08 | 6.401e-09 |
| full-df--ammonium | 1.008e-10 | 1.729e-08 | 9.176e-09 |
| full-df--hydroxide | 1.198e-09 | 3.801e-09 | 1.866e-09 |
| full-df--formaldehyde | 6.138e-10 | 3.718e-08 | 1.483e-08 |
| full-df--nitrogen | 1.649e-09 | 1.396e-09 | 8.060e-10 |
| full-df--water-dimer | 1.140e-09 | 2.507e-08 | 9.004e-09 |
| full-df--hydroxide-diffuse | 2.845e-09 | 1.898e-08 | 8.928e-09 |
| full-df--hydrogen-stretched | 4.691e-10 | 4.852e-08 | 3.374e-08 |
| correlation-df--water-stretched | 1.456e-08 | 1.458e-08 | 8.243e-09 |
| correlation-df--ammonium | 1.004e-10 | 3.352e-08 | 1.662e-08 |
| full-df--nitrogen-stretched | 2.088e-08 | 1.133e-07 | 5.909e-08 |

Small-population errors are maxima over the two default complete calls.

## Device, software and allocation provenance

| GPU | Actual device UUID | Driver | Core SHA-256 |
| --- | --- | --- | --- |
| NVIDIA H100 80GB HBM3 | `GPU-05a9b210-be43-f1ba-c2ed-7c61f2c44b19` | 595.58.03 | `03f62c5c671e4f6691d67aebf327126e92fa7a7589687615e3160dbf53c1808a` |
| NVIDIA GeForce RTX 5090 | `GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d` | 580.95.05 | `05d8ce6027f79fe463a7bfe2dcc699026e2236bd6186e8a4d8ad960bd9257a3e` |

Both Release builds use CUDA 12.9, GCC 12, the `portable_cuda` AOT profile and the original H100 stationary-AOT-off build settings; only target architecture is SM90 versus SM120. Source identity, full file inventory, exact configure commands, selected compiler-command receipts, launcher/version and before/after statistics are retained or recovered from frozen Git. Successful verbose compiler streams remain ignored; their byte counts/digests, actual representative commands and exception lines are retained in `build-verbose-trace-receipts.json`. Each loaded core reports the common native source identity before science. No source patch was applied.

H100 image: `generativeqc-science-py311-cuda129:20261015`, with a task-private Python 3.11 environment overlay. RTX 5090 uses a native USTC host and a task-private Python 3.12 environment overlay, without a container. Exact package versions are retained in each build receipt. Both use private LP64 `scipy-openblas32==0.3.30.0.6`; the existing H100 base environment remains unchanged. This differs from its old OpenBLAS dependency and is therefore a new matched configuration, not a relabelled old curve.

Both set one math thread and explicitly set `CUBLAS_WORKSPACE_CONFIG=:16:8` before native imports. Numeric budgets remain 32 GiB for QM7 and 8 GiB for the small populations; the old RTX 5090 publication already used the same QM7 budget. **No numeric budget was changed.** Numeric capacities are not measured peak VRAM/RSS; allocator-exact peak telemetry was not collected.

H100: qz Job `gqc-cbs-dual-h100-1106-v1`, CQ-NexChem, `开发区-H100-cuda13.2版本-119核`, one GPU / 10 CPU / 200 GiB and 8 GiB shared memory. Requested priority 4 is recorded by the platform as NORMAL/20. The six-hour combined build/measurement ceiling includes a finite three-hour science window. The authenticated running allocation and platform preflight log bind the actual source, core, input, producer and worker/device identity before the first solver call.

RTX 5090: USTC node1, CPU build job 16607 followed by GPU job 16608; one GPU, 10 CPU and 200 GiB, with a three-hour measurement ceiling. Requeue is disabled and the retained scheduler records show zero restarts. All C1–C7 points remain on the same recorded UUID on each GPU. No automatic scientific retry or replacement Job was used.

A 5090 build-time sccache service warning records a local compiler fallback. It is retained verbatim; the build still passes the source/core identity check. Cache snapshots are execution provenance and are not advertised as a complete-build hit percentage after that service event. Compilation is outside endpoint timings.

## Retained work and CPU replay

Each GPU retains **73 primary E+F calls, 20 separate startup E+F calls and 13 energy-only calls**. Every E+F composite and constituent has finite physical-shape Cartesian forces, including startup and the second tight repeat; every energy-only force field is absent. Per-call constituents, execution groups, CC/Lambda work and physical residuals remain in losslessly recoverable raw records.

All new scientific observations retain their exact original bytes and digests. Reversible JSON/IEEE-bit deltas use the immutable historical H100 capsule or an already recovered new H100 member only as storage dictionaries. Historical native values never certify a numerical gate: acceptance still uses the pinned independent references and declared same-source controls. Every reconstructed observation must match its original digest before numerical replay.

The full 14742-file source inventory is reconstructed byte-for-byte from frozen Git. Worker per-case/aggregate verification outcomes remain in the supervisor and stdout/stderr; detailed quantitative gate reports are regenerated from the exact new observations and independent references. This avoids duplicating derived reports or successful verbose compiler output while keeping the unchanged storage limit. The compressed member inventory authenticates every retained/reconstructed raw member.

Use unoptimized Python 3.11+ with NumPy and a full Git clone containing the source and reference commits. This runs retained numerical readers only, with no native solver, CUDA, compiler or network work:

```sh
python benchmarks/results/cbs-h100-5090-7109b3d2-20261106/replay.py
# Optional lossless recovery to a new, nonexistent directory:
python benchmarks/results/cbs-h100-5090-7109b3d2-20261106/replay.py --recover /new/empty/path
```

Numerical/device evidence, required CI, independent exact-head PR review and protected delivery are separate states. No default/performance promotion or broader parent-issue completion is implied.
