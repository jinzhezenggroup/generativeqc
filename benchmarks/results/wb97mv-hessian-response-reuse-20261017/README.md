# WB97M-V Hessian: bounded response reuse and parallel AD reductions

## Scope and result

Measured clean revision `51a87a04fdb8ef8e17bd8f6b848f2fcdc77f60af`, based on
`743c06bf80db5f04b43d78cb6b3a3ea423577c73` (merged PR #2590).
Slurm job **9950**, node **n1**, one RTX 5090, CUDA 12.9; GPU4PySCF **1.8.1**,
PySCF **2.14.0**, CuPy **13.6.0**. Slurm visibility was preserved. The node was
shared; this is not an exclusive-node or statistically general performance claim.

These are **STO-3G spherical, deliberately coarse 8/4/8 diagnostic grids**,
matched primary/NLC/CPHF moving grids, direct RKS. They are not chemically
converged quadrature, production-size scaling, UKS/DF/ECP speed rankings or a
general claim to outperform GPU4PySCF. UKS/DF/ECP correctness is tested separately.

Every engine/case has a first observed fresh-process sample plus three
artifact-primed fresh-process samples, with alternating engine order. Each sample
performs fresh SCF and publishes a complete host Hessian and JSON. The comparison
below uses the median of **each native sample's preparation + Hessian sum**
against the median reference Hessian, not Python startup or SCF advantages.
First samples and all phase/endpoint samples remain in `samples.json.xz`.

| Case | Native preparation + Hessian (s) | GPU4PySCF Hessian (s) | Reference / native |
|---|---:|---:|---:|
| water-dimer | 2.643700 | 2.873015 | 1.087x |
| formaldehyde | 1.102402 | 7.359160 | 6.676x |
| water | 0.446459 | 2.449993 | 5.488x |
| h2 | 0.128532 | 0.456874 | 3.555x |

| Case | Native full endpoint (s) | GPU4PySCF full endpoint (s) | Reference / native |
|---|---:|---:|---:|
| water-dimer | 4.855600 | 11.843334 | 2.439x |
| formaldehyde | 3.379704 | 24.554050 | 7.265x |
| water | 2.478935 | 9.185278 | 3.705x |
| h2 | 1.470686 | 5.617123 | 3.819x |

Native numerical execution remains linked AOT C++/CUDA compiler AD: no runtime
Python, runtime compilation, displaced SCF, numerical CPU/PySCF fallback,
intermediate panel downloads or posthoc Hessian symmetrization. The Python
benchmark controller/reference and independent qualification oracles are not
part of the native child. PTX JIT was disabled. Original SCF energy/density,
screening and response tolerances remain 1e-13/1e-11, 1e-14 and 1e-12/1e-11.

## Accuracy, work and resources

All 16 native/reference full-matrix pairs per case pass 3e-6 absolute error;
maximum observed error is **3.65142727166e-08 hartree/Bohr²**. All native matrices
also pass 3e-12 absolute/relative comparison with the four stored prior native
samples for each case. That prior evidence is
`../wb97mv-hessian-dispatch-20261017/samples.json.xz`; it is not relabelled as a
same-node clean timing baseline for this change.

| Case | Directions | Physical induced Fock actions | Cache columns | Cached actions | Physical residual checks |
|---|---:|---:|---:|---:|---:|
| water-dimer | 18 | 76 | 40 | 370 | 18 |
| formaldehyde | 12 | 56 | 32 | 244 | 12 |
| water | 9 | 28 | 10 | 90 | 9 |
| h2 | 6 | 13 | 1 | 0 | 6 |

In the dimer, physical induced Fock work falls **406 -> 76**, not just launch
overhead: 40 physical cache columns plus two physical Fock actions per nuclear
direction. Every direction retains a true physical stationarity-residual check.
Cached GMRES matrix actions are separately counted; a zero RHS may require none.
The operator's default cap is 128 KiB under the existing response/aggregate
budget. Larger spaces, allocation failure or explicit disable use physical GMRES.
One-electron parallel slots and moving-grid graph staging similarly retain
bounded ordinary-CUDA fallbacks. FP64 cooperative tree reductions deliberately
change summation order and are covered by independent oracles and strict parity.

| Case | Native sampled total-device GiB | Reference sampled total-device GiB |
|---|---:|---:|
| water-dimer | 16.858 | 2.664 |
| formaldehyde | 16.826 | 2.664 |
| water | 16.791 | 2.680 |
| h2 | 16.774 | 2.654 |

Memory is **50 ms sampled total-device use**, including SCF reservations,
contexts and other device consumers, not exact isolated peak or Hessian arena
bytes. Native total-device use is substantially higher than the reference in
these cases. All samples pass the existing **24 GiB** watchdog and **180 s**
endpoint deadline; these limits were not enlarged. No clean compilation-cost
measurement or exact owner-attributed GPU peak was made, so the validation
envelope deliberately retains `performance: not-run` for formal broad promotion.

## Larger controls and qualification

- Final-build ethane, job 9954: native first/primed endpoint 8.825015/8.896051 s; sampled total-device maximum 16.895 GiB. Native-only, not a qualified reference comparison.
- Final-build methanol, job 9954: native first/primed endpoint 5.248561/5.271887 s; sampled total-device maximum 16.858 GiB. Native-only, not a qualified reference comparison.

The earlier paired ethane/methanol attempts failed in GPU4PySCF SCF at the
original 240-cycle limit and original tolerance. They have **no reference
Hessian** and cannot establish a cross-engine speed ranking. Both failed
journals and stderr are retained; no easier reference tolerance/initial guess
or native cached-state substitution was introduced.

- Final rebuild job **9952**: 221 component tests passed, 2 skipped; seven selected coupled/independent tests passed. Jobs **9955/9956** additionally pass final RKS/UKS DF+ECP+spherical complete-Hessian cached/physical/serial fallback parity.
- Final rebuild job **9953**: memcheck and initcheck on coupled cached CPKS,
  spherical parallel/scalar one-electron response and moving-grid graph/serial/
  tight-capacity fallback; zero sanitizer errors.
- Host: 40 passed, 163 GPU skips; compiler structure: 565 modules, zero errors.
- Final CLI/library/probe hashes, cache version/statistics, incremental build
  log, compiler commands, source delta and exact qualification scripts are in
  `provenance.json.xz`. These are incremental cached builds, not clean-build cost.
- Final rebuild job **9958**: instrumented dimer kernel/API summaries, retained
  separately from clean timings. Kernel-time sums identify remaining integral,
  VV10 and geometry work; they are not disjoint endpoint phase timings.

Negative evidence remains explicit: early deployment omitted pinned Libxc inputs
and the checkout marker (harness failures); same-order cooperative prefetch did
not remove the reduction bottleneck; serial contraction epilogue fusion retained
low occupancy; the large 21-test coupled cross-product job 9920 hit its 20-minute
deadline and is not called a passing suite. Its zero-RHS cache-action assertion
was a test mistake, corrected without changing numerical gates (job 9934 passed).
The final bounded selection checks RKS/UKS independent direct and DF/ECP/spherical
HVPs, strict direct full-matrix fallback parity and an independent H2 full Hessian.
An initial sanitizer wrapper deployment failed under a deliberately empty PATH;
the corrected direct-ELF wrapper passed. Interrupted/redundant jobs and all
completed samples remain labelled. Intermediate and final profiles are
instrumented diagnostics, never clean timings or speedup evidence. Full traces and
binaries stay in ignored artifacts; no release/archive was published.

## Reproduction

`reproduce.sh` checks the exact measured binary hashes, archives the measured
clean source commit and reruns the paired benchmark. Supply recorded prebuilt
artifacts, the reference environment and CUDA/cuTENSOR runtime paths:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:15:00 env \
  SOURCE_CHECKOUT="$PWD" NATIVE_ROOT=/path/to/measured-artifacts \
  REFERENCE_PYTHON=/path/to/reference-env/bin/python \
  REFERENCE_LIBRARY_PATH=/path/to/cutensor/lib:/path/to/cuda/lib \
  bash benchmarks/results/wb97mv-hessian-response-reuse-20261017/reproduce.sh
```

For source rebuilds, use the measured revision and the native Hessian AOT build
contract in `docs/developer/dft_hessian.md`: Release, sm120, CUDA enabled,
`GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON`,
`GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON` and an empty inventory override for
the complete orbital-f/auxiliary-g closure. Verify sccache, else ccache, with
`--version` first and set both CXX/CUDA compiler launchers explicitly. The
recorded build used ccache 4.5.1 and fast compile OFF. Build the CLI and the five
probes listed in `provenance.json.xz`. Rebuilt binary hashes can differ; preserve
new build identities instead of calling them the measured artifacts. Run every
real-GPU qualification/benchmark/sanitizer under finite `srun` with its assigned
visibility. There is no runtime compiler-cache or Python requirement in the CLI.
