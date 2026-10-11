# Issue #2136: FP64 Lambda core retention and independent audit GEMM

## Frozen protocol

Complete cold spherical ethane DF-CCSD(T) E+F: aug-cc-pVTZ / aug-cc-pVTZ-RI,
230 AO, o=9, v=221, q=488, conventional unscreened RHF, correlation-only DF,
symmetric metric inverse square root with relative cutoff `1e-10`, and a 64 GiB
complete numeric budget. Production receives only molecular input, never oracle
orbitals, factors, amplitudes or response.

The source is master `957fd60b6fffa267918ab597ceb12fe2b1a8cd81` plus only
#2136's reconstruction patch; #2128 is already merged. Earlier prototypes and
the screen on a different GPU are not pooled into these measurements.
Qualification used an isolated detached worktree and left the original dirty
user worktree untouched.

Slurm 2749 on n2/node2 uses `main`, `gpu:pro6000:1`, assigned visibility, and a
finite four-hour limit. GPU: PRO 6000
`GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, driver 595.91.07, 600 W.
Build: CUDA 12.9.1, Release/sm120/portable-CUDA AOT, fast compile off, verified
ccache 4.5.1. Independent reference: PySCF 2.14.0.

Eight W/cadence/candidate configurations each execute twice in fresh processes,
first forward order, then reverse. `r0` disables both optional optimizations;
`r1` requests and actually admits both. This is a combined candidate, not a
cache-only or audit-only speedup. The mandatory independent audit still uses
original expanded retained equations and original virtual amplitude VJP, not
the solver's staged cuts or cached core values.

## Complete clean measurements

Seconds are medians; brackets are the two observed extrema, not confidence
intervals. Complete native E+F parents and separate process startup/teardown
timings are retained at full precision.

| W | Cadence | Control E+F seconds | Candidate E+F seconds | Lambda medians control/candidate | E+F saving |
| --- | ---: | --- | --- | --- | ---: |
| FP64 | 1 | 669.223 [668.506, 669.940] | 584.742 [584.733, 584.751] | 270.885 / 186.902 | 12.624% |
| FP64 | 30 | 593.991 [593.955, 594.028] | 518.652 [518.648, 518.656] | 196.191 / 120.836 | 12.684% |
| FP32 | 1 | 660.575 [660.498, 660.652] | 576.975 [576.966, 576.983] | 270.757 / 186.887 | 12.656% |
| FP32 | 30 | 586.116 [586.094, 586.137] | 510.962 [510.954, 510.970] | 196.211 / 120.850 | 12.822% |

All Lambda solves use 21 iterations: 42 actions at cadence one, 22 at cadence
thirty, independent of candidate selection. The `<120 s` stretch target is
**not reached**: do not round 120.836 or 120.850 down and claim it.

Strict W/cadence thirty parent medians:

| Seconds | Control | Candidate |
| --- | ---: | ---: |
| RHF | 129.208 | 129.218 |
| DF source construction | 3.616 | 3.605 |
| CCSD | 99.874 | 99.873 |
| (T), including force response | 39.366 | 39.374 |
| Corrected Lambda + parameter/factor VJP | 196.191 | 120.836 |
| Source nuclear response | 3.785 | 3.786 |
| Orbital/nuclear response | 121.952 | 121.961 |
| Complete native E+F | 593.991 | 518.652 |

Complete saving is 75.339 s versus 75.355 s within Lambda, not RHF/CCSD or
orbital fluctuations. CLI-process medians are 594.368 / 519.007 s. Nested
action times must not be added to GMRES or Lambda parents.

## Scientific qualification

All 16 endpoints pass independent same-Hamiltonian energy (`3e-9 Eh`) and
directional-force (`3e-7 Eh/bohr`) gates at both `1e-4` and `3e-5` Bohr.
Centered FD geometry is verified against the frozen input. The fresh PySCF
energy reference is `-79.71851664319477 Eh`; no reference state enters production.

| Check across all configurations | Maximum error | Acceptance |
| --- | ---: | ---: |
| Independent total energy | 2.637e-11 Eh | 3e-9 Eh |
| Independent directional force, both steps | 7.589e-9 Eh/bohr | 3e-7 Eh/bohr |
| All 24 components versus common strict baseline | 2.368e-11 Eh/bohr | 3e-7, plus stricter 3e-9 sanity |
| Translation defect | 1.088e-12 Eh/bohr | 3e-8 Eh/bohr |
| Independent Lambda residual | 6.117e-13 | 1e-9 |
| Exact Z residual | 1.357e-13 | 1e-9 |
| Stationarity | 7.673e-12 | 1e-9 |

The full-component comparator is a common native strict baseline, not an
independent analytic oracle for every component. Independent forces are two
directional energy finite differences; global RHF stability is not certified.
Original stricter native checks are unchanged.

The frozen library passes 76 GPU tests, including tails, provider/budget refusal,
nonfinite/no-publication, equal-shape changed Hamiltonians, small denominators
and cancellation from common Fock shifts. Fifteen memcheck cases have zero
errors. Host checks pass 36 regressions (54 conditional skips), 13 ownership
checks, native GMRES restart/stagnation/exhaustion contracts, and 501-module
dependency validation. The initially mismatched fallback test now explicitly
matches Q batch two; its bit-equality assertion was not relaxed. Cancelled
prototype/source/GPU-allocation-switching runs do not enter the timing samples.

## Work, resources and profiles

Strict W/cadence thirty changes GEMM requests 42,726 -> 45,986, semantic summands
40,429,634,645,741 -> 38,135,799,179,463, and generated/audit launches
231,740 -> 183,273. Logical packing-output bytes **increase**
1,596,162,795,584 -> 1,881,569,130,992 as scalar audit contractions become GEMM.
Complete capacity grows 6,844,452,897 -> 10,081,975,949 bytes, including exclusive
core/audit arenas of 1,710,877,464 / 1,525,829,744 bytes. Capacity is not measured
VRAM peak; summands are not hardware FLOPs; packing bytes are not DRAM traffic.
Resource refusal preserves original bounded execution. Only allocation shortage
permits retry, never nonfinite, numerical, binding or driver errors.

New instrumented complete endpoints run after the clean campaign in the same
allocation/library/GPU. `profile-summary.json` records immediate Lambda phases
and nested physical actions separately. Nsight Systems records observed whole
E+F CUDA launches/API/copies, not Lambda-only traffic; kernel/API sums can overlap
and are not wall time. Launch register/block metadata is not achieved occupancy.
Hardware-counter availability is reported explicitly; no driver/Slurm permission
bypass is attempted.

Largest remaining parents are RHF (~129 s), orbital/nuclear response (~122 s),
Lambda (~121 s), and CCSD (~100 s). Further owners should be ranked by removable
complete-parent time, not another forward-W microkernel. Q-resident staging or
new audit batching needs separate resource/lifetime and scientific qualification.

## Reproduction and retention

`publication.json` selects this compact local review bundle. Logs, retries,
compiler products, source archive and profiler databases stay ignored at
`n2:/data/jzzeng/qc-2136-lambda-master-20261009/` and `.artifacts/issue2136/`.
No Release, release tag/asset or external recovery host is created.

`summary.json` binds binary/input/oracle/FD SHA-256 identities. Reconstruction
patch decoded SHA-256 is `a0f90de6cac8733515f4488d0681a64b8d0abea6868386f1b54c880bbdf550c4`;
library SHA-256 is `50beaeb8f14d767135a9a925a00b9f1485bbd740ce002a6190c86ffe8fe664b6`.
Independent equations, lowering/reuse plans and schedule identities stay separate.

In a separate clean checkout of the frozen base, with the retained bundle at
`$evidence` and CUDA 12.9.1 at `$CUDA_HOME`, reconstruct and build:

```bash
gzip -dc "$evidence/measured-source.patch.gz" | git apply
ccache --version
cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_AOT_PROFILE=portable_cuda \
  -DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build-cuda --target generativeqc -j8
mkdir -p build
ccache g++ -std=c++20 -O2 -DGENERATIVEQC_HAS_CUDA=1 -Iinclude -Isrc \
  -I"$CUDA_HOME/include" -c benchmarks/df_ccsdt_force_endpoint.cpp -o build/endpoint.o
g++ build/endpoint.o build-cuda/libgenerativeqc.so \
  -Wl,-rpath,"$PWD/build-cuda" -o build/endpoint
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=04:00:00 python benchmarks/df_lambda_core_reuse_ablation.py \
  --endpoint build/endpoint --library build-cuda/libgenerativeqc.so \
  --input benchmarks/results/rccsd-diis-ring-1900/ethane230.input \
  --oracle "$evidence/oracle-energy.json" \
  --finite-differences benchmarks/results/df-lambda-gemm-20261004/oracle-energy-fd.json \
  --output .artifacts/issue2136/reproduction --repetitions 2
```

Match captured compiler/Python dependencies and keep assigned Slurm visibility.
`samples.json.gz` losslessly stores identical endpoint metadata once: decompress and merge
`shared_endpoint_fields` with each observation's `endpoint`. Use the shared
publication reader for plain/gzip members.

The envelope accepts numerical/complete-endpoint evidence, not formal performance
promotion. Two repeats per cell meet #2136's ablation requirement but not the
shared five-pair promotion gate; complete solver trajectories and measured VRAM
peak are not exported. No confidence interval or automatic promotion is inferred.
