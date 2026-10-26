# Decision: retain the response reuse audit; reject scalar fusion promotion

Status: rejected (experiment); remaining forward residency work is unresolved
Date: 2026-10-27

## Problem and decision

Issue #2516 requires source-grounded evidence before further CCSD(T) response
fusion. At `fa5a5c5613e816924bfe0b03734ecb3bad2cf5f9`, the local reverse W,
V and panel programs have **no common contractions** after existing TensorIR
optimization. Separate scalar response and full-Fock resolvent programs do share
work. Composing them with ordinary multi-output TensorIR and existing exact CSE
reduced device work, but improved complete cold E+analytic-F wall time by only
0.2–0.4%. Keep production unchanged and retain the experiment as negative evidence.

This is **not a global no-op finding or a recommendation to close #2516**.
Forward W/panel recomputation and parameter/core sharing remain measurable
opportunities. Narrow subsequent work to generic bounded producer residency and
cost selection in #783/#1889, with CC binding and complete scientific/endpoint
qualification in #2516. Do not add a CC-specific CSE engine or a second equation
implementation. #1763's existing scalar fusion and W reuse, and #1626's three
already-retired RCCSD(T) projections, are not gains from this audit.

## Actual path and legality

The DF force entry is `src/methods/df_ccsdt_force.cu`, not conventional
`rccsdt_force.cpp`. It invokes `pullback_and_fock_df_cuda` in
`src/cc/df_triples_cuda.cu`, then corrected Lambda and parameter/factor publication
in `src/cc/df_lambda_cuda.cu`, then DF source/nuclear and orbital response.
The prepared programs come from `occupied_triples_response.py`,
`occupied_triples_fock.py`, and `tools/generate_df_lambda.py`.

| Boundary | Physical identity, demand and lifetime |
| --- | --- |
| Forward panel | The same occupied `Bov_i` view and immutable `Bvv`, Q extent and FP64 contraction produce a v-cube. Three resident panel slots are evicted and rebuilt during the Fock traversal. An occupied index identifies a value only within this fixed-amplitude/factor epoch. |
| Forward W | Ordered `(i,j,k)`, the occupied panel, `t2_kj`, `ovoo_ij` and `t2_mk` determine the cube. Different ordered triples are not interchangeable merely because shapes agree. Existing repeated-index copies and sharing between Fock and triangle pullback already avoid some construction. |
| Reverse W | Four demanded cotangents: panel, `t2_kj`, `ovoo_ij`, `t2_mk`. The gathered scalar seed is permutation-specific. Repeated physical destinations still require each ordered accumulation. |
| Reverse V | Four demanded cotangents: `ovov_ij`, `t1_k`, `t2_ij`, `fov_k`. Its seed and operands differ from reverse W; similar dimensions do not make these four products common. |
| Reverse panel | Two products from the accumulated panel seed. `bar_bov_i` is strided; dense `bar_bvv` is accumulated over occupied panels before the upstream symmetric-factor boundary. It is not a third copy of either moment VJP. |
| Scalar response/resolvent | 36 `v_OCC_VIR` inputs bind identical ordered occupied/virtual views; 37 add nodes are common. Six denominator inputs remain distinct from resolvent gap inputs. Preserve ordered FP64 subtraction, signs and the existing 6/2/1 multiplicity exactly once. |
| Retained parameters/final core | Eight parameter programs consume the same converged Lambda and immutable primal inputs. The final factor publication uploads that same Lambda and recomputes core seeds. Intermediate Krylov vectors, fresh residual replays and original-equation audits are separate epochs/validation computations, excluded from reuse. |

Each native owner submits its dependent contractions/copies on its own stream;
scratch is reused only after its consumer in that order. A logical common node
does not establish that its physical buffer survives across prepared programs.
Sharing would need explicit producer/output identity, layout, Q/tile range,
stream dependency and lifetime, including resource refusal. The audit admits
only these strict FP64 frontends and rejects execution precision contracts;
it is not a mixed-precision composition mechanism. Exact CSE retains ordered
operands, coefficients, index identities and dtype; no reassociation or symmetry
rewrite is inferred. Never reuse independent validation work.

## Disjoint source-matched ledger

The baseline NVTX-only overlay names physical panel/W bindings and prepared
consumers. CUDA runtime correlation IDs associate device activity with the
innermost enclosing range on the submitting thread. Each activity is assigned
once. Parent host durations are **not** added to child GPU times. Profiled device
times are diagnostic ceilings, not clean endpoint speedups.

230 AO, o=9, v=221, Q=488, n2 RTX PRO 6000, Slurm job 3579:

| Region | Calls | GPU kernels | GPU seconds |
| --- | ---: | ---: | ---: |
| Forward panel | 357 | 714 | 2.499309264 |
| Forward W | 1,953 | 7,812 | 6.932915115 |
| Resolvent | 405 | 405 | 1.678147211 |
| Existing fused scalar response | 165 | 165 | 1.521451827 |
| Reverse V | 990 | 9,900 | 0.155946762 |
| Reverse W | 990 | 8,910 | 8.069100463 |
| Reverse panel | 405 | 1,620 | 6.262032700 |
| Fock vv | 405 | 1,620 | 3.139299217 |
| Fock oo | 45 | 270 | 1.232758775 |
| Remaining joint kernels | 1 parent | 2,191 | 0.013233565 |
| Eight retained parameters | 8 | 480 | 0.268752698 |
| Factor publication, including final core | 1 | 3,222 | 3.074587124 |

Joint response kernel sum is 31.504194899 s, excluding parameter/factor rows.
The joint parent also submits 497 copies: H2D 263,060,792 bytes, D2H 263,450,340,
D2D 41,189,373,576; total 41,715,884,708 bytes and 0.065153633 device seconds.
These are copy payloads, not measured DRAM traffic. Complete endpoint reports
retain packing, GEMM summands, provider capacities, Q batches and refusal flags.
Other numerical allocations and pack kernels are not relabeled as removable
traffic merely because their bytes are large.

Ranked surviving opportunities, with conservative interpretation:

1. **Forward W retention:** 729 ordered physical identities, 1,224 repeated
   constructions, 4.343361098 s repeated device time (about 2.02% of the 215.225 s
   baseline wall). This is an unattainable-without-cost upper bound. Retaining all
   W cubes would consume 62,949,797,352 bytes for values alone, effectively full
   rank-six storage. Only a bounded subset with generic lifetime/cost planning
   is admissible; no such schedule was implemented or qualified.
2. **Forward panel retention:** nine identities, 348 repeats, 2.435951768 s
   repeated device time (1.13% wall ceiling). Expanding three slots to nine costs
   another 518,105,328 bytes. It may reduce useful Lambda/replay capacity or Q
   batches; holding all panels was not attempted. These device rows are disjoint
   from W, but jointly achievable savings still require a complete schedule.
3. **Scalar response/resolvent composition:** only 165 of 405 resolvents can share
   the existing triangle response. Their measured baseline time is 0.689868381 s
   (0.32% wall ceiling); additional fused work lowers the net saving. Tested below.
4. **Parameter/core composition:** exact CSE removes 47 einsums across eight
   parameter programs (65,578,232,203 logical summands), or 85 with the final core
   (98,468,902,265). These are structural ceilings at runtime extents, not FLOPs
   or hardware work. The eight-parameter phase is only 0.268752698 s (0.12% wall);
   the 3.074587124 s factor phase also contains unavoidable factor work, so its
   full duration cannot be claimed as removable. A finer producer ledger and
   generic cross-program storage plan are prerequisites to a promotion claim.

The audit uses codegen representatives `(2,3)` and scales their existing
contraction trees to runtime extents; it does not re-plan contractions at those
extents. The ten products in local W/V/panel pullbacks remain distinct.

## Rejected experiment and endpoint gate

The retained reconstruction patch composes `fused_tile_program()` and
`resolvent_scalar_program()` into one ordinary multi-output Program, followed by
existing optimization. For compatible triangles, native binding publishes X/Y
into existing Fock pages. Other pages retain the resolver and all original
resource fallbacks. It adds no numeric buffer, equation, tolerance, precision
change or Q-batch change. It is retained evidence, **not enabled code**.

Modeled per-point work changes from response 763 operations/216 reads plus
resolver 231/186 to joint 804/216: 190 logical operations and 186 logical reads
saved. This is not a DRAM bandwidth measurement. Profile job 3590 reports 165
joint kernels (1.680167431 s) and 240 remaining resolvents (0.987997812 s).
The scalar family drops from 570 to 405 kernels and 3.199599038 to 2.668165243 s.
Response register count is unchanged at 244/thread; resolver remains 162/thread.
CUPTI reports localMemoryPerThread=0 in both variants; its localMemoryTotal
field is 221,773,824 in both, so do not interpret that field as measured spills
or claim a separate spill-traffic measurement.

Clean, interleaved cold processes, startup through teardown, medians in seconds:

| AO | Samples per arm | Baseline wall | Experiment wall | Gain | Baseline triples | Experiment triples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 4 | 5.045 | 5.025 | 0.40% | 1.040212739 | 1.0261416255 |
| 56 | 4 | 42.180 | 42.045 | 0.32% | 19.9010614395 | 19.7816289555 |
| 230 | 2 | 215.225 | 214.775 | 0.21% | 31.8169282905 | 31.2532404155 |

`common/performance.py::assess_comparison` requires at least five paired samples
and improvement beyond max(2%, estimated noise). These exploratory samples have
neither the required count nor a 2% descriptive gain. They are **not a performance
pass**, and no significance is claimed. Compilation succeeded with ccache;
compile elapsed wall time was not measured. More repetitions of this low-ceiling
fusion are not a substitute for a larger complete-endpoint opportunity.

Runtime allocation event inventories are identical. Summing tracked live bytes
over all CUPTI memory kinds gives a peak of 9,450,642,566 bytes for both. Sampled
whole-device memory peaks at 27,489 MiB for both; this includes context/driver and
other device usage and is not process-owned capacity. The endpoint's modeled
host/device numeric capacity is 11,961,676,969 bytes, CC Q8/Lambda Q32 with three
panels in both arms. These three memory measures must remain distinct.

## Numerical scope and reproducibility

The experimental GPU suite passes 99 tests with three skips; report tests pass
39; host tests pass 12 with one skip. Compute Sanitizer memcheck passes one
multi-CTA scalar test with zero errors. Two corrected-Lambda/factor-chain tests
need an additional opt-in fixture environment and the allocation-refusal test
needs a scheduled allocation interposer; those were not run. Do not infer their
coverage from the passing molecular endpoints. No acceptance tolerance changed.

Baseline/experiment total energies are identical for all retained molecular
samples. Maximum full-vector force differences are 7.11e-15, 1.60e-14 and 3.29e-14
for 28/56/230 AO. Complete molecular Lambda, Z, stationarity and primal replay
residuals are retained. Fresh independent PySCF 2.14.0 original-equation energies
and random normalized force directions for 28/56 AO use steps 1e-4 and 3e-5 bohr;
energy errors are below 3.2e-12 Eh and directional errors below 1.1e-8, within
the existing atol=rtol=3e-7 force gate. These are directional checks, not
all-component independent finite differences. The 230-AO reference energies and
two-coordinate/two-step independent force checks are explicitly historical
references from 2026-10-10, never historical A/B timings credited to this change.

Early cohorts `clean-3578`, `compare-3580`, and canceled `clusters-3582` are
excluded: legacy ELF DT_RPATH caused both apparent arms to load the candidate.
Final comparisons use variant-specific endpoint binaries linked with
`--enable-new-dtags`, per-arm `ldd` assertions and binary hashes. Primary clean
cohorts are `compare-final-3585` and `clusters-final-3586`. The earlier valid
prototype cohorts are not pooled with final-source samples.

Evidence, inputs, exact experimental source/test patches, profiler overlay,
analysis and original command recipes are retained in
`benchmarks/results/ccsdt-response-reuse-20261027/`. Large SQLite/Nsight traces and
build streams stay outside Git. CPU audit:

```bash
PYTHONPATH=python:. python -m benchmarks.ccsdt_response_reuse_audit \
  --occupied 9 --virtuals 221 --auxiliaries 488 --output /tmp/2516-audit.json
```

Revisit after a generic bounded residency schedule can address the larger
forward opportunities without reducing Q batching, weakening independent audits,
or changing ordered cotangent accumulation. Require source-matched complete
endpoint evidence and all promotion gates before enabling it.

References: #2516, #783, #1889, #682, #1763, #1626;
`docs/maintainer/performance_engineering.md`.
