# Decision: eliminate generic high-order Coulomb tangent simplexes

Status: implemented lowering; no molecular endpoint performance qualification
Date: 2026-10-11

## Problem and actual ownership

The continuation of #2244 starts from master `f80f732e2`. Order-six spatial reuse
does not remove the generic high-order derivative's `CoulombAuxiliary<Dual3,L>`:
each recurrence state stores a primal and three tangents. The simplex has
`choose(L+4,4)` entries and is prepared again for each independent atom seed.

Do not conflate total angular order with execution ownership. Admitted full-range
s/p/d order-seven tasks use the compiler's cooperative shared scalar DAG;
materialized dddd/order-eight tasks likewise use shared scalar responses. Missing
resident prerequisites, mixed-f tasks and other radial/optional recurrence routes
retain generic consumers. Their private recurrence is the target of this change;
the already shared producers, native admission and launch widths do not change.
Existing reports of 255 registers in production kernels are not evidence that
reducing block width solves their bottleneck.

## Decision

For `Dual3` and total orders 7--12, the compiler stores one scalar degree-`L+1`
Coulomb simplex and reconstructs each requested jet from its three raised
Cartesian roots. This is the same spatial-response identity used by the existing
materialized/cooperative compiler consumers, not a second integral recurrence.
The view keeps all nine geometry seeds, including non-diagonal forward directions.
Hermite coefficient jets, Gaussian normalization, prefactor multiplication and
the incumbent six-loop contraction/accumulation order remain authoritative.

Full, long-range and short-range moments use their existing owners. Optional
Hermite convolution reads the same response view, with its selection unchanged.
Reachable scalar preparation includes the original component box and single-axis
response faces plus dependencies; bounds are clipped to the degree-`L+1` simplex.
Poisoned unused cells prove that every response read is preceded by a write.
Malformed domains retain the complete simplex before unsigned bounds are used.

The scalar response capacity reaches degree 13 solely to differentiate existing
through-f/order-12 inputs. `kMaximumAngularMomentum` and the primal order limit
remain unchanged. Low orders, scalar values, single-direction AD, mixed precision
and existing shared-storage consumers retain their routes. No global tensor,
extra quartet traversal, cache requirement, native selector or thread change is
introduced. Primal preparation is still repeated across independent atom seeds
on this generic route; this change does not claim to eliminate that repetition.

## Work and resource evidence

Storage counts are representation bounds, not molecular work counters:

| Total order | Old jet cells / bytes | Scalar response cells / bytes |
| --- | ---: | ---: |
| 7 | 330 / 10,560 | 495 / 3,960 |
| 8 | 495 / 15,840 | 715 / 5,720 |
| 12 | 1,820 / 58,240 | 2,380 / 19,040 |

The additional response degree increases scalar recurrence cell count while
removing three stored tangent channels. Neither this table nor successful
numerical gates establishes fewer molecular primitives or endpoint speedups.

Finite n1/main RTX 5090 Slurm jobs 7476 (baseline), 7477 (corrected candidate) and
7478 (final candidate plus memcheck) each execute the same 288 shell-pair requests:
orders 7/8/12, full/LR/SR, four individual atom seeds, repeated/combined atoms,
translation, non-diagonal directions and all reachable/convolution combinations.
Independent normalized Gaussian-quadrature jet gates are `rtol=3e-10` and
`atol=2e-11`. Maximum absolute errors are `1.8041124150158794e-16` baseline and
`1.5959455978986625e-16` candidate. Candidate memcheck reports zero errors.
Final-source job 7485 repeats the 288 independent GPU checks and memcheck after
formatting, with the same numerical error and resource reservations. The complete
focused host suite passes 1,146 tests with 10 explicitly opt-in GPU skips;
compiler/shared-SCF structure audits report zero dependency errors, and focused
Ruff lint/format and whitespace checks pass.

The final linked **probe**, not the production shared library, reports unchanged
226 registers/thread at the same 128-thread launch width. CUDA function local
reservations change as follows:

| Probe order | Baseline local bytes/thread | Candidate local bytes/thread |
| --- | ---: | ---: |
| 7 | 19,376 | 12,784 |
| 8 | 26,960 | 16,848 |
| 12 | 83,568 | 44,368 |

These are static reservations, not executed local-memory traffic. CUOBJDUMP
distinguishes linked `STACK` (the same byte counts) from `LOCAL` (zero in
these probe images); a zero `LOCAL` field does not mean no local-memory traffic.
The NCU attempt in the finite following Slurm job fails with `ERR_NVGPUCTRPERM`;
no occupancy, local-load/store or causal stall claim is made. Single CUDA-event intervals are
retained as diagnostics, not repeated endpoint benchmarks or speedup evidence.

Raw source-matched baseline/candidate trees, compiler-cache statistics, commands,
generated headers, executables, requests, all jets, validation receipts,
`linked-probe-resources.txt`, `memcheck.log` and the failed NCU receipt are under
`/data/jzzeng/qc-direct-response-20261011-f80f732e/` on n1. The final-source
production-only reconstruction patch is retained there as `production.patch`,
SHA256 `b60f9325c92475dd27dcf67dfd0977874406791b031cc6024171be018a923e24`.
Final-source production inputs and the CUDA probe match the local checkout by
SHA256. The initial candidate's
failed compile is retained separately: host-only `std::min` in shared CUDA bounds
was replaced with an inline host/device bound helper rather than enabling relaxed
constexpr flags. No successful receipt overwrites that failed attempt.

## Rejected alternatives and remaining gates

### Source-matched production qualification and conditional PR decision

The follow-up request makes PR submission conditional on a demonstrated
96-atom PBE0 improvement. Both complete production libraries are rebuilt from
master `f80f732e2e1d8d4fbf1a25079baf5990d7aeabc3`, with only the four production
files in `production.patch` applied to the candidate. Release/sm_120, fast
compile disabled, CUDA 12.9.1, C++20 and verified ccache 4.5.1 are identical;
generated actual compiler commands invoke the launcher. Configure/build logs,
before/after cache statistics and source/binary hashes are retained separately.
Both complete builds exit zero. Final native image SHA256 values are:

- Baseline: `470c71730dcdfd6f3b4b6745757befec20127014258de4ce98c79fb7dc82450d`.
- Candidate: `4196093b69035ad1fab5326666eb4204f4f4341dbc294cdbd9109f91bb5b6d2a`.

The stationary PBE0 AOT image is byte-identical in both arms:
`969d92da3008b710c287c673d778194485443ba2517f5b0b334d99bdf89192b5`.
Separate compiler factories/artifact identities and loaded-native-map checks
prevent silently binding both owners to one source tree or native image.

Final linked production resource receipts contain the same 106 bounded-kernel
symbols. Sixty kernels change their stack reservations; all changed kernels
retain 255 registers/thread. Representative linked `STACK` bytes/thread are:

| Production consumer | Baseline | Candidate |
| --- | ---: | ---: |
| Private order-7, FullSources | 26,824 | 20,168 |
| Private order-8, FullSources | 34,440 | 24,264 |
| Largest generic derivative wrapper | 91,256 | 54,312 |
| Cooperative bounded order-7, FullSources | 176 | 176 |
| Default pure materialized order-8, FullSources | 752 | 752 |

The private order-7/8 examples are restricted variants of the retained
fallback, not a claim that the RKS endpoint executes those kernels. Static
resources do not demonstrate dynamic local-memory traffic or occupancy.

The complete synchronized public E+F comparison uses 96 atoms/768 spherical
def2-SVP AOs, the retained 48x16x32 Becke grid, identical automatic native force
routes, device final validation in both arms and no profiler instrumentation.
The same process/CUDA context retains two source-matched native images and two
independently converged owners. Each phase freezes owner-local native density
and coordinate blobs, primes once and measures five ABBA-interleaved pairs.
The independent retained GPU4PySCF/PySCF reference is unchanged; no reference
or host numerical work enters the production timed endpoint.

Qualification attempts are not pooled or erased:

- Job 7501, `paired-default-96-v1`: the original 1e-10 density tolerance
  yields different SCF stopping iterations (22 baseline, 17 candidate) and
  cross-arm density drift `4.083333671189848e-11`, exceeding the predeclared
  `2e-11` matching gate. No timing cohort is admitted.
- Job 7507, `paired-default-96-v2`: equally tightened 1e-13 density tolerance
  fails baseline construction at the finite 100-iteration SCF limit. The
  failed physical output is retained; this is not a candidate speed regression.
- Job 7516, `paired-default-96-v3`: both arms use 1e-12 density tolerance,
  retaining all other solver and physics settings. The solver-contract revision
  is explicit in the receipt and is not relabeled as the original 1e-10 protocol.
  All 30 complete numerical calls and 20 timed samples pass, with five pairs per
  phase. This stricter matched-density experiment is the sole admitted cohort.

| 96-atom complete endpoint | Baseline median (s) | Candidate median (s) | Improvement | Required |
| --- | ---: | ---: | ---: | ---: |
| warm | 14.266116638 | 14.285259157 | -0.1342% | >2% |
| moved-warm | 14.350531306 | 14.291240718 | +0.4132% | >2% |

The requirement is `max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))`
for both phases. Maximum independent E/F errors are
`9.094947017729282e-12 Eh` / `2.1289636720212002e-11 Eh/Bohr`, against gates
`1e-8` / `1e-7`. Cross-arm density drift is `1.5260015473472777e-12` warm and
`1.6675549829869851e-12` moved-warm. Every timed endpoint has one iteration,
one Fock build, warm-start use without fallback, and equal available semantic
AO/grid/force work. Frozen NPZ file hashes, blob hashes and before/after values
are independently rechecked locally and on n1. These work counters do not
certify executed screened J/K quartet counts.

Neither phase clears the improvement gate. Under the initial conditional
submission request, no PBE0 speedup PR was authorized by this evidence. The user
subsequently explicitly requested a PR for the resource/representation change
despite the absent PBE0 speedup. Submit it only with that scope: keep the negative
endpoint result visible and do not turn stack savings into a performance claim.
Construction/setup/priming times remain diagnostic; no cold speedup is claimed.
All raw samples, seed NPZs, drivers, failures and derived checks are retained
under `endpoint-evidence/` in the n1 evidence root above and copied to the ignored
local `.cache/direct-response-endpoint/` directory.

### Executed default-route hotspot diagnosis

Separate finite Slurm job 7517 profiles exactly one candidate warm complete
PBE0 E+F call with Nsight Systems 2025.1.3 after independent cold convergence
and priming. This diagnostic retains the original 1e-10 solver density tolerance;
it is not pooled with the stricter paired cohort. Cold/prime/profiled calls all
pass the independent E/F gates, and the captured call has one iteration and one
Fock build. The loaded native image matches the qualified candidate SHA256.

The `p0b/public-stationary-force` NVTX interval lasts 10.515444055 seconds and
contains 72,335 kernel launches, with summed GPU intervals of 9.075893591 seconds.
Representative actual force hotspots are:

| Executed kernel family | Launches | Summed GPU interval (s) | Share of force GPU intervals |
| --- | ---: | ---: | ---: |
| Spatial s/p/d low-order force | 1 | 2.556985369 | 28.1734% |
| Dominant phased Becke pair | 4,608 | 1.313448709 | 14.4718% |
| One-electron shell-warp gradient | 2 | 1.066740423 | 11.7536% |
| Dedicated order-seven force | 1 | 0.823228909 | 9.0705% |
| Pure materialized order-eight force | 1 | 0.377303157 | 4.1572% |

Order seven executes `direct_order_seven_force_kernel<false, Combined>`, not
the generic private-state fallback nor the cooperative bounded kernel in the
static table. Its observed launch is 256 threads, 1,360 CTAs and 214
registers/thread. Order eight executes the bounded specialization
`<false, Force, true, 8, Full, true, false, true>`, also 256 threads/1,360 CTAs,
with 234 registers/thread. Only that pure materialized specialization of the
bounded force family appears in the captured force interval. No changed generic
bounded derivative kernel symbol appears. These ownership/routing observations
explain why the fallback stack reduction does not move this default PBE0 endpoint.

Broader final-linked receipts additionally cover the dedicated order-seven and
spatial low-order families. The actually selected order-seven Combined kernel
retains 214 registers/thread and a 144-byte `STACK` in both images; the selected
pure materialized order-eight Full kernel retains 234 registers and 704 bytes.
The largest measured Direct-force hotspot, the Combined spatial low-order
kernel, retains 255 registers and 16,840 stack bytes. Thus neither the active
high-order consumers nor the remaining active 255-register low-order consumer
are improved by this generic representation change. These complete routing
inventories are retained as `linked-production-force-routes.json` per arm.

All profile intervals are explicitly intrusive: the tool warns that device-side
CUDA Event completion tracing is enabled. They are not clean endpoint speedups,
and summed GPU duration shares are not full endpoint wall-time shares. NCU
hardware counters remain unavailable, so no local-traffic, occupancy or stall
claim follows from this trace. The raw `.nsys-rep`, SQLite export, complete kernel
inventory, numerical outputs, environment, drivers and independently replayed
analysis are retained in `endpoint-evidence/profile-default-96-v1/` and the
matching ignored local cache directory. Future PBE0 derivative optimization
should target the actually selected dedicated order-seven and pure materialized
order-eight producers, without disabling them to manufacture generic-path gains.

### Integration with later master and resource-only PR

The user explicitly authorizes the resource-only PR after reviewing the absent
PBE0 speedup. Integrate onto master
`56bedd45719b539df2a26205260ac2b5ebd99760`, rather than reverting its newer
independent-center response owner from #2267. That owner shares scalar Coulomb
and relative-pair Hermite states across independent atoms for full/LR orders
7--11. This change is complementary: it lowers the remaining generic `Dual3`
storage, including order twelve, short-range and selected reachable/convolved
consumers, while retaining arbitrary non-diagonal geometry seeds. No shared
force-owner selection or scheduling changes are introduced.

Fresh post-integration checks pass 838 focused host cases with two opt-in GPU
skips, 520 compiler modules with zero dependency errors, 239 shared SCF modules
with zero dependency errors, and focused Ruff lint/format checks. The scientific
CUDA ownership inventory checks all 354 native files. Existing classifications
and runtime CUDA line counts remain unchanged. The shared native recurrence
grows by 24 nonblank/noncomment scientific lines net (+40/-16 including the
Boys bound replacement). This growth is bounded response-domain closure and
clipping for the existing CPU/CUDA recurrence, not a second recurrence or radial
formula. Only its existing ownership shard's rationale is updated.

Finite node1/main RTX 5090 job 7539 replays the independent 288 shell-pair cases
on the rebased source and reports memcheck zero errors. Its source revision is
`ebe1903b8fe954a0b45f3f9c3c7092311072a50b`; subsequent PR-note/documentation
updates do not change scientific or probe source bytes. All four production
inputs and both probe sources match the final checkout by SHA256. The selected
compiler cache is verified and reused, with before/after statistics retained.
Raw source hashes, generated headers, compile commands, jets, numerical receipt
and sanitizer output are under `endpoint-evidence/pr-rebased-7539/` on n1 and
the matching ignored local `.cache/direct-response-pr/` directory.

These compatibility checks do not requalify the old molecular timings or linked
production resources against later master. The production tables and PBE0
cohort above remain explicitly sourced to `f80f732e2e1d8d4fbf1a25079baf5990d7aeabc3`.
No later-master full-library resource, cold, endpoint speedup or hardware-counter
claim is made. The PR is a representation/resource change, not default promotion
based on the negative PBE0 performance gate.

### Remaining work

- Changing block widths or register caps does not eliminate derivative state and
  would alter the qualified cooperative lane ownership.
- Extending the order-six Wick expansion into a second order-7--12 all-center
  implementation repeats science and revives previously rejected schedules.
- Restricting the response view to diagonal seeds silently narrows the AD API.
- Ignoring the reachable halo or excluding order 12 leaves a correctness hole or
  the largest private generic frame intact.

Further production claims still need authorized hardware counters and independent
molecular J'/K' and complete synchronized cold E+F gates. The qualified warm and
moved-warm measurements above show no significant PBE0 gain. Keep the unchanged
255-register production problem separate from the measured probe.
Further optimization should investigate repeated per-atom primal/Hermite work and
the actual remaining contraction traffic, not infer occupancy from stack bytes.

## References

- #2244 and the order-six spatial reuse decision.
- `2026-10-06-direct-pair-scalar-spatial-responses.md`.
- `2026-10-06-cooperative-weighted-shell-force.md`.
- `docs/developer/direct_pair_recurrence.md`.
- `docs/maintainer/performance_engineering.md`.
