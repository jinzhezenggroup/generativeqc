# Diagnosis: remove repeated work and expensive fallback routes

Status: proposed
Date: 2026-10-21

## Scope and decision

Follow-up to the [default-setting investigation](../implemented/performance/2026-10-20-readme-default-alignment-and-ranking-reversal.md)
and the [unified README measurements](../../../benchmarks/results/readme-unified-20261021/README.md).
The implementation priorities below are proposals, not completed optimizations.
The root README, official timing populations, grids, convergence controls and
acceptance gates are unchanged by this investigation.

Keep the common observable acceptance contract. Optimize the native derivative
algorithms, Fast ownership/qualification and resource-dependent dispatch before
changing scientific controls. A smaller SCF iteration count or FP32 label does
not imply less executed work. Cache choices must account for the consumer and
actual amortization; disabling all caches is not a useful general fix.

## Measured endpoints

Except where explicitly labeled as historical or instrumented ablations, these
are new n1 RTX 5090 diagnostics on the publication's frozen source
`0fe06f31c6c96182973405b58f5047f3f177fff4`. Core SHA-256 is
`f89e0b13b8c5e028c33fe087501f65aeb919e5789841599d5feab59840d6dff0`.
The source check verifies 716 Python/runtime/driver inputs against that revision,
with zero mismatches. Every GPU command uses finite Slurm allocation and its
assigned visibility, two CPUs and 48 GB host memory, with one math thread.

The new component diagnostics use one cold and one frozen-density warm call.
They do not replace the publication's three-repeat warm medians. All completed
diagnostic samples are rechecked by `readme_defaults_qualification.qualify`
against the original independent references and receipt-bound reuse manifest.
Both engines retain full moving-grid response and the published default
XC/VV10 quadratures. No GPU4PySCF Fast profile is used.
All 15 complete diagnostic populations (30 E/F calls) pass. One additional
30-second capacity probe intentionally stops before completing an endpoint;
it is admission evidence only, not a timing or accuracy result.

| Case, 96 atoms | Native cold / warm (s) | GPU4PySCF cold / warm (s) |
| --- | ---: | ---: |
| PBE DF Standard | 68.538 / 29.876 | 12.409 / 4.527 |
| PBE0 DF Standard | 104.354 / 55.859 | 25.840 / 13.223 |
| PBE0 DF Fast | 211.078 / 178.814 | Same reference |
| PBE0 Direct Standard | 48.607 / 8.414 | 20.257 / 6.911 |
| PBE0 Direct Fast | 38.487 / 7.929 | Same reference |

Native component traces and reference SCF/gradient timers are opt-in diagnostic
instrumentation. Native inclusive regions must not be added across nesting.
An asynchronous operation's host completion interval can charge a preceding
stream dependency to the next component; see the range-exchange finding below.

## DF force response: auxiliary g and repeated J/K traversal

PBE DF96 warm takes 29.876 s, but SCF takes only **0.536 s**. Its DF force
response takes **24.316 s**, including **23.893 s** of three-center derivative
contraction. All separately traced s/p/d/f shell kernels sum to **0.812 s**.
The remaining **23.081 s** is consistent with the generic auxiliary-g branch
and dispatch overhead. This subtraction is a residual attribution, not a
separately measured exact g-kernel duration. There are five g remainder panels.

Auxiliary g means angular momentum l=4, not grid level or a hardware limit.
The unified source admits these functions and preserves accelerated s/p/d/f
shell response, but the g remainder still contracts per AO element. It lacks
the lower-shell producer reuse. The g-only kernel filters other auxiliary
classes inside a traversal of the full response-weight range.

PBE0 DF96 warm SCF takes **1.056 s**, while two DF response calls total
**48.462 s**. The exchange call costs 24.792 s and the Coulomb call 23.670 s.
Their three-center derivatives cost 23.523/23.259 s; traced shell work accounts
for only 1.068/0.821 s. The same geometry's expensive derivatives are traversed
separately for K and J, with 57 and nine g remainder panels respectively.

This is not a large raw-tensor upload: response records zero tensor host-to-device
bytes, borrows the 8,542,322,688-byte whitened tensor, and reuses the final fitted
projection for K. PBE0's GPU4PySCF warm SCF/forces cost 4.465/8.758 s; native
SCF is already faster in this case. Fixing SCF tolerances alone cannot remove
the dominant native force cost.

Ownership explains the split. `PreparedFockPlan::energy_derivative_components_with_cuda_df_state`
consumes the one-shot fitted projection for K before J may overwrite the shared
scratch. The resident-density borrowing contract is J-only. A fused derivative
pass therefore needs a deliberate multiple-consumer or combined-adjoint design;
simply changing two calls to one would violate current component/lifetime contracts.

Relevant code:

- `src/scf/cuda/df_gradient_bridge.cu`: s/p/d/f shell selection and g remainder.
- `src/scf/cuda/df_derivatives.cu`: `AuxiliaryGOnly` per-element contraction.
- `src/scf/fock_prepared.cpp`: borrowed-state K-before-J consumption.
- `src/scf/cuda_fock_provider.cpp`: generated response and J-only density borrowing.

## DF cold preparation and the larger resource cliff

PBE DF96 cold preparation takes **25.559 s**, including **22.542 s** to
materialize the resident three-center tensor. Raw generation costs 13.284 GPU
seconds; metric eigen-projection and scaled rotation cost 4.601 s each.
The trace records 782 raw-generation calls, 15 whitening panels, 30 GEMMs,
295,296 packed pairs, and approximately 15.445 trillion whitening FLOPs.
PBE0's corresponding materialization costs 21.889 s.

The current symmetric eigen-based whitening performs two matrix products.
GPU4PySCF uses Cholesky first. Those methods have different rank/threshold
contracts; changing factorization requires independent numerical and rank
qualification, not merely replacing a slower function name.

The larger r2SCAN-3c DF96 failure has an additional value-generation/resource
cause. Prior source-bound evidence in the earlier note reports a 135.361 s
first RI-J build, about 14.54 GB for values, 7.14 GB for response, 1,183,744
ordered pairs and auxiliary tile 185. Consumer panels repeatedly regenerate
raw sources. These older timings are not new measurements of the unified
binary. The unified campaign still times out, while WB97M-V DF96 fails its
prepared-force resource requirement. Neither is explained solely by g forces.

## Fast fallback: two strict solves and owner reconstruction

PBE0 DF96 Fast is a **FP64 Standard-threshold candidate**, not a successful
FP32 force route. Its warm endpoint is accounted for by:

| Work | Seconds |
| --- | ---: |
| Recreate candidate owner and restore its seed | 26.258 |
| Candidate energy and forces | 62.498 |
| Suspend candidate | 0.057 |
| Separate tighter FP64 audit, including its preparation and forces | 90.056 |
| Complete endpoint | 178.814 |

The candidate executes two Focks and one force evaluation; the audit executes
three Focks and another force evaluation. Four J/K response invocations total
97.487 s. Two tensor materializations total 44.186 s. Those component times
are nested within the table, not additional costs.

`PrecisionPreparedBatch._resume` rebuilds the native prepared owner. The audit
factory exports the candidate's seed and closes its owner to respect bounded
GPU storage, then constructs a distinct reference owner. Seed reuse therefore
does not preserve the expensive preparation. This ownership choice explains
why warm still pays cold-like preparation twice.

HF Direct96 has the same unqualified-force fallback. Fast warm is 16.849 s:
candidate 6.312 s, audit 9.002 s, resume 1.534 s, suspension 0.019 s. The two
warm-density validation regions total 3.029 s **inside** those costs. The
publication's Standard warm median is 3.003 s. Candidate and audit both use
FP64 and each executes three Focks plus forces.

Relevant code is `python/generativeqc/precision_policy.py` (the
`fp32_gradient_not_qualified` / `budget_or_domain_requires_fp64_audit` route)
and `python/generativeqc/_precision_batch.py` (`_resume`, `reference_factory`).
Do not silently delete observable validation to advertise Fast. Qualify a
native final-state route, or make a documented Standard fallback avoid a
redundant independent endpoint when its scientific contract permits that.

## Fast mixed-J dispatch and Direct cache profitability

The retained same-card Slurm 11629 diagnostic already isolates r2SCAN-3c
Direct96: Standard/Fast cold 82.284/274.864 s despite 22/15 Focks. J costs
16.801/224.291 s; XC costs 27.317/11.741 s. Warm strict J is 0.759 s for both.
`direct_jk_value_dispatch` excludes generated Coulomb when `mixed_coulomb`
is true, selecting a slower generic/canonical route. FP32 arithmetic is not
the only changed variable. Generated or qualified strict J may be preferable
for an otherwise Fast solve; dispatch and precision ledgers must agree.

New PBE0 Direct96 cache ablations change only
`GENERATIVEQC_KS_RESIDENT_VALUES=0` and pass the same independent gates:

| Profile | Default cold / warm (s) | Cache disabled cold / warm (s) |
| --- | ---: | ---: |
| Standard | 48.607 / 8.414 | 41.048 / 8.227 |
| Fast | 38.487 / 7.929 | 60.617 / 7.816 |

Standard default pays 10.486 s in `ks_begin` to prepare the optional source;
15 SCF iterations do not recover that cost. Fast benefits strongly from its
cache, avoiding expensive mixed recurrence. These are single observations on
separate scheduler cards, not a statistical speedup claim or permission to
globally disable residency. They establish opposite qualitative profitability
in the two routes. Standard's 15 cold iterations versus the reference's nine
also matter, but are not the only reason for its cold gap.

## WB97M-V: range exchange, not an XC host-compute regression

Standard Direct96 takes 642.378/45.293 s cold/warm, consistent with publication
638.889/45.180 s. Its 28 cold Focks include 23 provisional and five full-target
Focks, with four VV10 pair sweeps. Warm performs one target Fock and hits the
pair cache, so there is no fresh VV10 pair sweep in that SCF call.

The nested XC operation reports 529.580 s host completion during cold, but its
CUDA events cover only 128.711 s. **Do not attribute the difference to CPU XC.**
The enclosing `ks_iteration` GPU region instead assigns 447.240 s to `ks_jk`
and 128.715 s to `ks_xc`. Primary J/K accounts for 32.530 s, leaving about
**414.710 s in the range-exchange interval**. In warm, J/K costs 15.960 s,
primary J/K 1.192 s, and the range residual is **14.767 s**; XC is only 1.875 s.

`enqueue_range_correction` submits a separate range owner and orders the KS
stream behind its completion. The following XC launch blocks while prior
range work drains. Its host interval consequently includes earlier work even
though the XC start event is after that dependency. Compare ordered parent
GPU regions, not nested host completion totals, for attribution.

At 48 atoms, the profiled Standard owner retains 3.831 GB of provider storage
and executes resident canonical kernels; the warm range interval is 0.097 s.
At 96 atoms, retained provider storage is only 0.449 GB. Automatic range
residency has a 4 GiB value ceiling in `CudaKsPlan::Impl::prepare_phase_source`.
This exposes an expensive streamed range-exchange route at the larger size.

Slurm 11644 changes only `GENERATIVEQC_KS_RESIDENT_MAX_BYTES=17179869184`.
It still retains just 0.449 GB and takes **637.573/44.991 s**, with the same
28/one iterations and passing independent E/F checks. Raising the ceiling
alone therefore does not resolve this case. A separate 30-second progress
probe records the decisive admission state:

```text
required_source_value_bytes = 14985773728
source_value_allowance_bytes = 14689091564
free_device_bytes_after_preparation = 14967767040
canonical_compensation_elements = required_compensation_elements = 1280000
admission = canonical domain or value capacity refusal
source_values_built = 0
```

Even with a higher policy ceiling, the complete cache exceeds the safe allowance
by 296,682,164 bytes. Available device memory is an observation, not evidence
identifying which other allocation owns it. The default requests all value
buckets and refuses that whole cache. Investigate the context/resource inventory
and automatic selection of profitable partial buckets; do not bypass the
reserve or advertise a blanket ceiling increase as a measured fix. The existing
per-bucket streamed fallback is a useful boundary for that future work.

GDB sampling (Slurm 11642) confirms CUDA launch/wait stacks. The machine's
perf policy disallows CPU perf sampling; no system policy was changed.
A diagnostic CUPTI activity collector (11643) on WB48 records 83.902/6.549 s
of GPU kernel durations inside 87.755/7.012 s cold/warm endpoints, with zero
dropped records. CUDA API time overlaps device time and must not be added.
Both debugger and CUPTI endpoints pass the same observable gates but remain
instrumented diagnostic populations. No native code was rebuilt for profiling;
only the external CUPTI collector was compiled through verified sccache 0.16.0.

## Priorities and invariants

1. Qualify Fast routes that avoid duplicate strict endpoints and preserve
   expensive geometry preparation across required checks, within the same
   bounded lifetime/resource contract.
2. Add shell-level auxiliary-g derivative reuse, then share derivative source
   traversal across J/K consumers while preserving separate physical outputs.
3. Avoid the unprofitable mixed-J dispatch route; qualify the selected arithmetic
   and keep work/precision provenance accurate.
4. Make full-range and long-range source admission depend on bounded storage,
   reuse and execution cost. Preserve a correct bounded streamed fallback.
5. Reduce DF materialization/whitening and large-case source regeneration without
   silently changing factorization rank, auxiliary basis or force response.

Promoting any new default still requires its own complete endpoint, independent
E/F, moved-geometry, batch and constrained-memory qualification as applicable.
These sampled static diagnostics alone do not establish a universal policy.
The historical ranking also benefited from artificially tight reference SCF
controls: the earlier same-binary PBE0 ablation changed reference cold from
79.58 to 29.24 s and 37 to nine cycles by restoring convergence defaults alone.
That historical effect is distinct from the native costs measured here.

## Evidence retention

Full scripts, outputs, valid component traces, source checks and independently
recomputed gates are retained under ignored local
`.artifacts/slow-diagnosis/` and
`n1:/data/jzzeng/readme-slow-diagnosis-20261021/`.
`summarize.py` reconstructs endpoint intervals from post-timer checkpoints;
`qualify.py` recomputes the unchanged observable gates.
`check_evidence.py` checks source/library/driver identities, unchanged ablation
settings, source verification, complete-call qualification, and lossless CUPTI
record collection, and writes a file-hash inventory. The standalone ignored
`.artifacts/slow-diagnosis-20261021.tar.xz` bundles this evidence for review.

One original wrapper (Slurm 11631) exits 127 **after** Python completed and
saved both WB96 endpoints, because its shell file was updated while Python
ran. Preserve that failed wrapper outcome. The complete original results,
valid trace boundaries and successful independent gates remain diagnostic
evidence; this is not relabeled as a clean successful scheduler job. Subsequent
ablations use a separate immutable runner.
