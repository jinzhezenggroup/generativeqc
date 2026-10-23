# Bounded DF Hermite charge partitions

Measured source: `4f6558075d6484ac65b6b6852f3f71765757f84a`.
The explicit `GENERATIVEQC_DF_MD_CHARGE_SPLITS=1..64` control partitions source
primitive groups and reduces their independent partials in ascending order.
Unset retains one partition. Partials reuse dead orbital primal-potential
storage; insufficient capacity retains the original contraction. No additional
device allocation, screening, precision, basis, grid or convergence change.

The FP64 sum order changes and is independently qualified. These results do not
promote a default or claim that DF cold has caught GPU4PySCF. On water96,
spherical def2-mTZVPP with r2SCAN-3c, Standard DF, five alternating fresh-process
split-1/split-32 pairs each execute complete energy/analytic forces and three
warm calls. One median warm observation per process contributes to statistics.

| Phase | Split 1 median / MAD (s) | Split 32 median / MAD (s) | Improvement |
| --- | ---: | ---: | ---: |
| Cold | 52.552040 / 0.048872 | 43.591161 / 0.072325 | 17.051% |
| Warm | 14.898640 / 0.141796 | 12.208637 / 0.064395 | 18.055% |

Both gains exceed the unchanged maximum of 2% and twice the sum of relative
MADs. The cold/warm floors are 2.000%/2.958%. Eight earlier screen processes
remain separate; neither their timings nor the Fast process is pooled into
these five-pair populations. The earlier screen selected 32; 64 did not improve
its single observation. Every observation is retained.

All 104 native calls in 31 processes qualify. Static calls use the audited
dense-grid and CPU/GPU references in `../readme-final-20261022/`. Standard/Fast
moved r2SCAN-3c calls use one fresh GPU4 DF process at the identical displaced
geometry (atom one z +0.001 Bohr). Those comparisons establish sampled schedule
equivalence, not independent moved-grid convergence. Standard moved maximum
force error is 1.179e-7 Eh/Bohr; Fast is 1.273e-5 Eh/Bohr, both within the
unchanged gates. PBE, PBE0, B3LYP, r2SCAN and M06-2X add Standard/Fast cold/warm
compatibility coverage. WB97M-V is outside this pure-J streamed consumer.

The immutable Release sm120 package was built on n1 with CUDA 12.9.1 and verified
sccache 0.16.0 after authenticating 5,984 source inputs. All five stationary
manifest targets were rebuilt. All device work uses finite Slurm RTX 5090 jobs
and one numerical host thread. The separate `source-inputs.json.xz` preserves the exact input manifest; the receipt bundle preserves
core/standalone package hashes, build controls, original runners, full energy and
force arrays, semantic work counts, cache statistics and validation logs.

Validation: 1,163 host cases; native CUDA ownership checks; integrated metric/J
and independently displaced Coulomb force gates (16 cases with split 8 and
again 16 with split 32), including deficient-metric fallback. Eight separate
multi-group charge, derivative and graph/replay cases run under Compute
Sanitizer, each reporting zero errors. Earlier preformat private GPU testing
(186 cases) is retained in ignored artifacts and is not labeled final-binary
evidence. Shared compiler-cache statistics can include other users.

```bash
PYTHONPATH=python:. python benchmarks/results/df-md-charge-splits-20261023/reproduce.py \
  > .artifacts/df-md-charge-splits-reproduced.json
```

The script rechecks all native calls and recomputes repeated-process statistics.
See the [partition decision](../../../.agents/notes/proposed/2026-10-23-df-md-charge-source-partitions.md)
for ownership and promotion boundaries.

The bundle also retains the exact source-tree patch from the published e8af
ancestor to the measured 4f65 source. Applying that patch to an isolated e8af
checkout recovers the measured source without an external archive; the complete
input inventory authenticates the resulting build inputs. Later integration
receipts must remain separate from the five-pair measured population.

## Current-base integration

`integration.json.xz` retains a separate build and endpoint bridge at
`af19ae37a09b965c560408f818b68c8aeb9073cd`, including all 28 native calls.
The 16 integrated metric/J/force tests and eight partition derivative/replay
tests under Compute Sanitizer pass. Standard/Fast cold/warm coverage for five
other DF functionals and both moved r2SCAN-3c trajectories pass the unchanged
independent gates. Moved maximum force errors are 1.1785e-7 and 1.2784e-5
Eh/Bohr for Standard and Fast. This bridge is compatibility evidence; its
single observations do not enter the historical five-pair speedup statistics.

The build input inventory is stored as an authenticated delta against the
historical inventory: 118 changed/new entries recover all 6,000 input hashes
and the exact original manifest bytes. The bridge also retains compiler-cache
statistics, the full build log, immutable package hashes and original runners.
`reproduce.py` authenticates this delta and independently rechecks every call.
Both larger historical bundles are explicitly registered by exact hash and
size in the existing evidence review; all storage caps remain unchanged.
