# Native CCSD(T)/CBS E+F reuse qualification

Native implementation: `c6a489ac27305be24592bed9b54f6c936e00ad0c`, based on
master `f2cf62d9b`. The evidence-only commit does not change native code.
Every retained endpoint's source hashes were checked against this pinned tree,
before integration of the PR's parallel portability/CI fixes and newer master.
These records do not claim byte-identical source or a new GPU qualification of
that later integrated head.
The raw CBS run label `native-live-capacity` identifies that source snapshot;
it is not a Git revision. The CUDA library SHA256 is
`fdf5680fe5c6b3a51f3912502ff846b75e6df52fe611212b82f70c2a6a8ce92d`.

Qualification uses Slurm `main`, `node1`, one `gpu:5090:1`, four CPUs, and finite
time limits. Hardware is NVIDIA GeForce RTX 5090; driver 580.95.05, CUDA toolkit
12.9.86, SM120, Release. Numerical budgets are 256 MiB per native group, not a
whole-process/device memory guarantee. Measured whole-endpoint allocation peaks
remain unavailable; do not substitute planned capacities for measurements.

## Integrated host smoke

After integrating remote PR head `0103a7fcba9acccefb491495a45af47d4442d0c0`,
the pre-log tree is `933cd35d94e13b3549139d4c27ef0e08eb4aca20`.
`src/methods/mp2_method.cpp` and `src/posthf/mp2_force.cpp` remain byte-identical
to the qualified snapshot. The host companion type gains the parallel
Clang/libstdc++ construction fix; the force header gains namespace formatting.

The CPU library/native MP2 gradient contract rebuild uses ccache 4.5.1 and
passes. Host portability and CBS output-retention smoke: **4 passed, 1 skipped**
(Clang is unavailable locally). The integrated CPU water-algebra complete E+F
passes both independent constituent/composite oracle and real reuse checks:
**2 passed**. Native API contracts and public API docstrings also pass. The
three `integrated-*.log.gz` files retain this host evidence; no additional GPU
benchmark or integrated-head GPU qualification is implied.

## Complete endpoints

All times include construction, SCF, correlation, response, analytic forces,
publication and group cleanup. Each row retains one cold and three complete
replay samples; these are **not** retained prepared-batch warm timings. Oracle
checks and a separate semantic-work replay are outside the timed samples.

| Case | Execution | Cold (s) | Median replay (s) | Reference groups |
| --- | --- | ---: | ---: | ---: |
| H2 DZ/TZ | Grouped, larger MP2 force cache disabled | 74.9774 | 73.5223 | 2 |
| H2 DZ/TZ | Independent components, force cache enabled | 4.5681 | 2.6971 | 5 |
| H2 DZ/TZ | Native groups, force cache enabled | 4.7219 | 3.4577 | 2 |
| Water algebra | Grouped, larger MP2 force cache disabled | 8.1232 | 6.7557 | 2 |
| Water algebra | Independent components, force cache enabled | 1.8928 | 0.6158 | 5 |
| Water algebra | Native groups, force cache enabled | 1.6168 | 0.4165 | 2 |

The first comparison isolates the optional ordinary CUDA MP2 force primal cache
with the **same native library and grouped coordinator**, using
`GENERATIVEQC_MP2_FORCE_MO_CACHE=0/1`. The correction group's borrowed CC blocks
remain enabled in both. Median complete-endpoint gains are 21.3x for H2 and 16.2x
for the water algebra case. They are not measurements against a different
master library, nor universal molecule/basis speedups.

The second comparison isolates same-basis grouping with the same cache-enabled
native library and independent coordinator snapshot. Water improves 1.48x.
**H2 grouping is 28% slower than the independently coordinated cache-enabled
endpoint**, despite reduced source work: the generic stationary HF companion
derivative consumer costs more than ordinary HF's specialized force route.
This is a retained limitation, not evidence that every reuse mechanism always
reduces endpoint time. Common response solves and derivative fusion remain
future work; there is no fully GPU-resident multi-basis claim.

H2 uses physical spherical cc-pVDZ/cc-pVTZ and the lower-basis correction. Water
uses a **synthetic pair** and STO-3G correction with nonzero triples; it tests
algebra/analytic derivatives, not predictive molecular CBS accuracy. The native
CCSD(T) force domain still limits the correction calculation to 12 public AOs.

## Semantic work

These are actual completed `native_mo_blocks` journal counts, covering MO
production, not all SCF or derivative integral work. Whole endpoint timings
include those other consumers.

| Case | Execution | AO source reads | Transform FMAs | Reused MO block reads |
| --- | --- | ---: | ---: | ---: |
| H2 DZ/TZ | Grouped, larger force cache disabled | 1167 | 9929620646 | 230 |
| H2 DZ/TZ | Independent, cache enabled | 31 | 72571614 | 1378 |
| H2 DZ/TZ | Native groups, cache enabled | 24 | 72133034 | 1378 |
| Water algebra | Grouped, larger force cache disabled | 578 | 4754428 | 574 |
| Water algebra | Independent, cache enabled | 16 | 388402 | 1148 |
| Water algebra | Native groups, cache enabled | 9 | 253414 | 1148 |

Every counted source read is device-backed in these runs. Both grouped cases
report zero companion MP2 energy AO reads/transform FMAs, borrowed OVOV and
borrowed force symmetry-cover reads. The larger MP2-only group retains 4420880
bytes for H2 and 9048 bytes for water. Reuse counts include separate response
actions; they do not imply one shared CC/MP2 response solve.

## Acceptance and boundaries

- Independent pinned PySCF 2.14.0 references gate **every constituent** and the
  composite at `1e-9 Eh` / `2e-6 Eh/Bohr`, with independently reconverged complete
  energy directional derivatives in the native test suite.
- Across these grouped samples, maximum composite energy errors are
  `3.62e-13 Eh` (H2) and `2.56e-13 Eh` (water); maximum force errors are
  `5.72e-10 Eh/Bohr` and `1.75e-7 Eh/Bohr` respectively.
- Final CPU selection: **68 passed, 4 skipped**, including actual borrowed-block
  endpoint tests. Expensive physical H2 DZ/TZ CPU forces remain explicit opt-in.
- Allocated CUDA selection excluding the expensive disabled-source F case:
  **28 passed**. This includes full CBS E+F, directional derivatives, native ABI
  rejection/recovery, true nonconvergence and unchanged scientific outputs.
- Native MP2 gradient contracts pass, including the regression that inflates
  the completed SCF peak above the force budget but admits the actual live
  reference payload and compares its forces with independent ordinary HF.
- Separate allocated vendor-workspace-aware tight-budget test: **1 passed**.
  It preserves an admitted legacy endpoint and warm seed across two replays,
  observes resource-only source fallback with zero optional MO cache, and checks
  complete E+F parity/capacity. It does not prove cache-construction fallback
  under every possible vendor allocation refusal.
- Separate default cached F-shell MP2 qualification completes all cold/prepared
  E+F samples against a freshly executed PySCF oracle: maximum errors
  `1.78e-14 Eh` / `1.14e-13 Eh/Bohr`. Its 0.9364 s prepared-warm median is not a
  CBS complete-replay timing and is not used for the CBS speedup claims.
- The combined allocated suite including `GENERATIVEQC_MP2_REFERENCE_SOURCE=0`
  on the F-shell case **times out at 15 minutes**. The raw compatibility source
  route is unchanged; this run is not a pass or evidence for its performance.
- Public/native API and force documentation selection: **73 passed, 39 subtests**;
  both API checkers and Sphinx warnings-as-errors pass.

## Reproduction and retention

Run `benchmarks/ccsdt_cbs_endpoint.py --device cuda --case h2-dz-tz --oracle
--work --repeats 3 --revision <source> --output <fresh-path>` through finite-time
`srun`; repeat with `water-algebra` and the force-cache disable control. Never
unset or override Slurm's assigned device visibility. The frozen independent
coordinator and its qualification wrapper are retained compressed beside the
measurements; unpack them under `.artifacts/ccsdt-cbs-20261012/` to replay that
specific comparison. They are not production code.

The six source/library-matched CBS JSON records, separate F-shell MP2 record,
focused GPU/CPU/contracts/native/resource logs, timeout log and tight-budget
journal are losslessly gzip-compressed here. Full local build and diagnostic
artifacts remain ignored under `.artifacts/ccsdt-cbs-20261012/`.
