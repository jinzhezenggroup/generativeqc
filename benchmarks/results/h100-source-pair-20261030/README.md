# H100 post-#2887 source-pair qualification

The frozen #2888 candidate improves the 96-atom PBE0 DF endpoint relative to
the post-#2887 baseline, but still trails GPU4PySCF. Of these four physical
PBE0 cases, only Direct 48 passes the reference win rule. This is evidence
for issues #2891/#2892, not completion of the 30-case target in #2889/#2893.

| Physical case | GPU4/candidate Standard | GPU4/candidate Fast |
| --- | ---: | ---: |
| DF 48 atoms | 0.746262 | 0.742674 |
| DF 96 atoms | 0.448375 | 0.448758 |
| Direct 48 atoms | 1.303818 | 1.453114 |
| Direct 96 atoms | 0.650446 | 0.728874 |

Each entry is the median of three process-paired ratios of the warm medians.
A win requires median >1.05 and all three ratios >1. Standard/Fast and repeat
calls do not add votes. For DF 96 Standard, baseline/candidate is 1.198610;
candidate process medians are 4.376272/4.366584/4.369282 seconds and reference
medians are 1.954534/1.957869/1.966369 seconds. Direct 96 remains unfavorable.
No production/default policy is promoted by this record.

## Scope and identities

- Baseline: `15ab66a68a061355fda833a93e88484dbfee6360`.
- Candidate: `2241fa6a76698c09573c540b2822316ef7cdb908`. These observations do
  not qualify later #2888 heads or an integrated merge commit.
- Driver: `f83e5c75204b3e2b8c1f8f6d5dd9c3fe5e6691b3462768b1778b97e47659b42a`.
- Independent clean SM90 builds, core/AOT artifact identities and actual
  H100 80GB HBM3 SM9.0/MIG-off identities are in the build/campaign receipts.
- Pilot: 15 populations/60 E+F, one process per engine/case, GPU UUID
  `GPU-41699bcb-d373-9f0b-0025-d27bf1e0b24c`. Includes small HF Direct/PBE0 DF
  controls and large PBE0 DF/Direct controls; exploratory only.
- Source pairs: 60 populations/240 E+F, three processes per source/arm/case,
  UUID `GPU-00d796f2-3bb6-d245-52ac-478b21a4104b`. Four PBE0 physical cases,
  baseline/candidate Standard/Fast and stock GPU4PySCF 1.9.0.
- Separate intrusive diagnostics: 8 populations/32 E+F, UUID
  `GPU-fc041bfb-a47c-3be6-fdd0-f5be7fede407`. DF 48/96 source pairs, candidate
  Fast 96, projection-off 96 and Direct 96 controls. All diagnostic seconds
  are excluded from clean performance comparisons.

Every population has one cold and three warm complete energy/analytic-force
calls. Warm seeds are each engine's own frozen cold density. CUDA work is
synchronized and complete forces are returned to the host. The paired
allocation is 20 CPU/200 GiB with 16 GiB shared memory and one thread per
BLAS/OpenMP consumer. All 332 calls pass the unchanged independent direct
oracle gates: E/atom <=1e-5 Eh, Fmax <=1e-4 Eh/Bohr, FRMS <=3e-5 Eh/Bohr.
The original method-specific oracle reuse audit is retained; it explicitly
records that the full bytes of fourteen older denser-grid producer scripts
were not recovered. Oracle vectors, geometry and published auxiliary inputs
remain available in the current tree.

## What the diagnostic establishes

Native prepared force capability is true for every diagnostic call. Candidate
default DF calls report `response_final_fitted_projection_reused=1` in the
same valid stream `force_response` record that reaches
`atom_coordinates=3*atoms`; the baseline and projection-off control report 0.
The admission counter is retained separately and is not used as proof of
completed arithmetic.

Both J and K responses remain separate. Candidate DF 96 uses eight bounded
K derivative panels and two J panels. The archived component traces show
substantial three-center derivative contraction and pseudo-density product
intervals after projection reuse. Panel counts alone do not establish
repeated raw forward-integral generation. Inclusive GPU-event intervals
include stream idle time between submissions; host/GPU views overlap and
must not be added or interpreted as achieved occupancy. These observations
motivate further attribution/optimization; they do not establish a joint
J/K contraction's correctness or performance.

## Offline replay and storage

From the repository root:

```bash
uv run --no-env-file --no-project --python 3.12 --with numpy \
  benchmarks/results/h100-source-pair-20261030/replay.py > summary.json
```

The reader checks exact case sets/order, source/core/AOT/input/device bindings,
every force coordinate against the independent oracle, all qualifier fields,
three process-pair statistics, trace bytes, native component validation and
completed-row projection reuse. Explicit exceptions keep checks active with
`python -O`. The storage representation is:

- `observations.json.xz`: full energy/timing/convergence/settings/precision
  observations, plans, original qualification reports and shared provenance.
- `forces.npy.xz`: all force coordinates in little-endian FP64; no rounding or
  downcast. Lists are reconstructed before checking each original normalized
  JSON digest, so a representation change cannot silently change a value.
- `original-json-bindings.json.xz`: original private JSON byte counts/hashes
  and normalized-public JSON hashes.
- `traces.jsonl.xz`: all eight CUDA and host JSONL streams concatenated with
  exact byte boundaries in the manifest. Each recovered stream is byte-identical
  to its original. `producers/*.py.xz` retain the executed Python adapters and
  campaign/runtime producers without executing them during replay.

Only the private shared-storage root prefix is replaced with
`<H100_PORTABILITY_ROOT>` in JSON string values and dictionary keys. Public
JSON is therefore not byte-identical to private originals. That replacement
changes the qualifier's canonical `receipt_sha256`; replay recomputes this
identity and requires every other qualifier field to match exactly. Numerical
values, ordering, source revisions and original artifact identities are retained.

The eight large transient progress streams, scheduler/build logs and profiler
databases are not published here. Original progress hashes/sizes are explicit
in the manifest and private originals remain in existing run storage. The
public reader does not claim to replay those omitted progress streams. This
bundle publishes the complete E+F population and CUDA/host component streams
needed for the statements above within the unchanged evidence budget.
