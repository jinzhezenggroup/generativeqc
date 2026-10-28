# Native prepared HF projection transaction

Scoped CPU R4/R6 and matrix-capacity evidence for #2227. This does not qualify
CUDA/ECP transfer, full resource/RSS authority, general progressive scheduling or
completion of the parent issue. It is independent of #2841's plan admission.

Integrated baseline: `d51bebb7807c2c7a754bc96a104a121435517688`. The candidate uses a
separately built library and the production reconstruction in `source.patch.xz`.
`provenance.json` records verified reconstruction of all 11 changed production
paths, compiler/package/runner identities, build settings and compiler-cache
observations. `samples.json.xz` binds the actual source files and both library
hashes. No same-library substitution is used for this native ownership change.

Both builds use Release GCC 11.4.0, scalar CPU linear algebra, CUDA disabled,
default CPU stationary force AOT enabled and verified sccache 0.16.0. Generated
compiler commands invoke the launcher. The final observed shared-server snapshot reports 160 requests, one hit and
159 misses, with no cache errors. Snapshots are not isolated compile-speed
measurements or per-task deltas. This task did not reset caches or weaken identities.

## Complete endpoint protocol

Each ABBA and BAAB block performs three warmups and 21 samples per case. There
are 1,008 timed endpoints, with one OpenMP/OpenBLAS thread requested. Imports and
process startup are excluded; owner construction and ordinary cleanup are timed.

- RHF H2 and UHF H2+, each Direct/DF: source STO-3G solve, basis projection,
  target def2-SVP E/F, independent fixed-density target audit and cleanup.
  Two genuinely different bond geometries alternate. Each sample starts fresh.
- Ragged RHF H2/He, each Direct/DF: source and target construction, source solve,
  projection, cold target E/F, both source/target warm replays, both moved-coordinate
  E/F replays, complete detached density receipts and cleanup. The H2 bond changes
  by 0.035 Bohr; He translates by the same amount. Reuse occurs within a sample,
  never between samples.

Corresponding samples have exactly equal scientific/work signatures: energies,
full force/density arrays, iterations, Fock counts, restart and final disposition.
The progressive cases also retain equal independently rebuilt physical residuals.
The ragged public result does not expose a physical residual; those values remain
null, alongside actual density-update RMS values, rather than invented zeros.
All 12 median ratios pass the unchanged 1.05 gate:

| Case | ABBA candidate/baseline | BAAB candidate/baseline |
| --- | ---: | ---: |
| RHF Direct | 1.007395 | 1.008096 |
| RHF DF | 0.972081 | 0.958523 |
| UHF Direct | 1.007396 | 1.016291 |
| UHF DF | 0.969593 | 0.964688 |
| Ragged RHF Direct | 1.007735 | 1.010495 |
| Ragged RHF DF | 0.976967 | 0.977166 |

Raw timings, min/max and relative median absolute deviations are retained. This
is a scoped non-regression result on a shared CPU host, not an exclusive-host,
kernel-speedup or whole-application ranking claim.

An initial extended water/def2-SVP ragged timing attempt was stopped after about
833 worker CPU seconds before its first block published any comparison. It is
recorded as **incomplete**, not passing or failing the performance gate. Its
reason and harness hash remain in provenance. The completed matrix above makes
no water/OH performance claim; their numerical qualification is separate.

The original qualified `a3ec1f8fd8116fc70cf945ab2e7dd97f1e0d42a6` run is
retained under `pre-integration/`, including all 1,008 samples, production
reconstruction and provenance. Its 12 ratios also passed (maximum 1.012170),
with no scientific/work mismatches. Mainline #2789 and #2842 were then integrated,
and both libraries, all qualification and the full matrix above were rerun.

## Other qualification

- 126/126 CTests, including a public-header SDK transfer across three fleets
  sharing one context; RHF and UHF with an empty beta spin.
- 304 focused Python passes plus 25 subtests and 18 explicit CUDA deselections
  in one integrated CPU-only run. An earlier older-protocol capacity injection
  selected the new path; explicit compatibility selection preserves that test.
- Independent PySCF 2.14.0 moved water/OH RHF/UHF Direct/DF energy and full-force
  comparisons pass absolute gates of 1e-8 Hartree and 1e-6 Hartree/Bohr. Existing
  system projection checks also cover Cartesian/spherical representations.
- Real native exact/below capacity, strict/non-strict later-item rejection,
  malformed coordinates, frozen/imported/stale eligibility, concurrent source
  borrowing and publication rollback checks pass. Qualified Python calls forbid
  source-density export, Python metric/projection work and compatibility retry.
- Installed SDK binary passes under `env -i PATH=/nonexistent`, with only the
  installed library/provider search path. No Python/compiler runtime dependency.
- Sphinx warnings-as-errors, pinned ty 0.0.82 and all applicable hooks pass.

The native capacity is a conservative simultaneously-live **matrix numeric**
bound, including staged proposals and final physical validation. It excludes
prepared arenas, caller buffers, topology metadata and opaque provider memory.

## Plan-admission integration

Integration with master `78e0c8afab504bbdf7ecd73ede2e4179195eaf49` retains
both #2841 plan admission and this fleet transaction. The public inventory now
contains 133 C functions, and the progressive endpoint test forbids both Python
planning and Python projection compatibility on admitted native execution.
The integrated CPU Release build passed 127/127 CTests, 233 focused Python tests
plus 25 subtests, and the installed public-header SDK under
`env -i PATH=/nonexistent`. This is a fresh integration correctness check;
the complete endpoint timings above remain measurements against `d51bebb78`,
not new timings for the later master. No GPU qualification is claimed here.

## Reproduction

Build the baseline and candidate independently with the settings in provenance,
then run:

```sh
python benchmarks/results/hf-fleet-transfer-20261028/benchmark.py \
  --baseline /absolute/baseline-checkout \
  --candidate /absolute/candidate-checkout \
  --baseline-library /absolute/baseline-checkout/build/libgenerativeqc.so \
  --candidate-library /absolute/candidate-checkout/build/libgenerativeqc.so \
  --output /absolute/ignored-directory/samples.json
```

The harness saves completed blocks, retains every timed sample and fails on any
scientific/work signature difference or median ratio above 1.05. XZ records are
lossless; read with `json.loads(lzma.decompress(path.read_bytes()))`. Hashes bind
this historical receipt and do not add permanent required-CI source fingerprints.
