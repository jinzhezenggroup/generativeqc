# MGGA integer-power experiments, 2026-10-14

This is the **historical** decision for the original common CSE source and
tested scopes. New source-matched master qualification, with SCF and unqualified
spin/precision paths preserved, is retained separately in
`benchmarks/results/m06-rks-force-powers-20261014/`. The regressions and stricter
stress failures in this capsule remain unchanged; these are not master timings.

**Decision: do not change production defaults or the master README.** Bounded
integer-power lowering speeds point evaluation, but both tested endpoint
variants have a retained complete-endpoint regression. The compiler exposes an
explicit FP64-only experiment; ordinary SCF and force generation stay unchanged.
The durable rationale is in
`.agents/notes/rejected/2026-10-14-mgga-integer-power-defaults.md`.

## Matched full endpoints

All native endpoint runs use node n5's Slurm-assigned RTX 5090, Release/auto AOT,
FP64, matrix tile 16, the ordinary README moving-grid/DF/basis protocol and its
unchanged 1,800-second whole-point limit. Every population contains cold, five
frozen-seed warm, moved and five moved-warm calls. Reference files at both
geometries are independently retained and verified by the shared driver.

The common native source is `786c31a48`, containing an **unmerged, separately
qualified equal-spin branch CSE prerequisite**. Candidates add only the recorded
integer-lowering change. These are incremental comparisons on that common
source, not a new measurement of master `810f081ef` or the published GPU4PySCF
timing populations. Do not credit the preceding CSE gains again.

| M06-2X phase | Baseline (s) | Global lowering (s) | Force-only lowering (s) |
| --- | ---: | ---: | ---: |
| DF/48 cold | 37.869489 | 35.188102 | 37.431879 |
| DF/48 warm | 7.124970 | 6.514494 | 6.670259 |
| DF/48 moved | 28.664111 | 26.583964 | 28.288082 |
| DF/48 moved-warm | 7.118959 | 6.519952 | 6.669358 |
| DF/96 cold | 123.680956 | 114.099151 | 122.614808 |
| DF/96 warm | 29.957500 | **31.703341** | 29.000015 |
| DF/96 moved | 92.432596 | 85.607959 | 91.500335 |
| DF/96 moved-warm | 29.948927 | 27.516511 | 28.989035 |
| Direct/96 cold | 136.580794 | 133.262118 | 136.208958 |
| Direct/96 warm | 17.251976 | 16.272472 | 16.772929 |
| Direct/96 moved | 94.625997 | 88.309197 | **106.983009** |
| Direct/96 moved-warm | 17.272345 | 16.271191 | 16.787956 |

Warm entries are medians of **all five** samples. Their full ranges are in the
compressed evidence. Cold includes calculator construction and preparation:
the table uses `complete_seconds`, not only synchronized execution `seconds`.

- Global DF/96 warm executes four rather than three SCF iterations/Fock builds
  in every replay: **5.8% slower**. Global direct/96 cold has 19 rather than 18.
- Force-only direct/96 moved executes 14 rather than 12 iterations/builds:
  **13.1% slower**, despite byte-identical SCF point source. Its cause has not
  been independently isolated; source equality does not establish identical
  compiled full-image behavior or deterministic work across separate cohorts.
- Every other force-only phase retains the available SCF/AO/grid work. Warm
  savings are 6.4%, 3.2% and 2.8%, but do not justify universal promotion.
- All **108 returned E/F calls** pass the unchanged independent 1e-8 Eh /
  1e-7 Eh/Bohr gates; observed maxima are 2.0464e-11 / 1.5727e-10. Scientific
  agreement is not a guarantee of reduced work or endpoint speed.

No #2486 scheduling knob is retried: host AO acquisition, generic dddd routing,
order-seven pages 32 and matrix tile 16 remain the incumbent choices.

## Diagnostic boundaries

Post-CSE M06 point evaluation on RTX 5090 drops 1.131505 to 0.696010 ms for
65,536 points, approximately 38.5%. A separate, **unpromoted** r2SCAN extended
addition-chain experiment saves about 15.5% on RTX PRO 6000. Neither figure is
a complete E+F speedup or an update of the README hardware comparison.

Preserve the additional strict-tail failures. At rtol 2e-10, the independent
113-bit Libxc Maple oracle still finds four common M06 baseline/candidate
derivative failures at 65,536 diffuse points. Binary64 Libxc has its own larger
cancellation errors there. The existing scientific point/finite-difference/
boundary gates pass, but the stricter diagnostic has **not** passed. No
tolerance is relaxed and no failing row is removed. Both M06/MN15 GPU memchecks
report zero errors. The copied dirty-source point CLI remains globally failed
for provenance even though its individual method gates pass.

The initial incompatible-glibc endpoint failure and missing-executable sanitizer
attempt also remain in the evidence. Accepted endpoint cohorts use the same
hashed loader/runtime bundle already retained by #2486 in all arms. Native
production code receives no compatibility shim or reference-oracle dependency.

## Retained evidence

`summary.json.xz` contains every returned full-precision endpoint row, all warm
ranges, actual SCF/Fock counts, available AO/grid/force work, independent oracle
errors, source/consumer/library hashes, Slurm receipts, compiler/cache/loader
provenance, raw point timings, stricter failures and reproduction scripts.
Unavailable inner J/K counters remain null, not zero. Baseline jobs are
n5/1568–1570; global and force-only jobs are n5/1571 and n5/1572.

Only explicit compiler calls opt into the experiment:

```python
emit_split_hybrid_device_body("M06-2X", lower_integer_powers=True)
```

The option is boolean and FP64-only. Ordinary entry points leave it false;
float or non-boolean opt-in requests fail closed. Reproducing the two full
native ablations additionally requires their embedded source patches and the
same common CSE source/build configuration. No commit, PR, Release or release
asset is created by this experiment.
