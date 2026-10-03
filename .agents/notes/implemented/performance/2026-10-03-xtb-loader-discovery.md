# Decision: skip unneeded loader discovery and isolate constructor timing

Status: implemented
Date: 2026-10-03

## Problem and decision

After structural CUDA topology bootstrap, small cold endpoints still spent time
in repeated Python loader discovery. Every constructor scanned all optional
NVIDIA provider directories even when all six SONAME groups already had retained
successful handles. Explicit native-library overrides also scanned lower-priority
wheel locations before selecting the known existing override.

Discover provider directories only upon reaching a missing group. Retain the
existing dependency order and successful handles, and retry missing or unloadable
groups on every later call. Yield native candidates lazily in the existing order:
environment override, bundled library, source build. Each load gets a fresh
iterator. Native ABI binding, library loading, device facts and profile selection
still execute on every call. This introduces no device/path/profile result cache.

The initial preload remains required by the
[CUDA wheel provider boundary](../architecture/2026-09-18-provider-free-cuda-wheels.md).
Skipping it solely because a native SDK happens to be installed would change
provider precedence and break the wheel's lazy SONAME resolution contract.

## Timing correction

Investigation also exposed a benchmark lifetime error: assigning a new
`Calculator` to `calc` inside the constructor timer destroyed the preceding
calculator there. Consequently all earlier multi-case receipts charged the
previous molecule's cleanup to the next constructor. This affects historical
`construction_seconds` and derived `cold_total`, including the
[bootstrap note](2026-10-03-xtb-topology-bootstrap.md). First-process and
singlepoint/warm/changed measurements are unaffected by this particular error.
Those historical rows must not be described as isolated constructor timing.

Release the last result and calculator after all samples, before the next case,
and record that cost separately as `cleanup_seconds`. It is cleanup after the
whole cold/warm/changed sequence, not a cold-only lifecycle measurement.
`cold_total` remains constructor plus first call. Reports carry
`calculator-cold-v2-separate-cleanup`; comparisons reject mixed timing contracts.
Old/old comparisons remain readable and explicitly labeled `legacy-mixed-cleanup`.
Do not subtract an estimated cleanup cost from historical measurements: remeasure
both engines under the same contract.

## Validation

Loader tests exercise already-loaded groups without directory access, missing
providers, failed then repaired loads, alternate SONAME groups, explicit path
changes, fallback after an override disappears, repeated device selection and
independent diagnostics. All 49 packaging/profile tests pass, including the
native source-identity ABI gate. The 24 benchmark tests include both engine
entry points with a fake clock and a finalizer costing 100 seconds, proving that
both constructors still measure exactly one second and cleanup is reported
separately. Accuracy/SCC/geometry comparison gates remain in force.

CMake and native harness builds use ccache. Final n2 runtime/bootstrap tests pass
43/43. Both n1 RTX 5090 and n2 RTX PRO 6000 pass 41 public tests and 110 complete
energy/force samples under the corrected comparator. All energy, force and SCC
count gates pass. Settings remain FP64, fresh SCC, 300 K, Broyden 8/0.4, maximum
300 iterations, and energy/charge tolerances 1e-10/1e-8. No scientific work,
CUDA schedule or numerical tolerance changes in this increment.
Energy matches the parent exactly and maximum force change is 4.163e-17 Eh/bohr;
maximum xTBloom differences are 5.684e-14 Eh and 5.156e-15 Eh/bohr.

## Corrected endpoint evidence

Constructor plus first call, milliseconds; each is one cold sample:

| GPU / case | Parent loader | Candidate | xTBloom |
| --- | ---: | ---: | ---: |
| n1 / H2O | 14.389 | 13.325 | 13.591 |
| n2 / H2O | 13.238 | 12.734 | 12.935 |
| n1 / NH3 | 14.713 | 16.066 | 15.166 |
| n2 / NH3 | 13.600 | 13.257 | 13.374 |
| n1 / HF | 11.597 | 10.513 | 10.553 |
| n2 / HF | 10.504 | 10.027 | 9.966 |
| n1 / water32 | 246.086 | 233.552 | 260.396 |
| n2 / water32 | 228.481 | 227.383 | 240.129 |
| n1 / water64 | 363.896 | 362.745 | 395.992 |
| n2 / water64 | 349.393 | 348.929 | 383.249 |
| n1 / first-process H2 | 453.975 | 502.485 | 380.914 |
| n2 / first-process H2 | 336.681 | 345.278 | 280.626 |

All ten first-singlepoint, repeated and changed-geometry medians beat xTBloom on
both GPUs. Combined cold totals win 8/10 on each. First-process initialization
remains slower and variable; n1 NH3 and n2 HF also still trail in this cohort.
These results do not establish universal cold-start superiority or a repeatable
large-molecule gain from a Python discovery change. All samples remain recorded.

A separate cProfile probe of 20 later constructors on n2
drops from 38,661 calls / 21 ms to 15,901 calls / 9 ms; the removed discovery
functions no longer appear. This diagnoses host work, not end-to-end GPU speed.
The first constructor still spends about 342 ms in native device probing and
61 ms preloading providers. A construction-only Nsight summary omits most of
that initialization; it cannot support a complete CUDA API work claim.

Ignored receipts:

- `.artifacts/{n1,n2}/loader-discovery-v2/`: corrected parent, candidate,
  xTBloom and comparison JSON. Earlier `loader-discovery/` reports retain the
  preceding-cleanup contamination and are diagnostic history only.
- `.artifacts/{n1,n2}-loader-endpoints-v2.log`,
  `.artifacts/loader-final-host-tests.log`, `.artifacts/loader-benchmark-tests.log`,
  `.artifacts/n2-loader-final-tests.log`: acceptance logs.
- `.artifacts/n2-{bootstrap,loader}-construction-profile.log`: cProfile evidence.
- `.artifacts/loader-python-identities.txt`: changed Python source SHA-256s;
  remote `.artifacts/python-before-loader/` preserves the parent export.
- Within each node, both Python variants use the same native binary. n1 uses
  bootstrap SHA `affe3faf3beb45bcd341e108eb3f58654bc14b5ed2188fe7616535d04c9556f1`;
  n2 uses final build `7e93f097958fe2d5115448c587164c7b4b001271895922d7643f66e30cf9bde1`.
  GFN2 CUDA objects are unchanged in the latter source-identity refresh.
- xTBloom revision `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`, binary SHA
  `6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

## Revisit when

Further ABI-handle retention needs explicit identity, lifetime, diagnostics,
threading and monkeypatch contracts. Do not introduce a global ordinal cache
or assume an earlier profile remains valid to save constructor time. First-process
startup and the remaining native scientific schedules require independent work.

## Historical first-process timing qualification

The later [PR #1742 comparator correction](https://github.com/jinzhezenggroup/generativeqc/pull/1742)
found that identity verification triggered xTBloom's lazy native/provider load
outside both timed phases. The historical first-process comparisons above omit
that reference loading cost. The corrected v3 comparator includes lazy loading
and retains separate cleanup; warm/changed timings and numerical gates keep
their original scope. Its measured endpoint begins after Python package import,
not OS process launch. The later corrected measurements include subsequent
stack changes and are not an isolated remeasurement of loader discovery.
Original timing rows are unchanged; no universal startup superiority is claimed.
