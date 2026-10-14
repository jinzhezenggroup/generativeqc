# Exact two-electron response reuse qualification

These are complete, same-library off/on WB97M-V energy/force populations at
the original FP64, 48×16×32 grid, 100-iteration, 1800-second point protocol.
Every cold, five fixed-seed warm, moved and five moved-warm call passes the
independent 1e-8 Eh / 1e-7 Eh/Bohr gates. The experiment remains **default-off**;
these observations do not replace the default README curves or establish the
remaining 96-atom acceptance associated with #2426.

| Atoms | Off cold | On cold | Off warm | On warm | Off moved | On moved | Off moved-warm | On moved-warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 7.13146 | 7.13483 | 0.38498 | 0.33264 | 5.40272 | 5.40894 | 0.38466 | 0.33246 |
| 12 | 30.61254 | 30.32307 | 1.65264 | 1.21116 | 19.46161 | 19.51819 | 1.65911 | 1.22443 |
| 48 | 253.57256 | 253.55527 | 11.40746 | 7.49369 | 212.45508 | 212.35508 | 11.40477 | 7.48548 |

Times are complete endpoint seconds; warm columns are five-sample medians.
At 48 atoms the native integral-response interval decreases from 4.06665 to
0.15912 seconds in the warm median. Cold/moved still compute all sources.
Actual-hit traces separately prove skipped two-electron evaluations for exact
keys, with current one-electron/Pulay evaluation retained. UKS warm solves can
change density bits and correctly miss; repeated authenticated UKS snapshots
exercise real hits without a tolerance-based key.

The measured source is `b5ca4306e0ae01776b0b5d17b1365db751602201`.
The compiled source is `78961f12db3482a9e79f409c435237a96c693237`;
only two test files changed, and all 1731 native identity inputs are identical.
The native inputs also match historical `0d12d0596a526edd703f8ef5c95192c3e18e0917`.
Later `11e16d800a87b26d031544c25733addc43bfcf4f` changes one `std::min` template
argument for Apple integer-type portability; it was not part of these GPU
measurements.
Library SHA256: `6e180869daee6a32aa504c700d7ab8cfb83a6b47af8d53d4496436e06e81f5ab`.
Native identity: `6326a312702b21bf338cdff066511df843da451a50455e96f5a14f117417494c`.
Measurements use n4 RTX 5090 Slurm jobs 913/914. Each arm explicitly sets
`GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE` to 0 or 1, retaining other defaults.

Qualification receipts retain:

- n4 job 911: six independent molecular E/F and reconverged displaced-energy
  cases, stale-state recovery and RKS reuse pass. The initial UKS test's
  incorrect assumption of identical warm densities fails and remains recorded.
- n4 job 912: corrected RKS/UKS endpoint parity and exact-snapshot reuse both
  pass, including geometry changes, explicit derivative routing and disable.
- n1 jobs 8445/8446: memcheck and synccheck each pass all three focused cases,
  with zero errors. Instrumented timings are not performance observations.
- n2 job 2982: NVML/driver mismatch aborts preflight before any test; this is
  retained as a failed launch, not a GPU qualification.

`evidence.json.xz` losslessly contains every successful timing sample, complete
force vector, independent cold/moved oracle, trace, JUnit result and relevant
build/cache/scheduler receipt. `manifest.json` binds stored and decoded hashes.
With Python 3.10+ and its standard-library LZMA support, run from the repository
root in an isolated environment:

```bash
python -m venv .artifacts/exact-response-audit-venv
.artifacts/exact-response-audit-venv/bin/python -m pip install defusedxml==0.7.1
.artifacts/exact-response-audit-venv/bin/python benchmarks/results/exact-integral-response-20261014/verify.py
```

The audit uses hardened XML parsing from `defusedxml` and requires no native
library, GPU or new measurements. It recomputes all 72 E/F gates, six complete
populations, medians and qualification outcomes. The original partial
reference/48 timing stays partial; the separately reconverged moved oracle
supplies accuracy only.
