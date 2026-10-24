# Decision: compose landed native transfers after the DF-defaults CI tail

Status: implemented
Date: 2026-10-12

## Actual failed source and workload

PR #2359 run 38231471125 tested merge
`4a3a1abefefa18336d7836754020bc9ab3b40101`, with parents current reviewed
head `dc2d1ee721d13ff2fc64d5561d76c9b45f8a9187` and then-master
`f2cf62d9b5bee139b3d8815ba78272f9770aa3c2`. Its selector was the
pre-transfer blob `c390a400e7405592ffb8e6033d1213a0ad5b1ce5`.
Core-c job 114763636490 reached the unchanged 30-minute test-step limit at
10:47:20 UTC with corrected-Lambda cases still active. There was no visible
numerical assertion failure. The separately reviewed allocation-error
recovery executable and historical-evidence reconstruction/mutation controls
passed in this run; their source is retained unchanged.

Observed complete pytest times were core-a 1575.92, core-b 971.62,
runtime-heavy 1010.11, posthf 1766.76, compiler-heavy 1238.17 and ecp-forces
622.79 seconds. Core-c timed out at 1800 seconds. The three native UCCSD
cases on C account for 391.522 visible start/result worker-seconds; the six
native TensorIR CPU cases on A account for 251.660. These are observed
worker intervals, not full fixture costs or a wall-time prediction.

## Mechanical composition

Integrate actual landed master
`61a0e715fde35566722870389af8e137036232db` with the reviewed DF-defaults
head. The master selector and ownership regressions are retained byte for
byte. Its already reviewed native transfer moves complete
`test_cc_native_tensor_cpu.py` and `test_uccsd_native.py` into the existing
compiler-heavy lane, retaining their original core inventory slots. This
directly removes native work from the observed A/C tails. No new shard rule,
frontier plugin, corrected-force source, timeout or worker change is added.

Compiler dependencies, loadfile scheduling and its existing diagnostics
remain as documented in the original native-transfer decision. Complete
numerical/negative cases and fresh owner lifetimes are preserved. The prior
compiler duration plus transferred worker intervals is not a bound for the
combined critical path; final hosted CI and the normal merge queue decide
whether it meets the unchanged deadline.

All reviewed production, allocation-recovery, numerical-test and retained
source-evidence blobs remain exact. The promotion inventory mechanically
unions the head's standalone-Lambda/complete-force distinction with master's
independent ragged-XC record; no qualification claim is broadened. Local
execution remains blocked by executor ENOSPC, so static inventory/source
checks are not a claim of local Bash or pytest execution.

## References

- [Native transfer](2026-10-12-uccsd-native-compiler-transfer.md)
- [DF response defaults](2026-10-12-df-force-response-defaults.md)
- [Failed hosted run](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38231471125)
