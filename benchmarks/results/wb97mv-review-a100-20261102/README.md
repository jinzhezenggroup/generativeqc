# A100 qualification of the WB97M-V review fixes

This packet belongs to scientific source
`a53c7618fcd59b88693a693a08a36f7ad3bfa51969f6564b46561e95e04505fa`.
It covers the diagnostic-selection, optional-factor fallback, occupied/raw input
exclusion and mandatory D3/D4 preparation-order fixes in PR #2932. Exact/mixed
long-range treatment and automatic generic LR response remain the defaults;
these timings explicitly select fitted LR and the shell response.

## Complete matched production endpoints

Build 224730 produces separate OFF production and ON native-fixture packages.
Three fresh A100 pairs (224784–224786) each measure a complete cold endpoint and
three warm calls from each engine's own frozen cold density, with alternating
engine order. Native timed processes reject PySCF/GPU4PySCF imports. Geometry,
orbital basis, retained def2-SVP-JKFIT auxiliary input, thread controls and the
fitted LR operator match the independent GPU4PySCF arm. All 48 cross-sample
energy/force comparisons pass the unchanged common gates. Full vectors and
unrounded samples are retained.

| Pair | Native cold (s) | Reference cold (s) | Median warm reference/native |
| --- | ---: | ---: | ---: |
| 224784 | 88.926 | 86.374 | 1.157 |
| 224785 | 90.759 | 85.520 | 1.170 |
| 224786 | 89.306 | 87.708 | 1.177 |

Native warm calls take about 25.1–25.2 seconds versus 29.1–29.7 seconds; the
median paired ratio is 1.170. Cold native calls remain about 2–6% slower. This
is paired static-geometry qualification, not a new dense-grid reference or an
all-functional/all-device performance claim. No sm120 execution of these review
fixes is inferred from earlier-source 5090 results.

The native cold count of 28 Fock builds consists of 23 provisional operator
steps and five full-target evaluations. VV10 records four pair sweeps and one
cache hit. The separate Huckel initializer uses seven preliminary Fock builds.
GPU4PySCF reports ten SCF iterations; both warm paths use one. These counts
explain why raw iteration ratios alone do not predict endpoint cost. All final
physical acceptance gates remain unchanged; the remaining cold gap is visible.

## Native and molecular controls

Job 224764 passes all four native controls (range shell pairs, range exchange,
warm occupied factor and semilocal CUDA) and 132 Python cases, with fourteen
PBE-D4 failures and one skip. The narrow WB package did not contain PBE.
Adding PBE sidecar libraries in 224788 still leaves all fourteen CUDA D4 cases
failing: the frozen core's static native registry also lacks that profile.
Both original runs and their complete failures remain in the packet.

Job 224790 builds a separate OFF D4 qualification core with PBE in the static
registry and passes all 38 original D4 controls. Its scientific source is the
same, but its binary identity/profile inventory differs from the timed WB
package. Original timed core and AOT files remain frozen. The D4 retry does not
transfer WB timings onto the expanded core. Its resource gates retain the
independent energy/force oracle, minimum-capacity admission, optional fallback,
geometry/replay and failure/recovery checks.

The reader maps only identical test IDs to their later retry and retains each
original run's status. The final unique population is 146 passed and one skipped,
with fourteen explicit recoveries; all four native controls pass. The earlier
failures are never rewritten into successful runs.

## Replay and storage

```bash
PYTHONPATH=python:. python benchmarks/results/wb97mv-review-a100-20261102/report.py \
  --output /tmp/wb97mv-review-a100-replay.json
```

The unchanged exact FP64 v3 codec authenticates every decoded text against its
original checksum. Production/fixture identities, runners, endpoint vectors,
all failures, JUnit populations and both D4 package attempts remain recoverable.
Verbose successful initial-build compiler transcripts are represented by length
and checksum in `compile_log_inventory`; their original archives remain ignored
locally. D4 retry build transcripts are retained in full. No numerical rounding,
Release publication, external backup or evidence-budget increase is used.

See the [profile-package rationale](../../../.agents/notes/implemented/compatibility/2026-11-02-profile-scoped-cuda-qualification.md).
