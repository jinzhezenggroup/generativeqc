# Source-bound A100 acceptance for PRs 2902 and 2900

The task owner authorized A100 in place of H100 for these incremental stages.
This evidence belongs to the measured scientific sources below; it does not
qualify an arbitrary later source or the separate all-functional performance
objective. Exact/mixed long-range treatment remains the default.

| PR | Scientific source identity | Original production build |
| --- | --- | --- |
| 2902 | `4b8ab33e04c4347589fdaa232c6eb3a3e8a079c5385d827f0a39dea564a01f51` | 224633 |
| 2900 | `5aa7bf5569c69f173232442e9cf7ba52be80fcb3381b618a8254215e0ae62ffc` | 224634 |

`production` packages have `BUILD_TESTS=OFF`; separate `fixture` packages have
hooks enabled. Native and molecular controls use the fixture. Timed complete
PBE0 endpoints use the production core and hash-bound AOT resources. Each receipt
retains the actual LSF job, assigned A100, driver, source and binary identities,
input geometry, thread controls and runner. No Slurm identifier is fabricated.
The later master integration changes only CI cache tooling; its scientific
identity was recomputed and is unchanged.

The native composition and stationary binding binaries pass normally for both
sources. They cover independent J/K composition, signed transport, RKS/UKS,
ragged/batched items, changed geometry, capture refusal, failure/drain/recovery,
borrowed lifetimes and explicit combined-source admission. PR 2900 additionally
emits ten RSH cases and 240 independent CPU range-value stencils. The reader
recomputes their five-point differences at both steps, checks stability and
compares all 18 derivative components of host, resident and borrowed paths.
The maximum observed error is 1.535e-12 against the unchanged 2e-9 gate. Analytic
CPU Coulomb rows remain distinct from the two finite-difference exchange rows.

Corrected molecular jobs 224709/224710 each pass 145 tests with one skip.
Original loader failures and obsolete-fixture failures remain failures: tests
must query actual snapshot resources, load a hook-enabled fixture for injected
faults, and use separate native/compatibility batch owners. Interleaving a
compatibility response between native cold/warm calls consumes a one-shot lease
and cannot qualify the native replay sequence.

Unfiltered sanitizer qualification passes in jobs 224781/224782: both complete
native binaries pass normally and under memcheck on each source, all eight exit
codes are zero, and each of the four memcheck summaries reports zero errors.
All child processes are instrumented; no API or kernel filter is used. Runtime
copies change only the executable loader/RUNPATH, with every executable,
read-only constant and CUDA section hash checked against the frozen fixture.
The core and AOT files are unchanged.

This old cluster combines a 535.154.05 kernel driver with CUDA 12.9 cuBLAS. An
independent program that only creates/destroys a cuBLAS handle reproduces the
`cuCtxGetLimit` API warning while cuBLAS returns success. The clean memcheck
runs use unmodified NVIDIA Compute Sanitizer 12.4.127 and the official CUDA 12.9
575.57.08 forward-compatibility user-mode driver, restricted to the task's
runtime directory. The system installation/kernel driver is untouched. Original
loader failures, transfer failure and vendor-probe warnings remain retained and
unqualified; the old warning logs' default print limit is disclosed by those
logs. Clean runs use `--print-limit 0`. Timed endpoints retain their original
runtime and must not be relabelled as compatibility-driver timings.

Public Direct jobs 224765/224766 separately pass 29 production tests and three
fixture fault cases each. They cover small RKS/UKS independent E/F, minimum
fleet budgets, cold/warm/moved geometry, ragged failure/recovery and sentinel
output preservation. Earlier jobs 224758/224759 pass the same production cases
but crash in an obsolete test wrapper that passed a public resource handle to
the private ledger ABI. The corrected fixture uses the prepared owner's native
binding; scientific sources/binaries are unchanged. Independent review remains
a separate merge gate.

## Complete PBE0 endpoint population

Every case has three fresh native/reference process pairs, a complete cold
constructor+prepare+SCF+force+export call and three warm calls from each engine's
own frozen cold density. Both use the declared stock grids with a fixed native
SCF tile of 1024. Full vectors pass the common energy/force gates against the
original independent dense reference, whose original hash and identity are
preserved. This static-geometry population is not a moving-geometry envelope.

The table gives median paired warm `GPU4PySCF / native` time ratios; values below
one mean native is slower. All cold/warm seconds, iterations, operator work,
full precision provenance and force vectors are recoverable from the capsule.

| PR | 48 Direct | 48 DF | 96 Direct | 96 DF |
| --- | ---: | ---: | ---: | ---: |
| 2902 | 0.989 | 1.676 | 0.608 | 1.077 |
| 2900 | 1.149 | 1.710 | 0.680 | 1.090 |

The 96-atom Direct regressions remain visible. These ratios do not establish
that the overall 20-of-30 objective or the user's broader all-GPU target is met.
Other functional timing cohorts remain separate from this focused publication.

Six newly executed dense GPU4PySCF references cover 48/96-atom M06-2X,
r2SCAN-3c and WB97M-V. Each reproduces the original two levels and passes both
new-versus-old comparisons and the stricter quarter-gate grid-stability test.
WB97M-V explicitly uses XC levels 5/6 and VV10 levels 4/5. These are independent
reference calculations, never production inputs. The reader recomputes all
18 comparisons rather than trusting stored acceptance booleans.

## Storage and replay

Run without a GPU:

```bash
PYTHONPATH=python:. python benchmarks/results/a100-native-stages-20261102/report.py \
  --output /tmp/a100-native-stage-replay.json
```

The five shards use the unchanged exact FP64 v3 codec. Every decoded text is
authenticated against its original checksum; no scientific value is rounded.
Shards separate native controls, failed molecular fixtures, corrected molecular
fixtures, public Direct lifecycle and endpoints/references, keeping each stored file below 1 MiB.

All complete endpoint vectors, test logs, JUnit results, corrected-run resource
streams, native failure logs and executable recipes are retained. Only verbose
intrusive `policy-*.jsonl` timing streams from the two obsolete failing molecular
fixtures are represented by lengths, checksums, record/event counts and scope in
`intrusive_stream_inventory`; their original archives remain ignored locally.
Those omitted streams cannot supply a new profiling or resource acceptance
claim. Full resource traces from the corrected successful runs remain available.
NVIDIA helper ELF capsules are non-scientific binaries: their hashes and
reconstruction generator are retained, rather than duplicating executables in
Git. This storage choice does not change numerical or device gates.

Historical pre-AO and public-force-policy payloads are independently recoverable
from their existing Git snapshots. No Release, tag, external backup or evidence
budget increase was used.
