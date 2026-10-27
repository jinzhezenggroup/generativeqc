# Native prepared admission and bounded XC capacity

This publication qualifies a bounded #2227 R2 cutover. Native prepared owners
retain a public numeric ledger across preparation, independent ports, execution
and destruction. Strict FP64 all-electron s/p PBE/PBE0 KS with an explicit numeric
cap reserves its mandatory fleet before optional XC storage. External selected
inventories, unsupported controls and native capacity misses retain their bounded
incumbent paths. Public CUDA DF KS forces remain unsupported.

The measured baseline is `1ec57de7eefd523f1a079351faf142fce6b74135`; the candidate
adds the retained `candidate.patch.xz` (decompress before applying), SHA-256
`591cd8b728e89abd06cc63c25d0ddb6e95c3db9547de6274886cad5eddf11512`.
Published master `e1e244197411b58da42aa8a2cbce49014d7ed3ff` has the identical
baseline tree. Documentation/evidence edits after measurement are not part of
that frozen patch. The old 51/100 failures remain failures in the
[historical diagnosis](../prepared-admission-xc-diagnosis-20261026/README.md).

## Reading and replay

From the repository root:

```bash
python benchmarks/results/native-prepared-xc-promises-20261026/replay.py
```

The shared publication reader authenticates plain/gzip/XZ members. Replay checks
every saved science/resource observation, all original 100 timing/work
comparisons, exact/full/adjacent/one-byte cap controls, unsupported force
refusals and SDK metadata lifetimes. It returns nonzero for any original timing
failure, preserving the 1.05 gate. No GPU or native library is loaded.

The publication manifest's numerical decision is distinct from that paired
nonregression result in `samples` and `summary`. The shared validation envelope's
speedup-promotion schema requires separate compile-cost and complete promotion
budgets; this campaign does not claim that promotion. Its endpoint acceptance
criterion is the original #2227 1.05 nonregression test, with unchanged published
iteration work, 21 samples and three warmups per block in both ABBA and BAAB.

The original 20 SDK routes include cold/warm/moved prepared endpoints. The 30 CLI
routes include both geometries in separate complete processes; process launch,
preparation and normal cleanup are timed. The target PATH excludes Python and
compiler executables. All original requested properties are retained. SDK v1
grid arrays match the authenticated original inputs; the current CLI PBE v4
default uses new independent PySCF 2.14 references, checked by reconverged finite
differences at two steps. Energy and force gates remain 1e-8 Hartree and 1e-6
Hartree/bohr.

## Reproduction inputs

Restore byte-identical frozen scripts, protocol, model/oracle inputs and SDK
drivers into a new directory without executing them:

```bash
python benchmarks/results/native-prepared-xc-promises-20261026/replay.py \
  --extract-frozen .artifacts/native-prepared-xc-reproduction
```

The extractor verifies every original SHA-256. `frozen-inputs` retains the earlier
cap data required by the unchanged historical checker on import; its successful
old caps are not substituted for current-source cap measurements. JSON stays
structured with explicit byte-reconstruction formatting. Source text is retained
separately in `frozen-sources` so formatters cannot rewrite a measured runner.

Create independent source trees with `git archive` of the measured base (or the
tree-identical published master), applying `candidate.patch.xz` (decompress before applying) only to the
candidate. Build/install with the recorded Release configuration, CUDA 12.9,
architecture 120, `GENERATIVEQC_CUDA_FAST_COMPILE=OFF`, and verified sccache 0.16+
launchers. The original/final build workers and complete SDK drivers are retained.
The final worker records incremental provenance because only a completed build
tree was reused; all earlier installations, archives and patches stayed intact.

Frozen workers include their original absolute n1 paths. For a new measurement,
use fresh directories and installation prefixes, adapt path constants and
regenerate input/product manifests before submission; never run them against
the retained receipt directories. Preserve model, property, cap and timing
protocols. Run GPU build qualification, controls, profiles and timings through
finite `srun` on n1/main with `--gres=gpu:5090:1`, preserving assigned
`CUDA_VISIBLE_DEVICES`. The recorded timing invocation uses `--nodes=1
--ntasks=1 --cpus-per-task=2 --time=05:00:00`; profiling is a separate 40-minute
job. Compilation/profiling never contributes samples to clean timing.

## Evidence boundaries

`summary` retains qualification and failed-attempt identities. `controls`
retains all final fixed-bound raw publications and errors plus preflight and
metadata-lifetime observations. `samples` retains every full-precision timed
observation, CLI warmup, per-call duration and comparison. Repeated commands
are shared without changing order or numerical values.

`profiles` retains all 160 trace identities, kernel/API distributions, copy
bytes/counts, graph activity and tracked allocation lifetimes. All 80 paired
kernel counts agree. No recorded CUDA runtime errors or inconsistent lifetimes
were observed. Nsight 2025.1.3 emitted a CUDA 13-driver/12.9-tracing-library warning
for every trace; this diagnostic is retained. Kernel counts are not Fock counts;
effective precision, complete opaque CUDA/provider storage and physical RSS
are not inferred. Numeric ledger peaks reset per binding and do not themselves
describe an entire CLI process peak.

Large profiler databases, build outputs, full diagnostic streams and failed
attempts remain in ignored `.artifacts/xc-current/` and the recorded n1 run
directories. No external artifact host is required for numerical replay and no
Release/tag or publishing workflow is involved. The 21-AO private DF UKS fixture
failed final-state validation both with and without a resource ledger; it remains
unqualified. Small DF private-port tests do not expand public force capability.
