# Portable DF Rys and coalesced stationary J/K gradients

The automatic derivative selector now prefers qualified exact-target tuning,
then the portable 23-class FP64 Rys/compact profile. It has no GPU-model
allowlist. Missing mathematical classes and explicit legacy retain polynomial
execution. Ordinary full-range restricted hybrids combine J/K cotangents before
one three-center/metric derivative contraction. Component diagnostics reject
coalesced sources; unrestricted/range-separated consumers keep their channels.

`report.py` reads `observations.json.xz` and reports complete energy-plus-force
warm medians. Every matrix population has one cold call and three warm calls,
one math thread, unchanged stock quadratures, and the frozen engine-local cold
density. All times, energies, iterations/Fock counts, available grid/nonlocal
work, independent per-repeat errors and original receipt identities are retained.
Full raw receipts and omitted force vectors remain in ignored
`.artifacts/a100-catchup-20261031/`; the compact summary cannot reconstruct them.

## Measured scope

- Candidate integration: 32 DF populations each on A100, RTX 5090 and RTX PRO
  6000, covering the eight README methods at 3/12/48/96 atoms. A100 explicitly
  selects the portable derivative candidate. Blackwell uses exact SM120 auto;
  its preceding full matrices forced the Direct portable profile. Those
  comparisons are not an isolated coalescing ablation.
- Default integration: a separate complete 32-case A100 matrix with automatic
  portable selection; two final-binary HF/PBE0 DF96 populations on each
  Blackwell device. Preserve their distinct source and binary identities.
- Each complete matrix retains 27 qualified and five failed independent gates:
  HF DF fitting energy at all sizes, and M06-2X DF3 default-grid force error.
  No scientific gate was changed. The 48/96-atom def2-SVP cases contain 384/768
  orbital functions. Historical SM120 controls remain separately retained in
  `issue435-jkfit-rys`; their original target provenance is not rewritten.
- A100, 5090 and PRO 6000 each pass 18 independent tests on both candidate and
  promoted-default builds. Coverage includes RHF/UHF, panel/group/packet
  consumers, auxiliary-f shells, hybrid original/moved-geometry forces and
  constrained-memory fallback. Candidate native transport/controller tests pass
  eight compiled profiles on each device; default Blackwell integration repeats
  them. Host provider/binding tests pass 124; final policy/dispatcher tests 65.
- Both Blackwell candidate builds pass nine unfiltered CUDA memcheck tests.
  A100 passes nine memory tests and three race tests with
  **`--report-api-errors no`**. CUDA 12.9 cuBLAS on driver 535 performs an
  unsupported `cuCtxGetLimit` feature query, reproduced by standalone cuBLAS
  create/destroy despite successful creation. Unfiltered native API failures,
  vendor probe and initial loader failures are retained separately. This is
  not an all-API-errors sanitizer pass.

Default A100 warm seconds at 96 atoms, compared with the preceding portable
baseline and retained GPU4PySCF endpoint:

| Method, DF | Previous | Automatic portable | GPU4PySCF |
| --- | ---: | ---: | ---: |
| HF | 3.126004 | 1.976332 | 1.417968 |
| PBE | 3.664199 | 3.674189 | 2.294710 |
| PBE0 | 7.163701 | 4.260415 | 3.205529 |
| B3LYP | 7.224516 | 4.318768 | 3.193353 |
| M06-2X | 7.565493 | 4.666087 | 4.050520 |
| r2SCAN | 4.247439 | 4.250100 | 3.146223 |
| r2SCAN-3c | 6.263321 | 6.259109 | 4.031535 |
| WB97M-V | 34.379809 | 31.930048 | 29.231100 |

These default-grid observations do not establish universal accuracy-qualified
ranking. HF retains its fitting failure. Native and GPU4PySCF WB97M-V DF have
different long-range fitting contracts. The retained 5090 reference uses n4,
while the new native runs use n5; the two physical RTX 5090 hosts are explicit
in original provenance. PRO 6000 has no GPU4PySCF reference. All A100 DF96 cases
still trail that reference; the broader all-functional target remains open.

## Controls and negative evidence

The initial A100 small M06-2X/r2SCAN timing increases did not reproduce in two
fresh ABBA rounds (12 warm samples per arm/method); all samples remain retained.
The promoted-default PBE DF12 matrix has an initial warm timing increase;
two fresh cross-binary ABBA rounds do not reproduce it. The previous portable
default takes 0.176911 s versus 0.162946 s for the promoted default (12 warm
samples per arm); all eight independent endpoint gates pass, with paired force
differences below 1.23e-13 Eh/Bohr. This is a broader production regression
control, not an isolated qualification-flag toggle. Initial observations remain
retained separately from the recheck.

Increasing force tiles and geometry scratch to 2048 points/64 MiB improves
several semilocal endpoints but regresses WB97M-V DF96 to 81.674956 s. The clean
Rys candidate takes 31.959681 s, and a same-build legacy control 34.386219 s.
All 16 enlarged-grid populations pass unchanged numerical gates. **Those grid
overrides are rejected and absent from production.** Their exact cause requires
further work; shared semilocal/nonlocal batching is a candidate diagnosis.

## Provenance and storage

The XZ bundle preserves measured native identities, the candidate-to-review
source differences, and separate automatic-integration receipts. Formatting
and provenance-only edits are distinguished from compilation. A final
unqualified-target precedence fix leaves current production policy emission
byte-identical; it is not relabelled as part of the earlier measured binaries.
The inherited `513385791` CuMetal pragma repair is likewise separate from these
DF measurements.

CI run [41100772725](https://github.com/jinzhezenggroup/generativeqc/actions/runs/41100772725)
compiled the inherited repair at `df4125ab733ab9f8f1af831501bcce5e7dccd938`.
Its 102 SM120 shard-6 kernels have exactly the qualified REG/STACK/SHARED/LOCAL
resources and unchanged flags/toolchain. Only generated/preprocessed source
identity changed after removing the invalid pragma. Local current-source
generation matches the measured shard hash, so this change refreshes those
two ledger identities using that actual CI measurement. Resource thresholds
and the checker stay unchanged. The bundle retains the report identity and
measurement; 288 related host checks pass, two device-dependent checks skip.

To keep the existing evidence storage budget, the historical Rys 96/384-AO
controls were losslessly gzip-compressed. Their original retained bytes and
SHA256 values are verified by decompression; no numeric field was removed.

See the [portable selection decision](../../../.agents/notes/implemented/performance/2026-10-31-portable-df-rys.md),
[coalesced gradient decision](../../../.agents/notes/implemented/performance/2026-10-31-coalesced-df-hybrid-gradient.md),
and [grid investigation](../../../.agents/notes/proposed/2026-10-31-stationary-grid-batching.md).
