# Guarded diagonal XC contraction, 2026-10-16

PR #2567 keeps canonical half-panel preparation and reuses an exactly doubled
shared right panel only under a uniform AO/panel/**input-weight** range witness.
Unsafe tiles retain original two-legged arithmetic. Productive four-jet
pair/point multiplications are **8 → 5**, a symbolic expression count, **not
hardware instructions or measured total FLOPs**. The vote replaces an existing
barrier; no global scratch, approximation or CTA/page-width tuning is added.
FP32, spin, mapping, tails, ordered scatter and bounded fallbacks remain.

## Sources and protocol

- Baseline: #2563 scientific source `ef02ef322e809a905cca09e7d2b3323b9d1972ae`.
- Standalone: `5940546542d0cf77b60c8e59a2df32f8852f8d0f`, exactly two scientific
  emitter changes (`xc_contraction_cuda.py`, `xc_tile_batch_cuda.py`).
- Integration: standalone plus six force-compiler files of #2558 at
  `719214668aeb69146174a7039490fd95d5607533`, composing #2541/#2549/#2556/#2558.
  Those files are **not part of #2567**. No other changes from those branches
  are overlaid. Master advanced independently; #2563 subsequently merged.

These are frozen source-specific pairs, **not current-master measurements**.
Finite Slurm node1 jobs **9053/9054**, 85/80-minute limits, use distinct assigned
RTX 5090s, preserved visibility and eight host threads. Each of 16 populations
keeps the original **1800-second limit**, cold + **all five** warm + moved +
**all five** moved-warm calls. r²SCAN-3c uses canonical def2-mTZVPP, 544/1088
spherical AOs and unchanged moving Becke 48×16×32/atom. PBE0/HF controls use
matched def2-SVP fixtures. Exact Direct J/K, strict FP64, screening, convergence
and numerical gates are unchanged. No prewarming, trimming or count normalization.

Release: CUDA12.9.1/nvcc12.9.86, GNU11.4, sm_120; native/shell/stationary CUDA
AOT enabled, CPU stationary-force AOT disabled. Verified ccache4.5.1 and explicit
CXX/CUDA launchers reuse existing caches. Receipts bind canonical 1761/1761/1762
input inventories, loaded probes/library hashes, generated inputs, build settings
and complete stationary bundles. Eight additional integration Libxc CPU files
differ only in identity strings, not arithmetic.

## Complete energy + analytic forces

Seconds include preparation and returned host forces. Warm entries are medians
of all five clean calls, never intrusive profiles. Entries are baseline → candidate.

| Experiment/case | Cold | Warm | Moved | Moved-warm |
| --- | ---: | ---: | ---: | ---: |
| Standalone r²SCAN96 |308.882366 → 296.468955|83.002791 → 82.456957|203.825642 → 196.732393|83.045661 → 82.485471|
| Standalone r²SCAN48 |76.307738 → 73.115471|21.471919 → 21.282187|57.295039 → 54.758131|21.473707 → 21.330949|
| Standalone PBE0/48 |23.445 → 23.575|4.586 → 4.602|22.401 → 22.509|4.574 → 4.583|
| Standalone HF/48 |10.980 → 10.969|0.746 → 0.745|7.228 → 7.230|0.745 → 0.746|
| Integration r²SCAN96 |315.298063 → 254.163584|82.237315 → 42.716903|201.994474 → 155.577024|82.349748 → 42.753795|
| Integration r²SCAN48 |75.577681 → 61.975930|21.318676 → 11.375935|56.567079 → 44.667711|21.280618 → 11.426388|
| Integration PBE0/48 |23.346404 → 22.419434|4.532278 → 3.984294|22.110113 → 21.633868|4.527317 → 3.973177|
| Integration HF/48 |10.877592 → 10.900262|0.739338 → 0.738534|7.154281 → 7.161102|0.738204 → 0.739822|

96 standalone improves **4.02/0.66/3.48/0.67%**, integration
**19.39/48.06/22.98/48.08%**. Maximum observed 48-control regression is 0.55%.
One cold/moved observation is not a statistical confidence interval.

Actual cold/moved iterations/Fock builds: standalone96 **24/24,13/13**;
integration96 **25/25 → 24/24,13/13**; both48 **19/19,13/13**;
PBE0 **17/18,12/12**; HF **21/22,16/17**. Every warm/moved-warm is **1/1**.
The integration cold gain includes a changed trajectory; it is not an isolated
kernel gain. Extra physical builds are retained, not normalized away.

## Profile and executed work

Separate intrusive integration kernel device seconds; profile cold counts are
**25/25 → 24/24**, warm **1/1**:

| Device group | Cold baseline → candidate | Warm baseline → candidate |
| --- | ---: | ---: |
| SCF Direct J |103.396457 → 99.693109|4.133620 → 4.150129|
| XC/AO/grid |111.792559 → 94.272500|4.472156 → 3.927544|
| Generated low |4.594054 → 4.595359|4.595138 → 4.595546|
| Order-6 spatial J′ |21.728971 → 13.674095|21.726820 → 13.762362|
| Order-7 cooperative J′ |27.811374 → 1.987174|27.807967 → 1.987465|
| Order-8 materialized J′ |5.726122 → 0.449788|5.725769 → 0.450183|
| Other stationary/Becke |11.413950 → 11.411625|9.236757 → 9.236795|
| One-electron metadata |2.258665 → 2.247787|2.252450 → 2.252712|

Native **software executed-invocation census**, not hardware counters:

| Order | Bra preparations | Ket preparations | Component contractions |
| --- | ---: | ---: | ---: |
|6|unavailable|unavailable|unavailable|
|7|2,606,140 → 1,027,752|3,147,016 → 1,250,544|1,937,135,584 → 754,811,008|
|8|231,424 → 95,976|231,424 → 95,976|270,222,624 → 107,642,016|

Primitive products stay **2,400,504,912 / 1,250,544 / 95,976** for orders6/7/8;
7/8 Coulomb preparations stay **1,250,544 / 95,976**. Enumeration/admission and
published scientific work are unchanged. Order6 executed components stay
**1,121,296,224**; unavailable preparation counters remain null, never zero.

The reader exposes mutually exclusive host wall ledgers and device-window
partitions (including no-kernel activity), separately from summed device kernel
times. All ledgers reconcile. Asynchronous order markers do not own their host
wait: never add device time to wall time. Profiles complete in
315.626506→254.447390s cold, 82.428649→42.842425s warm, not clean medians.
Standalone diagnostic cold is also **25/25 → 24/24**, unlike its matched clean
pair; XC device time is 112.868318→95.059515s cold, 4.508174→3.960121s warm.

## Qualification, limits and retention

All **192 clean + 8 diagnostic endpoints** are independently checked at collection
and packaging against original geometry oracles, including D4/gCP and moving-grid
response, at unchanged **1e-8 Eh / 1e-7 Eh/Bohr** gates. Clean maxima:
**2.546585164964199e-11 Eh / 6.738787197528573e-11 Eh/Bohr**.
Host/native: **783 passes**, 44 GPU opt-in skips; compiler audit **541 modules,
zero dependency errors**. Actual emitted CUDA, including FMA-on/off, passes
**44 tests each** under memcheck/synccheck/racecheck, zero errors/hazards/warnings.
Both complete AOT bundles pass three public-composition/independent-grid oracle
tests. Production diagonal consumers: **48 registers, zero stack/spills,
8992 body shared bytes (10016 including CUDA's 1024-byte reservation)**.

Full-coefficient preparation was rejected for finite overflow and silent
cancellation/underflow changes. The final fallback matches all three adversarial
controls exactly; a huge weight defeats a final-panel-only witness. Correct
four-RHS density fusion was slower and rejected. Corrected-source ABBA contraction
probes improve ~26–29%, not endpoint time. Dynamic counters are denied
(`ERR_NVGPUCTRPERM`). Node2 driver/NVML mismatch prevented tests; node1 replaced
them. Two initial revised-source attempts lacked AOT sidecars and returned no
forces; failed receipts are retained, never promoted as E+F timings.

Run `python benchmarks/results/xc-diagonal-potential-20261016/summarize.py`:
NumPy only, no GPU/network/calculation/output file. The unchanged 64-MiB cap
requires an explicit compact scope: **24 candidate96 clean calls + all 8 profile
calls retain every force bit**; the other **168 clean calls retain every scalar
sample/count/error/gate and collection-time independent-validation receipt, not
offline-replayable forces**. Their complete raw JSON/force streams remain in
ignored local campaign directories, hash-bound, not deleted. The reader rechecks
all retained energies, medians/counts and 32 full force gates; it does not claim
168 receipt-only force gates were independently replayed offline.

Force storage uses lossless FP64 XOR byteplanes in standard NumPy ZIP-LZMA.
Shared storage anchors and **separate independent oracles** are hash-bound to
`md-j-sparse-support-20261016`; anchors are never references. Canonical retained
projection hashes verify all reconstructed bits, not omitted verbose diagnostics.
Top16 kernels are named; **all** other launches/phase times are aggregated.
Full SQLite hashes, final provenance, failures and exact site recipes are stored
in `evidence.json.xz`. Reproduce in fresh ignored directories with
`GQC_XC_REPRO_ROOT`, staged frozen sources/bundles/references and finite `srun`;
recipes retain site installation paths, not a portable environment installer.
This task's preceding long MD qualification narrative is losslessly compressed
as `qualification.md.xz`; no foreign data or scientific coverage is removed.

The path is default **on this PR branch** under existing admission, not a claim
of default composed behavior on master; the force overlay remains unmerged.
42.716903s is still **1.61×** historical GPU4PySCF26.59s (16.13s remaining), cold
254.163584s **1.50×** historical169.87s, not new competitor measurements.
Next cold bottlenecks are ~99.69s Direct-J and ~94.27s XC device work at24 builds;
prioritize bounded exact density-first J recurrence reuse. Warm order6 remains
~13.76s: first instrument missing preparation counts, then share primitive/DAG
preparation across contractions. Do not repeat density-fusion or scheduling-only
trials, enlarge caps or relax screening. No merge, Release or tag is performed.
