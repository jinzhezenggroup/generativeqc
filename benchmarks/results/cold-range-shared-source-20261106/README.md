# Cold strict range-source construction

Issue: #2976. Default promotion qualified on 2026-11-07 after integrating master
`b26a19f3c`. No fresh GPU4PySCF latency ranking is claimed.

## Task and timing boundary

96 atoms, 768 spherical def2-SVP AOs, neutral restricted WB97M-V, full Direct,
Standard FP64, unchanged grids and convergence. Analytic forces include grid
response. Cold measures constructor + prepare + SCF + forces + host export;
warm is one frozen engine-local cold-density call in the same owner. Each arm
starts in a fresh process, uses its scheduler-assigned device and has tracing
disabled. The production driver rejects PySCF and GPU4PySCF imports.

This is one bounded paired cohort per image/device, not a timing distribution or
a broad tuning sweep. The original producer remains selectable with
`GENERATIVEQC_DIRECT_RESIDENT_COMPONENT_SOURCE=1`; leaving it unset selects the
candidate. Both arms retain resident replay. Disabling the LR lease is a
different experiment.

## Source binding

| Cohort | Integration base | Native source | Status |
| --- | --- | --- | --- |
| Initial | `9fbb47085` | `362fb48621a638f873d605a466cde08f7e250b585e48f52df68904e76e3e3e0a` | Complete |
| Final | `b26a19f3c` | `33b9650bd7a90bcc6c6b986512f75e2f39c7ea875860dbcf36f04293210870f2` | Complete |

Final ELF SHA256: `948b333ef7abe5f4f75b427e3cb8a4b702ce0ff135f9d76685dbac87ccaac170`.

The final implementation shares #3026's compiler-owned consumer and uses a
compile-time `ResidentValues` sink. Ordinary Fock kernels do not acquire a
per-component runtime source branch. The initial image used that runtime branch;
its timings are retained as initial evidence, never relabeled as the final image.

## Complete endpoint results

Final master-integrated cohort, seconds:

| Device / allocation | Component cold | Shared cold | Saved | Component warm | Shared warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 / LSF 226235 | 106.309547 | 97.375213 | 8.934334 (8.40%) | 14.906582 | 15.007078 |
| RTX5090 / Slurm 16621 | 184.298838 | 178.414226 | 5.884612 (3.19%) | 13.620037 | 13.619380 |

A100 warm changes by +0.67%; RTX5090 warm is effectively unchanged (-0.005%).
Final paired energy errors are at most 9.1e-13 Ha, and maximum force differences
are below 8e-13 Ha/Bohr. Both devices pass the predeclared cold ratio <0.98 and
warm ratio <=1.03 gates. Leave the shared producer enabled by default, with its
bounded scalar fallback and explicit component-source rollback.

Initial cohort retained separately, seconds:

| Device / allocation | Component cold | Shared cold | Saved | Component warm | Shared warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100 / LSF 226233 | 106.451900 | 97.446155 | 9.005745 (8.46%) | 14.903431 | 14.998046 |
| RTX5090 / Slurm 16620 | 184.496538 | 178.625509 | 5.871029 (3.18%) | 13.614480 | 13.644642 |

Both cold arms perform 16 Focks (11 provisional + 5 target) and four actual SCF
VV10 sweeps. Warm performs one Fock and no new pair sweep. Initial paired energy
differences are zero; maximum force differences are below 1.1e-12 Ha/Bohr.

The selection also makes an already-supported exact uncached range-K consumer
reachable through source-only shell metadata. Endpoint savings therefore include
source preparation and that consumer; they are not isolated producer timings.

## Scientific and resource gates

- Independent Libcint/PySCF signed SR/LR raw K matrices cover RKS/UKS, zero and
  negative densities, screened and unscreened sources, complete/partial residency
  and a stricter consumer threshold. The final cohort adds an f-shell fixture.
- Compact source slots retain their sorted canonical ranks and exact zeros. A
  finite audit rejects uninitialized publications. Stricter consumer-threshold
  censuses fall back to the AO path, preserving all immutable candidate rows.
- Source-only primitive metadata and the temporary quadratic inverse are bounded.
  The inverse fits alongside retained values, is stream-fenced and freed; optional
  allocation refusal retains the original component producer.
- Independent static complete E/F reference values are reused only for numerical
  acceptance, not latency ranking. Original source/ELF identities, all force
  components and semantic work remain in the compressed evidence.
- No GPU-name specialization, precision relaxation, CPU production oracle,
  Direct-to-DF substitution or convergence/grid changes are involved.

## Retained negatives and replay

Slurm 16619 rejected the earlier source
`8cefeb4f193949830d35e383a57446fdffb1caf500604881a4faf50a43830cf9` at the
stricter-threshold census audit. The fix preserves that audit and changes dispatch;
it does not weaken acceptance. Earlier cancelled compilation (143), a post-link
Bash provenance-driver failure (2), and pre-submission launcher failures remain
recorded. No endpoint is claimed from those attempts.

`python benchmarks/results/cold-range-shared-source-20261106/replay.py`
checks the compressed source-bound receipts, complete endpoints, all exported
forces, independent source/reference gates, scheduler identities and work counts
without a GPU. Native source receipts retain integration bases, exact overlay
hashes and reconstruction patches; build receipts authenticate the actual ELF and
restoration of the borrowed compiler workspace.

The final GPU source gate passes three independent fixtures with zero skips;
RTX5090 small-fixture memcheck reports zero errors. A100 separately passes the
96-AO raw-source probe before its endpoints. Host qualification comprises 250
scoped tests and 137 default-inventory tests, plus compiler/SCF ownership and
applicable pre-commit checks. `summary.json` records all paired observables and
independent reference errors; `evidence.json.xz` retains the original receipts.
Oracle progress journals retain their semantic source counters and original
SHA256 hashes, rather than redundant timestamps. Complete endpoint records,
all force components, source patches and failed independent-gate logs are retained.
