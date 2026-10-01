# Compiler-owned CPU value ERIs: matched RHF endpoints

Date: 2026-10-01. Single-threaded FP64 CPU, exact four-center in-core RHF,
energy-only. No GPU, density fitting, converged-density reuse, or runtime
scientific compilation. Shared virtualized Intel Xeon Platinum 8573C host;
these small molecular cases do not establish general scaling or GPU performance.

## Complete endpoint results

Seconds, medians. Cold and changed geometry have three independent processes;
warm has six fresh-calculator/core-guess calls across those processes.

| Case | Phase | Before | Candidate | PySCF in-core | Before / candidate |
|---|---|---:|---:|---:|---:|
| H2/STO-3G | cold | 0.032598 | 0.027958 | 0.005759 | 1.17 |
| H2/STO-3G | warm | 0.002776 | 0.001739 | 0.003545 | 1.60 |
| H2/STO-3G | changed | 0.001718 | 0.001683 | 0.005699 | 1.02 |
| Water/STO-3G | cold | 0.127224 | 0.028783 | 0.012990 | 4.42 |
| Water/STO-3G | warm | 0.066887 | 0.006158 | 0.012513 | 10.86 |
| Water/STO-3G | changed | 0.060875 | 0.007046 | 0.015410 | 8.64 |
| Water/def2-SVP | cold | 1.818436 | 0.146575 | 0.032961 | 12.41 |
| Water/def2-SVP | warm | 1.660519 | 0.094615 | 0.049404 | 17.55 |
| Water/def2-SVP | changed | 1.696943 | 0.092298 | 0.036368 | 18.39 |
| Formaldehyde/def2-SVP | cold | 13.616591 | 0.925144 | 0.092999 | 14.72 |
| Formaldehyde/def2-SVP | warm | 15.123341 | 0.930459 | 0.106732 | 16.25 |
| Formaldehyde/def2-SVP | changed | 14.572484 | 0.892696 | 0.079241 | 16.32 |

All samples and outliers are retained. Warm before/candidate ranges are:

- H2: 1.617–4.584 / 1.372–4.746 ms; overlapping ranges preclude a robust gain claim
- Water/STO-3G: 56.0–92.5 / 5.23–9.55 ms
- Water/def2-SVP: 1.231–5.046 / 0.0810–0.1848 s
- Formaldehyde/def2-SVP: 13.826–25.879 / 0.7809–1.0370 s

Host noise is substantial; no confidence interval or universal speedup is
claimed. Candidate remains 1.9x and 8.7x slower than PySCF on the two larger warm
def2-SVP cases. This is an intermediate optimization, not performance parity.

## Scientific contract and numerical gates

All 144 endpoints converged. Maximum all-repeat errors are **1.4211e-13 Ha**
against the old native implementation and **4.1212e-13 Ha** against independent
PySCF/libcint. Native SCF iteration counts are unchanged in every phase: H2 2,
water/STO-3G 8, water/def2-SVP 13, formaldehyde 14. PySCF reports 1/8/13/15 plus
its native extra post-SCF check; its actual J/K calls are retained.

- Identical neutral singlet geometry in Bohr and spherical AOs
- Exact common exponent/coefficient decimals from the bundled BSE basis pack;
  `cases.json` freezes the independent PySCF input
- Core-Hamiltonian guess each call, DIIS space/history 8, max 100 SCF iterations
- Both require |delta E| < 1e-10 Ha and RMS(delta D) < 1e-8
- Screening frontend setting 1e-13; neither equal screening work nor identical
  implementation schedules are claimed
- Every numerical-library pool is recorded as one thread; PySCF 2.14.0
- Three independent processes, engine order reversed in the middle round;
  four endpoints each: cold, warm, warm, changed geometry (last atom x +0.02 Bohr)
- Timer includes Calculator/Molecule setup, integral preparation, SCF and result
  collection. Backend Python import is separately retained. Process-cold does
  not mean disk-cache-cold. No benchmark processes overlap, and compilation and
  tests completed before the timed campaign

Both native and PySCF auto routes materialize ERIs. Do not call this a matched
integral-direct comparison. The unchanged exact native quartet census is in
`work-counts.json`; the change lowers arithmetic and temporary allocation rather
than changing requested integral or SCF work.

## Compiler boundary and qualification

The existing value-only shell-class DAG and scalar factorization emit 313
center/axis symmetry representatives covering all 10,000 ordered s/p/d
Cartesian component tuples. HF and DFT share the same native integral provider.
Derivatives, f+ values and range-separated values retain their prior explicit
paths; the independent RawSource recurrence is unchanged.

- 60,000 actual compiled primitive values against independent libcint, six geometries
- Independent review: another 60,000 passing stress values and 1,998 Boys values
  against 100-digit arithmetic, including the T=20 boundary
- Million-Bohr translations retain an existing conditioning limitation:
  candidate 1.23e-9 vs baseline 1.33e-9 maximum error; this negative row is explicit
- 57/57 native tests pass after the rebase, including contracted Cartesian and
  spherical values, derivative fallback values, and an f-shell value fallback
- 14 public HF/hybrid-DFT tests pass; six CUDA-only cases skip without hardware
- Additional compiler regression suite: 59 passed; source-identity suite: 13 passed
- Shared HF/PBE/PBE0 energy checks in two bases differ by at most 9.95e-14 Ha
  from baseline, with identical SCF counts; no DFT speedup is claimed here

Generated header size is 3,793,074 bytes; exact SHA is in the identity file.
Native text grows by 1,562,183 bytes (17,947,458 to 19,509,641); the debug-bearing
shared library grows from 286,502,032 to 302,708,712 bytes. A forced native rebuild
of the changed integral translation unit plus relinking took 290.8 s with maximum
child RSS 4,046,072 KiB on this host. This is an offline build cost, not hidden
endpoint preparation. No ISA-specific flag, fast-math flag, or precision change
was introduced.

## Identity and reproduction

Measured candidate: `bc94d5a02b75754fc9952edb1bcc306fad67db68`, based on merged
master `b8dd212` (#1650). Baseline is the original comparison's master `6ccecb1`,
using the full-tree-identical `07fc955` binary. #1650 changes compiler import
ownership only: all 115 pre-existing scientific generated files remain
byte-identical; only build identity changes. Both builds use GCC 14.2.0,
RelWithDebInfo (`-O2 -g -DNDEBUG`), CUDA OFF, the same OpenBLAS provider, and
unchanged AOT defaults. The native candidate source inventory was recomputed
and matched to the loaded library before measurement.

`observations.json.gz` retains every endpoint and runtime metadata.
`summary.json` retains per-phase medians/ranges, all raw timing samples, energies,
iterations, and all-repeat errors. `measured-inputs.json.gz` retains the exact
original script, basis payload and invocation driver, with their SHA256 hashes.
The readable `benchmark.py` is a formatting, typing, bound-callback and portable
path adaptation of that exact measured script; scientific settings and timers
are unchanged. `run.py` accepts explicit worktree/library paths:

```sh
python run.py --baseline-source /path/to/base --baseline-library /path/to/base/libgenerativeqc.so \
  --candidate-source /path/to/candidate --candidate-library /path/to/candidate/libgenerativeqc.so \
  --output attempts/reproduction
```

No binaries, generated scientific headers, full build logs, Release assets or
release tags are committed.
