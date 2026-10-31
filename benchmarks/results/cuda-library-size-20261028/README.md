# CUDA library size qualification

This campaign keeps the 80 DF angular classes, all admitted mathematical
lowerings, schedules, precisions and architecture images. It removes already
resolved relocatable ELF inputs from private Direct/GFN2 **host-link copies**,
and avoids instantiating mixed-precision DF launches whose existing production
policy always selects strict arithmetic. Original archives remain intact.

The source base is `568293b7867dd077fc10063951c8ce442ac83172`. The receipts retain
the production overlay, source hashes, full configuration/compiler/link commands,
toolchain/cache records, binary sections and identities, decoded executable-image
hashes, raw endpoint samples, independent references and unsuccessful diagnostics.
Hardware is Slurm node2 PRO6000, SM120, CUDA 12.9.86/GCC 11.4 with verified
sccache 0.16.0. All real-device work used finite `srun` jobs and preserved assigned
device visibility. No release or distribution was published.

## Size and executable preservation

| Artifact | Bytes |
| --- | ---: |
| Baseline complete library | 699,244,888 |
| Same objects, pruned Direct/GFN2 host archives | 586,920,280 |
| Same rest of library, plus 20 corrected DF objects | 537,659,848 |
| Independently rebuilt integrated library | 537,665,248 |

The matched complete library saves **161,585,040 bytes (23.1%)**. This is the
same single-SM120, 73-XC-registration configuration used in the preceding
[XC qualification](../xc-structural-default-20261028/README.md), with optional
stationary-force/Hessian/shell bundles disabled. It is a complete loadable native
library, not every possible release feature or architecture combination.

Across all 80 DF objects, file bytes decrease from **343,371,360 to 293,984,736**;
their device-image bytes decrease from **328,152,800 to 279,079,800**. Within the
20 affected classes, kernel instances decrease **348 to 174**. Every surviving
kernel's machine-code section is byte-identical in the matched comparison.
Direct and GFN2 each retain their original executable cubin byte-for-byte after
pruning. These are one-off measurements, not required source fingerprint tests.

The independent CMake build completed in 1,720.56 seconds, including unrelated
cache misses and overlapping qualification work. The selected 20-object compile
span was 27.03 seconds. Neither is a controlled before/after cold-build comparison;
do not infer a whole-project compilation speedup from these times. Pruning itself
does not reduce device compilation work; the discarded DF instantiations do.

## Runtime and scientific gates

After compilation and other GPU work ended, five randomized paired processes
covered each of Direct FP64, DF FP64 and public DF Fast on 24/96-AO water cases.
Each process includes Calculator/preparation plus first E+F in the cold timer,
three frozen-seed warm E+F calls, and one geometry-moved E+F call. This gives
**300 complete calls across six cohorts**. All samples remain in the receipts.

The predeclared gate is 1.02 for each cold/warm/moved cohort, comparing medians
of process medians. Maximum ratio is **1.0088434653**. Iterations, Fock counts and
reported mixed/retry work match. Maximum paired energy/force errors are
**4.55e-13 Eh / 1.09e-13 Eh/bohr**, within 1e-9/1e-8 gates.

Independent PySCF Direct/DF energies and analytic forces are evaluated in separate
processes, outside production timing. All cold/warm/moved results pass energy
1e-7 Eh and strict/mixed force 1e-6/1e-5 Eh/bohr gates. The largest independent
energy error is 7.05e-12 Eh and force error 7.34e-8 Eh/bohr. These sampled gates
do not establish universal performance or mathematical coverage.

The integrated CMake binary passes two native tests (CUDA runtime; independent
DF derivatives including full/symmetric/packed weights, representations, clipping,
packet tails and strict fallback for a mixed request), 17 HF-DF/Direct/GFN2
endpoint and transaction tests, and eight mixed spdf FCI force tests. The initial
FCI run skipped because `threadpoolctl` was missing; after installing that oracle
dependency, all eight passed. The initial matched-library GFN2 attempt lacked
its build-metadata path; the integrated build passed with its actual metadata.
Both unsuccessful observations are retained.

Six compiled archive contracts pass real-only, PTX-only, combined, multi-SM and
multi-config builds, original-archive retention, incremental updates and OFF.
Device execution also passes forced PTX JIT with cross-unit device calls. Host
tests cover missing-tool/other-backend fallback, invalid closure/tool errors and
monolithic CLI registration/rescan behavior. Host suite receipts are retained.

## Reproduction

No compiler, GPU or network is needed to replay the measurements and gates:

```bash
python benchmarks/results/cuda-library-size-20261028/reduce.py
```

`receipts.json.xz` is ordinary LZMA-compressed JSON. `production_overlay` applies
to the recorded base; `final-artifacts`, `compiler_commands`, link commands and
`recipes` retain the exact build/launch inputs. Toolchain prefixes are local
execution provenance and must be adapted when reproducing on another machine.
The original/native artifacts and generated-source trees remain ignored local
qualification outputs, not checked-in binaries.

Before rebasing, the campaign compacted whitespace in seven historical JSON
records to fit the unchanged 80-MiB evidence budget. The receipts retain those
original before/after identities and value-equivalence checks. Mainline #2842
subsequently recovered enough storage headroom, so the final change leaves all
seven historical records untouched and retains this campaign within the same cap.
