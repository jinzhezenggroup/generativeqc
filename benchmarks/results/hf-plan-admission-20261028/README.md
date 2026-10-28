# Native progressive-HF numerical plan admission

This is a scoped CPU ownership/non-regression receipt for #2227. It does not
qualify a new scientific domain, GPU execution, general adaptive scheduling or
complete runtime migration.

Baseline Python: `a3ec1f8fd8116fc70cf945ab2e7dd97f1e0d42a6`.
Candidate production file hashes and the actual native library SHA-256 are in
`samples.json.xz`. `provenance.json` records the runner/compiler/package versions
and a verified `source.patch.xz` reconstruction against the baseline commit. The
patch contains only changed production sources and build wiring. Both Python trees used that same freshly built CPU library;
the baseline retains its Python planning policy and the candidate queries the
new native policy. Scientific native sources are unchanged by this slice.

The Release build uses GCC 11.4.0, the scalar CPU linear-algebra backend, CUDA
disabled, default CPU force AOT enabled, and verified sccache 0.16.0. Generated
compile commands invoke sccache. Cumulative local cache counters were 313
requests/313 misses after the initial CPU build, and 414 requests/407 misses/7
hits after enabling the default force AOT. These counters are cache provenance,
not compile-speed measurements. Initial AOT-off validation had one expected
missing-force CLI failure; the default-AOT rebuild passed all 126 CTests.

The timed endpoint includes fresh source/target calculators, target snapshot,
plan construction, source SCF, projection, target E+F, independent fixed-density
physical audit and native owner cleanup. Imports are excluded. Two distinct
H2/H2+ geometries alternate; no prepared density is carried between samples.
Each block performs three warmups then 21 samples for RHF/UHF × Direct/DF.
ABBA and BAAB produce 672 timed endpoints. One thread is requested for OpenMP
and OpenBLAS. No isolated-kernel, exclusive-host or speedup claim is made.

All corresponding samples have exactly equal published source/target energies,
full target force and density arrays, physical residual, source/target iteration
and Fock-build counts, projection status, restart origin and final disposition.
All eight complete-endpoint median ratios pass the unchanged 1.05 limit:

| Case | ABBA candidate/baseline | BAAB candidate/baseline |
| --- | ---: | ---: |
| RHF Direct | 0.998716 | 0.999212 |
| RHF DF | 1.002847 | 1.004343 |
| UHF Direct | 1.002069 | 0.998535 |
| UHF DF | 0.991048 | 1.002424 |

Separate qualification: 261 focused Python tests plus 25 subtests, including
independent PySCF 2.14.0 RHF/UHF Direct/DF energy/force checks; 126/126 CTests;
installed public-header SDK query with empty PATH; Sphinx warnings-as-errors;
CI-pinned ty 0.0.82. New executable tests cover exact/below limits, integer/cost
overflow, absent estimates, malformed ABI, atomic output, concurrent decisions,
plan identity, explicit older-library compatibility and native refusal without
Python retry. Numeric admission does not certify final observable accuracy.

Reproduce with two checkouts and a qualified candidate CPU library:

```sh
python benchmarks/results/hf-plan-admission-20261028/benchmark.py \
  --baseline /absolute/baseline-checkout \
  --candidate /absolute/candidate-checkout \
  --library /absolute/candidate-checkout/build/libgenerativeqc.so \
  --output /absolute/ignored-directory/samples.json
```

The harness retains every timed sample and fails if scientific/work signatures
differ or any median ratio exceeds 1.05. The frozen JSON is losslessly XZ-compressed;
read it with `json.loads(lzma.decompress(path.read_bytes()))`. Its file hashes bind
this historical receipt, and are not permanent required-CI source fingerprints.
