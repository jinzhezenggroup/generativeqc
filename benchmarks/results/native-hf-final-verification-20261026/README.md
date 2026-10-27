# Native progressive-HF final verification

This bounded #2227 R4/R6 migration moves scalar final acceptance into a native
C ABI shared by the Python controller and SDK callers. Physical fixed-density
Fock/residual assembly and general progressive orchestration remain Python.
The benchmark rejects any use of the Python final-policy compatibility path.

Measured base: `e1e244197411b58da42aa8a2cbce49014d7ed3ff`. Apply the decompressed
`candidate.patch.xz` to a separate copy of that revision. Its SHA-256 is
`be350b5a5cc890ec4d5b02f60bb1fdb0f204f67ae129444b9b6efb51d94028b4`.
Later evidence and Agent Note additions are outside this frozen measured patch.

## Replay and acceptance

From the repository root, with Python and NumPy:

```bash
python benchmarks/results/native-hf-final-verification-20261026/replay.py
```

The shared reader authenticates every published member. Replay recomputes all
1536 energy/force/density comparisons including warmups, unchanged iteration
and Fock counts, and all eight timing decisions. Energy, maximum force and
density absolute gates are 1e-8 Ha, 1e-6 Ha/bohr and 1e-7, respectively.
Maximum errors are 2.7312e-14, 8.2876e-12 and 1.5956e-11.

All eight candidate/baseline median ratios pass the original 1.05 bound; the
maximum is 1.0202265228437473. There are four RHF/UHF x Direct/DF CPU cases, two
genuinely different geometries, 21 samples and three warmups in each block of
both ABBA and BAAB. Every duration, observable and work count is retained.
The 84.5-second pilot budget predicted the 85.37-second completed campaign.
This is a bounded nonregression result, without a significant-speedup or
compile-cost promotion claim. The shared publication decision declares the
independent numerical acceptance separately.

Each timed observation includes both geometries' calculator/target planning,
source and target solves, density projection, physical verification and cleanup.
PySCF runs separately, before collection. Its results never enter production
execution. Production verification correctly remains `unverified` when no
independent observed accuracy is supplied. The controller still establishes
the target and publishes the unchanged physical diagnostics.

## Reproduce collection

Restore the exact original scripts without executing them:

```bash
python benchmarks/results/native-hf-final-verification-20261026/replay.py \
  --extract-frozen .artifacts/native-hf-final-verification-reproduction
```

The frozen capsule contains `endpoint.py`, `campaign.py`, the original audit,
source manifest and pilot cost estimate, with individual original byte hashes.
The reconstruction patch and reference JSON preserve their original bytes in
XZ; the full samples use token-preserving JSON compaction followed by XZ.
No custom scientific decoder or external archive is required.

Create separate baseline/candidate checkouts from the pinned base and apply
the patch only to the candidate. Build each in `build/native-ks-catalog` using
CPU Release, Ninja, GNU 11.4.0, CUDA OFF and a verified sccache launcher; the
recorded local prefix was `/home/jzzeng/miniconda3`. The configure and test/cache
receipts are retained in `summary`. Use the frozen runner's `oracle` mode in a
separate invocation to generate references, then `campaign.py --baseline ...
--candidate ... --output ...` with fresh paths. Preserve the protocol; adapt
only local paths and available CPU affinity. Set OMP/OpenBLAS/MKL thread counts
to one. The recorded affinity was cores 46 and 47 on an AMD EPYC 7K62 host.

The qualification includes 121 CTests, 146 Python tests, four focused native
rechecks after formatting and Sphinx warnings-as-errors. Native metadata tests
run with `PATH=/nonexistent`. Initial setup/formatting failures are documented;
the frozen measured production source follows the corrected checks.

This small CPU fixture does not establish scaling, CUDA coverage, complete
provider/RSS accounting, native physical-audit ownership or full #2227 closure.
No scientific equation or new backend admission changes in this migration.
