# Native CPU fixed-density HF physical audit

This bounded #2227 R4/R6 migration moves the final fixed-density physical audit
into a native owner shared by Python and C SDK callers. It reuses PreparedFockPlan,
Fock assembly and the existing mean-field energy and commutator implementations.
The admitted CPU path prepares one Fock, computes physical residual maximum/RMS,
exports overlap and closes its native owners. The benchmark rejects Python
physical-audit compatibility. General progressive scheduling remains Python.

Measured base: `02562ba426c1d6d9458b1f3b7af3bc9f84d907c4`. Apply the decompressed
`candidate.patch.xz` to a separate checkout of that revision. Its SHA-256 is
`58b11f20aa893a4da2a373cd4e7fed35d6cb60fe2ce6d31550ca5933596f1f2e`.
Later publication and Agent Note edits are outside the frozen measured patch.
This independent patch does not include #2814's scalar final-verification API.

## Replay and acceptance

From the repository root, with Python and NumPy:

```bash
python benchmarks/results/native-hf-physical-audit-20261026/replay.py
```

The shared reader authenticates every member. Replay recomputes all 1536
energy/force/density comparisons including warmups, unchanged iteration and
Fock counts, and eight timing decisions. Absolute gates are 1e-8 Ha for energy,
1e-6 Ha/bohr for maximum force error and 1e-7 for density. Maximum errors are
2.7312e-14, 8.2876e-12 and 1.5956e-11, respectively.

All eight candidate/baseline median ratios pass the original 1.05 bound; the
maximum is 0.9906245353813529. Collection covers four RHF/UHF x Direct/DF CPU
cases, two different geometries, 21 samples and three warmups in each block of
both ABBA and BAAB. Every duration, observable and work count is retained.
The 85.84-second pilot estimate preceded the 83.22-second completed campaign.
This is bounded nonregression evidence, without a significant-speedup or
compile-cost promotion claim. The publication decision records independent
numerical acceptance separately.

Each observation includes both geometries' calculator/target planning, source
and target solves, density projection, physical audit and cleanup. PySCF results
come from a separate invocation; byte-identical reference reuse is documented in
`reference-origin.json` inside the frozen capsule. Oracle values never enter
production execution. Production final verification remains `unverified` without
independent observed accuracy supplied to it.

## Reproduce collection

Restore the original scripts without executing them:

```bash
python benchmarks/results/native-hf-physical-audit-20261026/replay.py \
  --extract-frozen .artifacts/native-hf-physical-audit-reproduction
```

The capsule retains the collector, campaign, original audit, source manifest,
pilot observations/estimate and reference origin with original byte hashes.
The patch and reference preserve their exact original bytes in XZ; samples use
token-preserving JSON compaction followed by XZ. No custom scientific decoder
or external archive is required.

Create baseline/candidate checkouts from the pinned base and apply the patch only
to the candidate. Build each in `build/native-ks-catalog` using CPU Release,
Ninja, GNU 11.4.0, CUDA OFF and verified sccache 0.16.0. The local dependency
prefix was `/home/jzzeng/miniconda3`; configure/test/cache receipts are retained
in `summary`. Run the collector's `oracle` mode separately if regenerating the
reference, then `campaign.py --baseline ... --candidate ... --output ...` with
fresh paths. Preserve the protocol; adapt only paths and available CPU affinity.
OMP/OpenBLAS/MKL each use one thread; recorded affinity is cores 46 and 47 on
an AMD EPYC 7K62 host. Initial candidate compilation recorded 369 misses and
zero hits; use of the cache is not a claim of cache hits.

Qualification includes 121 CTests and 81 Python tests plus 25 subtests, Sphinx
warnings-as-errors, ownership/API audits and an SDK physical audit executed with
`PATH=/nonexistent`. Separate independent science tests cover perturbed density
and moved geometry in d-containing Ne+H2, RHF/UHF x Direct/DF, Cartesian and
spherical representations. Initial oracle setup/normalization failures and their
corrections are retained; production equations and acceptance gates were unchanged.

Numeric capacity uses the existing CPU HF envelope plus bounded audit workspace,
with exact/below-limit and transactional publication checks. It is not RSS or
complete provider allocation accounting. Python resource preflight remains an
additional frontend gate. CUDA, ECP, general progressive/force-aware control and
full #2227 closure remain outside this scope. Missing protocol/unsupported domain
retain explicit compatibility; native execution failure never retries Python.
