# Frozen DF factor precision qualification

Issue [#1840](https://github.com/jinzhezenggroup/generativeqc/issues/1840)
qualifies the saved ethane230 frame, with 230 spherical orbital AOs, 488
auxiliaries and 9 occupied orbitals, using aug-cc-pVTZ / aug-cc-pVTZ-RI.
The acceptance rule is unchanged:

```text
abs(native - reference) <= 3e-10 + 3e-10 * abs(reference)
```

All seven original quantities (`Bov`, `Bvv`, `ovov`, `ovvo`, `oovv`, `ovoo`,
`oooo`) must pass elementwise. `Boo` is an additional check. The replay also
captures and tests actual GPU `Bov`/`Bvv` before the existing physical pair
projection; projection cannot hide a failure. Published blocks use that
projection on both sides. No RHF/CC solve or energy-agreement substitution is
part of this qualification.

## Inputs and independent reference

Obtain the retained `pr1781-ethane230-frozen-309669bb-n2.zip` through an authorized
transfer. It is 1,353,823,563 bytes, with SHA-256
`8415c7dd8e81caf7702f423785aff8e925f24b4d1010992226ffdfe8c7305ae2`.
No public download URL is supplied. Verify the archive and extract a pristine
copy; the tools then audit every contained manifest entry. Keep all large arrays
outside Git. A diagnostic log changed in an old extraction is a failed audit,
not permission to ignore that entry.

The original reference integrals and FP64 transformation contained amplified
errors. Preserve that reference and its failure record. The correction uses
PySCF 2.14.0 / libcint 6.1.3 with a separately compiled root interposer, plus
independent analytic Gaussian center derivatives for total angular degree at
most two. The root interposer uses quad moment recurrences with FP64 final
eigensolver/output, checked against 90-digit incomplete-gamma moments. The
analytic helper uses host extended precision and has independent 90-digit
tests. These are validation-only dependencies, never production fallbacks.

The recipe preserves the exact saved C, basis/order, the two original one-ULP
coordinate conversion differences, relative cutoff `1e-10` and rank 488.
It performs MO-first extended accumulation with FP64 final whitening, without
pair symmetrization. The original saved reference must not be called a passing
reference after this correction. Inspect `reference.json` for hashes, moment
errors, root-call counts and rank/cutoff.

## Reproduction

From a checkout with its CUDA library built, set `PYTHONPATH=python:.` and make
`ccache` and the host C/C++ compilers available. Install the pinned validation
dependencies, including NumPy, PySCF and mpmath. Each output directory must be
new. Generate the reference under a finite CPU allocation:

```bash
srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=4 --time=00:10:00 \
  env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m tools.generativeqc_validation.df_frozen_oracle \
  --package /path/to/pristine-package --output /path/to/qualified-reference
```

Run the native source under a finite GPU allocation. On n2 the GPU resource is
`pro6000`; use the resource type belonging to the actual validation host.
Preserve Slurm's `CUDA_VISIBLE_DEVICES`:

```bash
srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 \
  env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m tools.generativeqc_validation.df_frozen_replay \
  --package /path/to/pristine-package \
  --reference /path/to/qualified-reference/reference.npz \
  --output /path/to/native-replay --library "$PWD/build-cuda/libgenerativeqc.so" \
  --cuda-root /path/to/cuda
```

The replay builds validation-only ELF interposers with ccache, retaining the
actual GPU raw high/low rows, root, eigenvalues and factors. It checks exact
single traversal, both component byte counts and the native rank. A failed
gate exits nonzero after writing `candidate-result.json`; raw arrays are in
`candidate_native.npz`. Interposer copies and synchronizations invalidate
performance interpretations of the reported timing. Retain the actual build
configuration, final source and library hashes with the result.

Small gates in `test_df_compensated_precision.py`, `test_df_frozen_oracle.py`,
`test_df_cc_molecular_source.py` and `test_df_source_metric_response.py` protect
independent primitive arithmetic, factor/block layout, capacity boundaries and
fixed-rank response. They do not qualify complete large-system forces.
Benzene264 requires its own original-gate frozen comparison. Numerical review
of the reference correction remains a separate acceptance requirement.
