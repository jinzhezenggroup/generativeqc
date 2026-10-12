# Array API conformance gate

The experimental public namespace `generativeqc.experimental.array_api` targets
Array API **2025.12**, with CPU eager execution, all thirteen standard dtypes and
both optional `linalg` and `fft` extensions. This is not a claim that the bounded
TensorIR compiler or native chemistry backends implement the entire standard.
See the [user contract](../user/experimental_array_api.md) and
[dispatch/ownership boundary](../developer/array_api_frontend.md).

## Reproduce qualification

`manifests/array_api_conformance.json` owns the suite/spec revisions and settings.
Use a clean upstream checkout with its specification submodule initialized:

```bash
git clone https://github.com/data-apis/array-api-tests.git /tmp/array-api-tests
git -C /tmp/array-api-tests checkout 6c0b59f9ecd654f0c356614f2e421c9993f676d2
git -C /tmp/array-api-tests submodule update --init
python -m pip install 'generativeqc[test,array-api-test]'
python tools/check_array_api_conformance.py /tmp/array-api-tests
PYTHONPATH=python python -m pytest -q tests/python/test_array_api*.py \
  tests/python/test_experimental_array_api.py
```

For an uninstalled source checkout, install NumPy >=2.1.3, array-api-compat >=1.15.0,
typing-extensions >=4.0
and the dependencies in the `test`/`array-api-test` extras without building the
chemistry library. Native compiler tests additionally require a C++ compiler
and sccache/ccache. The runner supplies the source `PYTHONPATH`. No GPU is used
by the eager gate or default local tests.

The runner verifies both source revisions and clean tracked files, selects
2025.12, runs 100 deterministic Hypothesis examples per property, removes
environment-based dtype/test exclusions, and supplies empty skip/xfail files.
There are no library-specific skips or expected failures; upstream tests may
skip conditions they cannot exercise, and `-rs` reports their reasons. No
extension, dtype or data-dependent-shape category is disabled. CI tests the
minimum NumPy release on Python 3.11 and current NumPy on Python 3.13.

## Acceptance boundary

An eager change must pass the complete upstream gate and local scientific
boundary regressions. Independent upstream value, shape, dtype, object protocol,
IEEE special-case and signature tests are the acceptance reference. NumPy and
array-api-compat are production eager dependencies, not conformance oracles.

Changes must also preserve exact scientific constants, finite compiled-domain
checks, TensorSpec identities, autodiff admission, and no implicit foreign-device
transfers. Eager support for integers, complex values, dynamic shapes or new
functions cannot authorize unsupported TensorIR/native lowering. Unsupported
capture raises instead of executing eager work during tracing.

## Compiler and autodiff gate

`tests/python/test_array_api_native_autodiff.py` qualifies native CPU forward,
combined energy/gradient, JVP and snapshot VJP. It disables reference execution
and AD entry points and checks analytic nonlinear derivatives, independently
contracted RHF Fock matrices, finite differences, cache reuse, domain separation
and fail-before-toolchain resource gates. This is separate from eager conformance.

CUDA planner/source and missing-toolchain tests need no GPU. Real-device tests
require `GENERATIVEQC_ARRAY_API_CUDA_TEST=1` inside a finite Slurm job on an
allowed node, preserving assigned device visibility. They are separate from
the CPU qualification claim.
