# Execution and backends

Execution backends, precision, tensor kernels, state transport, hardware scheduling, and extension boundaries. Backend-specific plans must preserve shared scientific semantics.

**Suggested starting path:** [CUDA tensor execution](../tensor_cuda.md) → [Precision](../tensor_precision.md) → [Extensions and backend boundaries](../extensions.md).

## Browse this subsystem

```{toctree}
:maxdepth: 1
:caption: Execution and backends

../tensor_cuda
../cpu_linear_algebra
../cuda_time_estimator
../tensor_precision
../lowering_providers
../state_transport
../opencl_backend
../extensions
../xtb_native_ownership
../extending/index
```
