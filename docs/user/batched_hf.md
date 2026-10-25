# Batched HF calculations

Use `prepare_batch` to run RHF or UHF on several molecules and reuse the
prepared calculation when their coordinates change. The batch can contain
different molecule sizes; results are returned in input order.

## Example

```python
from generativeqc import Calculator

systems = [
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    [("He", (0, 0, 0))],
]

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
with calc.prepare_batch(systems, warm_start=True) as batch:
    cold = batch.execute(strict=True)
    warm = batch.execute(strict=True)

print(cold.energies)
print(warm.energies)
```

Coordinates use Bohr; energies are in Hartree and forces in Hartree/Bohr.
Use `device="cpu"` for a CPU build. For geometry updates, checkpoints and
batch lifetime, see [checkpoint/restart](checkpoint.md) and
[prepared execution](../developer/prepared_execution.md).

## Execution and failures

- RHF and UHF support mixed-size batches with per-molecule convergence,
  iteration counts and failure status. An invalid or unconverged item does
  not invalidate the others. Use a non-strict execution to inspect
  individual failures.
- A successful run saves a density for later calls on the same prepared
  system. Changed coordinates invalidate geometry-dependent caches but
  can reuse the density as an initial guess.
- If a warm guess fails ordinary convergence checks, the runtime can try
  one fresh-density solve. CUDA out-of-memory and driver failures do not
  trigger this retry. A failed call does not overwrite the last successful
  density.
- CPU batches run compatible systems in worker groups. CUDA batches keep
  the supported s/p/d/f contracted Cartesian and real-spherical HF work
  on the GPU; different shapes are grouped internally.

The backend diagnostic labels are `cpu_reference` and `cuda`.
`hybrid_cuda` remains an older ABI value and is not reported by current runs.

## Solver and fitting diagnostics

CUDA uses a small-matrix eigensolver through 16 AOs, batched Jacobi for
17–32 AOs, and cuSOLVER for larger matrices. When a larger solver cannot
run in a captured CUDA Graph, the calculation uses an ordinary CUDA stream
for that solve; it does not change the HF method.

`PreparedBatch.last_eigensolver_diagnostics()` records the solver chosen for
each bucket. Density-fitted batches also expose
`PreparedBatch.last_density_fitting_metric_diagnostics()`, including metric
rank, condition number and workspace estimates.

For implementation details, see [CUDA Fock execution](../developer/fock_cuda_execution.md)
and [density fitting](../developer/density_fitting.md). Measure complete
cold and warm batch times with `benchmarks/batch_throughput.py` rather than
inferring throughput from an individual kernel.
