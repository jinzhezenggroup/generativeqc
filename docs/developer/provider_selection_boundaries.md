# Provider-selection ownership

Scientific and method owners describe the operation they need. They do not select
cuBLAS, cuBLASLt, cuTENSOR, CUTLASS, CUB, cuSOLVER, cuSPARSE, NCCL, or an
equivalent generated implementation by name.

The target production boundary is:

```text
scientific / method semantics
        |
        v
canonical TensorIR / solver operation
        |
        v
scientifically admitted precision variants
        |
        v
provider-neutral lowering request
        |
        v
provider + layout + fusion + resource candidates
        |
        v
prepared lowering binding
        |
        v
execution
```

Provider names are valid inside provider implementations, diagnostics, and
explicit qualification/benchmark pinning. They are not production policy inputs
to MethodIR owners, CC/HF/DFT/GFN method APIs, or generic schedule search.

## Migration debt guard

`tools/check_provider_selection_boundaries.py` freezes the current named
implementation selectors in
`manifests/maintenance/provider_selection_boundaries.json`.

The remaining guarded debt contains two method/runtime families. Tensor CUDA
cooperative reductions have already moved provider identity below `TensorSchedule`:
the schedule describes cooperative execution shape, while a typed
`ReductionLoweringBinding` selects generated CUDA or CUB for qualification and
artifact binding.

- CC/DF-CCSD(T) still carries `matrix_gemm`, `df_matrix_gemm`, and
  `lambda_matrix_gemm` through method/solver surfaces. This is migration debt
  under #1890.
- RHF/SCF still carries `use_cublas` through policy and prepared execution.
  This is migration debt under #1890.

The guard scans production native sources, the scientific compiler package, and
source generators. A new named selector fails unless it is explicitly reviewed
and classified. Removing or reducing an existing selector also fails until the
manifest is updated, so migration progress cannot leave stale allowlist entries.

This guard complements `check_vendor_boundaries.py`. The vendor-call guard
answers "where may a CUDA library symbol appear?"; this guard answers "where may
an upper layer choose that implementation?". Passing one does not imply passing
the other.

## Completion condition

The migration is complete when production scientific/method APIs carry semantic
requests or prepared bindings instead of implementation booleans/strings, and
provider pinning remains only in explicit diagnostics/qualification code.
