# Provider-selector boundary guard

Date: 2026-10-06

Agent: ChatGPT
Model: GPT-5.6 Sol

## Decision

Add a structural guard for named provider/implementation selectors rather than
attempting to remove all existing selectors in one mechanical change.

The current compiler has a valid provider-neutral lowering contract, but several
older execution paths still choose an implementation above that boundary.
Examples include Tensor CUDA `reduction_provider`, CC
`df_matrix_gemm`/`lambda_matrix_gemm`/`matrix_gemm`, and RHF
`use_cublas`.

Directly deleting these fields in one patch would simultaneously change source
identity, artifact cache keys, CUDA emission, schedule tuning, CUB qualification,
CC runtime APIs, and SCF prepared-state behavior. That is not an architecture-only
refactor and would require independent endpoint qualification.

## Implemented boundary

`tools/check_provider_selection_boundaries.py` scans production compiler/native
sources and generators for a narrow set of named implementation selectors.
`manifests/maintenance/provider_selection_boundaries.json` records every
current occurrence with an owner contract and migration reason.

The check is exact-counted. New files/selectors, increased debt, and stale
entries all fail. Provider-call ownership remains independently enforced by
`tools/check_vendor_boundaries.py`.

## Follow-up

- #1886/#1889 remove `reduction_provider` from TensorSchedule/search identity and
  bind generated/CUB through the shared lowering candidate/binding mechanism.
- #1890 removes CC GEMM booleans and RHF `use_cublas` from production
  method/policy APIs.
- #934 retains the structural check after those migrations so the architecture
  cannot regress.

No scientific equation, numerical tolerance, provider implementation, default
schedule, or runtime selection is changed by this patch.
