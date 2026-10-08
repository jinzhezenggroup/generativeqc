# Maintainer Guide

Maintain scientific correctness, reproducible performance, resource discipline, and repository health. Use the core workflows for recurring work; subsystem qualification documents define specific acceptance contracts.

Keep the distinction clear: the current rules and qualification procedures live here; raw measurements belong under `benchmarks/results/`, and historical investigation or discarded designs belong under `.agents/notes/`.

## Core workflows

```{toctree}
:maxdepth: 1
:caption: Core workflows

validation
performance_engineering
evidence_retention
resource_planning
roadmap
generated-files
```

## Scientific qualification

```{toctree}
:maxdepth: 1
:caption: Scientific qualification

df_frozen_precision
dft_mp_v1_contract
f_shell_validation
hybrid_cuda_acceptance
ccsdt_cpu_bundle_qualification
pbe0_xc_tile_qualification
stationary_large_domain_qualification
```

## Runtime and work evidence

```{toctree}
:maxdepth: 1
:caption: Runtime and work evidence

cuda_ownership
vendor_boundaries
cpu_autotuning
source_work_audit
replay_allocation_receipts
native_structured_materialization
residency_receipts
producer_work_receipts
```
