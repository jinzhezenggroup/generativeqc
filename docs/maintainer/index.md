# Maintainer Guide

Use this guide to keep GenerativeQC scientifically trustworthy, reproducible, performant, and maintainable.

- [Validation gates](validation.md)
- [Frozen DF factor precision](df_frozen_precision.md)
- [Performance engineering](performance_engineering.md)
- [PBE0 SCF XC tile qualification](pbe0_xc_tile_qualification.md)
- [Large-domain stationary CUDA qualification](stationary_large_domain_qualification.md)
- [Scientific evidence retention](evidence_retention.md)
- [CUDA ownership](cuda_ownership.md)
- [CUDA vendor boundaries](vendor_boundaries.md)
- [Resource planning](resource_planning.md)
- [CPU autotuning](cpu_autotuning.md)
- [F-shell validation](f_shell_validation.md)
- [Implementation roadmap](roadmap.md)
- [Generated documentation/data](generated-files.md)

Historical investigation belongs in `.agents/notes/`; current operational truth belongs here.

```{toctree}
:hidden:
:maxdepth: 1

validation
df_frozen_precision
performance_engineering
pbe0_xc_tile_qualification
evidence_retention
cuda_ownership
vendor_boundaries
resource_planning
cpu_autotuning
f_shell_validation
ccsdt_cpu_bundle_qualification
dft_mp_v1_contract
hybrid_cuda_acceptance
stationary_large_domain_qualification
source_work_audit
replay_allocation_receipts
native_structured_materialization
residency_receipts
roadmap
generated-files
```
