# CUDA XC capability-based admission

Refs #1850 #1423 #1847.

Reusable CUDA XC fast paths now consume immutable point-program capabilities instead of functional names or numeric family thresholds. This refactor deliberately does **not** promote any new default or numerical domain.

Capability states are `Qualified`, `QualificationRequired`, and `Unavailable`: the middle state distinguishes structurally reusable mechanisms that still lack endpoint/numerical evidence from true implementation/provider gaps.

| Mechanism | Qualified today | Qualification gap | Implementation/provider gap |
| --- | --- | --- | --- |
| component scaling | PBE | LDA | r2SCAN, B3LYP, WB97M-V, generated split hybrids |
| FP32 AO / FP64 storage | LDA, PBE | r2SCAN, B3LYP, WB97M-V, generated split hybrids | response remains strict FP64 |
| FP32 density / FP64 accumulation | LDA, PBE, r2SCAN | B3LYP, WB97M-V, generated split hybrids | fitted Coulomb AUTO remains unsupported |
| semilocal response | LDA, PBE | none | r2SCAN/WB97M-V/generated MGGA response; B3LYP point-response lowering |
| CUDA Graph replay | LDA, PBE | r2SCAN, B3LYP, WB97M-V, generated split hybrids | exact/range exchange, nonlocal and DF provider capture remain separately gated |

The current point evaluator explicitly rejects non-unit scaling for meta-GGA/composed/generated branches. The response program owns only LDA/PBE point response and forbids meta-GGA response, so those stay `Unavailable` rather than being mislabeled as qualification-only work.

Preserved fail-closed boundaries: #1847 local-AO admission is intentionally not duplicated; WB97M-V MolecularV1/VV10 and range-provider checks stay method/provider-specific; AUTO with density-fitted Coulomb remains rejected; host-unfused generated/global-hybrid XC remains unsupported; exact/range exchange and nonlocal graph capture are not promoted.

Curated capabilities are manifest-owned. Generated split hybrids receive one point-program-class capability record independent of method identifier. No performance or default-promotion claim is made by this refactor.

Agent: ChatGPT
Model: GPT-5.6 Sol
