# Decision: cross-functional force active-AO policy

Status: guarded automatic selector implemented; no positive production profile
Date: 2026-10-04
Owners: #1853 and #1598

Ordinary stationary CUDA force and composite RSH/nonlocal force now enter one
method-name-free active-AO policy. The selector keys on architecture, derivative
order, spin blocks, execution composition, Hamiltonian/provider state,
atom/AO/grid workload, tile policy, resident-grid capability and admitted
host/device budgets. Both consumers reuse the existing ResidentAoMapCache; no
second producer or scientific kernel is introduced.

The automatic policy is the production entry point, but the qualified profile
registry is intentionally empty. Retained WB97M-V evidence shows only about
1.6% complete 48-atom warm improvement and remains about 0.65% slower than its
matched reference; 24-atom timing ranges overlap. That is below the repository
5% complete-endpoint promotion threshold. #1833 also lacks a current-source
complete GPU campaign. Therefore auto resolves to dense today.

A future positive profile must carry exact source/device evidence and bound the
structural workload/resource domain. Cold, repeated warm, moved and moved-warm
energy/force endpoints need independent numerical gates and actual selected
versus dense AO-square counters. Every profile miss, unsupported derivative
order, missing resident grid or unsupported Hamiltonian remains dense.
Overlapping profiles are an error.

Agent: ChatGPT
Model: GPT-5.6 Sol
