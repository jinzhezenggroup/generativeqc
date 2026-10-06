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


## 2026-10-05: superseded consumer and retained qualification

Merged #1881 already contains #1833's ordinary force-map consumer, bounded
cache ownership, dense fallback and telemetry. Its shared workload resolver
owns public ordinary and composite selection. Do not restore #1833's separate
atom-count-only automatic cutoff while reconciling that older branch. Preserve
the remaining telemetry and prepared-policy mutation regressions independently.

The earlier statement that #1833 lacks a current-source GPU campaign is
historical. The frozen Slurm5757 composition at `d40aeafee0bb1e73fc130a9a253efdbf9beb31e8`
retains 216 independent energy/force pairings in
`benchmarks/results/pbe0-scf-local-ao-20261004/integrated-summary.json`.
It qualifies that source composition, not each PR's isolated gain or the moving
master binary. The sampled-jet cutoff remains heuristic, not a certified
force-error bound.

Merged #1934 later retained complete public-profile 48/96-atom warm improvements
of about 18.4%/30.7%, so the original below-threshold rationale is not a summary
of all later evidence. Its earlier 96-atom cold regression of 8.393% and 48-atom
moved regression of 1.052% remain visible. See
`benchmarks/results/pbe0-public-force-policy-20261005/README.md` for the exact
source/device scope, ordered timing protocol and numerical/work receipts.
These results do not establish profitability for every eligible molecule or
resolve profile interpolation and representativeness. The production profile
registry remains empty; #1853 owns the guarded-default decision under #1598.
No new GPU measurement, performance claim or default promotion accompanies this
regression-test preservation.

Agent: dot


## 2026-10-06: guarded positive production profile

The retained public-profile campaign in merged #1934 supersedes the earlier
empty-registry promotion decision for one structural domain. Production now
registers `sm120-ordinary-rks-second-jet-v1` for all-electron, direct,
ordinary RKS second-jet workloads on `sm_120` with:

- 48–96 atoms;
- 384–768 AOs;
- 1,179,648–2,359,296 grid points;
- fixed 256-point tiles;
- at least 512 MiB device and 256 MiB host force budgets.

The profile keeps the measured `1e-16` sampled-jet cutoff and a 16 MiB
resident-map cache allowance. Matching remains structural; no functional or
molecule name appears in the selector. Every architecture, spin, derivative
order, composition, provider, workload, tile or resource miss remains dense.

This scope deliberately does not extrapolate beyond the retained evidence
envelope. In the current-composition campaign the force-map candidate improved
48/96-atom warm complete E+F by about 18.4%/30.7% and also improved the recorded
cold/moved observations. Earlier narrow-composition cold/moved negatives remain
retained and are the reason the promotion is bounded rather than global.
Supplemental reversed-order cold observations likewise remain evidence, not a
claim of universal profitability.

Agent: ChatGPT
Model: GPT-5.6 Sol
