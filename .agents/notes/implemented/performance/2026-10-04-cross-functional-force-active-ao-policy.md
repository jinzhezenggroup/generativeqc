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
registers `sm120-ordinary-rks-second-jet-v2` for all-electron, Direct,
ordinary RKS second-jet workloads on `sm_120`.

Admission is deliberately **not** an atom/AO/grid benchmark window. The runtime
uses the predicted dense contraction work

```text
grid_points * AO_count^2
```

and admits map discovery once that work reaches the smallest retained strongly
positive endpoint, 173,946,175,488 point·AO² (the 48-atom campaign). This is a
continuous cost crossover: a different molecular size or grid shape can match,
while a nominally large molecule with too little dense work stays dense.

The remaining predicates are genuine execution-capability/resource guards:
ordinary composition, RKS, second AO jet, Direct/all-electron, `sm_120`, fixed
256-point tiles, at least 512 MiB device and 256 MiB host force budgets, and the
current 2048-AO implementation capacity. The profile keeps the measured
`1e-16` sampled-jet cutoff and a 16 MiB resident-map cache allowance.

No functional or molecule name appears in the selector. UKS, first-jet,
composite, DF, ECP, another architecture/tile policy, insufficient resources,
or work below the crossover remains dense. Future evidence should refine this
cost model or qualify additional capabilities rather than adding benchmark-shape
ranges.

In the current-composition campaign the force-map candidate improved 48/96-atom
warm complete E+F by about 18.4%/30.7%. Earlier narrow-composition cold/moved
negatives and supplemental reversed-order cold observations remain retained;
they motivate guarded admission rather than a global unconditional switch.

Agent: ChatGPT
Model: GPT-5.6 Sol
