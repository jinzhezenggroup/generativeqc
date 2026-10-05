# Proposal: post-order-five route toward a ten-second exact-direct endpoint

Status: proposed; component budgets are targets, not performance forecasts
Date: 2026-10-05

## Measured basis

Slurm5905 qualifies order-five over the order-four composed baseline, retaining
#1830/#1833/#1847/indexed policy. The complete 96-atom E+F warm and moved-warm
medians are 26.989124 and 26.014840 s, with all 288 independent repeat pairings
passing. The matched reference is 13.629942 and 10.086490 s. Actual iterations
remain in every record. These are not unmodified-master timings.

The clean warm wall components are 7.632423 s stationary derivatives and
8.795291 s grid response, leaving about 10.56 s for the remaining endpoint.
The first milestone is a measured complete endpoint below 20 s; ten seconds
requires large improvements across all three groups.

| Complete work group | Current seconds | Ten-second engineering budget |
| --- | ---: | ---: |
| Stationary derivatives | 7.63 | 3.5 |
| Grid/AO/partition response | 8.80 | 2.5 |
| SCF J/K, XC and remaining endpoint | 10.56 | 4.0 |

Source-matched Slurm5917 provides an intrusive warm trace, not another clean
speedup claim. Its kernel durations identify two-electron forces (6.55 s), J/K
values (4.43 s), Becke (4.04 s), AO geometry forces (2.67 s), XC kernels (1.94 s),
AO jets (1.12 s), one-electron forces (1.08 s), and matrix products (0.79 s).
These name-based device sums are not exclusive endpoint wall components.
The generated J and K streaming regions separately take 2.41 and 1.97 s.

## Prioritized experiments and stop conditions

1. Finish shipping qualified order-four/five source contraction in #1952.
   Next test grouping within the *existing admitted force page*. It targets
   different weighted programs serialized within scalar warps, retains one
   quartet enumeration, and adds no resident queue. The isolated candidate
   `48a288b7f` is based on the measured order-five control. Sorting overhead and
   changed reduction order require through-f, sanitizer, full E/F and trajectory
   gates. Keep it only for a complete endpoint gain. Do not repeat the rejected
   thirteen-pass angular schedule or compact-page experiment.
2. Prioritize #1892's actual J/K algorithms. Independent entry points, density
   screening and source preparation already exist. Investigate J-specific
   density precontraction into compiler-owned shell-pair intermediates, with
   explicit component masks that preserve admitted contributions. For K,
   inspect density-conditioned task ordering and the admitted domain before
   adding any screen. Shared value-source production feeding two independent
   contractions is another candidate where J/K admitted domains overlap; it
   must retain each consumer's predicates and outputs. Require recurrence/task
   counts and bounded resource/lifetime fallbacks. A new plan type alone is not
   progress against the measured 4.43 s.
3. Advance #1893 at the sparse consumer boundary. Current local maps already
   cut 96-atom SCF point/AO-square work from 1.392e12 to 7.567e10 and force work
   to 8.106e10. Do not call dense work unoptimized or count those delivered gains
   again. Smaller reusable local blocks, point-wise work inside union maps and
   source/consumer fusion need work ledgers, packing/launch costs, unchanged
   scientific thresholds and full endpoint qualification. #1951 is backend
   infrastructure with no established full PBE0 speedup.
4. Advance #1894's canonical normalized-product derivative primitive. #1950's
   logarithm scheduling targets a 1.50 s kernel inside the trace, and its
   8.6%/12.3% isolated result is not a whole-endpoint improvement. The scaled
   product alternative saved only 0.72 s at 96 atoms; these approaches overlap
   and their percentages cannot be added. Further work must remove substantial
   pair traffic or repeated consumers while retaining canonical AD, zero-factor
   semantics and ordered response. Grid ownership and moving-grid response stay
   inside the timed endpoint.

Do not spend the next full CUDA build on naive joint J/K derivative CSE: a
planning-only census saved roughly 5–8% arithmetic but increased live values.
Likewise, remember that transposing existing scalar AD reduced static work only
about 15–25%, and a two-zero witness omitted at most about 1.55% of actual
96-atom pair work. Neither justifies a claimed path to a multi-fold endpoint gain.

## Secondary positive candidates

Slurm5915's ordered-AO gather passes all 288 E/F pairings on the older composed
baseline. The affected 96-atom geometry component improves about 0.69 s, and
stable moved-warm improves about 0.91 s overall. Control warm samples fluctuate
from 33.36 to 38.55 s, so the observed 4.03 s median gap must not be attributed
entirely to this change. Cold slightly regresses and SCF counts differ.
Compose and requalify it with order five before shipping or promoting a default.

## Qualification contract

Every milestone uses frozen source/library identities, finite Slurm allocations,
48/96 cold/five-warm/moved/five-moved-warm calls, an independent full-density
exact-direct GPU4PySCF reference, complete moving-grid forces, physical residuals
and actual SCF histories. Preserve all outliers, failures and semantic work.
Never normalize by iterations, add gains from separate baselines, silently use
DF/COSX, relax science, or call a provider binding a measured performance win.
Automatic defaults remain a separate #1834 decision; #1895 stays open.

## References

#1895, #1892, #1893, #1894, #1834, #1950, #1951, #1952.
`benchmarks/results/pbe0-weighted-order5-20261005/` retains the incremental
endpoint and source-matched profile. Separate local candidate evidence remains
under the corresponding `/data/jzzeng/qc-189*-20261005/.artifacts/` checkouts.
