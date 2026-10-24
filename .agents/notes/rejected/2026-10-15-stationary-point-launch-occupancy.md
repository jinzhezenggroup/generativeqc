# Trial: distribute underfilled stationary point tiles across multiprocessors

Status: rejected for default promotion; source and candidate-only tests reverted
Date: 2026-10-15

## Problem

The stationary phased-force consumer evaluates a scalar XC program on each
bounded grid tile. Its fixed 128-thread producer exposes only two or four blocks
for the common 256/512-point tiles. Imported meta-GGA programs have substantial
instruction latency, so most multiprocessors cannot contribute to this stage.
This is distinct from the AO-discovery, matrix-tile, dddd and order-seven-page
candidates already rejected or retained opt-in by #2486. Do not rerun those
experiments or re-credit #2505/#2511 integer-power lowering or #2464 branch CSE.

## Rejected candidate and boundaries

Keep 128 threads when its block count already reaches the device's SM count;
otherwise use one 32-thread warp per block. Read the SM count from the device
properties already queried at owner construction, retaining at least one as a
defensive fallback. No device query is added to a tile's execution.

Only the existing precomputed stationary FP64 point producer changes launch
geometry. Its phased-storage, atom-count, producer-witness, external-seed and
sticky-error gates remain intact. Every point still executes the same scalar
program exactly once and writes its own scratch slot. Later geometry kernels
and reductions are unchanged. There is no cross-thread point arithmetic to
reassociate, new allocation, larger scientific domain, or method/spin whitelist.
The original nonphased, FP32, resource-constrained and SCF paths are unchanged.

An unconditional 32-thread policy is not justified: the diagnostic MN15/UKS
65,536-point population on RTX PRO 6000 is about 1.19% slower. The adaptive
candidate retains the original 128-thread launch there. Do not subtract this
negative pilot from the campaign or label its point timings as molecular gains.

An independent mandatory-expression-hoisting pilot changes emitted SCF source
and is not combined with this force-launch-only candidate. Its source patch and
complete observations remain in ignored run storage; no SCF arithmetic change
is proposed.

## Candidate validation

- 201 focused host regressions pass, including independent force consumers,
  executable launch-error injection, partial tiles and both sides of the SM
  occupancy boundary. The existing owner-budget test verifies the retained SM
  count. Compiler structure checks 540 modules with zero dependency errors.
- The configured pre-commit checks pass for the four changed source/test files.
- Actual force-point bodies for M06-2X and MN15, RKS and UKS, compare bitwise
  across 256/8192/65536 points on RTX 5090 and RTX PRO 6000 under finite Slurm
  allocations. Alternating-arm event timings are diagnostics, not endpoint gains.
- A cached, compiled point-launch memcheck covers 40 method/spin/count cases,
  partial warps, output guards and exact outputs; Slurm n2/3031 reports zero
  errors. An earlier harness attempt failed loading NVRTC before the point
  launches. Its zero-error sanitizer footer does not make that setup failure a
  pass; the failed log is retained separately from the successful retry.

## Endpoint protocol

The control is master `5f80ba629b5f20412215cdc690f8fa458ac3fdf8`. Both arms
are fresh cached Release builds of the archived source, with the candidate's
source patch retained separately. Native and packaged stationary libraries must
match their own generated identities; no provenance gate is bypassed.

Qualify complete README M06-2X RKS def2-SVP energy plus host-returned moving-grid
forces, both DF96 and direct96, cold + five frozen-own-seed warm + moved + five
moved-warm calls. Retain preparation time, all calls, independent 1e-8 Eh /
1e-7 Eh/Bohr gates and actual SCF/Fock work. Reverse arm order for small gains.
Independent reference densities must never seed production. Unavailable integral
execution counters remain unknown, not zero or inferred equal.

A 48-atom perturbed water-cluster/sto-3g consumer exercises phased geometry for
M06-2X RKS and closed-shell UKS, with PBE/PBE0/r2SCAN RKS controls. Its independent
PySCF SCF and analytic moving-grid references use the production grid but not its
density. Every returned call must pass the existing small-fixture 2e-9 Eh /
2e-7 Eh/Bohr gates. Closed-shell UKS is not a polarized molecular speed claim.

An earlier 36-atom cluster is a nonphased fallback control, not qualification of
the new producer: the existing public auto policy starts phased geometry at 48
atoms. Keep its returned calls and independent gates, without claiming a launch
speedup. Two node1 launches also failed before Python execution because the old
node5-only interpreter wrapper was absent. The corrected campaign uses a local
wrapper, identical for both arms on each node; original failed prefixes/logs
remain separate. Missing scientific work is unknown, not normalized to zero.

Raw source/build receipts, protocols, successful and failed diagnostics and full
JSON reports are under `.artifacts/xc-point-warp-20261015/` locally and
`/data/jzzeng/qc-xc-point-warp-20261015/` on the participating nodes. These
ignored diagnostics are not a promoted benchmark campaign or README update.

## Complete endpoint rejection

All eight README populations finish twelve calls and pass every independent
energy/full-force gate. The complete host-returned E+F warm medians are:

| Node / order | Route | Control warm / moved-warm (s) | Candidate warm / moved-warm (s) |
| --- | --- | --- | --- |
| n5 / control first | DF96 | 29.000441279029474 / 28.97691157902591 | 29.279504052945413 / 29.280535019934177 |
| n1 / candidate first | DF96 | 28.581369 / 28.451770 | 29.125356 / 29.125190 |
| n5 / control first | Direct96 | 16.776370483916253 / 16.817957579973154 | 16.735877 / 16.747872 |
| n1 / candidate first | Direct96 | 16.783567 / 16.777999 | 16.925492 / 16.908658 |

DF96 regresses in both orders; the direct result changes sign. All corresponding
warm calls retain three DF Focks or one direct Fock. Do not normalize away cold
or moved work: the n5 direct cold calls take 18 versus 19 Focks, and moved takes
13 versus 12. Those calls remain in the raw evidence rather than becoming a
same-work speedup claim. The full-precision JSON, not rounded table entries, is
the timing source of truth.

The force components alone improve slightly, while the endpoint remainder
increases. Subtracting the force timer does not isolate an exclusive SCF timer,
so this is not proof of a changed SCF equation or a specific bottleneck. All
SCF/XC/grid object files are byte-identical between builds: 27 of 555 objects
change, confined to stationary force objects/device links and build provenance.
Neither this identity check nor a point-probe win overrules measured E+F losses.

The 48-atom control completes 60 calls; the candidate returns 36, including
all M06 RKS/closed-shell UKS calls, before job 8525 reaches its finite time limit.
The 36-atom fallback control completes 60; its candidate returns 54 before job
8523 reaches its limit. Missing gates on partially referenced rows remain
unknown, not passing. These partial campaigns do not qualify a cross-functional
default. M06 RKS at 48 atoms does not show a reliable endpoint improvement.

A persistent two-process interleaving diagnostic (n5/1593) is stopped manually:
it retains both owners concurrently and changes residency/resource preconditions
relative to the single-owner benchmark. Only its completed control cold call
is known; unfinished candidate work remains unknown. It is not a clean paired
qualification or evidence that the candidate's kernel causes that stall.

## Separate output-order diagnostic

A derivative-before-energy emission-order probe retains all graph nodes,
operand order, lazy branch semantics and the original 128-thread point launch.
It applies generically to coalesced FP64 outputs, not a functional whitelist,
and does not modify production sources. M06 and MN15, both spins, at
256/8192/65536 points compare bitwise in Slurm n2/3038. Compiler register/spill
reports do not improve and alternating event medians show no useful reduction.
Do not start an endpoint campaign or promote source-order churn for this result.
Setup failures from missing remote source/assets/cache dependencies remain
separate from the successfully prepared, verified-sccache probe.

## Consequences and revisit

Retain the existing 128-thread production policy. No PR is justified by this
candidate, and #2505/#2511/#2464 remain the baseline, not newly earned gains.
Revisit only with actual production-kernel/complete-endpoint evidence identifying
a material causal improvement, matched semantic work and unchanged numerical,
resource, failure and fallback contracts. Do not retry a uniform 32-thread policy,
the same SM-count heuristic, or output-root ordering on the strength of the
simplified point benchmark alone.
