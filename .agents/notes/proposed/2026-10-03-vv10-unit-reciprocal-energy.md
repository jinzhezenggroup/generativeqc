# Proposal: use a unit-numerator reciprocal for bounded VV10 energy

Status: experimental; complete endpoint comparison pending
Date: 2026-10-03

The current one-denominator closure computes phi=-1.5/D, D=gi*gj*(gi+gj).
An explicit optional compiler policy instead computes phi=-1.5*(1/D), then
reuses phi in the same derivative factors. All operations remain FP64, without
fast math, reduced-precision buffers or approximate intrinsics in source.
This changes energy rounding as well as the derived partials, so it has a
separate lowering identity and cannot inherit claims that the energy IR root
is unchanged. Only the existing bounded VV10 feature dispatch selects it;
the ordered exceptional-domain closure, CPU and rVV10 remain unchanged.

A standalone sm_120 compilation with verified ccache and --fmad=false shows
152 SASS instructions for a unit reciprocal and multiply, versus 184 for
constant-numerator division (including compiler slow paths and NOPs). This is
static selection evidence only, not a measured speedup. The previous complete
profile shows VV10 feature and force pair kernels occupying 10.608 and 12.998 s
of a 47.301-s intrusive 24-atom warm endpoint, motivating a targeted test.

105 host tests pass, including independent 90-digit scalar energy and all
consumed energy partials, exact out-of-domain fallback, original policy root
identity, distinct reciprocal-energy identity and signed-weight semantics.
The candidate library is
`f10568b65cdc6df1fcc369733fccae3fe5f4962c5bb6a95704dfdb225cb02d9a`, parent
`a4c705fd5` plus archived full/LR, staging and unit-reciprocal patches. It includes
master `74c89369c` and its optional SPD-allocation fallback correction.
Six independent complete WB97M-V tests pass on node1 before a same-allocation
comparison of `vvcombo` and `vvunit`, full24 with three warm repeats. Those
complete results, larger and changed-geometry gates remain pending.

Both compared binaries include partner staging, already rejected as a default
after 49.364 -> 49.391 s warm medians in a separate controlled comparison.
Holding it identical isolates this energy policy; a promotion candidate must
remove that unhelpful staging dependency and requalify the actual final stack.
The baseline `vvcombo` also contains the full/LR derivative proposal #1736.
Do not attribute that proposal's benefit to this scalar change.

Source archive, patches, native binary hash, ccache receipts, static compiler
probe and test/results files live in ignored
`.artifacts/wb97m-vv10-unit-reciprocal/` in the composed worktree. Promotion still
requires every complete sample to pass 1e-8 Eh and 1e-7 Eh/Bohr gates, with honest
cold/warm work counts. Molecular active-pair counts remain unavailable.

## Complete matched evidence

Node1 job 5448 ran both binaries sequentially on one scheduled RTX5090 with
8 CPUs. Full24 warm medians were 46.375876 -> 45.339046 s (1.0229x);
cold 323.056338 -> 311.721160 s, both 18 iterations. All warm repeats used
one iteration. Every one of five candidate pairs passes, max energy
2.785328e-11 Eh and force 4.163248e-10 Eh/Bohr. Reference warm median is
27.258260 s: this does not establish reference superiority.

Node1 job 5451 passes the rebuild/stale-state test and the moved12 independent
CPU-oracle gate. Cold/priming/warm/moved times are
166.592983 / 23.188834 / 22.537709 / 91.752514 s, with 22/1/1/11 iterations.
All samples pass, max energy 1.3074e-12 Eh and force 8.9654e-10 Eh/Bohr.
Node2 job 2113 is qualifying default48 separately.
