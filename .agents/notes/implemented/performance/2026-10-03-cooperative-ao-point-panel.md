# Decision: reuse inactive Becke shared storage for parallel AO pullbacks

Status: implemented; bounded scalar fallback retained
Date: 2026-10-03

## Problem

PR #1716 fixes strict PBE0 convergence and improves complete force endpoints,
but the 96-atom default remains 7.62 times slower than GPU4PySCF. Its cooperative
geometry kernel distributes Becke work while thread zero still computes every
AO pullback and moving-grid contribution. Increasing tile sizes alone does not
remove that serial work, and an unconditional 1024-point tile exceeds the
unchanged additional-consumer budget for the largest case.

## Decision

Factor the existing XC point setup and per-AO pullback into shared helpers.
The scalar path preserves their original sequence. In the cooperative kernel,
borrow the already admitted dynamic Becke pair-state storage before that state
is live. Store one XC point value and a three-component gradient per active AO.
Distribute independent AO evaluations over the existing block, then give each
atom one writer and let thread zero accumulate point motion. Both reductions
visit AOs in their original order, including noncontiguous atom ownership and
arbitrary active-AO maps. No floating-point atomics or alternative XC/AD algebra
are introduced.

A block-wide barrier ends every AO panel read before Becke overwrites the same
bytes. The required panel must fit the actual retained-pair or four-row-strip
capacity; otherwise use the scalar AO helper and the existing cooperative Becke
schedule. Point concurrency, thread count, fixed control size, shared-memory
admission, global scratch, tile size and source channel layout do not change.

Each spin/AO pullback is still evaluated once per grid point. The candidate adds
bounded shared writes/reads and atom-to-AO metadata scans, not additional source
evaluations. The metadata scan is a possible performance cost, so this proposal
requires a complete-endpoint improvement rather than assuming parallelism wins.

## Historical isolated qualification

The host-thread harness executes the emitted kernel with real generated Becke
helpers, 12/33/96/128 atoms, arbitrary reversed active maps, noncontiguous AO atom
ownership, nonbinary signed inputs, external seeds, tails and sticky failures.
The 12-atom/1024-AO case exercises insufficient shared storage; 96/768 and
128/1024 exercise the larger panel. Instrumentation checks exactly two spin
pullbacks per active AO and point. AO and moving-grid channels are bitwise equal
to the scalar path under the host harness's explicitly noncontracted arithmetic.
All storage guards remain intact. The 17 focused host/ABI checks pass.

Node1 Slurm job 5408 additionally passes 14 actual PBE0/r2SCAN RKS/UKS geometry
schedule tests through 128 atoms, including constrained resources, changed
geometry and tails. Its 24-atom public PBE0 diagnostic has two clean warm calls
of 5.84177 and 5.86172 seconds, versus the earlier default campaign's 6.64176
second median. These are preliminary different-run observations, not a new
matched full-protocol speedup claim. All four diagnostic calls, including cold
and the separately instrumented replay, pass the retained independent oracle:
energy error 4.55e-12 Eh and maximum force error 1.15e-11 Eh/Bohr. The diagnostic
does not exercise the full moved-geometry publication protocol.

The frozen candidate compiler file SHA-256 is
`1ed3b32d253428ea022e774bffa872666849d98a73266584c0785163767bfc7a`.
Node1 job 5412 completes the 96-atom diagnostic: cold takes 619.747 s with
31 SCF iterations, two clean warm calls take 98.5927/98.5831 s with one iteration,
and a separately instrumented replay takes 99.1123 s. Every call passes the
independent gates, with maximum energy/force errors 1.033e-10 Eh and
2.418e-11 Eh/Bohr. These are not matched cold or full moved-protocol timings;
the large endpoint is still about seven times slower than the reference.

Job 5411 passes the two 96/128-atom geometry probes under memcheck, initcheck,
synccheck and racecheck: zero errors, and zero race hazards/warnings. It finishes
successfully in 20m19s within the original 25-minute allocation. At this stage
full cold/warm/moved qualification had not run. The recorded native library remains the
compensated #1716 binary; this candidate changes the separately generated
stationary consumer. Its emitted source and compiler hash must be retained in
addition to the loaded library identity.

## Profiling and composed qualification

The unchanged default is independently profiled on node3 in Slurm job 12097,
using the exact compensated library and a CUDA-profiler range around only the
fourth already-warm public execute. Two clean warm calls take 107.880/107.951 s;
the instrumented replay takes 112.804 s and is not a clean timing sample.
All four calls pass the same independent energy/force gates. Nsight's kernel
totals identify 29.238 s in the bounded shell derivative, 28.425 s in cooperative
geometry, 17.294 s in density-product GEMMs, and 9.517 s in the grid feature
kernel. These diagnostic kernel totals are not added to host wall phases.

Thus the AO panel cannot by itself resolve the large-system gap. The resident
RKS upload explicitly creates identical spin densities, but the grid consumer
still executes both identical density-product GEMMs. The separate feature
kernel also remained serial across AOs. Those observations motivated the
separately documented spin-product reuse and cooperative feature schedule.
The original profiler journals and CSVs remain under the earlier
worktree's `.artifacts/pbe0-large-20261002/remaining96*` paths.

The composed implementation now completes all 72 native plus 72 independent
reference endpoints on master `74c89369c`, with the unchanged 256-point public
tiles and numerical/resource policy. Native 96-atom original/moved warm medians
are 81.308650/81.542779 s; it is still about eight times slower than the new
reference. The complete [campaign and raw evidence](../../../../benchmarks/results/pbe0-grid-reuse-20261003/README.md)
distinguish this final build from all earlier isolated diagnostics.

Final-master job 12101 passes native J/K derivative/fallback tests and 14
independent PBE0/r2SCAN RKS/UKS geometry tests through 128 atoms; job 12102
passes all 138 grid device tests. The final eight-module host run passes 672
tests. A separately frozen master52 profile reduces geometry-kernel time from
28.424543 to 18.814593 s with the same 9,216 launches. This is instrumented
attribution for the composed changes, not a stand-alone AO-panel wall speedup.

## Rejected shortcuts

- A new global or uncharged shared panel would change the resource contract.
- Atomics or unordered tree reductions would unnecessarily change AO summation.
- Assigning an atom to a thread while evaluating all AO mathematics inside the
  atom-selection loop serializes lanes through divergent atom predicates.
- Merely increasing the public tile ignores the 96-atom budget cliff and does
  not fix serial AO work.

Local diagnostic artifacts are in `.artifacts/pbe0-ao-cooperative/`; the earlier
default campaign remains in `benchmarks/results/pbe0-def2-svp-20261003/`.
