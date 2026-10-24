# Decision: acquire the cold determinant inside SCF before reusing its force state

Status: implemented
Date: 2026-10-22

## Problem and diagnosis

Direct cold/warm ratios alone do not explain the comparison with GPU4PySCF.
Native HF has one Fock application on frozen warm replay, while stock GPU4PySCF
has two effective-potential calls. Cold additionally exposes slower repeated
operator work and a redundant native force-finalization route.

The preceding solver-policy binary's 96-atom diagnostic has native HF
22.414/2.923 seconds cold/warm versus GPU4PySCF 12.059/3.397 seconds. Native
completed SCF iterations cost roughly 1.4–1.5 seconds each. Reference late
effective-potential calls cost roughly 0.7–0.9 seconds, plus about 0.016 seconds
for each eigensolve. These are different scope boundaries, not isolated kernel
speed ratios. Native Hückel setup is 0.927 seconds versus reference MINAO 0.187
seconds, so initial guessing alone cannot explain the gap.

PBE0 illustrates a different balance: native 31.681 seconds includes 4.0 seconds
of preparation, 16.309 seconds of native batch execution, and about 11.2 seconds
of force/export ownership (including nested snapshot/decode). Reference cold is
22.934 seconds, with 16.228 seconds in its SCF owner and 6.695 seconds in its
gradient owner. Both have nine cycles. Native warm SCF is cheaper because it has
one Fock, but force/export remains slower. Do not present every Direct warm case
as a native win, add nested regions together, or call these old instrumented
profiles a new clean-master comparison. Raw identities and exclusive ledgers
are retained with the new evidence.

## Decision and invariant

The Hückel seed publishes density but no matching orbital frame. Previously that
fact disabled force-ready reuse for the entire solve, even though each accepted
SCF proposal already copies its orbitals into retained scratch alongside its
density. A cold solve has no finite previous-energy baseline and cannot converge
on its first iteration. Its mandatory first accepted update therefore establishes
the missing frame before any force-ready exit is possible.

Retain the existing force-ready route for internally owned cold seeds. At a later
converged iteration, retain D_n and its unextrapolated F[D_n], and publish the
already retained C_n. Preserve the physical pre-DIIS residual, energy and density
conditions and the complete derivative consumer. No tolerance, approximation,
allocation, mathematical producer or iterative work is changed.

Unproven external seeds, moved/partial warm batches, explicit final-Fock rebuilds,
incremental execution and physical-reference exports keep their existing bounded
fallbacks. A one-iteration cold request remains unconverged and publishes no
forces. Failed eigensolvers/iterations cannot certify an orbital frame.

## Evidence and rejected alternatives

[Source-bound compact records and optional raw replay](../../../../benchmarks/results/direct-cold-frame-20261022/README.md)
retain 72 independently qualified clean E/F calls. At 96 atoms, three interleaved
fresh-process populations per engine reduce native cold median 22.480 → 20.106
seconds (10.56%), with 11 SCF iterations and 13 → 11 target-basis Fock applications.
Both cold variants additionally use seven preliminary Hückel Fock builds;
warm replay uses none. GPU4PySCF uses eleven cold/two warm effective-potential
calls, a distinct counter from native target/preliminary Fock builds.
Warm medians are 2.917 → 2.913 seconds. Same-allocation stock GPU4PySCF remains
11.958/3.414 seconds, so most of the cold gap remains. Independent candidate
force error is at most 6.38e-7 Eh/Bohr across all sizes/replays. Nine GPU tests,
44 host benchmark tests, repository hooks and zero-error memcheck pass. The new
tests check scientific outputs and executed semantic work, not source spelling.

Existing incremental J/K is not a usable default shortcut: on the unchanged
baseline HF becomes 26.322/5.603 seconds and PBE0 42.127/roughly 12.9 seconds.
Incremental execution retains stricter stationarity and extra full-density
confirmation; density-weighted screening limits it to one delta between full
refreshes. Separate HF J/K takes 52.954 seconds cold; adding component Rys K
takes 80.461 seconds, both with eleven cycles. All ablations pass the independent
gates but are slower and remain rejected. Their observations use separate
allocations and are not headline interleaved timing populations.

The compact records retain every timing/error value used for the claims, work
counters, maximum residual diagnostics, source reconstruction and binary/input
identities. Original force arrays and full logs remain in ignored local storage
and on n1, bound by the artifact checksum; optional replay requalifies them.
This fits the existing checkout evidence budget without retiring historical
evidence or changing the storage policy.

The optional full artifact retains the exact `measure.sh`, `test.sh` and
`sanitize.sh` runners. The paired measurement invokes
`python -m benchmarks.readme_defaults standard --method hf --mode direct`
with each frozen library and uses the `gpu4pyscf` arm for the reference. It
passes `--atoms`, `--warm-repeats 3`, the retained input directory, explicit
source revision and a separate output for every process. Run any repetition
through a finite `srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1
--time=00:40:00` allocation, preserving its CUDA visibility, and use new output
paths instead of overwriting the retained receipts.

The initial new test used an unsupported `initial_guess="huckel"` string and
failed before native calls. Correcting it to the public auto policy, with an
assertion on actual Hückel provenance, changes no native source or binary.
Retain the failed log and corrected test identity separately from the source
manifest used for compilation.

## Remaining work

The dominant remaining HF opportunity is cheaper repeated full Fock work; PBE0
also needs preparation and force/export work reduction. Improving incremental
execution requires qualification of its final-state ownership and screening
error policy, not blindly enabling its current switch or removing a check.
Revisit this cold-frame policy if SCF ever permits first-step acceptance without
a trusted energy baseline, or changes how accepted density/orbital generations
are retained. The paired first-step-failure and cold/moved/warm scientific tests
must continue to cover that boundary.
