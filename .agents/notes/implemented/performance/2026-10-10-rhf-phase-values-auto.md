# Decision: Default exact RHF values to bounded automatic routing

Status: implemented
Date: 2026-10-10

## Problem

The [phase-local source](2026-10-10-rhf-phase-canonical-values.md) removed
repeated ERI evaluation in the qualified ethane cold endpoint, but requiring
an environment opt-in indefinitely would hide that benefit from normal use.
Resource legality alone does not establish payoff for warm references or
already fast generated shell schedules.

## Decision

Unset and explicit `auto` now select a conservative cold/domain heuristic.
It requires a fresh bucket, an f-containing domain with uncovered generic
Fock work, at least 128 Cartesian AOs and an iteration limit of at least
eight. The Cartesian threshold implies at least 256 MiB of canonical
values and keeps smaller low-setup references out of automatic construction.
The iteration limit excludes short requests, not a prediction that SCF
will perform eight iterations. A new supplied guess is not a proof of a
converged reference; known warm/reused buckets are rejected independently.

`0` forces the original schedule. `1` retains the original forced-resident
request, including smaller/warm eligible references, without bypassing
memory or scientific gates. Other values are rejected on eligible calls.
Fully generated or lower-angular schedules do not pay a phase setup merely
because they have sufficient memory. No molecule name or geometry key is
used for automatic dispatch.

Inactive owners use ordinary reusable bucket graphs. Only admitted values
use local graphs; lazy ordinary capture handles a bucket whose earlier
forced/cold execution had only resident graphs. Source lifetime, FP64
compensation, final physical Fock, census, finite audits and failure
propagation remain unchanged.

## Rejected alternatives

- Default every eligible call to the old `1`: memory-fit and a large iteration
  limit do not establish payoff, especially on warm or generated routes.
- Build on every warm call: the source is intentionally not retained across
  executions, so a quickly converged replay would pay a fresh construction.
- Change SCF convergence or skip final audits to predict reuse: scheduling
  must not alter the scientific acceptance contract.

## Evidence and revisit

Host boundary/injection tests protect parsing, cold/domain/size/limit refusal
and the existing resource/numerical contracts. Independent allocated-device
tests must distinguish automatic cold admission from warm, smaller,
lower-angular and tight-budget refusal. Complete default-auto endpoint
qualification uses new source/binary identities; historical opt-in samples
are not relabelled as measurements of this policy.

The heuristic deliberately leaves potential wins unselected. Broaden its
domain only with source-matched full endpoints and short-convergence cases;
it is not a universally optimal or per-iteration online cost model.

## Source-matched automatic qualification

The clean measured revision is `2db14f19fc1a460989785a0d2e21e45fa0a5dac5`,
based on master `15e052d800085c61373937ed9c81c06cc6ca3548`. The new library
SHA256 is `6d771d67542d82e47d4442b19f6a4b9ad74afd971d2f4ff32b80485b03c20ded`.
These are new measurements with the environment **unset**, not relabelled
opt-in results. The later master LR-acceptance receipt at `742dff879` changes
no production compilation inputs for this qualification.

Slurm jobs 7118–7120 run fresh-process A-B-B-A on three distinct n1 RTX 5090
UUIDs, six complete native ethane E+F samples per side. A explicitly disables
values; B leaves selection unset. Complete native median changes from
430.517632 to 351.972712 seconds: 18.24430% reduction and 1.223156x speedup.
RHF including the native initial DF guess changes from 97.303432 to
18.458025 seconds. Every per-UUID comparison has positive reduction, all
samples pass the same numerical gates and the observed reduction exceeds
the retained descriptive noise floor. This is descriptive scoped evidence,
not a formal universal promotion envelope or an inferential confidence interval.

All trajectories retain RHF 12 iterations and final physical Fock 13, CC
20/38 iterations/evaluations, Lambda 21/22 iterations/actions with Q32,
Z 12/13 and two exact shell derivative passes. One-second sampled NVML total
device peaks are 31,149/31,151 MiB, below the recorded 32,607-MiB device;
these are not exact allocator/owned-buffer peak measurements. New cached
compilation takes 210.98 seconds and does not claim a full uncached build cost.

Candidate energy error is at most 2.416e-12 Eh against retained independent
PySCF ethane energy, and the 24-force error is at most 8.153e-10 Eh/Bohr
against the retained qualified native vector. The retained independent
two-coordinate/two-step FD checks also pass. Those ethane references are
retained, not fresh full ethane oracles. Five new allocated-device tests on
n4/job719 independently generate water, methane and methane-SPD RHF
original/displaced energies and check automatic admission, warm/size/angular/
budget refusal, geometry changes, forced rollback and graph reuse.

The separate n4 node-level profile proves automatic admission, 575,639,415
completed source values and 13 resident physical Focks. Source preparation
takes 11.680399 seconds; 364 resident kernel launches total 2.974123 device
seconds, with no canonical recurrence or bounded-direct Fock kernels after
preparation. Complete transient reference admission is 5,594,556,007 bytes.
Exact two-electron nuclear derivatives still take 89.508454 seconds and are
not changed by this optimization. These instrumented timings are not pooled
with clean n1 samples. The methane lifetime/geometry probe separately passes
full memcheck on n1/job7122 with zero errors and zero leaked bytes.

The compact numerical publication retains all complete endpoint outputs,
comparisons, oracle identities, reconstruction patch and recipes at
[the auto qualification](../../../../benchmarks/results/rhf-phase-values-auto-20261010/README.md).
Raw build logs, sampled NVML streams, progress journals, Nsight report/SQLite
and sanitizer output remain in the separate local audit evidence bundle.
