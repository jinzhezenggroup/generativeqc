# Decision: consumer-owned bounded XC force powers

Status: implemented; generic scalar and complete-endpoint qualification passed
Date: 2026-10-15

## Problem

PR #2506 moved #2505's M06-2X/RKS-only performance exception into a generated
method/code/spin manifest. This relocates specialization rather than removing
it. The generic scalar IR already has a bounded, shared integer-power pass;
curated semilocal generators apply it after AD, but imported split hybrids did
not. The historical global M06 DF96 warm regression adds a fourth Fock build
instead of three. It cannot be repaired by a method whitelist, hidden solver
work, different initial densities, or relaxed scientific tolerances.

## Decision

Keep the consumer boundary, remove the method boundary. The FP64 stationary
force consumer selects bounded power expansion for every imported program and
both spins, using the existing IR pass after independent polarized AD and
before equal-spin binding. No functional name/code/spin participates in the
schedule selector. Remove the schedule manifest/generated-registry plumbing.
Curated semilocal programs already use this same bounded pass and do not change.

SCF point generation and float scalar ordering stay unchanged. A stationary
derivative schedule is not the self-consistency map: rewriting the latter can
perturb a convergence boundary enough to add a whole expensive Fock build.
This consumer distinction is generic, not a benchmark- or molecule-specific
exception. No #2486 AO/dddd/page/tile negative experiment is repeated.

## Required gates

- Exact source preservation for every registered SCF point and float graph;
  exact preservation of #2505's already-qualified M06 RKS force source.
- Independent Libxc Cartesian partials, displaced-energy checks, tau/2 ABI,
  and actual RKS/UKS force emission for both registered imported hybrids.
- Real-GPU scalar acceptance of both actual production spin consumers through
  finite Slurm, including existing independent boundary/oracle gates.
- Paired timings and semantic work: distinguish scalar-kernel measurements
  from complete E+F endpoints, and retain every sample and negative verdict.
- Revisit the DF96 extra-build population without re-enabling the known-bad
  global SCF rewrite. Never credit incidental baseline iteration variability
  as a solver improvement or claim a universal hardware speed guarantee.

## Evidence and history

The policy-only admission refactor remains historical rationale in
`.agents/notes/implemented/architecture/2026-10-15-xc-scalar-schedule-qualification.md`.
The original negative experiments and #2505's 252 native observations remain
unchanged and recoverable through their existing checksummed Git snapshot.
The generic FP64 force schedule is qualified below; the rejected global SCF
schedule remains disabled. No functional/spin admission list replaces it.

Initial host acceptance: 64 tests pass with the pinned Libxc 7 environment,
including actual Cartesian force source for both imported methods and spins,
unchanged independent partial/finite-difference gates, and scalar ABI probes
bound to both production spin consumers. No real GPU result is inferred from
these host tests. The initial probe harness failed compilation because two
namespaces have distinct C++ result types; a generic output sink fixes that
harness error without changing generated scientific code or tolerances.

### Generic source and real-GPU gates

Scientific candidate: `bbef4e534d31d050474bfee570b606eb10b01c90`.
Control: merged #2505, `2b256f353b3803d9ffc7fe1e418fd3176e934634`.
During qualification, #2506 merged at `fdc3cd54b2dbefeafa325959d54dfd7aeeb8c53b`
on 2026-10-15 at 09:48:02 Asia/Shanghai. Its entire tree equals the original
`7427602379759611ea05345b0e3598b6240198f8` PR parent already in this feature's
history. A follow-up PR integrates that exact merged baseline without
reapplying its obsolete whitelist or relabeling the measured source revisions.
A one-off source receipt preserves 14 consumer sources byte-for-byte, including
every SCF/float graph, M06 RKS FP64 force, and all five curated geometry sources.
Only M06 UKS FP64 force and MN15 RKS/UKS FP64 force change. These fingerprints
are qualification receipts, not permanent required-CI source-spelling tests.

Slurm 3022 on node2, RTX PRO 6000 Blackwell/sm_120, CUDA 12.9.1: both methods
pass the original independent Libxc 7 partials, finite differences and restricted
embedding gates; M06 also passes its authenticated 113-bit Maple empty-spin
boundary gates. The probe
body must belong to both actual production spin consumers. Both methods pass
compute-sanitizer memcheck with zero errors. The final focused host suite has
64 passes, including the functional-alias test; compiler structure reports
540 modules and zero dependency errors. No scientific tolerance is relaxed.

The Cartesian force-point campaign has eight alternating rounds, 100 device
repetitions, and 256/8192/65536 points, retaining all 192 timing observations.
M06 UKS point medians improve 31.76--31.92%; MN15 RKS improves 49.63--58.23%,
and MN15 UKS 46.87--58.67%. These are not molecular endpoint speedups. The
already-qualified M06 RKS graph/output is identical: its -0.12% to +0.04%
timing changes are noise and receive no new performance credit.

Retain the stricter, non-admission pair diagnostic
`abs(candidate-control) <= 1e-12 + 5e-12*abs(control)`: M06 UKS has one
exceedance at 8192 and three at 65536, all in tau/2 partials. Independent
113-bit reevaluation shows two candidate tails better and two worse than the
control. Candidate/strict-wide-limit ratios for the worse rows are 1.864814
and 2.003368 (absolute errors 2.997e-11 and 1.361e-11). They remain recorded
diagnostic failures, not passes; the original independent scientific gates
pass. All other method/spin/count combinations have zero strict pair tails.

### Complete open-shell endpoints

Two source-matched populations on node2 reverse arm order, each with cold,
five frozen-own-seed warm, moved, and five moved-warm calls. The public CUDA
Calculator uses water/sto-3g, direct FP64, a 24x8x16 atom-centered grid,
1e-12 energy/1e-10 density/1e-12 screening tolerances, and neutral RKS or
charge-one doublet UKS. Native preparation and synchronized execution are
timed; independent PySCF SCF plus analytic moving-grid force is not. The
reference does not supply a native initial density.

UKS warm complete E+F improves 4.618% forward and 5.061% reversed;
moved-warm improves 4.691% and 5.095%. Warm Fock work stays two builds.
The unchanged RKS warm control differs by at most 0.18%, not a new speedup.
Every successful fixture passes independent 2e-9 Eh/2e-7 Eh/Bohr gates.
MN15 has no qualified packaged complete-force profile in this inventory:
do not enable a capability to obtain an endpoint number, or relabel its point
timings as a complete MN15 speedup.

Slurm 1580/1581 on node5, RTX 5090: matched README M06 RKS DF96 cold,
five frozen-own-seed warm, moved, and five moved-warm endpoints pass every
independent 1e-8 Eh/1e-7 Eh/Bohr gate. Both arms have 18 cold, 3 warm,
12 moved, and 3 moved-warm Fock builds. Warm medians are 35.657592 s control
and 35.658724 s candidate (+0.0032%); moved-warm changes +0.0387%. This
unchanged scalar consumer is a nonregression control, not a new speedup or a
second claim for #2505. The historical global-SCF three-to-four-build
experiment remains rejected, not re-enabled for this test. Cold complete
time changes +0.635% with identical work, not evidence of a new speedup.

The matched direct96 endpoints also pass every independent gate. Warm and
moved-warm Fock work stays one build, with median timing differences +0.0349%
and +0.0856%, respectively. Cold work is 19 control versus 18 candidate;
moved work is 14 versus 12. These consumers have identical SCF and force scalar
graphs, so the apparent cold -4.585% and moved -12.172% improvements are baseline
trajectory variability, not optimization credit. This does not establish that
every stochastic cold/moved SCF iteration boundary has been eliminated.

### Build identity and failed setup retention

Both arms explicitly reuse the immutable native SCF library built from
`8021e2d92717b7035448227936a4e960dc25c0b6`; `git diff` from that revision to
merged control `2b256f353` is empty for `src`, `python`, `cmake`, and the root
CMakeLists. New matching RKS/SPD and both-arm UKS stationary artifacts are
compiled with existing ccache and the same Release/sm_120/fmad-false contract.
The redundant all-native rebuild is deliberately stopped after these relevant
artifacts compile, not represented as a completed build. No binary is relabeled
as coming from a newer source revision.

Keep the early missing-packaged-UKS failure and the stale-RKS-AOT plan-identity
rejection, including completed preceding calls and the recorded failure row.
They are setup failures, not scientific speedups. Rebuild/install the matching
artifact rather than weakening provenance checks or inventing a fallback.
The first strict batch raised before serializing the failed UKS item; its work
is unknown, never recorded as zero. Subsequent calls serialize status/work
before checking completion. Native SCF is identical between arms; the UKS
stationary binary is genuinely different and authenticated by its manifest.

Native library SHA-256:
`5c080c49d086890bec0ad1b5390d29b47c78943dcf0b4caf23b410b778e2d9c1`.
Stationary artifact SHA-256 receipts (control UKS is built from `2b256f353`,
candidate artifacts from `bbef4e534`):

```text
control UKS   2e56a0ad7212bebb4d0676f36a7dcb743f4a0ce033fd66ef9002cefa36f87033
candidate UKS e2d47e60c825591f14b89b17c1c47ab5105b55d2082f64e086e32ae4d9ceb458
candidate RKS 5660cef839eb3434fac01d3133d63ebd4187c485f829a9ee7c89f57ab732644d
candidate SPD 7487bf1bbf2db493d12de9e06cd8eeb42239ef92c8ab322c70e72985a30769a2
```

Successful small endpoints use Slurm 3024/3025/3026, reversing arm order in
3026. Setup failures are Slurm 3023 (absent UKS package) and 3024 (stale RKS
package). Full logs/JSON keep that distinction instead of deleting these runs.

### Evidence encoding and retention

The new ledger is
`benchmarks/results/xc-force-powers-generic-20261015/ledger.json.xz`, schema 1.
The 2756-byte file has SHA-256
`24b8754f13a619338571161fc7ab157913b0b5eb2a94744bb3c3ea3c88c3b82c`.
It retains all 192 point timings and 157 endpoint rows: 144 completed paired
qualification rows, 12 completed setup-prefix rows, and one serialized failed
status row. The additional early UKS setup exception has unknown unrecorded
work, as described above. No setup prefix contributes to the paired speedup.
The ledger uses XZ-compressed JSON. Numeric columns are lossless
`[scale, deltas]`: accumulate non-null integer deltas from zero, then decode
with `math.ldexp(total, -scale)`; null does not advance the accumulator. Point
columns encode the original binary32 100-repetition CUDA event span; divide
decoded values by 100 to reproduce each stored per-point-call millisecond
sample exactly. Endpoint `seconds`/`prepare` reproduce original binary64
values; `complete_seconds` is their sum. Each file keeps native row order,
iterations, all Fock counts, statuses, nonconverged rows, warm/fallback bitsets,
and per-row independent energy/force errors. Successful small files contain
12 RKS then 12 UKS rows; failed setup files retain their available prefix.
The phase order is cold, five warm, moved, five moved-warm for each method.
Point arm columns keep round order; round arm order alternates, starting with
control. No median replaces the underlying observations.

To reproduce point inputs, reset `np.random.default_rng(20261015)` for each
method/spin/count. Draw two densities as `10**rng.uniform(-9, 1, (count, 2))`,
then Cartesian gradients as
`rng.normal(size=(count, 2, 3))*rho[..., None]**(4/3)*0.1`, and kinetic
ingredients as `rho**(5/3)*rng.uniform(0.5, 2, (count, 2))`. RKS copies
alpha ingredients into beta after these draws. Emit the actual Cartesian
force point with `_emit_composed_point` and the registry code/MethodIR, as in
`tests/python/test_equal_spin_xc_codegen.py`; retain energy, two density
partials, six Cartesian gradient partials, and two tau/2 partials. Use
128-thread CUDA blocks, three warm launches, then the paired event protocol
above, with `nvcc -O3 --fmad=false -std=c++17 -arch=sm_120` and the verified
compiler cache. The independent acceptance command is
`tools/validate_split_hybrid_cuda.py --consumer stationary_force`, with its
required authenticated Libxc 7 archive and CUDA arguments. Run every real-GPU
invocation through finite Slurm, never by overriding assigned device visibility.

The small molecular fixture is `ATOMS` in `tests/python/test_dft_complete_cpu.py`;
move atom 1 by `(0.015, -0.01, 0.02)` Bohr. Use `max_iterations=200`, public
`prepare_batch(..., warm_start=True)`, and freeze the owner's own seed with
`set_warm_start_updates(False)` after cold/moved execution. The untimed oracle
is that file's `independent_global_hybrid_gradient` on the authenticated native
AO/quadrature snapshot, with `xc="450,236"`. The README96 protocols use
`benchmarks/run_readme_dft_benchmarks.sh native`, `README_BENCHMARK_METHODS=m06-2x`,
`README_BENCHMARK_SIZES=96`, `README_BENCHMARK_REPEATS=5`, separately for
`README_BENCHMARK_DENSITY_FITTING=cuda` and `none`, with existing independent
reference JSON. Do not mix incompatible artifact plan identities between arms.

The 64 MiB results budget is unchanged. Whitespace-only minification of the
three directly related #2505 recovery descriptors frees 460 bytes; only the
two bound descriptor byte counts/checksums change. The 252-row historical
ledger and pinned full snapshot are unchanged. No negative result, unrelated
evidence, Release, tag, external backup, or release asset is removed/published.

Detailed raw arrays, compiler commands, GPU receipts, setup failures, and
native/reference JSON remain under ignored
`.artifacts/xc-force-generic-20261015/` and the same campaign directory under
`/data/jzzeng/` on nodes 1/2/5. Fixed scalar report SHA-256 receipts:

```text
gpu-acceptance.json       3dcb817cca3d4e1100178830b541c4c248010a1e53368e44e70eedaf5264de91
point-timings.json        4c32f4364e96f10eb8d92c0782ed1807f25b63e45cff13156bf911ad5609eb26
source-preservation.json  5d0570cfa3afad4e7c0d961c259ad1f00b2040b0cf08dff656a5dd1d19948097
strict-pair-wide-tail.json 116c56f3807522e95ff0c198f4c6846f84202eec61d2171f03e8f4c285a1d3ed
```
