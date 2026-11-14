# Decision: publish the complete composed-default README cohort, including cold losses

Status: implemented
Date: 2026-11-11

## Scope and identity

The requested composition is master
`3c2c6a37617ea89691de00ca1c9586d07bda0307`, PR #3064 head
`ebe21c712ce8d51d73fc37837ac08407496f726c`, and PR #3065 head
`3782e5124951b6ed8272dfae8d66c17cce99d2ba`. Both PRs were open when frozen;
only a local integration branch was merged, not the upstream PRs.

Clean measured commit: `6eb5fceda4d7c0e40aea371929eef8c05db773a8`.
Native input identity:
`97f578e7ec023269762fd1d0306aa2a4f45a1f60778c1fb08131afbaeb5e4ed1`.
Production ELF SHA256:
`485cac533c9c12de3463bf0bf052b704501274599f60f11e39c7b1716c3907a4`.

A fresh CMake build directory compiles the complete Release/SM120 target with
verified sccache 0.16.0; this is not the earlier manually relinked XC image.
Actual launcher commands and before/after statistics are retained. The cache
reports 101 hits and 1253 misses across its CUDA subcompilation categories;
using a cache is not a claim that this build was mostly hits. An absolute-path
Python check authenticates the local expected input identity against the built
and executed ELF. Subsequent changes are tests, evidence and documentation;
production input identity remains unchanged.

## Complete acquisition and numerical acceptance

Build Slurm job 17124; original qualification 17129; four measurement jobs
17125/17126/17127/17128. All use finite main-partition allocations on node1,
preserving their assigned RTX 5090 visibility. Eight methods × Direct/DF ×
3/12/48/96 atoms give 64 consecutive same-GPU engine pairs, balanced first
engine order 32/32, one numerical host thread. Every process starts a new
calculator/plan and performs one complete cold E+force call and three warm
calls from its own frozen cold density. Min–max bars retain all warm calls.
Import/context startup and cache-miss compilation are outside the endpoint;
caches are retained, not artificially emptied. Applicable D4/gCP and host
force export remain inside timing. Native endpoint imports reject both
reference engines. Native convergence and spatial-order selectors are unset.
GPU4PySCF's SCF/convergence/auxiliary/grid settings are unchanged; the existing
common full-grid-response force contract is retained.

All 128 processes / 512 complete calls succeed, original campaign exit 0,
no retries or displaced samples. Full-vector replay freshly recomputes the
unchanged independent E/atom 1e-5, Fmax 1e-4, Frms 3e-5 gates from the reviewed
historical dense/CPU oracles. Scientific inputs, auxiliary inputs and the
explicit environment-reuse contract must match. Both engines qualify 58/64
pairs: all DF HF sizes and Direct/DF M06-2X3 remain failures, not hidden wins.
All 1024 cross-engine comparisons pass but do not replace independent gates.
The public compact replay uses recorded unrounded oracle-error scalars; it
does not claim a new full-vector subtraction.

Direct wins 29/32 cold and 32/32 warm; DF wins 27/32 cold and 32/32 warm.
Scientifically qualified wins are 51/58 cold and 58/58 warm. Direct
ωB97M-V96 warm is 9.375783152606674× faster. CBS is unchanged.

All eight cold losses are at 96 atoms (native/reference seconds):

| Mode | Method | Native | GPU4PySCF |
| --- | --- | ---: | ---: |
| Direct | B3LYP | 23.111533 | 21.713763 |
| Direct | r2SCAN-3c | 34.006462 | 32.012435 |
| Direct | M06-2X | 25.905336 | 24.946405 |
| DF | HF | 17.520099 | 17.494772 |
| DF | PBE0 | 24.683477 | 22.807901 |
| DF | B3LYP | 25.925769 | 24.268533 |
| DF | r2SCAN-3c | 28.663298 | 23.400643 |
| DF | M06-2X | 29.647315 | 27.969656 |

These are single-process observations, not confidence intervals or universal
rankings; the almost tied DF HF96 point is still a loss. No successful slow
pair is replaced and no reference control is loosened.

## Why the old cold-win count does not carry forward

The earlier RHF-only README cohort measured `8c9acd542` on `d35ca77a4`, before
master's #3060 force-stationarity repair. The new composition includes that
repair. `standard_ks_stationarity_allowance` no longer grants the energy-only
allowance to an imminent full-range force consumer, and
`early_ks_final_probe_stationary` preserves its physical maximum-residual
ceiling. Force intent is retained even when no extra workspace is needed.

Observed DF96 cold physical Focks in the old/new source-bound cohorts:

| Method | Earlier | Composed |
| --- | ---: | ---: |
| PBE0 | 9 | 15 |
| B3LYP | 9 | 16 |
| PBE | 10 | 15 |
| r2SCAN | 9 | 14 |
| r2SCAN-3c | 10 | 16 |
| M06-2X | 10 | 16 |

Warm calls retain one native physical Fock. Direct RHF48/96 retain 10/11
cold physical Focks. The additional DFT cold work is consistent with the
stronger force contract, not evidence that XC bisection changes quadrature
or that the RHF default reverted. This campaign is not a same-binary,
single-factor ablation of #3060/#3064/#3065; do not attribute every latency
difference to one policy, or add/subtract earlier PR timings to predict it.
Do not roll back the force gate merely to restore the old cold-win count.

## Qualification boundaries and retained failed attempts

Physical RHF convergence/finalization, graph/non-graph and reduction-width
device guards pass. Default/explicit-MO/strict and tighter-control HF48
independent-oracle guards, unsupported-request rejection and prepared-policy
identity checks pass. The original Python qualification has 285 passed and
16 skipped (four optional restart tests and twelve explicit state probes).

A separate opt-in state run, job 17130, has four passing original range-only
RKS/UKS Cartesian/spherical production controls. Eight UKS mask probes fail
their prerequisite before SCF: this production build intentionally omits
`tensor_structured_scf_qualification_for_test_v1`. The complete failed run
and JUnit records remain retained. These eight cases are neither passing
numerical qualification nor evidence of a measured force failure; no
test-hook binary is substituted into the performance cohort.

The inherited #3065 core-c CI failure also exposed an incomplete host fixture:
its extracted replay predicate now references the bound orbital-gradient
policy, but the fixture had neither the plan field nor its request value.
Add those neutral bindings, retaining the existing reducer lifetime behavior;
all four endpoint-contract tests pass. No production numerical code changes.

## Retention and replay

Current compact data and reproduction command are in
`benchmarks/results/readme-rtx5090-composed-20261111/README.md`.
Full vectors, acquisition, compilation, failed attempts and replay scripts
remain in ignored `.artifacts/readme-rtx5090-composed-20261111/` and the
matching `n1:/data/jzzeng/readme-rtx5090-composed-20261111` directory. The
clean source archive and exact ELF are retained separately; no Release,
release asset, tag, or external backup is created.

Keep the previous RHF-only cohort and all its displaced failed-pair samples.
To fit the unchanged master evidence cap, compact only related historical
October 26 receipt envelopes: remove JSON whitespace and recompress the
native/reference/qualification streams. Every string/number token and array
order is preserved. Exact original envelopes are independently recovered
from published master `3c2c6a37617ea89691de00ca1c9586d07bda0307`, with sizes,
SHA256s and token hashes in the private preservation receipt. Update only
attachment storage digests and the old compact cohort's reference-container
digests; all old timing/error/work and decision fields remain identical.
Publication metadata whitespace is compacted too. Historical exact replay
uses its isolated Matplotlib 3.11.2; new full and scalar replay use 3.10.9.
No unrelated evidence or scientific input is removed and the cap is not raised.

The same reference containers are also consumed by the retained GGA indexed-GEMM
and A100 README capsules. Their original `legacy_sha256` links initially still
named the pre-compaction bytes, which made their independent replay fail before
reaching any scientific checks. Rebind only those two container digests in each
capsule after independently comparing every JSON string/number token and its
order against the pinned master above. All 333 GGA and 388 A100 original receipt
texts and both FP64 streams remain byte-identical. The storage receipts pin the
original reference revision and changed fields; the capsule metadata and outer
manifests authenticate the current containers. No checksum, numerical gate or
failed result is removed; this is a storage-link repair, not renewed numerical
qualification.

## Revisit when

Optimize the now-observed full cold force endpoint while preserving physical
stationarity and independent force gates. Use a same-source controlled
ablation if causal per-policy savings are required; renew qualification before
changing stopping, grids, fitting, precision, or the retained bounded fallbacks.

## Superseding default-policy measurement

At the user's request, the default-policy decision is superseded by
[restricted KS orbital stopping](../numerics/2026-11-11-ks-default-orbital-stopping.md).
The preceding results remain historical evidence for `6eb5fceda`; they do not
justify silently imposing a more stringent force contract than the reference
engine's default. The original `1e-7` force-repeatability gate is retained under
an explicit strict request, not weakened or relabeled as a default guarantee.

A fresh complete cohort measures `886556ce5504dffa10679e624c963e795eb12e64`,
native source identity
`01e5b8baaa5bdb377a60689f184f5b16eaab0e1ba93a3f3a058e51db9900eba8`, ELF SHA256
`ff0c1473889c8781bd005af0f7b3e6a06571cfc9c1a9af1e9ed980fe86f61555`.
Jobs 17152/17153/17154/17155 acquire all 128 processes / 512 complete calls,
without retries or successful-point replacement. All three native selectors
are unset and the GPU4PySCF default settings compare exactly with the retained
baseline. Cold Direct wins 32/32, DF 31/32; warm wins remain 64/64. Both engines
still independently qualify 58/64 pairs, with all 58 qualified pairs faster in
both phases. All 1024 paired vector comparisons pass the unchanged gates.

The sole cold loss, DF HF96, remains visible at 18.182064924389124 versus
17.741809528321028 seconds; its compact DF HF owner was not changed. Default
DF96 physical Focks are now PBE0/B3LYP 9, PBE/r2SCAN 10, r2SCAN-3c 11 and
M06-2X 10. The strict-only early proof cannot bypass the global MO-gradient
gate; restoring the old work counts by skipping that gate would not be a valid
alignment. Direct WB97M-V96 warm speedup is 9.345219795538252. CBS is byte-identical.
These are complete endpoint observations, not a same-allocation single-factor
latency ablation or a universal ranking.

The current compact publication replaces only this figure pair's preceding
cohort; the exact earlier publication remains recoverable at `28ac57e819`.
No historical sample supplies a new timing. Full-vector/source/build receipts
are retained in ignored `.artifacts/readme-rtx5090-aligned-20261111/` and the
matching `n1:/data/jzzeng/readme-rtx5090-aligned-20261111`. The scalar capsule
stores full build-receipt digests instead of duplicating verbose compiler logs;
it does not claim a fresh vector subtraction. The fixed aggregate retention
cap is unchanged, and the concurrent receipt-integrity repair `6b3f98eae` is
preserved rather than overwritten.
