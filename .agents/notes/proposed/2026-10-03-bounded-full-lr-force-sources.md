# Proposal: specialized full-range and LR stationary source traversals

Status: experimental; larger and latest native qualification pending
Date: 2026-10-03

The bounded fused RSH derivative traversal evaluates three radial source kinds.
For the packaged omega=0.3 specialization, reuse the full-range helper's separate
J/K outputs and a unit-coefficient LR traversal. Publish the existing three
source meanings as J', short_coefficient*(Kfull' - KLR'), and
long_coefficient*KLR'. The full helper returns derivatives; the LR shell worker
returns force, so its sign is inverted exactly once before source composition.

The full helper now produces J/K in one traversal after #1716; this proposal
therefore uses two specialized source traversals and two successful host drains.
Other omega values retain the fused bounded RSH route. No dense quartet storage,
CPU scientific fallback, precision relaxation or new screen is introduced.
Both asynchronous host result lifetimes are protected on failed enqueues,
failed synchronization and C++ exceptions. Publication occurs only after complete
success; a failed invocation does not replace the caller's source vector.

## Evidence and provenance

75 host tests pass, including labelled nonzero J/K/LR outputs, independent
coefficient/sign checks, all failure/throw points, host-result lifetime, recovery
and the successful synchronization count. Verified ccache before/after receipts
and source/archive/compiler-command hashes are retained under ignored
`.artifacts/wb97m-full-lr/` in the review worktree.

Native library `03c2ae9d6d22dbf1f58e851819fb3e26f1b700b2c05e5a5fd8ba6c36d3800c80`
was built from parent `47ce26971` plus the recorded patch, including #1716.
Node1 Slurm job 5426 passes native Cartesian order-two derivative/CPU finite
differences and s/p/d/f SR/LR derivative gates, then six independent complete
WB97M-V RKS/UKS/displaced-energy tests. It uses eight CPUs, one scheduled RTX5090
and a finite one-hour allocation. The first launch failed before execution due
to a missing library SONAME symlink; that packaging receipt is retained.

The complete 24-atom water cluster (192 spherical def2-SVP AOs, 589824 unpruned
points, grid 48 x 16 x 32) passes all five paired GPU4PySCF samples: max energy
2.718e-11 Eh and force 4.162e-10 Eh/Bohr, against 1e-8/1e-7 gates. Cold is
327.467 s (18 SCF iterations), warm median 53.958 s (one iteration each), versus
GPU4PySCF warm 27.229 s. A separate concurrent node1 allocation with the same
parent, compiled without this patch, records cold 346.036 s (18 iterations),
warm 57.717 s, all five pairs passing. Different allocations/devices preclude a
controlled speedup claim; a same-allocation comparison and larger case remain
necessary before default promotion.

An older three-traversal prototype, before #1716, used binary
`64f9e224f92c45138ca2d23963b124ec57b2e8247c6dc8820cb8a6f75927310d`.
It passed six independent molecular tests; node1 job 5406 compared capacity and
that candidate in one allocation: warm medians 66.958 -> 64.975 s, all five
sample pairs passing. Its cold iteration counts differed. The old source,
profile and measurements remain under `.artifacts/wb97m-full-lr-force/`; they
do not qualify the new two-traversal implementation.

Actual screened molecular quartet/recurrence counts are not exported. The two
source traversals above describe the selected scheduling contract, not observed
post-screen work. Preserve this distinction in benchmark reports.

## Revisit when

A fused generated full/SR/LR shell consumer or another omega specialization has
complete endpoint evidence that beats this decomposition. Preserve all three
source meanings and independent strict energy/force gates when changing the
operator representation or screening policy.

## Controlled two-traversal comparison

Node1 Slurm job 5449 ran baseline `capacity2` followed by `full-lr2` on the same
scheduled RTX5090 with eight CPUs. Both used the full24 settings above and three
warm repeats. Baseline/candidate cold is 331.171878 / 330.307535 s, both 18 SCF
iterations. Warm median is 57.326409 / 54.321158 s (1.0553x), with one iteration
per sample. Candidate samples span 54.196943--54.730460 s; its GPU4PySCF median
is 27.380643 s. This improves our endpoint but does not beat the reference.

All five pairs pass for each variant. Candidate maximum energy error is
2.728485e-11 Eh and force error 4.160557e-10 Eh/Bohr; baseline maxima are
2.819434e-11 Eh and 4.163496e-10 Eh/Bohr. The final force-component records show
integral derivatives 18.012149 -> 14.975477 s and grid/pair drain
23.138527 -> 23.136647 s, consistent with the intended source change.
The compared native binaries include #1716 but predate #1732; the candidate
hash is the `03c2...` hash above. Full JSONs remain under the ignored evidence
folder; this evidence is not silently relabelled as a later build.

After #1725 merged, the branch was rebased onto master `9a5871dca`. A concurrent
review fix was preserved: when both exchange coefficients are zero, the full
helper's J row is published with zero exchange rows without evaluating LR or
forming potentially overflowing spin products. The updated implementation has 97 passing host tests and was rebuilt with
verified ccache. The new native hash is
`0b1d9a8511ebf177f49af94a3fbeaf2cd4359b4281b9f85c9cd874cfd5e43021`;
independent molecular and default48 qualification is running as `full-lr3`. Coupled larger
experiments include separate scalar changes and cannot isolate this proposal.

## Standalone repaired-source completed endpoint qualification

The exact `0b1d9a...` native/source `1d7fb4e0f` passes seven independent complete
molecular/rebuild/stale-state tests (node1 job 5459, 199.08 s). That job completed
default48: cold 1443.967429 s/21 iterations, priming 213.179017 s, three
one-iteration warm samples 213.507344, 212.919550, 213.093309 s (median
213.093309 s). GPU4PySCF warm is 172.248139 s. All five pairs pass, maximum
energy 2.978595e-11 Eh and force 5.890293e-10 Eh/Bohr. This standalone source
predates the geometry and VV10 denominator merges, and remains slower than the
reference; the result must not be relabeled as the subsequent integrated source.

Moved water-12/def2-TZVP on the 24x8x16 diagnostic grid passes all independent
full CPU-oracle samples, including fixed-final-state force replay. Cold, priming,
warm and moved are 166.944908 / 22.662385 / 23.144913 / 92.331291 s with
22/1/1/11 iterations; maxima are 1.307399e-12 Eh and 8.964989e-10 Eh/Bohr.
Raw receipts remain in `.artifacts/wb97m-full-lr-review/results/`.

Master `0d9d763e3` is then integrated, adding the separately qualified geometry
schedule and VV10 denominator changes. The fresh integrated binary and real-device
qualification must be identified separately from the standalone results above.
