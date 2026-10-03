# Qualification: resident AO maps and indexed WB97M-V forces

Status: opt-in integration; complete endpoint qualification passed, latest-master 96-atom timing pending
Date: 2026-10-03

## Composition and ownership

The earlier [force composition](2026-10-03-wb97mv-large-endpoint-stack.md) removes
extra long-range work but does not itself establish a large-system reference
advantage. This integration adds the shared resident AO discovery/cache/local
contraction interfaces (#1770, #1772, #1774), the WB97M-V force consumer (#1778),
native SCF discovery (#1780), and explicit indexed independent-force pages
(#1767). The original VV10 storage and range-force owners remain unchanged.
Individual component gains must not be added together.

All new selectors remain opt-in. Force AO selection uses cutoff 1e-16 and a
64 MiB map-cache allowance; native SCF discovery uses its explicit environment
control and reports its own preparation/work counters. Indexed scheduling is
captured when its native owner is constructed. Dense/unindexed fallbacks remain.
The fixed SCF experiment host cap is not admission against a remaining public
host budget, and this composition does not promote automatic screening.

## Completed observations retain their source identity

The jointly mapped, unindexed integration 0132d7584 is retained in README PR
#1786 with every cold/priming/warm pair, source/binary hashes and replay scripts.
Its 24/48/96-atom complete warm medians are 26.330918/97.112717/387.179006 s,
versus paired reference 27.397887/103.042334/403.542077 s. Fresh same-source
3/6/12-atom points complete the HF-style six-point curve. All 45 pairs across
nine dense/joint variants pass the original 1e-8 Eh / 1e-7 Eh/Bohr gates.
Finite n5 Slurm 1412 now completes its 96-atom dense control: warm median
457.225580 s versus its own paired reference 404.221890 s. Joint-map warm
latency is 15.32% below this same-binary dense control, with one SCF iteration
in every warm/priming call. Dense complete cold is 5016.225258 s, versus joint
4092.520636 s and their respective references 1849.279613 / 1851.051863 s.
Native dense/joint cold trajectories take 23/25 iterations, so their cold
difference is not an isolated contraction-work comparison. Both remain slower
than the reference. This source is still the frozen 0132/b82 composition.

The indexed composition 9ad97c2c4 is separately published as
`benchmark/wb97m-indexed-20261003`. Its 1353-input source identity is
`2b1ae789cad9489b239e2af211b41859f4d5464b99813b2c4859dd8bfc022728`;
library SHA-256 is
`d4801b8231af9f2cabb36baede1c8708193e93f736d5887aee5ae31b299edb8f`.
Finite n1 RTX 5090 Slurm 5585 completes all five 96-atom indexed/reference pairs.
The independent verifier checks raw E/F arrays, convergence, source/library
identity, actual AO selection, all reference XC backend flags and cold totals.
Maximum errors are 1.174e-10 Eh / 4.853e-10 Eh/Bohr.

All three native warm samples are 364.772766, 364.468820 and 364.769040 s;
paired reference samples are 399.912986, 401.186207 and 399.054703 s. Every
warm/priming call takes one SCF iteration and reference XC is on GPU. Medians
are 364.769040 versus 399.912986 s, or 8.79% less native time. Complete cold
includes construction and preparation: 3707.315601 versus 1815.398540 s, with
23 versus 16 SCF iterations. No iteration normalization or sample omission is
used. Grid, method, basis and full energy/analytic-force semantics match #1786.

The same-binary 96-atom unindexed control also completes in Slurm 5585. All ten
OFF/ON cold/priming/warm pairs pass the independent verifier, with maximum
errors below 1.174e-10 Eh / 4.853e-10 Eh/Bohr. Unindexed warm samples are
380.325073, 380.922667 and 383.214918 s (median 380.922667 s); the paired
reference median is 399.923344 s. Indexed scheduling therefore reduces native
warm latency by 4.24%, separately from the complete 8.79% reference advantage.
Every priming/warm call takes one iteration and reference XC remains on GPU.
Unindexed complete cold is 3875.281943 s with 24 iterations, versus indexed
3707.315601 s with 23; their respective references are 1813.225496 /
1815.398540 s with 16 iterations. Do not attribute the cold difference entirely
to force scheduling or normalize away the trajectory difference. At 24 atoms,
the prior 26.662892/26.547909 s OFF/ON medians do not establish a robust gain
against the observed scatter. Actual indexed-claim counts are not exposed by
this runner; requested policy is not relabeled as a measured claim count.

## Master 1a4acc519 qualification and frozen timings

Frozen source 4079d30b4873ac41307f4105c0f778feeb9db83c includes actual master
1a4acc519 and both claim-reader barriers. The add/add regression-test conflict
is resolved by preserving triangular and indexed schedules, zero and 256-count
products, all skip combinations, and initial cursors 0/1/15. Four host claim
checks pass; six explicitly gated device cases skip in that host command.
All staged hooks pass. No new integral algebra or cutoff is introduced.

The 1353-input source identity is
`62861c1498e199dd9c3acf55e5c31e4f6b8898c81460372475ad31e0a9c85602`;
library SHA-256 is
`6dbab582dbc86f860c2db9ec0a4ffe2241756fd37c18f087ad4cb0961dd59009`.
All 450 compiler commands use verified ccache launchers, with checkout-root
normalization and before/after statistics. Two cacheable compilations miss;
cache usage is not presented as a cache-hit speedup.

The native discovery/WB test executable is byte-identical to the prior
sanitizer-qualified
`5956bd132175d701358722b8b7a204431a94962c84aec0011917af6a7f948c0e`.
The first deployment, finite n1 Slurm 5616, passes both native component targets
and all ten claim tests, then fails before force evaluation because the adjacent
`toolchain/ptxas` symlink was not copied. All seven endpoints fail; no numerical
qualification is inferred. The failed job/logs are preserved under
`failed-attempt1/`. The missing assembler is restored from the existing qualified
toolchain, with no source/library change. Finite n1 Slurm 5618 passes both native
targets and all ten claim tests, including synccheck/racecheck. All seven
independent full E/F/displaced-energy/stale-state cases pass with joint maps,
and all seven pass with force-cache allowance zero. Each mode checks 66
successful native calls and 467 XC submissions; every successful call actually
selects SCF maps. `qualification-verified.json` binds all source/binary identities,
raw test/work logs and the successful Slurm outcome. Suite durations are not
performance measurements. Sequential 24/96-atom OFF/ON timing now follows in
the same allocation; older timings are not assigned to this library.

An additional 65 host resource/cache/lease checks pass, with 34 real-device cases
explicitly skipped in that host command. The first host attempt retained 64
passes and one C++ compile failure because `/tmp` was full; the unchanged suite
passes using `/data` temporary storage and the ccache compiler wrapper. The
environment failure and its original log are retained.

All ten 24-atom OFF/ON cold/priming/warm pairs subsequently pass the independent
verifier, with errors below 2.388e-12 Eh / 4.920e-10 Eh/Bohr, one-iteration warm
results and reference XC on GPU. Unindexed/indexed native warm medians are
27.385214 / 26.819840 s versus their paired reference 28.489713 / 28.446338 s.
Complete cold is 249.422527 / 249.021745 s versus reference
120.385531 / 120.059483 s; both native cold solves take 18 iterations. These
results belong to frozen 4079/6dba, not the newer qualification below. The
96-atom sequence continues without restarting or changing that source.

The frozen 4079/6dba 96-atom indexed variant subsequently completes in Slurm
5618. Its five independent E/F pairs pass, with maximum errors below
1.151e-10 Eh / 3.205e-10 Eh/Bohr, one-iteration priming/warm calls and GPU
reference XC. Native/reference warm medians are 375.332872 / 411.107088 s;
complete cold is 3989.496849 / 1866.058777 s with 24/16 iterations. The 8.70%
warm time reduction belongs to this whole frozen composition. Its same-binary
unindexed 96-atom control is still running, so no isolated indexed percentage
is inferred from this pair. The independent one-variant receipt is retained as
`results/matched96-indexed-verified.json` in that campaign's ignored directory.

Slurm 5618 subsequently completes the same-binary unindexed control. All ten
96-atom OFF/ON cold/priming/warm pairs pass independent verification. Unindexed
native warm samples are 392.764871 / 393.840803 / 393.344613 s; its median is
393.344613 s versus reference 411.146876 s. The indexed 375.332872 s median
therefore uses 4.58% less warm time than its unindexed control, with equal
one-iteration replay work. Maximum errors across both variants are below
1.151e-10 Eh / 3.245e-10 Eh/Bohr, and reference XC remains on GPU. Unindexed
complete cold is 4004.997612 s versus indexed 3989.496849 s; both native solves
take 24 iterations, while their references take 16 and cost about 1865–1866 s.
The single cold samples do not establish a robust cold benefit. The completed
two-variant receipt is `results/matched96-verified.json`; all measurements keep
the frozen 4079/6dba source and library identities above.

## Master 9c54107ca and spin-boundary regression repair

The integration advances to actual master 9c54107ca via merge 5ac08497e.
Its new scalar eigensolver selection is explicitly CPU primary-HF only;
the prior CUDA timing campaign remains frozen and is not relabeled.
Independent diagnosis of the parent B3LYP regression identifies cancellation
in the shared VWN spin interpolation, rather than an AO-map indexing defect.
The exact fix from #1794 forms spin fractions from the supplied densities
before differentiation, without changing the formula or error gates. See the
[numerical decision](../implemented/numerics/2026-10-04-vwn-spin-fractions.md).

With this patch, the complete original native DFT CUDA regression now passes.
Finite n1 Slurm 5625 also passes 168 independent point/VWN cases, including
52 real-device boundary cases, and seven complete independent WB97M-V
energy/force/displaced-energy/stale-state tests with joint maps plus indexed
forces (66 successful native calls, 467 actual XC submissions). This is fresh
qualification of the new composition, not a cold/warm timing result.
The 1355-input source identity is
`ebdf07921464440e085b2925a1bd061ba9090788485d1cab01e3cada7122814c`;
library SHA-256 is
`5a1b86cef1c0e978118d3024dc36861ebe8cd326dee8a6fe944a6b34000a5461`.
All 451 compiler commands use verified ccache launchers. The qualification
receipt binds source, binary, logs, successful job outcome and actual work.

## Master dc6ea9940: fresh larger comparison

Merge 49f0f9bc0 incorporates master dc6ea9940's evidence-consumer changes.
All 1355 native build inputs remain byte-identical to the qualified master-9c
plus VWN repair above. All hooks and 54 evidence-consumer/source-registry
host tests pass. This establishes source equivalence for the existing library;
it does not relabel any earlier performance measurement.

Finite n1 Slurm 5640 completes a fresh 24-atom unseeded indexed endpoint with
this binary: warm 26.122775 s versus reference 27.374984 s, complete cold
240.379900 s versus 116.000208 s. All five independent pairs pass, every
warm/priming call takes one iteration and reference XC reports GPU execution.
The separate GPU preliminary-density controls and their complete cost are
recorded in the [cold experiment](2026-10-04-wb97mv-gpu-preliminary-density.md).

Finite n1 Slurm 5648 now runs the 96-atom unseeded counterpart from the same
frozen source/library and byte-identical measurement driver. It checks source,
binary and script hashes before device work, records construction/preparation
plus first execution, and retains all cold/priming/three warm pairs with the
unchanged 1e-8 Eh / 1e-7 Eh/Bohr gates. This comparison remains in progress.
Its receipts are under ignored `.artifacts/latest-master96-20261004/`; the
older master-1a OFF/ON campaign continues separately without being restarted.

Slurm 5648 subsequently completes successfully. The independent verifier checks
all five cold/priming/three-warm pairs against the unchanged numerical, shape,
convergence, reference-XC and complete-cold gates. Maximum errors are
1.178e-10 Eh / 4.850e-10 Eh/Bohr; every priming/warm endpoint takes one
iteration and every reference XC component reports GPU execution. Native warm
samples are 362.318830 / 362.283309 / 362.391458 s, versus reference
395.467209 / 395.366895 / 395.434545 s. Their medians are 362.318830 and
395.434545 s: the whole indexed composition uses 8.37% less warm time.
This is not an isolated indexed-scheduling speedup.

Complete cold is 3691.344927 / 1792.691183 s, with 23/16 native/reference
iterations. All construction/preparation and first execution costs remain
included; native cold is still about 2.06 times the reference. Actual force
and SCF point-AO-square work are 0.058249 and 0.054380 of their dense counts.
The force traversal covers 2,359,296 points in 9,216 tiles, including 768
empty tiles, and still performs both geometry consumers. The verifier receipt
binds raw report SHA-256
`3ac5483e156672ee490315d1928ba4bdb8dad4e02fafb5a49ad9b870bcdf0cae`
to frozen source 49f0f9bc0, identity ebdf0792 and library 5a1b86ce above.
These completed results do not replace the distinct six-point README source.

## Master 837c2a51c: shared planner integration

Merge 678f7eb88 incorporates actual master 837c2a51c, including #1773's shared
budget-aware tile search and the now-merged #1794 VWN repair. Only three native
inventory paths differ from the previous composition: the ordinary stationary
runtime and the stationary/composite resource-planner modules. The new identity
is therefore distinct, even though the composite candidate order and mathematical
sources are retained. All 350 relevant host planner/resource/merge tests pass.

The new 1355-input identity is
`d198acd713d7c4218bb4cacd9acbd7eb646d5fc2b89c77d04bdb47901c74ad36`;
library SHA-256 is
`fe826ad7eb9293686d5bb3b5e97456f3a9c884a13256603e7573ea3a1703f25b`.
Verified ccache 4.5.1 launchers appear in all 451 compiler commands; retained
before/after statistics show six hits and one miss during this incremental build.
The native DFT test executable remains byte-identical to the master-9c/VWN
qualification, while the rebuilt library advertises the new source identity.

Finite n2 Slurm 2181 on RTX PRO 6000 passes the full native DFT regression and
seven independent complete WB97M-V E/F/displaced-energy/stale-state cases in
each of joint-map and zero-force-cache modes. Each mode verifies 66 successful
native calls and 467 actual XC submissions, with SCF maps selected on every
successful call. The independent post-run receipt binds source/library/script
hashes, actual work, device receipt, raw logs and successful Slurm completion.
These are numerical qualification results, not RTX 5090 timing claims.

Finite n2 Slurm 2182 now runs a complete unseeded 96-atom cold/priming/three-warm
comparison using this exact source/library. It depends on successful 2181 and
also checks both completed endpoint/work receipts before starting. Measurement
drivers are byte-identical to the frozen master-dc 96-atom campaign; only the
source/library/hardware receipts and orchestration differ. This RTX PRO 6000
measurement remains pending and cannot be mixed into the RTX 5090 README curve.
The n1 master-dc and master-1a jobs continue under their own frozen identities.

Slurm 2182 subsequently completes successfully on RTX PRO 6000. All five
96-atom cold/priming/three-warm pairs pass the independent unchanged E/F,
convergence, on-GPU XC and complete-cold gates. Maximum errors are below
1.169e-10 Eh / 3.198e-10 Eh/Bohr. Native warm samples are
342.975112 / 342.057653 / 342.122510 s, versus reference
368.990720 / 368.969166 / 369.130887 s. The 342.122510 / 368.990720 s medians
give 7.28% less native warm time on this GPU. Complete cold remains slower:
3620.209206 / 1679.026175 s with 24/16 iterations. Every priming/warm endpoint
takes one iteration. This qualifies the frozen d198acd7/fe826ad7 composition's
large complete endpoint; it does not add a PRO 6000 point to the 5090 figure.

Ignored `.artifacts/master837-20261004/` retains build, cache, source archive and
all qualification receipts; `.artifacts/master837-pro6000-96-20261004/` retains
the new complete timing campaign. No existing endpoint is relabeled.

## Master resident-component merges without changed build inputs

Merge `7bbbd9c54` incorporates actual master `79418329e`, including the reviewed
resident AO-column discovery and geometry-bound map-cache components. Those
component changes already exist in this integration: the merge tree has no
file-content difference from its first parent. Recomputing the complete
1355-input manifest preserves identity `d198acd7` above exactly. The existing
`fe826ad7` library and Slurm 2181 qualification therefore still describe every
native/compiler/Python build input in this composition; no rebuild or repeated
device test is represented as necessary or performed for this history-only
merge. The ignored `.artifacts/master794-20261004/input-identity.json` records
the full hashes and this scope. All running benchmark snapshots retain their
original commits and timings.

Merge `227f76fdd` then incorporates actual master `d9431c913`, the reviewed
README publication from #1786. Recomputing all 1355 native inputs again gives
the same d198acd7 identity; the changes are benchmark evidence and its consumers,
not native inputs. The existing build/qualification retains its original source
commit. The README now contains the six-point HF-style warm figure after
current-head publication LGTM and successful CI. Its explicitly identified
0132d758/b82e4686 candidate remains distinct from these newer measurements.
The merge identity receipt is in ignored `.artifacts/masterd943-20261004/`.

## Readiness boundaries

The earlier parent B3LYP failure is superseded by the full-regression result
above after the independent #1794 fix. The latest native-input composition now
has full regression, both independent fallback qualifications and a completed
96-atom endpoint comparison. Against master d9431c913, the combined retention
check passes the unchanged 64 MiB limit; the explicit change review contains
17 evidence files / 240,651 bytes, below the unchanged 2 MiB limit. The
source-bound checker receipt is retained in `.artifacts/masterd943-20261004/`.

These author-side gates allow final integration review. Component/overlap
disposition, current-head CI and a real current-composition LGTM remain required
before merge; the earlier scoped force-composition LGTM does not cover the new
consumers. The separate 96-atom preliminary-source experiment remains private
and pending. Neither a public CUDA preliminary API nor unsupported public
WB97M-V resource admission is promoted by this integration.

Ignored `.artifacts/scf-ao-pages/` retains every indexed/unindexed attempt and
the independent per-variant verifier. `.artifacts/indexed-latest-20261003/`
retains frozen source, build/cache receipts, transfer hashes, scripts, Slurm
records and new qualification. README #1786 remains the reviewable publication
of the completed unindexed joint comparison and its HF-style warm figure;
complete cold results remain explicit in the detailed evidence. The new
`.artifacts/vwn-spin-boundary-20261004/` retains the master-9c build and full
regression receipts.

## Master b909e14c1 follow-up

Master subsequently merges reviewed #1801, which skips unconsumed resident
PSSS packing only for Direct J/K and repairs standalone native DFT test linkage.
The integration takes that exact packing policy. The only merge conflict is two
wordings of the same XC-versus-grid ownership comment; retain master's wording
and the identical four-double expectation. The existing native-input identity
is now superseded for this union, so earlier endpoint records retain their
frozen inputs. A new ccache build and finite full-native/independent fallback
qualification are required before this newer composition is merged. The earlier
large endpoint remains evidence for its original source, not a new timing claim.

The frozen union 3d5323471 subsequently completes n2 Slurm 2190 on RTX PRO
6000: the full native DFT regression and all seven independent complete
E/F/displaced-energy/stale-state cases pass in each of sparse and zero-force-cache
modes. Each mode again records 66 successful native calls and 467 actual XC
submissions. There are no skipped qualified cases. All 451 compiler commands
use ccache. Native identity is
`7d94e015d9b63017c8dd01e16e91f3be327d92c02d513085d4e06d77eaf834bf`;
library SHA-256 is
`90d74009910c1f949a39dff7f802faa85abc1eb1e6c06104b411cc5602ea4864`.
The bound qualification receipt lives in ignored
`.artifacts/masterb909-20261004/qualification-verified.json`. Initial Slurm 2189
passed native regression but could not compile generated forces because the
copied toolchain directory lacked the ptxas link; preserve its unsuccessful
attempt before copying the complete existing toolchain and rerunning. This is
numerical composition qualification, not a fresh timing campaign. Current-head
CI and an actual current-composition review remain merge gates.

## Master 73755342d integration

This union subsequently includes #1800's CPU KS final-maximum closure and the
shared optional convergence-eligibility predicate from master 73755342d. The
merge has no conflict in the AO/index or force consumers. A fresh source-bound
build and qualification cover this composition; b909 and earlier numerical and
timing receipts retain their original identities. The evidence-verifier review
finding is fixed separately: all 33 acceptance predicates are unchanged but
remain active under Python optimization, with coherent bad-hash, failed-native-
qualification and inaccurate-energy controls rejecting under `python -O`.
