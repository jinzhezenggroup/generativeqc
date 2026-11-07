# Proposal: exact DPPP pure-J pair contraction and complete-mask proof

Status: proposed; default off, completed5090 endpoint gates pass, full qualification incomplete
Date: 2026-11-04

## Problem

The scalar DPPP force consumer forms 162 independent AO-component weights even
when pure Coulomb work has an exact ordered-pair outer product. A partial AO
Schwarz mask destroys that representation. Enumerating every AO quartet merely
to establish complete admission also repeats avoidable Cartesian-product work.

## Candidate

The existing weighted-ERI compiler can project two complete ordered pair arrays
before Coulomb contraction and differentiate the resulting scalar through its
existing FP64 physical leaf rules. This is not a low-rank density approximation:
each pair retains its directional density, diagonal/off-diagonal folding,
normalization and canonical orientation. Arbitrary-weight and mixed roots retain
their existing construction and derivative rules.

Native dispatch admits only DPPP, full-range pure J, strict FP64 and no bilinear
seed, behind `GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_PAIR_SEPARABLE=1`. There is no
GPU-name selector. The ordinary template does not own pair-array storage.
Partial masks or unsupported/nonfinite regrouping decline before any force
publication and retry the original strict nuclear consumer exactly once.

For finite nonnegative Schwarz factors, FP64 multiplication is monotone, so the
minimum product proves complete admission with the same strict `<` comparison.
The same-pair triangular domain contains its minimum-factor diagonal. Negative,
NaN and infinite factors retain the original per-quartet comparisons. Overflow,
underflow, zero, empty domains and nonstandard tolerances retain comparison
semantics; no screening threshold changes.

Diagnostic pair-factor loads, product proofs and exceptional fallback decisions
are distinct from actually executed AO-quartet comparisons. Diagnostic sampled
thread cycles are intrusive overlapping work, never device or endpoint time.

## Isolation and evidence boundary

This proposal is isolated on master `c776b6cc12855b68195e4be27d0604b1b524690e`.
It excludes the retained reverse-AD, CTA MD-radial, projection-cache,
relative-sum, density-potential, warp and pipeline experiments. It does not
reopen already merged #2959 queue work or change the production default.

Historical A100 screen225398 measured research source117324, not this isolated
PR source. Both old/new arms had pair contraction and the other research
controls enabled. Exact-mask v2 changed r2SCAN-3c complete warm E+F median from
7.475630 to 7.294360 seconds; fresh GPU4PySCF was 5.718142 seconds. The remaining
gap is 1.576218 seconds, not solved by the proof. Its separate census changed
623577888 actual AO comparisons to zero, with 107266704 pair-factor loads and
4013424 proof decisions; actual admission and primitive work stayed unchanged.
PBE0 performed zero candidate work. These are neither a current-master paired
gain nor hybrid K-prime improvements nor this PR's production qualification.

The source and native ELF of that normal SM80/O3 package are respectively
`1173240300ea67dc5de6f67a9afb743abdefcfe5d3d1e86a64fbb63ad7afaeb4` and
`7ad196e063212694692923307b45a927afa8ee2f0fce52bf2b2d473736749e05`.
The original screen callback SHA256 is
`c361707addff7445a9cbc9915280dc9d36498f9bd0d06c8c70e313d17f873f40`.
Full original receipts remain in the ignored research worktree artifacts;
substantive decisions and measurement identities are synchronized to #2976.

## Acceptance and next priorities

Required host gates exercise actual native admission, canonical orientation,
directional RKS/UKS pair weights, independent libcint energy derivatives,
coincident centers, signed scaling, masking, exceptional inputs and publication
fallback. They do not substitute for normal production CUDA endpoint/resource,
race/sanitizer, moved-geometry, current-base or physical RTX 5090 qualification.

The isolated source passed222 host/compiler/oracle tests across
`test_direct_scalar_separable_host.py`, `test_weighted_eri_separable.py`,
`test_direct_force_diagnostic.py` and `test_weighted_eri.py`. Compiler ownership
checks covered584 modules with zero dependency errors; all six changed Python
files passed Ruff lint and format checks. The initial extraction omitted the
pair ABI signature stanza; actual native compilation caught it before any
publication, and the corrected complete run passed without skips. Initial
failure logs are retained rather than relabeled passing.

On 2026-11-04 the local Slurm controller exposed only node3/main/gpu:5090:1,
drained for maintenance. Subsequent authenticated strict-host-key SSH checks
found independent idle RTX5090 controllers on the authorized n1/node1,
n4/node4 and n5/node5. The earlier global availability inference was wrong;
node3 remains prohibited, but physical qualification is possible on n1 using
its retained benchmark environment and finite `srun` allocations. Default-off
is not a proof of non-regression. Normal SM120 builds of both the isolated
candidate and its exact master base are being prepared for matched execution.

The next attribution targets are the remaining r2SCAN-3c complete warm gap and
Direct WB97M-V cold work, not another mask microbenchmark. Retained Direct
224903 WB97M-V observations were 163.920156 versus reference88.569923 seconds
cold, but 21.746654 versus28.590500 seconds warm. Native cold used28 iterations,
29 Focks,29 J/58 K applications,23 provisional/5 target nonlocal Focks and4 pair
sweeps. These actual Direct receipts must be distinguished from the separate
fitted investigation and are not current-source evidence.

Revisit default admission only after complete endpoint/non-regression evidence
on the actual isolated source, including both A100 and physical RTX 5090.

## Follow-up225399: actual Direct WB97M-V and endpoint attribution

The single exclusive A100 gpu22/gpu_chem allocation225399 completed exit0 on
2026-11-04 18:36:13–18:49:23 +08:00. Original callback SHA256 is
`c4a66985d2e042ab2b44207d8f5efe07d2209d35289ef8492f14f168295e4b81`
(38,872,833 bytes). The authenticated research source117324/ELF7ad196 is
unchanged: this is not the isolated PR's performance evidence.

Independent complete WB97M-V cold E+F observations were native151.124412 and
fresh GPU4PySCF89.192967 seconds; three same-owner own-density warm medians
were17.046731 and28.499975 seconds. The cold disparity is61.931445 seconds;
native warm is11.453244 seconds faster. Native cold executes28 iterations,
29 Focks,29J/58K,23 provisional/five target nonlocal Focks and four SCF pair
sweeps. Reference cold executes10 iterations/11 effective-potential builds.
Warm native executes one iteration/Fock,1J/2K and zero SCF pair sweeps. The
seven preliminary native Huckel Focks remain separate from target counts.
Independent all-observation-pair energy/atom, max-force and RMS-force errors
are6.779e-12,2.117e-7 and7.794e-8, passing unchanged1e-5/1e-4/3e-5 gates.

Exact matching Nsight Systems2025.1.3.140 imported the original cold/warm
WB97M-V and warm r2SCAN-3c captures offline. Profiles cover native execute,
not constructor/prepare or the complete public endpoint. Instrumented owners
are not extra clean timing observations. Kernel sums below may overlap and
are neither endpoint wall-time differences nor reference-side attribution:

- r2SCAN warm:7.401699 seconds of kernel sums; two-electron forces4.027791,
  grid/XC/geometry2.123596, Direct J0.325341, dispersion0.333081 and
  one-electron force0.187075 seconds. The low/order<=5 spatial consumer costs
  1.863326 seconds and cooperative order-six costs1.527931 seconds. The low
  consumer includes multiple classes: do not label all1.863326 as DPPP.
- WB97M-V cold:149.375301 seconds of kernel sums; J/K84.535040, VV10
  energy/gradient packets29.665941, grid/XC/geometry19.262378 and
  two-electron forces8.842986 seconds. Resident canonical J/K alone costs
  19.196077 seconds across420 launches. Five VV10 energy packet launches
  cost21.295175 seconds and one gradient packet costs8.370766 seconds;
  executed packets are not interchangeable with the four SCF sweep counters.
- WB97M-V warm:17.051911 seconds of kernel sums; two-electron forces8.942246,
  grid/XC/geometry5.479752, J/K1.896090 and VV10 packets about6 microseconds.
  The strict bounded hybrid force consumer alone costs7.493144 seconds.

The next substantive targets are therefore the strict order-six/pure-J force
consumer and low-order force ownership for r2SCAN, and WB97M-V cold J/K plus
nonlocal/solver work, with the bounded hybrid force consumer as its warm
target. This DPPP-only pure-J PR cannot solve hybrid K-prime or the cold
SCF-work disparity. Do not rescan the already rejected VV10 factor0.1 policy:
`implemented/performance/2026-10-18-vv10-inner-default.md` records that fewer
provisional Focks but more target/pair sweeps worsened the complete endpoint.

The local full normal build failed with root filesystem full; its owned Ninja
was interrupted and failure logs retained. Only owned build/artifact paths
were moved to `/data`; no shared cache was cleared and no unrelated process
was stopped. The n1 build reuses verified persistent sccache0.16.0, normal O3
and stationary AOT auto. A first n1 configure failed before compilation
because `.local/bin` was absent from the explicit PATH; the corrected build
retains that failure receipt. Physical screen results are recorded separately
when complete, never inferred from the old research source.

## Isolated-source RTX5090 screen15845 and default decision

Both actual normal SM120 builds completed exit0. Candidate/master native ELF
SHA256 values are respectively
`81fa4a1f0999387ab342083bff731bed4589330f4354c6bf7e8f1d0a17e19acf` and
`3b652ca2a2abc2b61569cc60f8393b492c773c2d31c04d73c77f22636a70a582`.
Candidate source is
`d6d8b71bfdca861247411ecc702e146c040d3217d23fb0e130749a29528b4daf`;
master source is
`66ac1beee51b8a0e1ae0d5def0550a0694a7046e14fd875ef3028a8086d3436b`.

One finite30-minute n1/main/gpu:5090:1 `srun` allocation15845 ran on node1
from20:03:53 to20:16:51 +08:00 on2026-11-04, retaining assigned visibility.
Its original329,796-byte archive SHA256 is
`f700d9f781bbc29260757122c2a5cdd1ce8a8ce0f19163d4bbd3445db2cf0bef`.
All eight clean complete-endpoint stages, the separate candidate census and
memcheck completed exit0. Racecheck hit its predeclared240-second deadline:
its stage/allocation/client exit is124. This is not a passing full qualification
or a reported race-free result. No repeat allocation was submitted.

Water96 complete E+F observations, one cold and three own-density warm per arm:

| Method / seconds | Exact master | Candidate off | Candidate on | Fresh GPU4PySCF |
|---|---:|---:|---:|---:|
| r2SCAN-3c cold |30.360071|30.399601|30.351788|32.557462|
| r2SCAN-3c warm median |11.113714|11.237232|11.023791|14.843803|
| PBE0 cold |16.352179|16.467007|16.497684|20.302583|
| PBE0 warm median |5.143053|5.152634|5.147925|6.700412|

Default-off/master warm ratios are1.011114 and1.001863: both pass the
predeclared5% endpoint screen, but this is not a zero-regression/statistical
claim. Pair-on/off warm ratios are0.981006 and0.999086. The roughly1.9%
r2SCAN gain versus same-binary off is only about0.81% versus actual master;
it does not justify promotion or a broad speedup claim. Cold is effectively
unchanged. Do not attach historical A1002.42% mask-only evidence to this table.

All-observation-pair independent gates pass: maximum r2SCAN energy/atom,
max-force and RMS-force errors are3.880e-12,1.350e-6 and4.497e-7; PBE0 errors
are2.790e-12,4.542e-7 and1.984e-7. Native paired max-force errors are below
9.660e-12 and4.264e-13 respectively. Native scientific settings and semantic
SCF work match all three arms: r2SCAN cold10 Focks/10J, warm1 Fock/1J; PBE0
cold9 Focks/9J+9K, warm1 Fock/1J+1K. The actual loaded reference derivative
image is authenticated with the same `ff4c73...` hash as225399. Native source,
packaged images, actual mapped library/inode and blocked oracle imports are
authenticated outside endpoint timing.

The separate two-execution census records8,026,848 attempts,7,905,888 accepts,
120,960 partial-mask refusals, zero numerical refusals, zero executed AO-product
comparisons,214,533,408 pair-factor loads and8,026,848 pair proofs. Primitive
products are25,237,056. Memcheck reports zero errors and its own census proves
228 accepted candidate shells. Racecheck produced no completed summary and
does not qualify safety. The original failing all-gates audit is retained;
a separately named completed-endpoint diagnostic passes numerical/work and
5% timing gates while explicitly keeping full qualification false. It is not
a post-hoc relaxation or passing replacement of the original acceptance.

Decision: keep the experiment default off and the PR draft. Clean-source A100,
completed race/resource/fallback/moved-geometry and broader timing gates remain
open. Do not spend another allocation to polish this small win while the large
force and WB97M-V cold owners remain the substantive targets.

Source inspection adds an important WB97M-V follow-up constraint: explicit
incremental Direct J/K accepts an exact RSH correction, but activating it
disables the frozen-VV10 initializer/outer schedule. A future delta experiment
must separate ordinary full-target/no-incremental and exact incremental arms
from the current default; a two-arm toggle would confound solver/nonlocal
changes with J/K savings. Keep coarse-screening/proposal relaxations disabled
for a same-cutoff study and retain fresh complete-target closure. This is a
design constraint, not a measured incremental speedup or default decision.

## Superseding qualification

The [seven-class complete-pair decision](../implemented/performance/2026-11-06-direct-complete-pair-projection.md)
supersedes the DPPP-only candidate after new scheduled A100/5090 complete
endpoints and independent science. It does not retroactively qualify this
earlier small/inconclusive experiment or settle WB97M-V hybrid performance.
