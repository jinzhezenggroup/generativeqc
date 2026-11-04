# Proposal: exact DPPP pure-J pair contraction and complete-mask proof

Status: proposed; default off, physical RTX 5090 qualification unavailable
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
drained for maintenance. The standing user prohibition on n3/node3 means no
5090 test or default promotion is authorized by that availability. Default-off
is not a proof of non-regression; this candidate must remain a draft until the
missing qualification is obtained.

The next attribution targets are the remaining r2SCAN-3c complete warm gap and
Direct WB97M-V cold work, not another mask microbenchmark. Retained Direct
224903 WB97M-V observations were 163.920156 versus reference88.569923 seconds
cold, but 21.746654 versus28.590500 seconds warm. Native cold used28 iterations,
29 Focks,29 J/58 K applications,23 provisional/5 target nonlocal Focks and4 pair
sweeps. These actual Direct receipts must be distinguished from the separate
fitted investigation and are not current-source evidence.

Revisit default admission only after complete endpoint/non-regression evidence
on the actual isolated source, including both A100 and physical RTX 5090.
