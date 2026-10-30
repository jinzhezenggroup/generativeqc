# Experiment: independently qualify existing Rys DF derivatives on SM90

Status: proposed
Date: 2026-10-30

## Problem and measured boundary

The H100 trace at `2241fa6a76698c09573c540b2822316ef7cdb908` executes
polynomial DF derivative kernels. Its intrusive PBE0/96-atom DF capture spends
1994.946 ms summed device time in three-center derivatives; the selected `102`
polynomial/packed kernel contributes 320.475 ms and has static REG=254/STACK=0.
These are kernel attribution and static resources, not clean endpoint speedups,
achieved occupancy or dynamic spill traffic. The exact namespace/template join
has been independently re-executed from the retained dumps and timelines.

The DF derivative production manifest has a qualified SM120 Rys profile and no
SM90 profile. Direct J/K's separately developed portable Rys classes in #2888
do not select the DF three-center derivative lowering. This experiment does
not change that PR's recurrence, one-shot final-U lease or bounded response work.

## Candidate and controls

An **unqualified** SM90 profile enumerates all 23 mathematically available
existing `RYS_SHELL_CLASSES`, with compact subgroups as an initial candidate.
It does not copy SM120 qualification or claim this schedule is optimal for
H100. The existing `GENERATIVEQC_DF_SHELL_POLICY=candidate` control selects it;
automatic and legacy selection retain the incumbent polynomial behavior.
Other architectures, unsupported classes, auxiliary-g, metric response and
range-separated response keep their current behavior. No recurrence,
precision, grid, fit or convergence changes are proposed.

The private preregistered plan is identified by SHA-256
`c6a8e73c84f995b946e8cff5940ffe8eb09c7b27b430d28bfaace0a470babf95`.
It defines 23 independent exploratory populations / 92 complete E+F calls:
HF and PBE0 DF at 48/96 atoms, each with incumbent polynomial, Rys compact,
Rys warp, Rys packed and a fresh same-H100 GPU4PySCF reference; plus three
small PBE0 DF controls. One explicit 1024-point public KS option is fixed
across these controls. Native and reference use unchanged independent oracle
vectors, scientific recipes, complete endpoint timing and finite limits.
All failures and unfavorable results remain visible. Repeats are not votes.

The manifest's generator digest binds the baseline source inventory at
`df4125ab733ab9f8f1af831501bcce5e7dccd938`; the profile digest binds the actual
earlier intrusive resource join. They record preparation provenance, not a
candidate compiled binary or an accepted performance report. Source-matched
build and executed-producer receipts must supply those identities separately.

## Admission and next decisions

No candidate build, H100 Rys execution or default promotion is claimed yet.
Independent full-vector numerical gates must pass before considering endpoint
benefit. A surviving global policy still needs larger/default-method coverage,
small/UKS/moved/lifetime and bounded-resource checks, derivative fixtures and
device sanitizer gates. Final integration must refresh the rapidly changing
main branch and rerun the fixed 30-case, three-process majority protocol.

Do not set `qualified=true` from availability, the SM120 profile, a successful
compile, an isolated class timing or a single favorable endpoint. Do not alias
SM90 to SM120 or select different winners for individual benchmark cases.

References: #2889, #2892, #2893; `docs/developer/df_tuning.md`;
`python/generativeqc_compiler/integral/df_rys_shell.py`.
