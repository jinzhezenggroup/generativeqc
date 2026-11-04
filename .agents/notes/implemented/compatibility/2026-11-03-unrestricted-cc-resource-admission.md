# Decision: unrestricted CC admission is resource-based, not a fixed AO count

Status: implemented
Date: 2026-11-03

The remaining force, CUDA and DF work recorded below is superseded by
[the unrestricted response decision](../numerics/2026-11-03-unrestricted-cbs-analytic-cuda-df-response.md).

## Problem

Both public UCCSD validation and the solver's capacity query rejected more than
twelve AOs. The generated equations already accept independent runtime alpha
and beta occupied/virtual dimensions. The numeric-capacity/work estimator also
already uses checked shape arithmetic before the UHF reference is constructed.
The dimension cutoff excluded otherwise admissible systems without identifying
a missing mathematical or integral implementation.

## Decision

Remove both hard-coded AO-count checks. Retain orbital-occupation validation,
all-electron s/p/d operator admission, checked arithmetic, complete endpoint
numeric-memory admission and finite worst-case scalar-work admission. Removing
an arbitrary dimension cutoff does not mean unbounded storage or execution.

## Evidence

`tests/python/test_uccsd_public.py` compares native UCCSD and UCCSD(T) with
independent PySCF on both a thirteen-AO He reference and a thirteen-AO Li
doublet. Supplied primitive shells are the only shared numerical inputs.
`tests/python/test_cbs_open_shell_native.py` additionally exercises a complete
Li-doublet CBS with a thirteen-AO correction and fourteen-AO upper basis,
including a nonzero correlation correction. Its energy gate is 1e-8 Hartree.
Large systems with an insufficient explicit byte budget still fail before SCF.

## Remaining request

This change removes the AO cutoff; it does not implement unrestricted analytic
forces, CUDA execution or density fitting. Native UMP2/UCCSD currently retain
separate CPU/conventional admission. The unrestricted gradient prototypes under
`tools/generativeqc_cc/` rely on the small reference export bridge and an exact
`ConventionalProvider`; they cannot be relabelled as scalable production DF/CUDA
endpoints. The existing native DF CC sources bind one restricted occupied/virtual
space and likewise cannot substitute for spin-labelled unrestricted sources.

Completing the full request requires native unrestricted CC/Lambda/triples and
orbital response, GPU graph execution, and spin-specific DF factors with their
metric/center cotangents. Each path needs independent complete-energy/analytic-
force gates, finite-difference checks of reconverged independent references,
failure publication/resource tests and complete endpoint qualification. A
production CPU/reference fallback or numerical force would not satisfy it.

The local Slurm cluster is `node3`, whose only GPU node is in DRAIN state.
This is not global unavailability: SSH queries found idle Slurm GPUs on allowed
hosts `n1`, `n2`, `n4` and `n5`. CUDA work must use remote `srun` there, with an
explicit finite time limit and the host's matching GRES (`gpu:5090:1` on n4/n5,
for example), preserving assigned device visibility. Do not use the maintenance
node or infer the remote fleet's state from local `sinfo` alone.

## References

Supersedes the twelve-AO admission statement in
[initial CBS companion ownership](../architecture/2026-11-02-open-shell-cbs-native-companions.md).
