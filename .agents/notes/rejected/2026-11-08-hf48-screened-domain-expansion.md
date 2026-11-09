# Decision: do not expand cold screened HF to the 384-AO domain

Status: rejected
Date: 2026-11-08

## Problem

HF48 defaults miss the additional independent Fmax1e-7 diagnostic. The existing
large-owner screened-FP64 proposal reaches a stricter target tail, suggesting a
possible correctness and performance fix through broader admission.

## Evidence and decision

Historical source9e292fde/ELF0bffef29, Slurm16823 and LSF229218 each ran four
fresh-process ABBA complete energy, analytic-force and host-export endpoints
plus three own-density-frozen warms. The candidate's15 Focks versus10 ordinary
Focks fix the force diagnostic (about1.1843e-9 versus1.8184e-7 Eh/Bohr), but
complete cold regresses about5.6% on both cards. Some A100 paired warms also
exceed3%. Neither stronger accuracy nor a flat median warm substitutes for
the declared complete endpoint and every-pair regression guards.

Do not promote broader automatic admission or ship an opt-in as a performance
success. Preserve both the default's independent failure and the candidate's
timing losses. The source predates capability-based XY dispatch; these timings
must not be relabeled as current-head qualification.

## Revisit conditions

A materially different handoff or work-reduction mechanism may be evaluated
independently. In particular, retaining bounded same-precision DIIS predictors
is not this experiment: it must rebuild physical target Focks, reset the physical
stop baseline and independently meet all scientific, resource and endpoint gates.

Full observations and replay:
`benchmarks/results/hf48-screened-proposal-rejection-20261108/`.
External memory: issue2976; draft PR3048.
