# Decision: measure the complete fresh-process PBE endpoint

Status: implemented; host-validated tooling, GPU qualification not run
Date: 2026-10-16

## Problem

The retained paired producer reports constructor, preparation and initial E+F
segments separated by interleaving and provenance work. Their sum cannot prove
continuous process-cold profitability. A warm owner also retains density and
CUDA state which cannot represent a cold solve.

## Decision

Launch a new interpreter for each arm/geometry/observation. Measure parent
wallclock through successful child completion, including imports, synchronized
E/F, provenance and cleanup. Preserve earlier result delivery separately.
Keep AOT assets and OS caches in their actual state; do not claim filesystem
cache coldness. A finite timeout reaps only the collector-owned child group.
Malformed output, failed or missing children retain failed observations.

Freeze the SDK Git tree, complete native identity, actual library, collector,
independent consumer, basis and reference. Separate collector tools from frozen
SDK production modules so adding qualification tooling does not modify an
active build. Recheck the inputs and native inventory after the campaign.

## Invariants

The method is full FP64, single-stage PBE0 on the original 48/96-atom protocol.
Use fresh density, explicit SCF/force256 and original absolute phased counts.
Require actual convergence, finite complete histories, observed J/K counts and
equal semantic work. Missing counters do not mean zero work. Preserve original
independent vector tolerances. One feasibility pair cannot satisfy the five
pairs per geometry or profitability gate. Clean timing does not replace NCU
dynamic local traffic or retained-owner warm/moved qualification.

## Evidence and consequences

Host tests exercise real subprocess lifetime, included post-delivery work,
timeout/malformed/nonzero exits, SDK package shadowing, frozen input bindings
and independent consumer negative cases. Immutable published vectors appear
only in explicitly synthetic parser fixtures, never as new cold observations.
No GPU run or performance promotion is implied by this tooling.

## Review correction

The initial host tests did not cover an existing output file, Python optimization
or actual precision/origin metadata. Review controls reproduced truncation of
existing evidence/input aliases and acceptance of invalid evidence under `-O`.
Reserve a new output exclusively, keep its descriptor for journal updates and
preserve the primary error when journal writing or closing also fails. Explicitly
reject optimized Python in both collector and consumer.

Retain and validate native precision, operator timeline, initial guess and restart
origin. A fresh MINAO/Hückel guess can legitimately set `initial_density_used`;
requiring that flag to be false would change the intended default cold protocol.
Require a cold origin with complete guess work and no discarded target attempt,
retry, refinement or provisional operator. Compare actual Fock/timeline/operator
counts, including finalization work. Synthetic fixtures remain explicitly
synthetic and do not relabel the retained vector population as a cold measurement.

Refs #2072 and PR #2250; current usage is described in
`docs/developer/xc_native_cuda.md`.
