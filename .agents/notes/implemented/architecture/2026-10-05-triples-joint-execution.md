# Decision: prepare occupied-triples W through a joint lowering portfolio

Status: implemented
Date: 2026-10-05

Extended by [optional cuTENSOR triples execution](2026-10-05-triples-cutensor-execution.md),
which adds a third provider and candidate-dependent resource admission while
retaining the scientific precision gates described here.

## Problem

The qualified W-FP32 work in #1864 coupled a CC-local precision enum directly
to SGEMM. Keeping that interface would make precision admission select a
vendor, and prevent generated execution from implementing the same request.
The existing strict path also dispatched raw BLAS callbacks from the method.

## Decision

Project the existing occupied-triples W TensorIR region into one canonical
request with four candidates: strict/mixed arithmetic crossed with library/
generated execution. The existing scientific equation, W-root identity and
qualification `issue1764/df-triples-w-fp32-candidate-v1` remain authoritative.
This is an AOT template; concrete extents and padded strides are validated by
the shared native executor before work begins.

The generated WExecution owns casts, contraction descriptors, provider context
and the strict/mixed bound launcher. Native triples supplies a shared scientific
PrecisionDirective and borrowed canonical inputs. It builds panels and W cubes
through the prepared object; it neither chooses a vendor nor dispatches dtypes.
Panel contractions remain FP64. In the admitted alternative, W products use
FP32 storage/compute/accumulation with explicit casts and FP64 combination.
V, denominators, epilogue and final reduction remain FP64.

The candidate set has no complete phase cost measurements. Selection explicitly
retains the qualified incumbent after legality gates; unknown timing is not
zero and cannot support a promotion. No per-tile plan or algorithm search runs.

## Resources and fallback

Admission charges the numeric arena, bounded host descriptor/algorithm tables
and the selected provider allowance. Shared BLAS uses no separate workspace.
Generated execution needs no handle or 96 MiB library allowance. Charging that
allowance to every candidate initially made the low-budget fallback ineffective;
endpoint tests now guard the actual minimum generated requirement.

The finite resource order tries one panel, generated execution at the admitted
precision, then strict generated execution. Optional provider preparation failure
can also retain precision with generated execution. Scientific execution failures
propagate without replay. Diagnostics report actual precision and schedule IDs,
provider/algorithm/version, cast and summand counts, resource fallback and retained
incumbent status. Host binding bytes are part of the published complete budget.

## Evidence

On n1, CUDA 12.9/RTX 5090 via Slurm main/gpu:5090:1, 30 tests pass, including
strict and mixed complete energies, generated-provider rejection injection,
independent H2O/NH3/CH4 energies, work accounting, repeatability, invalid inputs,
and both memory fallback levels. Strict tolerances are 3e-12 absolute/relative;
the retained mixed gate is 2e-7 absolute plus 2e-4 relative. Shared native
contraction tests exercise both dtypes/providers and layout/lifetime failures.
Host descriptor checks cover 27 o/v/q combinations including unit extents.
Compiler structure, pinned type checking and pre-commit checks pass. ccache
before/after statistics and endpoint logs remain in ignored local artifacts.

## Boundaries and revisit conditions

This migrates the standalone energy consumer. Legacy triples pullback and
Fock-response callbacks remain tracked by #1890; this note does not classify
them as provider implementations. Capture remains unsupported by the shared CC
binding until replay accounting exists. Complete cold/warm endpoint evidence,
additional provider families and broader production admission remain #1889 /
#1886 work. Revisit incumbent selection only with qualified complete phase costs.

## Rejected alternatives

Keeping the method enum or using a hidden provider Boolean would preserve the
same ownership problem. Rewriting W as a new matrix algebra would fork scientific
identity. Promoting a generated candidate on kernel timing alone would omit
casts, panel work, preparation, resources and result publication.
