# Decision: backend-following detached CUDA electronic properties

Status: implemented
Date: 2026-11-03

## Problem

The initial opt-in property implementation captured the correct accepted HF/KS
state but performed its analysis and point sampling on CPU even for CUDA SCF.
The requested public endpoint needs actual CUDA numerical execution, not a
CUDA solve followed by host property mathematics. This supersedes the host-only
choice in [the original note](2026-11-03-opt-in-electronic-properties.md).

## Decision

CUDA calculators dispatch property analysis to a detached native owner. Packing
uses `AoBasis`'s authoritative normalized public-AO expansions. Analytic position
integrals lift the ket polynomial in the shared CUDA compact overlap evaluator,
including f->g, without introducing a second Gaussian recurrence or changing
normalization. Device kernels compute total density, nuclear/electronic dipoles
and Mulliken populations. The existing symmetric-eigen provider and cuBLAS
square products implement the full-rank symmetric Lowdin transformation.

Point sampling reuses the compiled generated grid owner's FP64 AO and density
contractions, with value-only feature admission. A small same-stream consumer
combines the two spin density rows and checks the producer's sticky error before
publishing values. A Cube call retains that owner across its bounded point tiles;
only grid-coordinate construction and file output happen on host.

The additive C handle owns copies, not borrowed system/context/device pointers.
Callers serialize use, but can destroy the original context/system and use a new
same-device context. Native device guards and resource owners restore the caller's
device. Python density records retain backend/device choice and create independent
owners for later sampling, including after calculator cache closure.

## Explicit boundary

The existing accepted-state export remains a host snapshot. CUDA uploads this
snapshot; this is not a zero-copy resident endpoint or a performance claim.
Restricted/total/spin host result packaging does not evaluate integrals, AOs,
dipoles or populations. ECP and explicit resource-budget analysis remain rejected
until their scientific conventions and complete resource ledgers are qualified.
CUDA and point sampling admit normalized all-electron through-f inputs.

## Rejected alternatives

- Do not call CPU NumPy analysis or native host AO evaluators behind CUDA selectors.
- Do not introduce a hand-written AO polynomial/exponential evaluator alongside
  the generated CUDA grid algebra merely to supply values at user points.
- Do not replace analytic dipoles with quadrature or atomic-charge estimates.
- Do not silently truncate overlap eigenvectors for Lowdin populations; reject
  nonpositive eigenvalues rather than changing the requested partition.
- Do not add a second SCF solve or substitute a reference-package density.

## Acceptance gates

`tests/python/test_electronic_properties.py` exercises both backends against
identical-decimal-input PySCF/libcint RHF/UHF references, direct and fitted SCF,
RKS/UKS accepted-state operators, translation/charge conservation and detached
Cube output. `tests/python/test_electronic_properties_cuda.py` covers independent
contracted off-center Cartesian/spherical spdf operators and signed spin samples,
malformed C buffers and recovery, owner lifetime, and runtime host-fallback failure
sentinels. CUDA tests require explicit opt-in and a real Slurm allocation.

Independent operator/sample gates are 3e-12 (Lowdin 3e-11); independent SCF gates
are 2e-9 Ha energy and 3e-7 density/properties. No timing claim is made.

## Qualification evidence

Qualified on 2026-11-03 against master base `cd2b4514b` plus this working-tree
change. CUDA used GCC 12, CUDA 12.9.1, sm_120 and verified sccache 0.16.0.
The generated compiler command uses the selected sccache launcher.

- CPU native/Python property, API-contract and batch-output regression:
  **140 passed**, 35 CUDA cases skipped, 42 subtests passed.
- Native CUDA on node `n1`, two Slurm-visible RTX 5090 GPUs, job **15703**:
  **30 passed**, 33 non-CUDA cases deselected. The selection includes the
  explicit CPU-context rejection test as a CUDA admission boundary.
- Compute Sanitizer memcheck on the same two-GPU allocation domain, job
  **15704**: all **7** CUDA-specific tests passed, **0 memory errors**.
- Native/public API audits, Ruff, C++ formatting and Sphinx warnings-as-errors
  pass. Raw local qualification logs are ignored artifacts under
  `build/electronic-properties-validation/`.

The device-restoration gate deliberately resets the caller's visible ordinal
after calculator/context setup, whose existing device-selection policy is not
changed by this endpoint. Owner creation, analysis, sampling and teardown then
must restore that ordinal. The nonpositive-spectrum failure gate uses four
identical AOs: a single two-AO zero mode can round to a small positive eigenvalue
in cuSOLVER and does not reliably exercise a nonpositive-spectrum branch. No
new eigenvalue truncation threshold is introduced to force the test outcome.

## Revisit when

An accepted-state resident density lease and resource-budget receipt can be
retained through analysis without weakening identity, stream or lifetime checks.
That can remove the explicit host roundtrip, but must preserve detached public
sampling and independent numerical acceptance gates.
