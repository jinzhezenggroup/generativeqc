# Decision: seed prepared MP2 references from retained HF density

Status: implemented
Date: 2026-10-02

## Problem

Prepared conventional MP2 batches rebuilt every RHF reference from a cold guess even though the batch ABI already owns validated `HfWarmState` density snapshots. This made same-geometry replay and nearby-geometry scans pay avoidable reference iterations, and the default Python `warm_start=True` path was rejected outright for MP2.

## Decision

The first #1503 slice reuses the existing HF warm-state contract for MP2 reference seeding. A retained density is only an initial proposal: the ordinary strict RHF solver still evaluates and converges the target Hamiltonian before correlation runs. Changed geometry rebuilds the geometry-bound MP2 owner, then passes the previous compatible density into that new owner's RHF solve.

If a seeded RHF solve throws or fails to produce a converged physical reference, the endpoint retries from the ordinary cold guess. A new warm snapshot is committed by the batch only after the complete MP2 endpoint succeeds. Checkpoint restore validates dimensions, diagnostics, source geometry, and the density with the shared RHF warm-density validator before changing live state.

This slice deliberately does not reuse a prior physical reference without SCF, does not transport MP2/CC amplitudes, and does not complete the RCCSD/RCCSD(T) portion of #1503.

## Rejected alternatives

Directly accepting a previous converged HF reference was rejected for this slice because geometry/provider/source identity is not yet represented by the existing public warm-state ABI. Reusing CC amplitudes here was rejected because cross-frame orbital/amplitude transport belongs to #190.

## Invariants

- Warm state never establishes convergence for a new target.
- Numerical warm-reference failure retains a cold retry; resource admission still includes every live seed payload.
- Warm state is updated only after a complete successful MP2 endpoint.
- `warm_start=False` preserves the prior deterministic cold path.
- Missing checkpoint entries preserve neighboring retained states.
- No correlation or amplitude state is relabeled as HF warm state.

## Evidence

`tests/python/test_mp2_batch.py` covers cold-to-warm replay, same-geometry energy parity, changed-geometry seed use, checkpoint save/restore, explicit clearing, and the unchanged profiling rejection. Existing warm-disabled MP2 batch tests continue to exercise the old path.

### Review repair: bounded warm-state lifetime

The first implementation copied a candidate density before correlation while
retaining the previous seed, but still passed the entire numeric budget to every
phase and reported the old capacities. The endpoint now reserves both live
states' density/coordinate payloads before reference, energy, and force
admission, and adds that reservation to phase diagnostics. The iterative density
is moved into the candidate, and an unsuccessful reference result is retired
before cold retry. Frozen updates reserve only the previous seed; warm-disabled
execution keeps the previous budget. Admission failure preserves last-good state
and invalidates result diagnostics. This follows the per-item method budget,
not a new global batch/RSS promise.

Exact-source checkpoint execution also exposed a pre-existing generic validator
that rejected MP2's mandatory zero screening tolerance. Screening now permits
zero (the unscreened contract), while negative/nonfinite/non-numeric screening
and nonpositive energy/density convergence tolerances remain rejected.

Tests cover one-byte-below/exact-boundary energy and force admission,
frozen changed-geometry replay, atomic rejected/partial imports, failure after
reference success, and an independent PySCF moved-geometry energy/force oracle.
The opt-in real-CUDA test now executes actual warm replay and checks backend,
complete energy/force parity, and failed-item recovery. CUDA qualification is
NVIDIA_NOT_RUN until that gate runs on a real device; CPU success cannot replace
it. No endpoint speedup is claimed.

Review repair by Agent: dot

## Consequences

### CUDA density-export repair (2026-10-02)

The real NVIDIA warm-replay gate exposed an adapter mismatch: CUDA RHF exports
its validated AO density in `ScfResult.reference->density`, while its optional
`ScfResult.density` vector is empty. Moving only the latter into the MP2 checkpoint
published coordinates without a density. Every CUDA warm attempt then fell back
to a cold solve, checkpoint import rejected the incomplete state, and prior-seed
byte accounting reflected only the coordinate payload.

MP2 now retains the existing physical-reference density when the iterative
result vector is absent. This copies an already exported, validated host matrix;
it adds no device download and does not reconstruct or approximate the density.
The existing candidate/prior-seed budget charge still covers the complete matrix
and coordinates, and publication remains conditional on complete MP2 success.
The opt-in CUDA regression explicitly checks all four H2 density entries before
replay, in addition to the unchanged moved-geometry energy/force and fallback
gates. CPU fixed-capacity boundary tests remain distinct from CUDA's adaptive
correlation/force arena capacity diagnostics.

Repeated prepared MP2 energy calls can reduce RHF startup work without changing MP2 equations or final reference gates. Full same-geometry reference reuse and correlated-method warm lifecycle remain follow-up work and require stronger versioned identity/provenance.

## Revisit when

A versioned prepared-reference state can prove geometry, basis, occupation, backend/device, provider/source, and precision compatibility strongly enough to skip additional RHF work, or when #190 supplies independently validated orbital/amplitude transport.

## References

- #1503
- #1401
- #190

Agent: ChatGPT
Model: GPT-5.6 Sol
