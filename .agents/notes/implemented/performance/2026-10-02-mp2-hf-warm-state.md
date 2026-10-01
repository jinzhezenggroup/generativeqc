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
- Warm-state failure cannot make an otherwise valid endpoint less robust; cold retry remains available.
- Warm state is updated only after a complete successful MP2 endpoint.
- `warm_start=False` preserves the prior deterministic cold path.
- Missing checkpoint entries preserve neighboring retained states.
- No correlation or amplitude state is relabeled as HF warm state.

## Evidence

`tests/python/test_mp2_batch.py` covers cold-to-warm replay, same-geometry energy parity, changed-geometry seed use, explicit clearing, and the unchanged profiling rejection. Existing warm-disabled MP2 batch tests continue to exercise the old path.

## Consequences

Repeated prepared MP2 energy calls can reduce RHF startup work without changing MP2 equations or final reference gates. Full same-geometry reference reuse and correlated-method warm lifecycle remain follow-up work and require stronger versioned identity/provenance.

## Revisit when

A versioned prepared-reference state can prove geometry, basis, occupation, backend/device, provider/source, and precision compatibility strongly enough to skip additional RHF work, or when #190 supplies independently validated orbital/amplitude transport.

## References

- #1503
- #1401
- #190

Agent: ChatGPT
Model: GPT-5.6 Sol
