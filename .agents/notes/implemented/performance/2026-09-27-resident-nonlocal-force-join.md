# Decision: join resident nonlocal force production to stationary geometry

Status: implemented
Date: 2026-09-27

## Problem

The resident VV10 producer and strided stationary seed consumer existed on
separate branches. The complete force driver still downloaded density features,
compacted an active set on the host, downloaded pair seeds and uploaded each
geometry seed tile again.

## Decision

Use the full-grid producer under the same explicit force resource budget. Collect
its inputs from the semilocal feature leases and lend its device seed view to the
existing geometry consumer. Retain the producer until the final consumer drain.
Reset the stationary owner before pair submission so reset cannot insert a
producer-to-consumer fence. Keep scientific formulas in their existing owners.

## Invariants

The MolecularV1 threshold, ordered-pair semantics, all six inactive seed rows,
coefficient and source accounting do not change. Producer and consumer must share
one stream. Failed executions discard both owners. Geometry/model changes rebuild
all immutable owners. A dense pair capacity is not reported as a measured active
pair count. Enqueue timing is not labeled completed pair-kernel timing.

## Evidence

Device-free tests execute the real orchestration against strict lease protocols,
including odd tile tails, replay, invalid seed identities, cross-stream rejection,
no host publication and reset-before-pairs ordering. Python byte compilation was
run. GPU numerical and throughput qualification was NOT run (n3 was offline).
Existing complete WB97M-V RKS/UKS, finite-difference, independent-reference and
changed-geometry CUDA gates remain required before promotion.

## Consequences and rejected alternatives

This bounded path still collocates twice: pair seeds require all grid features.
Keeping every AO jet is a separate, explicitly budgeted caching optimization;
pretending a tile lease remains valid after the grid overwrites it is incorrect.
Do not weaken error publication or remove final-state validation to claim speedup.
The old fixed-grid public nonlocal API remains an independent compatibility path,
but no longer owns the complete WB97M-V force composition.

Refs #1423, #1479, #1482, #1488, #1497, #1504.

Agent: ChatGPT
Model: GPT-5.6 Sol
