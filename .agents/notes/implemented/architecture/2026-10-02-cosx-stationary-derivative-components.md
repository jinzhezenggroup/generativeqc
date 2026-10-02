# Decision: split stationary COSX J/K derivative sources

Status: implemented
Date: 2026-10-02

## Problem

The prepared COSX owner already evaluates the complete fixed-density RI/direct-J
plus COSX-K molecular derivative, but it returned only the summed two-electron
vector. The generic stationary DFT reducer requires physical source separation
so Coulomb and exact-exchange weights/provenance are never relabeled or applied
twice.

## Decision

Expose `energy_derivative_components()` from `PreparedCosxFockPlan` using the
existing `scf::FockEnergyDerivativeComponents` contract. Coulomb delegates to
the already-prepared J owner; exchange delegates to the existing complete COSX
molecular derivative and applies the resolved Fock coefficient exactly once.
The legacy summed `energy_derivative()` now recomposes those two vectors.

No new COSX mathematics, grid, coefficient, provider selection or allocation is
introduced. This is the provider seam required for the DFT stationary-gradient
consumer.

## Evidence

The existing independent COSX derivative comparison remains authoritative.
Native tests additionally require the split sources to have exact coordinate
shape and to recompose the existing combined provider result.

## References

Issue #246; follows #1007 and #1660.

Agent: ChatGPT
Model: GPT-5.6 Sol
