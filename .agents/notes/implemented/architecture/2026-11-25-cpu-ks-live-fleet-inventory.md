# Decision: CPU KS admission requires a native fleet inventory

Status: implemented

Date: 2026-11-25

## Problem

The phase-only CPU KS ABI reports a single item's bounds. In the absence of
the batch ABI, Python could quote that library (or an older library) by
summing item lifetimes itself. The result was a live capacity admission with
no guarantee that the selected runtime accounted for the same set of owners.
ECP shapes could similarly mix a compiled fleet inventory with offline ECP
workspace sizing.

## Decision

Use the native inventory of the elected CPU KS batch owner and require the
native ECP workspace query when a row contains ECP terms. Missing capability
refuses that plan before preparation; a native rejection is not retried.
Retain provider-free ordinary estimates for offline comparison and the old
phase-only C ABI for standalone diagnostics. A selected runtime cannot use
either as a substitute for a missing batch query.

The ordinary fleet still accounts for the conservative transient force floor
when the specialized dense owner is not elected. Packed and fitted owners
still use their own inventories. This decision does not qualify new force
consumers, change the scientific model, or alter CUDA compatibility.

## Evidence and references

- `test_ks_host_resources.py` compares the phase ABI to independent offline
  arithmetic. `test_ks_fleet_resources.py` verifies fleet parity and native
  rejection; `test_ecp_resources.py` covers mixed ECP/all-electron batches,
  exact and one-byte-short bounds, and refusal before preparation.
- See #2227 and [resource planning](../../../../docs/maintainer/resource_planning.md).
