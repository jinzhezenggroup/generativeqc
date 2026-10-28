# Decision: query warm-seed staging bounds before export

Status: implemented
Date: 2026-10-29

## Problem

Precision audit staging reconstructed AO and spin dimensions in Python. The
existing native export count query can materialize resident CUDA seeds, so it
cannot safely discover capacity before the caller checks its staging budget.

## Decision

Query `BatchWarmStarts::warm_density_size` and the prepared batch's ordered atom
counts. Reuse the native warm-state owner's representation and spin decisions;
do not copy scientific dimension equations into another planner. Aggregate with
checked byte arithmetic and publish transactionally. Never call `warm_state`
from the capacity query: that accessor may copy device data and allocate storage.

The all-slot envelope deliberately includes currently absent seeds. It stays
valid before SCF, after failed items and while frozen history is retained. The
Python caller still compares the bound against its numeric staging limit before
export and separately checks host validation reservations. Only a missing symbol
permits older-library reconstruction; native refusals propagate.

## Retained boundaries

This is a scoped #2227 R2/R4/R6 metadata cutover, excluding R3. It does not move
candidate/audit scheduling, independent history ownership, checkpoint format or
host reservation policy. Numeric density/coordinate bytes are not complete RSS,
allocator, validation-scratch or opaque provider-memory accounting.

## Validation

Compiled fault-injected owners forbid seed materialization and scientific work,
exercise individual and aggregate overflow, and preserve outputs on rejection.
Actual restricted/unrestricted HF/KS ragged owners cover cold and retained seed
bounds, exact/below-limit staging and same-owner restoration. Existing independent
precision endpoint tests remain the scientific and lifecycle qualification.
