# Decision: plan budgeted CPU KS singlepoints in the prepared owner

Status: implemented

Date: 2026-11-25

## Decision

The CPU KS singlepoint already runs as a one-item `PreparedBatch`. Give that
owner the requested energy/force outputs directly, as with CPU DF. Do not
create a calculator estimate and pass it back as though a separate caller
had supplied it. That required reselecting the grid, querying the electronic
and correction inventories again and revalidating composition before any
native preparation.

An internally chosen packed force inventory must bind the same force promise
as a supplied plan. Both paths check `ks_export`, select native derivative
ownership and bar compatibility force replay. External plans still go through
their identity and budget validation. CPU electronic-property exports and
CUDA preparation retain their existing routes.

## Verification

`test_cpu_ks_singlepoint_admission.py` compares exact/one-byte-short capacities
and energy/force results for dense, packed and fitted providers with an
explicitly prepared batch. The test forbids creating an extra public estimate;
it does not freeze inventory call counts or substitute scientific work.
