# Admit native DF source construction before allocation

Status: implemented
Date: 2026-10-04

## Problem

The internal CUDA DF-CC source admitted its transform/packing payload, then
constructed the raw source before inspecting that source's setup ledger.
Transactional rejection protected outputs but could exceed the numeric budget
first. With two orbital AOs and 100 supported auxiliary s shells, the original
24,212-byte admission was smaller than the 80,000-byte host metric alone.

## Decision

A CUDA-free, checked construction query runs before device selection and the
factory's argument copies. It includes both System argument copies, combined
systems, typed source metadata, identities, legacy warm/pair storage, public
transform temporaries, and host/device metrics. Actual owner and expansion sizes
come from the caller, so the capacity query does not assume a CUDA object ABI.

Reserve the source upload inventory, combined shell storage, pair arrays and
upload-ownership pointers before incremental packing. Explicitly skip the
unused resident-PSSS tables. Retain the later observed-ledger and downstream
phase checks, and include construction in numeric/device capacity diagnostics.
The small maximum-shell expansion allowance includes simultaneous vector
growth on the supported libstdc++/libc++ implementations; allocation-event
tests check the peak rather than only final capacities. Allocator/driver
overhead remains outside the existing numeric-buffer contract.

## Validation boundary

The host regression executes the real admission/source-packing prefixes with
a counted factory/device boundary. It checks the failing old budget, exact and
one-byte-short construction admission, overflow, varied s/p/d/f and primitive
counts, and bounded g-expansion scratch. It runs no CUDA numerical work.

No integral equations, metric cutoff, force domain or numerical tolerance
changes. Existing measured receipts remain historical; this repair is not a
new GPU qualification or timing claim. Later g metadata-only packing may omit
legacy storage included by this conservative bound without allocating it again.

This supersedes the post-construction-only setup admission described in the
[original source decision](2026-10-03-native-cuda-df-cc-source.md).
