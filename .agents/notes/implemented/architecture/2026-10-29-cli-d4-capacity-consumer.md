# Decision: native CLI consumes named D4 capacity

Status: implemented
Date: 2026-10-29

## Problem

The native D4 capacity query and global Python planning do not automatically
make the installed CLI's separate allocation inventory understand corrections.
Treating a correction-bearing carrier as plain PBE would omit retained owners;
admitting every program with that carrier would also misrepresent hybrids.

## Decision

Extend the existing CLI inventory only for named D4 programs whose compiled
electronic components, domain, spin and carrier match the existing PBE catalog.
Reject exchange, range, nonlocal and basis-bound composite additions. Obtain
the correction capacities from the public native query, add retained host and
device storage, and apply both global limits and the equal-share local cap.
No correction workspace formula or scientific parameter table lives in the CLI.

## Boundaries

This is a metadata-only energy inventory, without a context, GPU probe, Python
or scientific tensors. It does not admit forces, fitted KS, gCP composites or
the legacy PBE-D4 carrier. Its scope remains distinct from a Python ResourcePlan:
the latter also reserves the optional CPU compatibility-gradient consumer.

## Evidence and invariants

Executable installed-style CLI tests compare reported capacities with the public
D4 query and the ordinary PBE inventory, including repeated owners, spin cases,
exact and one-byte-short global/local limits and explicit unsupported contexts.
CUDA electronic leaves use the established host-only interposer in these tests;
the D4 metadata query remains real. These tests make no real-GPU execution claim.

## References

This consumes #2872 after its #2870 allocation prerequisite and advances #2227
R2/R5 only. R3 and completion of the parent tracker are outside this change.
