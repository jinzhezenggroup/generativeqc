# Decision: exclude named corrections from the electronic-only capacity promise

Status: implemented
Date: 2026-10-29

## Problem

The numeric-cap KS promise excludes the legacy D4 flag because its complete
bound accounts only for electronic owners. Compiler-named corrections instead
use `execution_plan.dispersion.kind`. They could enter the promise while their
retained tables, workspaces and gradient owners were omitted from that bound.
Optional XC admission could then rely on incomplete future-owner protection.

A real Slurm probe (14861) against the #2870 production binary reproduced this
for named PBE-D3(BJ): preparation published 105,701,171 unspent promised bytes
although independent correction storage was outside the electronic inventory.
The allocation ledger still enforced its numeric cap; the defect is the claimed
complete promise and resulting optional admission, not an unbounded allocator.

## Decision

Reject every nonzero named dispersion kind from this electronic-only promise,
in addition to the legacy D4 flag. Retain the existing capped allocation path,
shared ledger lifetimes, scientific method and public ABI. Ordinary unbudgeted
execution and uncorrected eligible KS promises retain their current behavior.
Selected-inventory admission already had no such promise and remains unchanged.

## Evidence and limits

Extend executable CUDA capacity tests across named D3(BJ), D3(0), and D4 RKS/UKS:
scalar and batch preparation must expose no incomplete promise; complete E/F,
warm/moved replay, one-byte cap refusal and final charge cleanup still work.
Compare E/F with the same complete unbudgeted owner and retain the existing
independent-reference tests in this native suite. Real-device qualification must
identify the actual source/binary; a host-only build is not CUDA evidence.

## Revisit when

The native bound includes every retained correction, replicated table, future
consumer and rebuild overlap, and independently qualifies optional-cache
admission under shared caps. Merely changing a selected inventory into a raw cap
or adding correction bytes observed after execution does not establish that
proof. This guard does not repair #2872's separate budgeted endpoint regression
or complete #2227 R2/R6. R3 is excluded.
