# Decision: retire idle Fast anchors before force-memory consumers

Status: proposed
Date: 2026-10-18

## Problem

The optional fixed-anchor arena remains charged after SCF. It can fit during
preparation yet prevent a later mandatory energy-plus-force allocation while the
KS owner is alive. The existing force-memory handoff retires optional ERIs only.

## Proposed change

After the existing exact-token, device and capture checks, synchronize the producer
stream under pressure, release the optional anchor, and query free memory again.
Retire optional ERIs only if the requested envelope still exceeds free memory.
Keep the handoff advisory: the force consumer still owns actual allocation.

Clear the arena, all derived views and anchor validity flags only after successful
free. Subtract the exact charge from state-device capacity and preserve cumulative
execution counters. Final D/W/XC, the token and last-good warm seed remain owned.
Later cold solves use ordinary Fast on this owner; no new optional allocation is
attempted after force consumers have acquired memory. Admission and numerical
policies are unchanged.

## Validation scope

The native Fast-owner regression exercises roomy and stale-token paths, pressured
retirement, exact state/provider accounting, preserved D/W/token/warm seed,
repeated handoff, warm replay and cold failure/recovery. Its conservative total-
device request avoids assumptions about stable globally free memory. Local
extracted-body host/UBSan controls cover anchor-first ERI retention,
stream/capture/free failures and combined anchor/ERI retirement;
they are diagnostic artifacts, not required-CI source-shape contracts. No new GPU
run or performance requalification is claimed by this local proposal.

Reference: PR #2602, review 5562708969. The existing small-domain performance and
broad-native-suite concerns are separate and remain unresolved.
