# Decision: refresh shared-Rys integration contracts without weakening gates

Status: implemented
Date: 2026-11-09

## Problem

Hosted CI for the master-integrated PR #3048 head `7e7f81648` exposes two stale
contracts. The compiled allocation/failure fixtures omit the newly retained
device FP32:FP64 property and its prepared-owner field, so fixture setup fails
before exercising any allocation or cleanup behavior. The shared fixture feeds
both optional-allocation and recurrence-stack failure populations.

The direct J/K AOT resource gate also reports a whole-manifest identity drift
after the shared-Rys capability addition. That failure is not a cubin resource
regression: hosted run `42496300823`, job `128779301462`, artifact
`13623957777` measures identical generated/preprocessed shard source, headers,
compiler flags, toolchains and all 102 kernel REG/STACK/SHARED/LOCAL records.
Only the complete production-shell manifest digest differs.

## Decision and boundaries

Bring the fake external runtime/owner structures up to date; continue compiling
the real preparation and destructor bodies with every existing failure,
recovery, stream-fencing and ownership assertion. No production policy changes.

Refresh only the ledger's manifest digest from the actual hosted measurement.
The validator's `measured_ledger` reproduces the updated committed ledger
exactly, and `compare` reports no remaining differences. Preserve every other
ledger byte and the fail-closed identity/resource checks. The full downloaded
report, compiler-cache receipts, compile records and cuobjdump output remain in
ignored task artifacts; this metadata refresh is not new molecular, numerical
or performance qualification.

## Evidence

The two allocation/failure modules and resource-ledger tests pass all 1,809
cases locally. No failed test or scientific population is removed or skipped.
The separate HF force-stationarity proposal still requires its scheduled
independent complete-endpoint acceptance.
