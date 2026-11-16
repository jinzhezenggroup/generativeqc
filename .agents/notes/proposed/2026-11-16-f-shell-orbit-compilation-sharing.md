# Proposal: qualify f-shell coordinate/center orbit sharing per consumer

Status: proposed; isolated host and sm120 complete-owner evidence only
Date: 2026-11-16

## Problem

The existing g-bearing value/gradient lowering shares coordinate and Gaussian
center orbits. Through-f lowering still duplicates much of that arithmetic,
especially in COSX's separately scoped value and gradient headers. Extending
the existing scalar IR lowering is preferable to introducing another algebra
implementation, but emitted source size alone does not qualify a build or
runtime optimization.

## Isolated experiment

Only a private compiler copy under `/tmp/gqc-wheel-f-orbits/compiler/` changes.
Share f/g value coordinate/center orbits, and f/g gradient y/z and center
exchange orbits, while preserving cheap s/p/d lowering, rounded pair geometry,
radial/Boys inputs, mathematical IR and the public angular inventory. The
original compiler and published native library remain unchanged.

Baseline emissions exactly match actual native generated headers. f/g
mathematical-inventory digests are unchanged. Header bytes before/after:

| Complete generated header | Before | Candidate |
| --- | ---: | ---: |
| Native through-g values | 4,341,862 | 1,976,037 |
| Native through-g gradients | 17,589,941 | 12,887,846 |
| Owned COSX through-f values | 3,100,663 | 737,581 |
| Owned COSX through-f gradients | 7,647,724 | 2,946,492 |

The combined COSX headers shrink from 10,748,387 to 3,684,073 bytes, about
65.7%. This is not a compiler-time or installed-wheel reduction.

## Complete sm120 owner results

Actual Release commands retain their optimization, profiles and compute120 PTX
plus sm120 ELF. Each valid candidate request has zero hits and four misses.
Original objects are authenticated against their recorded SHA256 digests.

| Complete owner | Original object bytes | Candidate object bytes | Candidate observed seconds |
| --- | ---: | ---: | ---: |
| One-electron values | 8,529,176 | 3,885,208 | 87.442 |
| One-electron gradients | 30,821,440 | 23,979,600 | 339.459 |
| COSX values | 2,635,000 | 3,722,328 | 18.254 |
| COSX gradients | 8,314,464 | 3,069,952 | 46.607 |

There is no matched original request-time experiment, so do not infer compile
speedups from these times. All original function identities remain, but several
resource records change. In particular, COSX value registers increase from
180 to 202 and stack from 176 to 320 bytes; its object grows **41.3%** despite
the much smaller header. The gradient owner's corresponding ESP-value kernel
also grows to 202 registers/320-byte stack, and contraction stack grows from
1,184 to 1,456 bytes. This broad lowering cannot be adopted on source-size
evidence alone. Investigate consumer-specific lowering and complete endpoint
performance rather than forcing identical code-generation policy everywhere.

A second complete COSX-gradient probe keeps the original through-f value
header and changes only gradient orbit lowering. Authenticated dependencies
confirm both headers. Its object is **5,559,360 bytes**, about 33.1% smaller
than the original 8,314,464-byte owner, with a **127.418-second** observed
request and zero hits/four misses. The ESP-value kernel retains its original
resources, but contraction stack still grows **1,184 -> 1,456** bytes and
probe-gradient stack **1,296 -> 1,536** bytes. This isolates the value-path
regression without proving that gradient sharing is an endpoint win. Preserve
the separate `cuda_cosx_derivative-gradient-only` receipts; do not compare its
single request time with a differently cached historical build.

## Independent qualification and evidence traps

- The two maintained emitted-arithmetic tests pass against independent PySCF
  2.14.0/Libcint raw, normalized, contracted and spherical through-f blocks.
- The maintained through-g emitted value/gradient and invalid-index suite
  passes **432 tests** using the private compiler. Together these are host
  arithmetic gates, not GPU, full architecture or native endpoint acceptance.
- The first value probe accidentally consumes the original header: a quoted
  sibling include inside the policy header wins over an extra include path.
  Preserve that invalid receipt rather than reporting its 2.419-second cache
  hit as a speedup. A corrected probe co-locates the unchanged policy and new
  math header; actual `nvcc -M` dependencies authenticate each candidate owner.
- `sm120-owner-comparison.json`, per-owner dependencies, input digests, cache
  receipts and objects remain under `/tmp/gqc-wheel-f-orbits/`. No production
  compiler edit, GPU endpoint claim or full-matrix claim is made.

## Acceptance boundaries

Retain all scientific/oracle, angular, architecture and resource/lifetime
coverage. An adopted lowering needs full supported-image compilation,
per-kernel/callee resource review and independent complete public GPU endpoints
with semantic work counts, including COSX values/forces and moved geometries.
Replace historical helper-count spelling tests with executable linkage behavior
if ownership changes; do not remove real integrity checks to make it pass.
