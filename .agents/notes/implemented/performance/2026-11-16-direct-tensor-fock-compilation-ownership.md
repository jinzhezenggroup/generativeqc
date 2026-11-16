# Decision: share cached-tensor and packed-Fock compilation ownership

Status: implemented; native/scientific gates pass, clean replay and full wheel pending
Date: 2026-11-16

## Problem and decision

The cached-tensor and packed-Fock launch owners retain the same Cartesian
recurrence functions in separate NVIDIA RDC objects. Device linking already
coalesces those definitions, but does not avoid compiling both copies.

Generate one private include-wrapper only inside the existing non-Apple NVIDIA
RDC archive. Keep both original source files, their public launch interfaces,
all generated numerical headers and the independently compiled non-RDC/CuMetal
path unchanged. The production source list owns the wrapper so optional
source-level compiler settings still apply. Do not combine the separately
qualified angular-force owner with launch-bound propagation constraints.

## Executable evidence

Retained inputs are clean code `d4953238df0d21eaae93092c325bcb7907b71cda`;
later published integration does not change native or compiler inputs. Evidence
is under `/tmp/gqc-wheel-weighted-owners/`, with **484** frozen input hashes,
original production commands, complete original objects and owned sccache
0.16.0 foreground-server receipts. No cache is cleared or disabled.

The complete sm120 pair changes from **67 function copies to 38** unique
functions, and **2,875,104 to 1,518,784 object bytes**. All original function
identities, resource records and individual SASS section bytes match exactly.
Its one observed compiler request is 24.713 seconds; historical overlapping
original tasks sum to 36.463 seconds. This is not a controlled cold-wheel
measurement.

For the unchanged **80;90-real;120-real**, including compute80 PTX, combined
object bytes fall from **3,895,680 to 2,082,368**, and identically extracted PTX
dump bytes from **3,805,168 to 2,045,753**. Function copies per architecture
change **62/62/67 to 34/34/38**. All per-architecture function identities,
register/stack/shared/local resources and individual SASS section bytes remain
identical. Each original request and the combined request records three cache
hits and five misses, so **54.787 versus 28.571 observed request seconds** is
explicitly a partially cached ownership comparison, not a cold compilation or
whole-wheel speedup. Removing already coalesced definitions does not imply a
smaller final library or wheel.

## Rejected grouping

A broader five-owner grouping also includes weighted-ERI, angular-Fock and
Schwarz launches. It preserves every function's resources and SASS bytes and
reduces objects from 23,819,400 to 20,657,984 bytes, but its 289.590-second
sm120 request does not improve on the historical 235.789-second task sum.
Weighted-ERI shares no retained weak function with the smaller pair. Do not
adopt that larger invalidation/optimization unit merely to reduce file count.

## Actual production integration

The sm120 Release library and CLI build and device-link successfully with
portable automatic XC AOT. The 203-second incremental request is not a cold
native build. Ninja records **444** dependencies for the actual generated
wrapper, including both original `.cu` files. The production object preserves
all **38** sm120 function identities, resources and SASS section bytes from
the original independent owners, not just from the prototype wrapper.

Slurm **17542** runs the actual linked library in an ABBA comparison with the
previous clean artifact. All **64 complete Direct-HF endpoints** pass independent
PySCF energies and forces: RHF/UHF, Cartesian/spherical, cold/warm/moved/restored.
All iteration, Fock-build and precision-work records match. Maximum independent
energy/force errors are **6.40e-14 / 8.04e-11**; maximum before/after differences,
including the baseline control, are **3.56e-15 / 1.23e-15**. Diagnostic execute
ratios have median **0.9990**, maximum **1.0061**; these are not whole-wheel or
general method speedup claims.

This first native artifact is explicitly built at dirty `2f1e783b8`, with the
recorded CMake-only patch. Its ELF SHA is
`0bd9fac2f766e080f78b0621d7999f5accfcc4b5e6d553bd17cf4beed0eba489`
and native source identity is
`b931e71405105aabe6eab24c749aa1379e563fa20cfc624f77486d9d816c732c`.
Do not describe it as a clean committed artifact.

All **64** matching native profile/lifecycle/source-admission/retry tests pass.
The broader CMake/codegen/resource suite initially has **913 passes, one skip**
and one CPU-configure failure because this host does not find its Conda OpenSSL
prefix by default. The unchanged failed test passes with the installed prefix
in `CMAKE_PREFIX_PATH`; no production behavior or coverage is changed. All
applicable pre-commit hooks pass. Refreshing the unmeasured AOT audit and Libxc
snapshot changes only their current source identities: no promotion, scientific
receipt, qualified endpoint, stale-context or acceptance field changes.

Clean committed native replay and full manylinux build, repair, installed-wheel
and matching-head CI acceptance remain separate gates.
