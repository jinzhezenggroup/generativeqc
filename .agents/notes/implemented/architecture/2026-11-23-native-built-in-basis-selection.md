# Decision: Resolve installed built-in basis names in the native catalog

Status: implemented
Date: 2026-11-23

## Decision

The native catalog resolves the name aliases used by the SDK and CLI. Expose
that operation without instantiating a system, so Python `Calculator` can use
it for both orbital and auxiliary names. Materialization and AO normalization
keep their existing owners; the selector does not grant scientific admission.

Python still attaches a `BasisSet` record for mathematical identity and
provenance. Only a missing native resolver permits old-library name lookup.
A selected native resolver's missing name or a catalog checksum mismatch must
not lead to a second lookup of the supplied alias. Standalone data APIs and
explicit custom or method-bound snapshots do not take the installed name path.

## Evidence

The native SDK test covers byte-counted aliases, invalid lengths and shared
query identity. Python tests check both basis roles, exact metadata and energy
parity, pack mismatch and the old-library boundary. No basis values, scientific
equations or precision policies change.

Forty alternating pairs on the installed CPU Release library compare the
native selector to the portable-name path under the same binary and warm
metadata cache. The timed endpoint includes Calculator construction, batch
preparation, energy execution and close. For PBE water with spherical def2-SVP,
def2-TZVP auxiliary and grid level 0, the paired median overhead was +0.02%.
Both arms produced the same basis metadata, −76.28565903143046 Hartree,
13 iterations and 15 Fock builds.

Related: #2227 (R5/R6).
