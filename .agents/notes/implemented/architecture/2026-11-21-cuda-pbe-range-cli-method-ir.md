# Decision: Transport explicit range-PBE MethodIR through the native CLI

Status: implemented
Date: 2026-11-21

## Decision

`hessian` and `hvp` may accept the compiler's canonical `MethodIR.to_payload()`
instead of a named method. Admit only the already qualified unscaled PBE +
SR/LR graph. Compare the entire semilocal semantic payload, including its
provenance and spin, to the linked catalog identity. Lower from that catalog
row and attach only the authenticated pair of exchange terms. Hash the complete
semantic graph for publication, as Python does; the supplied display name and
derived requirements do not authorize a calculation.

The native SCF still chooses the radial model before its first energy, and the
Hessian still borrows that accepted state. The CLI does not execute Python,
regenerate code or silently reinterpret the ordinary `pbe-rks` / `pbe-uks`
selectors. Avoid accepting a generic MethodIR graph that the linked stationary
endpoint cannot differentiate, and avoid a second user-facing exchange schema
with a different scientific identity from Python's.

## Evidence

- The host CLI contract validates compiler-emitted input, option conflicts,
  unsupported precision/providers, altered provenance and invalid SR/LR terms
  before device initialization. The ordinary CPU PBE CLI regression passes.
- Slurm 2295 (RTX 5090, SM120): actual CLI dense Hessian and HVP with Direct
  RKS, scalar-ECP RKS with DF primary and exact LR, and scalar-ECP UKS with two
  fitted radial operators and negative LR-minus-SR. The published identity
  equals the compiler's hash. Independently reconverged Libcint/Libxc moving-grid
  energy differences meet the `2e-6` Hartree/Bohr² HVP gate. The plain PBE
  selector still publishes the pure PBE identity and a distinct energy.
- Source-matched binary checksums, build/cache receipts and test logs:
  `build/hessian-gpu-sm120/pbe-ir-cli-*`. These results do not qualify non-SM120
  devices or arbitrary user-defined electronic/correction graphs.
- The full host KS final-state suite also passes after correcting the stale
  MN15 fitted-J/K expectation and checking the linked variational domains.
  Both domains still reject mixed Direct/DF identities; the variational domain
  rejects noncanonical partition order. See `pbe-ir-final-state-host.log` in
  the same build directory.
