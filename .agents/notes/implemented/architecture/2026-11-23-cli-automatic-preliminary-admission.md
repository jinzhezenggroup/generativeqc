# Decision: Make the CLI automatic cold start opt-in

Status: implemented
Date: 2026-11-23

## Decision

`run` and `resources` accept `--initial-guess auto` for expandable Direct FP64
CPU RHF/RKS targets. Model discovery calls the native Hückel and MINAO policy
descriptors in preference order. The CLI passes the automatic descriptor, not
an explicit seed kind: the prepared owner must still make the element, ECP
and population decision. CLI `none` remains the default and explicit providers
keep their existing admission rules and output fields.

The CLI borrows its existing basis topology for the same compact inventory as
Python. It can refuse the host budget before creating a context or preparing a
Fock owner. There is no second basis expansion or auxiliary source owner in
the dry run. A target decline carries no seed bytes; a local-cap skip retains
its provider ID but reserves no preparation storage. If execution publishes
no provider receipt, `run` verifies the target decline through the compact
query before reporting it. It does not fabricate iteration or timing counters.

## Evidence

On base `cae2959e1`, 152 CPU CTests and 335 installed Python tests passed.
Native CLI tests cover exact and one-byte-short local and total limits, fleet
seed accounting, rejection outside the CLI contract, and an ECP domain miss
with an independently analytic two-electron Gaussian RHF energy. Installed
CLI, Python and SDK PBE/H2 energy, force, capacity and fallback match at the
local limit and one byte below. The CLI and SDK execute with interpreters and
toolchains absent from the search path.

Ten interleaved CPU process pairs after warmup measured installed `resources`
plus exact-host-cap `run --forces`, including process startup and teardown.
On water, the default core policy changed by +0.15% (RHF) and −0.36% (PBE0).
New `auto` versus old explicit Hückel changed by +0.64% (RHF) and −0.56%
(PBE0). Energy, forces, SCF iterations, preliminary work and modeled peaks
matched. An intrusive PBE0 work trace separately confirmed 15 target Fock
builds for core and 13 for Hückel in both images; the preparation counted 7
additional Fock builds. There is no CUDA execution claim.

Raw receipts, binary digests, and compiler-cache statistics are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-automatic-preliminary-host`.

## References

- [Automatic CPU inventory](2026-11-23-cpu-automatic-preliminary-inventory.md)
- [CLI initial guesses](../../../../docs/user/native_cli.md)
