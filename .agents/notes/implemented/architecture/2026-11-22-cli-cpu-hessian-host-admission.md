# Decision: Enforce the composed CPU Hessian host cap in the CLI

Status: implemented
Date: 2026-11-22

## Decision

[The previous slice](2026-11-22-cli-cpu-hessian-resource-floor.md) reported a
composed host floor but only enforced a derivative-local cap. CPU `hessian`
and `hvp` now use the same composition as `resources` when the caller gives
`--host-bytes`. Execution borrows its expanded basis and already verified
derivative floor; the inventory does not re-expand the basis, normalize a
second system, or build an SCF or ERI plan. No-budget execution still skips
the aggregate inventory.

Compute the native effective limit first (headroom, then reserve). Refuse
before SCF if the full primal-plus-derivative floor exceeds that limit. The
optional allowance is the lesser of remaining host slack and the difference
between the requested derivative cap and its floor. Pass the selected
derivative cap to the existing stationary owner, which still admits optional
storage and work based on the accepted state. Do not infer any primal
release, or claim an allocator or process-RSS hard limit. Refuse incremental
Direct-J/K and opt-in packed ERIs rather than undercounting their storage.
This does not extend modeled admission to CUDA or add new primal inventories.

## Evidence

The CLI test exercises dense and HVP on bare PBE and corrected PBE0. At both
the exact floor and floor plus slack, dry-run and execution select identical
caps after headroom and reserve. The resulting derivatives match the direct
derivative-cap control. At one byte below the aggregate floor the endpoint
publishes nothing and rejects even when the configured SCF would fail after
one iteration. Separate checks reject packed ERIs and CUDA host-budget use.

Eight selected CPU CTests and all 18 installed CLI contract checks passed.
The initial two-library benchmark exceeded the 5% gate in four cases. A
binary/library crossover localized the timing difference to the loaded
shared-library artifact rather than the CLI. Byte comparison found only the
build ID and source hash differ between the two libraries. The cause of that
artifact-dependent timing remains unverified. Both CLIs were then tested
against the identical current library, using five interleaved complete
process pairs per case. All derivatives and work receipts matched. Median
changes ranged from -0.13% to +0.52%, with budgeted measurements comparing
automatic host admission against the same explicit derivative cap. Retain
both rounds in `/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-host-exec-bench`;
the controlled comparison does not erase the first round's measurements.
