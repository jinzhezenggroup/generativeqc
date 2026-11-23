# Decision: Account for packed CPU ERIs in the Hessian host budget

Status: implemented
Date: 2026-11-22

## Decision

`resources --property hessian|hvp` and the linked CPU endpoint now share the
packed byte flag parser and inventory. The two caps serve distinct purposes:
the packed cap gates just the retained FP64 orbit array, and the host cap
gates the modeled full endpoint peak. Charge `PackedSphericalEri::required_bytes`
once, not the caller's ceiling and not a second dense array. A cap miss is an
explicit refusal before SCF, not a retry with dense values. The existing linked
method and spherical s/p/d/f admissions still apply. This extends the prior
[B3LYP inventory](2026-11-22-cli-b3lyp-host-inventory.md), without changing
the public CPU host inventory bridge or the default provider.

`PreparedFockPlan` constructs the packed array, builds one-electron integrals
with dense ERIs disabled, then retains that source through SCF. The spherical
packed and dense builders share the shell-quartet generator and projection.
Their setup is serialized with one-electron preparation, and neither packed
J/K replay order needs an expanded ERI panel. Keep the existing conservative
setup allowance and overlap the complete primal inventory with the mandatory
derivative reservation: this change proves a smaller retained array, not a
smaller setup peak or an earlier release. It does not price RSS or qualify a
new scientific path.

## Evidence

On the patched checkout based on `cae2959e1`, 11 CPU CTests and all 19
installed CLI checks passed. PBE0 RKS and B3LYP-D3(BJ) UKS cover both dense
Hessian and HVP. Each test admits both exact cap boundaries, rejects a byte
short on either cap with no SCF trace or result, and verifies the native
retained array extent and agreement with the dense-provider result. The
native fleet test checks retention per copy and unchanged setup/SCF workspace.

Native primal capacity samples were 119,582 of 176,158 modeled bytes on HeH
and 4,669,414 of 121,456,422 on CsH/def2-SVP with scalar ECP. These samples
used the exact packed cap and matching SCF/grid controls; they do not observe
all setup scratch or process RSS. The corrected CsH HVP admitted at
325,860,438 modeled host bytes (758,640 packed bytes), reducing the dense
provider's modeled host peak by 4,899,608 bytes. Either one-byte miss
refused without output. The exact-fit endpoint matched the dense result
within `2e-8` and reported 15 SCF iterations, 17 Fock builds, 733,054
curvature ERI primitive visits and 26,940 ECP kernel blocks.

Seven interleaved process pairs per case on one EPYC 7K62 core used the
same shared library. The complete endpoints returned identical derivatives,
energies and work counts between the pre-admission CLI and the new CLI.
Across three default dense endpoints (LDA, PBE and corrected B3LYP) the
maximum median slowdown was 0.04%. Across packed PBE0 dense, corrected
B3LYP UKS HVP and corrected B3LYP with ECP HVP, the median changes were
respectively +0.07%, +0.84% and +0.23%. Raw evidence, including commands, resolved
plans, binaries' checksums and semantic work counts, is under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-packed-host-bench`.
