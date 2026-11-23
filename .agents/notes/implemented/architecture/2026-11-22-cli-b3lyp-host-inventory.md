# Decision: Compose the canonical CPU B3LYP primal and derivative inventories

Status: implemented
Date: 2026-11-22

## Decision

The linked CPU B3LYP aggregate was already executable, but the CLI
[host inventory](2026-11-22-cli-cpu-hessian-resource-floor.md) did not have
an admissible primal entry, so a combined budget could not be enforced.
Recognize the exact catalog entries for canonical B3LYP RKS/UKS, bare and
uncut two-body D3(BJ). Their electronic allocation shapes match the CPU
GGA jets and full-range hybrid J/K provider already sized by the native
KS host inventory. Charge the D3 owner only for corrected total programs.
Scalar ECP uses the existing native primal ECP bound and the exact derivative
floor queried on the normalized system. No SCF or derivative implementation
changed. Carrier names alone do not authorize a hybrid, custom blend, D4
composition or CUDA domain; force, DF and opt-in packed ERI remain excluded
from this combined budget.

## Evidence

Exact host and one-byte-short checks cover both spins, bare and corrected,
and dense and HVP products. They match the unrestricted results and method
identities. Eight CPU CTests and all 18 installed CLI checks passed.
On the existing real CsH def2-SVP/ECP fixture, corrected RKS HVP runs at
330,760,046 modeled host bytes, with 204,403,968 derivative bytes; one byte
less refuses without publication. The successful endpoint reports 15 SCF
iterations, 17 Fock builds, 733,054 curvature ERI visits and 26,940 ECP
kernel blocks. Explicit native sampling covered bare RKS, corrected UKS and
corrected ECP RKS: the respective sampled primal peaks were 45,248, 47,160
and 7,283,168 bytes, below 177,024, 172,146 and 126,356,030 modeled bytes.
These samples are not process-RSS measurements.

The matched-state independent B3LYP energy-curvature tests passed on H2 and
HeH+ (two tests, unscaled exact exchange).

Five interleaved process pairs per case on one EPYC 7K62 core used the
same current shared library, isolating the CLI change. The complete
endpoints returned identical derivatives, energies and work counts.
Median time changed by -0.09%, -0.05% and -0.01% for default dense LDA,
PBE and corrected B3LYP, and by +0.61% and +0.22% for corrected B3LYP HVP
and bare dense B3LYP under the same selected derivative cap. Receipts are
under `/data/jzzeng/gqc-2227-sdk-JsefEq0x/cli-b3lyp-resources-bench`.

The subsequent [packed ERI inventory](2026-11-22-cli-packed-eri-host-inventory.md)
supersedes the opt-in packed exclusion above without changing method admission.
