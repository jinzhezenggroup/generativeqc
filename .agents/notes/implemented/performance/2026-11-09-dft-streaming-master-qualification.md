# Decision: qualify DFT streaming independently on the new derivative master

Status: implemented; scoped default qualification, not all-README leadership
Date: 2026-11-09

## Problem and decision

The old shared-Rys DFT measurements predate master #3044's derivative/AO owners.
Do not merge unrelated HF ancestry, choose conflicting old Libxc snapshots or
reuse those timings as new-source qualification. PR #3051 isolates the DFT change
on 2bf3fd98d; the generic scalar-throughput, compiled-capability and actual-resource
policy is unchanged. No SM80/A100 profitability whitelist or new opt-in is added.

Normal Release CPU builds16861/16862 complete with verified sccache and preserve
both deployment targets and full ELFs. Control source bda7f76b / ELF f6b87f70 and
candidate source75d94b32 / ELF549ed9a2 are verified in every measured native process.
Exact-delta transport reconstructs the complete original libraries on A100.

## New evidence

LSF230450 / Slurm16863 each execute a complete control/candidate/candidate/control
population per PBE0/B3LYP Direct96, with own-density three-repeat warms. A100
cold means improve12.662143→11.734812s (-7.32%) and12.779418→11.867471s (-7.14%).
5090 changes+0.90% /+0.008%; all individually paired warms pass3%. Independent
dense/grid-stability/CPU spot, strict native E/F, Direct48 and actual automatic
dispatch checks pass without changing scientific work or accuracy.

Retain the historical raw-K oracle/sanitizer cohort as historical. Regeneration
proves an unchanged fixture executable body, not a new GPU fixture execution.
Preserve original build16857's timeout124, full ignored AOT stdout, cache/restore
receipts, every scientific population and all collector rows. An unconfirmed
optional bmod transport does not establish a changed limit or another allocation.
The unchanged evidence codec stores183 authenticated original texts within the
existing85MiB aggregate cap, without publishing Release assets.

## Remaining work

B3LYP still trails the same-allocation GPU4PySCF cold11.699543s by about0.168s
(1.44%). Its native warm is also slower. PBE0 reference cold165.394061s Direct /
30.152892s DF is anomalous and not evidence that those gaps are solved; the cause
is unmeasured. Additional B3LYP DF8.537541/8.701143s is a single paired point, not
balanced performance qualification or a streaming-driven DF default promotion.

Fresh cold-only attribution finds DPPP streaming~0.98s, PPPP~0.71–0.74s,
spatial-SPD low forces~0.69s and radial AO~0.48s, plus substantial Becke response.
Use these actual owners for the next work/data-movement change rather than
reopening rejected multi-gigabyte AO caches or widening weighted-Z for~0.1s.
Kernel sums cannot simply be subtracted from the complete endpoint.

## CI integration without another GPU campaign

The real source-executing Coulomb allocation probes need both the new device
property and its captured plan field in their external doubles. Unknown throughput
stays zero/incumbent; all1541 allocation/fallback cases pass after the two-field
repair. No production source or failure coverage is removed.

The hosted sm120 resource gate fails only its whole-manifest identity, because
the added streaming capabilities belong to other compiled profiles. Downloaded
artifact13623433085 from run42494598359 is SHA-256 authenticated. Its102 kernels,
resources, toolchain, flags, headers, shard and preprocessed identities match the
retained ledger exactly. The unchanged official measured-ledger constructor and
comparison accept a deliberate update of only `manifest_sha256`. Retain both the
actual failed report and the prior ledger so offline replay rechecks the equality,
rather than accepting an asserted boolean or disabling the gate. This is static
compilation/resource evidence, not an additional GPU scientific run.

## References

- Issue #2976 and PR #3051.
- `benchmarks/results/dft-streaming-master-20261109/` for new normal images;
  `benchmarks/results/dft-shared-rys-streaming-20261108/` for older images.
- `../../proposed/2026-11-08-dft-independent-master-qualification.md` for the split
  and `../compatibility/2026-11-08-dft-streaming-catalog-binding.md` for the strictly
  metadata-only Libxc catalog repair.
