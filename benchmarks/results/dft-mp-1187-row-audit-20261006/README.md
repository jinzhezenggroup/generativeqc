# #1187 frozen hybrid FP64 row audit

Initial audit: official `master` `21e682d1e24de606b85c76f174ad65e82cec66d5`.
Validation base: official `master` `162b4d88f83860792205a206c84bf04c3057b070`
with this PR's mapper, catalog, and regression tests overlaid.
Contract: `de6c847b1ed93e537c1422679ae3df53a4cca3cd13e7268483e419afbf2faa00`.
Scope: 20 mandatory PBE0 and 6 mandatory versioned B3LYP strict-FP64 rows;
the two B3LYP/O2 rows are optional. This is an evidence map, not a scientific PASS.

Run from the checkout root with `PYTHONPATH=python:.`:

```text
python -m tools.dft_mp_v1.map_1187_hybrid_rows
python -m tools.dft_mp_v1.map_1187_hybrid_rows --receipt /absolute/campaign/receipt.json
```

The catalog names retained #1716 (`pbe0-def2-svp-20261003`) and #1741
(`pbe0-grid-reuse-20261003`) reports. The mapper hashes each report and checks
its method/spin, actual atom/AO count, original and changed coordinates,
GridSpec, basis pack, and SCF targets. The older reports are independent
implementation evidence. They use a 48x16x32 grid rather than the frozen
72x24x48 common grid; their basis identity, SCF targets and geometry also do
not fully match. They carry no installed-production receipt for this contract.
They therefore contribute zero eligible rows. The map lists every missing row
and records incompatible candidates without promoting them.

For a new campaign, the mapper counts a row only when the shared validator
validates its installed-production campaign and the individual raw result passes
the shared semantic check. Other missing or invalid rows remain explicit; the
shared validator's overall BLOCKED status does not erase a valid individual row.
Receipt bytes are compared before and after audit and row validation; drift
rejects that receipt. Multiple receipts may combine only with identical source,
library, AOT artifact, adapter, build record, conditions, and hardware identities.
Paths may differ, but hashes must match. Incompatible identities clear aggregate
coverage and remain visible in the individual audits. These row passes do not
certify the overall performance gate or #1190's reviewed final acceptance.
Receipt inputs are resolved and deduplicated before auditing, including relative
and symlink aliases; the audit key retains the first supplied spelling. A producer
replacing a repeated input between audits cannot hide a prior campaign identity.
A campaign `RUNNING`, a static capacity result, or
an unbound historical benchmark remains missing. No installed hybrid receipt
was found in the checked-in evidence at this source revision.

The shared `validate.audit` currently requires RTX 5090/sm_120 hardware.
The authorized RTX 4090/sm_89 or H100/H200/sm_90 route needs one versioned,
reviewed hardware amendment and matching validator/plan support before its
receipts can qualify. That shared contract work is outside this issue-scoped
mapper's ownership. The 20 GiB budget and all scientific gates remain intact.

Validation used Python 3.11.16 in the exclusive qz CPU checkout at the base
above. After the repeated-input repair on 2026-10-07, the mapper and shared
contract suites reported 54 passed and 4 skipped
(the skipped input-generator checks require RDKit). Ruff check and format check
and the evidence-retention check passed. Mapper, catalog, and test SHA-256 hashes
matched the local files. This validation did not compile native/CUDA code or run
a GPU qualification campaign.
The three real-receipt repeated-path, relative-alias, and symlink-alias regressions
failed before the repair and passed afterward.
