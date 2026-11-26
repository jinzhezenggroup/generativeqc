# Decision: Use the existing Direct UHF quote for CLI run admission

Status: implemented
Date: 2026-11-25

The native CLI could quote a two-spin CPU HF fleet but rejected every UHF
`run --host-bytes` request before checking the quote. For all-electron Direct
CPU UHF, reuse the same inventory and effective-limit calculation before
starting the existing energy or energy-and-force endpoint. Do not add a second
UHF sizing formula or change the SCF or force calculation.

Keep fitted HF and ECP UHF quotes separate from this execution admission. The
host limit describes modeled capacity, not a provider lease, free memory, or
process RSS. Contextual validation and force capability remain with the prepared
owner. The CLI contract tests compare exact admission to uncapped execution and
check one-byte-short and excluded-domain refusals.
