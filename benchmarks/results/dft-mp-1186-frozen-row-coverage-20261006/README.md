# DFT-MP-v1 #1186 frozen FP64 row coverage

The generated `coverage.json` maps the 44 required LDA/PBE/r2SCAN FP64 rows in
`tools/dft_mp_v1/manifest.json` at contract SHA-256
`de6c847b1ed93e537c1422679ae3df53a4cca3cd13e7268483e419afbf2faa00`.
This baseline was generated from upstream master
`162b4d88f83860792205a206c84bf04c3057b070` with the issue-scoped mapper.
Each row carries the frozen input, changed-input and grid identities. `missing`
means that no full installed-production campaign receipt was supplied; it is
not a claim that the implementation cannot execute the row.

Generate the map from the repository root with:

```sh
python -m tools.dft_mp_v1.issue1186_coverage --output benchmarks/results/dft-mp-1186-frozen-row-coverage-20261006/coverage.json
```

Pass `--receipt PATH` to map a captured DFT-MP-v1 campaign. The tool delegates
campaign and per-row evidence checks to the existing validator before crediting
any PASS. A partial receipt leaves every other row visible as blocked or not-run.
Receipts changed by a live writer during validation are rejected.
This row-level report does not establish #1190 overall acceptance or scientific
review. The shared final gate still requires one exact merged source and an
independently reviewed raw receipt.

Retained evidence examined at this baseline:

- `benchmarks/results/issue172-r2scan3c-20260923/manifest.json` records an
  earlier H100 r2SCAN-3c campaign (`source_git_tree`
  `0e6350acbf193cde487c6772c2f783baf9cd86f2`) with small composite-method
  cases. Its manifest SHA-256 over the Git/LF bytes is
  `398069b03ce8bcf350608337efcecbe4312f6669907fb08105da26ce4161301c`.
  Its source, method, input and grid identities differ from this frozen contract.
- #1741's retained PBE0 and r2SCAN geometry regressions establish shared
  implementation behavior on real GPU hardware. The published PBE0 campaign
  uses a different method and its r2SCAN test summary is not a captured
  installed-production receipt for these exact frozen rows.
- #1186 comment `6006334590` independently reports AOT/resource/capacity
  coverage but no compatible frozen-row scientific receipts. The report keeps
  those 44 rows missing instead of promoting that qualification evidence.

No RTX 4090/H100/H200 evidence is relabeled as the original RTX 5090 hardware
contract. A versioned hardware amendment and corresponding shared validator
support are prerequisites to accepting a campaign on an alternate device.
