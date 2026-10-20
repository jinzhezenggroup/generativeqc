# Fast warm and final-consumer qualification

`validation.json` records the unchanged energy/force gates and all-repeat errors.
`receipts.json.xz` retains original native timings, independent dense references,
old Fast baselines, build/cache identities, and superseded failed test logs.
Native and oracle producer revisions remain distinct. No receipt was relabeled.

On n1 RTX 5090, source `798f9ddd7` passes four independent frozen/moved GPU tests
and both 96-atom PBE/PBE0 cold/warm populations. Fast warm runs one strict SCF
iteration, without a mixed restart or independent endpoint audit. Same-build
Standard/Fast PBE warm is 6.776/6.709 s (a small difference); PBE0 is 8.340/7.842 s.
This does not qualify all Fast methods, DF, or universal superiority to GPU4PySCF.
