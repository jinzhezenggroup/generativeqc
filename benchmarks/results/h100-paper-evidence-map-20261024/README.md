# Incremental paper evidence map — 2026-10-24

Refs #2739, parent #2724. This is a CPU/offline historical evidence index and
reader, **not the final H100 supplement**. Both issues retain remaining work.
No oracle, native endpoint, compiler, remote command or GPU run is performed.

`sources.json` pins complete inventories of seven historical/delivered families
and two CPU oracle fixtures by full commit, Git blob/mode, bytes and SHA256.
A separate prepared #2725 contract adds one blob; it is not delivered evidence.
Its lossless `prepared-protocol-manifest.json.xz` snapshot verifies against the
original Git blob/SHA256, allowing offline replay without an unmerged sibling's
Git object. It is a read-only contract copy, not a second protocol publisher.
The reader uses Git object bytes, never Windows checkout text or moving refs.
`coverage-status.json` preserves eight original paper claims verbatim, their
exact paper lines and code-repository `path@commit` evidence, with proposed
wording separately identified. It records eleven pending H100 core cells and
formal-review exceptions on externally merged #2721/#2719: COMMENTED is not
APPROVED. Old families, manuscript, figures and freeze ledger are unchanged.

## Reproduce

From the repository root, Python 3.10+ with the pinned Git objects available:

```sh
python benchmarks/results/h100-paper-evidence-map-20261024/reader.py --check
python -m unittest discover -s benchmarks/results/h100-paper-evidence-map-20261024 -p test_reader.py
```

Without `--check`, the reader prints the regenerated report to stdout. It writes
no evidence files and executes no archived scripts. `compact-report.json` is
the retained output. Its integrity PASS concerns only this offline index and
replay; `h100_core_status` remains INCOMPLETE. Explicit integrity/numerical
checks remain active under ordinary Python, `-O` and `-OO`.

XZ decoding caps memory at 96 MiB, compressed inputs at 16 MiB and expanded
outputs at 32 MiB per object. Missing EOF, trailing/concatenated streams,
duplicate JSON keys, unsafe/duplicate raw paths and byte/hash/mode mismatches
are rejected. Original metadata is validated without restoring historical paths.

## Replayed scope

| Evidence | Raw inputs | Scope |
| --- | --- | --- |
| #2685, merge `334e7f48` | Independent PySCF receipts/logs and native energies | 72 Standard/Fast constituent/composite comparisons; 5e-8 Eh; C1/C3/C4/C5/C6/C7, QM7 indices 0/8/48/183/293/5604 |
| #2721, merge `59434367` | Eight displaced PySCF receipts and native force arrays | 48 projections at 5e-6 Eh/Bohr; 24 translation controls at 1e-6; C4/C7, one seed-709 direction each, steps 1e-4 and 3e-5 Bohr |
| #2719, merge `1c7a3e96` | CPU endpoint returns and frozen reference arrays | 12 constituent/composite comparisons; physical H2 DZ/TZ and synthetic water algebra; gates 1e-9 Eh and 2e-6 Eh/Bohr |

Every numerical maximum/comparison is regenerated in `compact-report.json`.
No old PASS flag substitutes for numerical recomputation. Oracle composite
energies are checked against the CBS formula before their original raw values
enter finite differences. `math.fsum` fixes the accumulation algorithm across
Python versions and may differ in final rounding bits
from historical NumPy dot/sum; this is not a precision/significance claim.
These replays supplement the original, stricter source-reconstruction,
independent-reference and lifecycle verifiers retained at immutable locators.

GPU arrays retain native base `976f756b391bc65edb79f76df1a8c3fe807518a7`, patch
SHA256 `5a1a974bab09e0fe31e6186e6ec2129abf26961113aafcd3d9c81616eb2d7cad`,
library `65a401ed3db8fb303a5146f134cdb227af78928affe27e1f7571c09f270112de`.
CPU arrays retain source `bbd41a42764e8d22d15c1348f70a633961d4dcf5`, library
`67ffc728d989014fa383faa8a784b09ab607a30e284fd404d4c556dbb37176b4`.
Neither qualifies a current-master installed or H100 build.

## Preserved gaps and negative evidence

- C4/C7 directions are not full-vector alkane checks. Conventional CPU water
  and the frozen DF water-algebra claim A7 are separate synthetic fixtures.
- C2 original displaced-energy/driver recovery (#2705) and original paired
  VV10 full-force recovery (#2706) remain UNAVAILABLE. Historical summaries
  stay indexed without manufacturing new acceptance.
- Complete pinned family inventories preserve failed C1 oracle attempts,
  runner diagnostics, initial incomplete comparisons, capacity/build/test
  failures and original raw modes. Historical PR CI timeouts are not erased.
- Campaign attempted/complete/qualified/rejected/unavailable counts remain null
  because this offline mapper has no accepted H100 campaign inventory.
  `imported: 0` counts this map only. Job success, an image, compilation or a
  smoke test cannot populate a qualified receipt row.
- No timing ratio or repeat variance is generated. Future ratios require
  qualified arm pairs, raw samples, median and range. CPU oracle runtime is
  provenance, not a denominator. RTX 5090 single cold observations and
  within-process warm calls remain distinct from independent repetitions.
- Core H100 Direct WB97M-V cannot update paper C7 **DF** WB97M-V. D3 Lambda
  causality needs an accepted conditional #2736 study, not Standard/Fast cost.
  #2734–#2738 remain conditional and were not started here.
- No universal H100/H200 ranking, hardware-only attribution, true-CBS accuracy
  or measured 32 GiB peak-memory claim follows.

## Protocol and paper handoff

#2725 owns `h100-paper-protocol-20261024/` and execution admission. Its fixed
prepared contract is `manifest.json@df5c44a81c5d73210a8d2f7f09ba4930d7f67210`
(69,646 Git-LF bytes, SHA256
`dd96c17e702101d21352deb4f867ab0fbb7bce66700222d49fc67bbdef23185e`).
It is **prepared/unmerged**, not upstream-delivered or H100-qualified.
Its interface is `generativeqc.h100-paper-protocol.v1`, with cells[].id/input/
input_sha256/arms, sources, references and normative schedule; attempts use
`generativeqc.h100-paper-attempt.v1`. The reader verifies the eleven owner IDs,
canonical input hashes and arms; it does not admit or schedule experiments,
change HF old-manifest comparability gaps, or redesign scientific gates/repeats.

Paper snapshot `294a0817a75f7bd2d66db49f24a35db42e015c40` still freezes archive
`7522994e2e2e477e11c41cb2c2dbe74a2ce600e9`. Results lines 77–81, 177–186 and
468–474, plus `figures/scripts/results_cbs_cost.py`, retain C2-only/unchecked
markers. Proposed A9 wording adds delivered energies and C4/C7 directions
without rewriting those old markers or figures. Applying proposals to the
paper, its freeze ledger or figures requires the separately authorized slice.
