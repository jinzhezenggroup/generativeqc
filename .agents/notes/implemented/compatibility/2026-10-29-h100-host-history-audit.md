# Decision: separate historical HOST audits from live H100 numerical replay

Status: implemented
Date: 2026-10-29

## Problem

The WB97M-V H100 reader authenticated its actual 98 complete E/F returns and
independent references, but also required four superseded HOST preparation
capsules in every checkout. The resulting storage dependency prevented another
completed numerical report from fitting the unchanged 80 MiB evidence budget.

## Decision

Keep the complete actual runtime/scientific/control/source capsule and independent
oracles in-tree. Normal report v2 still recomputes every numerical, work, source,
release and population gate. It labels the optional historical HOST audit as
`NOT_RUN_HISTORICAL_AUDIT`, with its previously verified result explicitly recorded.

Use the existing Git snapshot restorer for the four HOST capsules and original
artifact inventory at protected ancestor `dd4fa29891e84f2a7be234c30c1dbc7a3fa2de89`.
All five original files were restored and verified before retirement. An explicit
`--host-history-root` audits those recovered bytes. The independent oracle can be
read from the current tree, under the original length and SHA checks; archived
HOST payloads have no missing-file fallback or implicit fetch.

## Invariants and evidence

- Retain all 98 actual returns, 72 formal returns, 18 formal processes and 10
  original NOT_RUN records. The failed pilot and unavailable HOST suite stay failed
  and unavailable. No timing, observable, work or scientific gate changes.
- Numerical replay requires neither Git history nor network access. Historical
  source recovery is a separate opt-in audit, not new device or scientific proof.
- Do not replace real scientific arrays with synthetic fixtures. Small synthetic
  HOST fixtures exercise damaged objects, duplicate members, missing recovery and
  changed reference bytes; the actual numerical report remains an executable test.
- Preserve original recovery identities and all existing storage limits. No new
  archive framework, external upload, GPU rerun or Release is introduced.

## Revisit when

Restore a historical HOST payload to the current tree only if a new active
scientific consumer needs its exact bytes offline. A hash/count-only storage test
or an old report alone is not that consumer.
